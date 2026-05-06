# bes_flow/compare_methods.py
#
# Benchmark comparison of five optical flow methods on the same test set:
#
#   1. PWCNet        (model.py)    — our SPWCNet-style model
#   2. BESFlowNetS   (model_s.py)  — our FlowNetS-style model
#   3. ODP                         — legacy strip-matching baseline
#   4. Farneback     (OpenCV)      — classical dense optical flow
#   5. RAFT-small    (torchvision) — pretrained DL baseline
#
# Usage
# ─────
#   python -m bes_flow.compare_methods \
#       --weights_pwc      checkpoints/pwc_best.pt \
#       --weights_flownets checkpoints/flownets_best.pt \
#       --cache            synthetic_data/dataset_well.h5 \
#       --output           outputs/comparison/
#
#   # Skip methods not yet ready:
#   python -m bes_flow.compare_methods \
#       --weights_pwc      checkpoints/pwc_best.pt \
#       --cache            synthetic_data/dataset_well.h5 \
#       --skip_flownets --skip_odp --skip_raft


import os
import time
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import CenteredNorm
import math

import torch
import torch.nn.functional as F

from bes_flow.config  import cfg
from bes_flow.model_pwcnet import PWCNet
from bes_flow.model_s import BESFlowNetS
from bes_flow.dataset import load_dataset_cache, BESDataset
from bes_flow.metrics import compute_all_metrics
from bes_flow.train   import predict_dataset
from bes_flow.odp import odp_chunk


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_pwc(weights_path, device):
    """Load PWCNet from a checkpoint."""
    model = PWCNet(
        max_displacement = cfg.max_displacement,
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  Loaded PWCNet ({n:,} params) {weights_path}")
    return model


def load_flownets(weights_path, device):
    """Load BESFlowNetS from a checkpoint."""
    model = BESFlowNetS().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  Loaded BESFlowNetS ({n:,} params) {weights_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm wrappers
# ─────────────────────────────────────────────────────────────────────────────
# Convention: every wrapper accepts framesA / framesB of shape
# (N, 1, H, W) float32 in [0, 1] and returns (N, 2, H, W) float32 in pixels
# with channel 0 = dx, channel 1 = dy.

def run_bes_model(model, dataset, device, batch_size=16):
    """
    Run a BES neural network on the test dataset.
    Uses predict_dataset() from train.py which reads frames directly from
    the dataset's numpy arrays without augmentation.

    Returns
    -------
    flows   : (N, 2, H, W) float32
    elapsed : float — wall-clock seconds spent in predict_dataset only
    """
    t0 = time.perf_counter()
    flows = predict_dataset(model, dataset, device, batch_size)
    elapsed = time.perf_counter() - t0
    return flows, elapsed


def run_farneback(framesA, framesB):
    """
    OpenCV Farneback dense optical flow.

    Parameters tuned for 64x64 BES images:
      pyr_scale=0.5, levels=3  — moderate pyramid depth for small images
      winsize=15               — large relative to image size for robustness
      poly_n=5, poly_sigma=1.2 — standard polynomial approximation
    """
    N     = len(framesA)
    H, W  = framesA.shape[2], framesA.shape[3]
    flows = np.zeros((N, 2, H, W), dtype=np.float32)

    elapsed = 0.0
    print("\n  Running Farneback...")
    for i in range(N):
        # Preprocessing: normalise to uint8
        fA = (framesA[i, 0] * 255).astype(np.uint8)
        fB = (framesB[i, 0] * 255).astype(np.uint8)

        t0 = time.perf_counter()
        flow_cv = cv2.calcOpticalFlowFarneback(
            fA, fB, flow=None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )   # (H, W, 2)  channel 0 = dx, channel 1 = dy
        elapsed += time.perf_counter() - t0

        flows[i, 0] = flow_cv[:, :, 0]
        flows[i, 1] = flow_cv[:, :, 1]

    return flows, elapsed


def run_raft_small(framesA, framesB, device, batch_size=16):
    """
    RAFT-small pretrained model from torchvision (>= 0.13).

    RAFT expects (B, 3, H, W) float tensors in [0, 1].  BES frames are
    single-channel, so the channel is repeated three times.
    """
    try:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    except ImportError:
        raise ImportError(
            "torchvision >= 0.13 is required for RAFT-small. "
            "Install with: pip install torchvision>=0.13"
        )

    print("\n  Loading RAFT-small (pretrained weights)...")
    raft = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device)
    raft.eval()

    N     = len(framesA)
    H, W  = framesA.shape[2], framesA.shape[3]
    H_up, W_up = 128, 128  # RAFT minimum resolution
    scale = H / H_up  # 0.5 — pixel rescaling factor

    flows = np.zeros((N, 2, H, W), dtype=np.float32)

    elapsed = 0.0
    print("\n  Running RAFT-small...")
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            # Preprocessing: channel repeat + upsample
            bA = torch.tensor(framesA[start:end]).repeat(1, 3, 1, 1).to(device)
            bB = torch.tensor(framesB[start:end]).repeat(1, 3, 1, 1).to(device)
            bA = F.interpolate(bA, size=(H_up, W_up), mode='bilinear',
                               align_corners=False)
            bB = F.interpolate(bB, size=(H_up, W_up), mode='bilinear',
                               align_corners=False)

            # RAFT returns a list of iterative flow estimates; take the last
            t0 = time.perf_counter()
            flow_predictions = raft(bA, bB)[-1]
            elapsed += time.perf_counter() - t0

            # Downsample back to 64x64 and rescale pixel values
            # Take into account flow scaling: 
            # A displacement of d px in 128x128 = d * (H / H_up) px in 64x64.
            flow_down = F.interpolate(flow_predictions, size=(H, W), mode='bilinear',
                                      align_corners=False)
            flow_down = flow_down * scale
            flows[start:end] = flow_down.cpu().numpy()

    del raft
    return flows, elapsed


def run_odp(framesA, framesB, nstep='default', smooth=15, mframe=2, mx='default', my='default'):
    """
    framesA, framesB shape (n_pairs, 1, H, W)
    flows output shape: (N, 2, H, W) float32 in pixels,
    channel 0 = dx, channel 1 = dy.

    """
    N, ny, nx  = framesA.shape[0], framesA.shape[2], framesA.shape[3]
    flows = np.zeros((N, 2, ny, nx), dtype=np.float32)

    # format inputs
    nstep_val = None if nstep == 'default' else int(nstep)
    mx_val = None if mx == 'default' else int(mx)
    my_val = None if my == 'default' else int(my)
    sm_param = int(smooth)
    m_frame = int(mframe)
    
    if nstep_val is None: nstep_val = int(2.0 * math.log(nx / 10.0) / math.log(2.0) + 0.5)
    if mx_val is None: mx_val = int((nx / 6.0) / 2 + 0.5) * 2 + 1
    if my_val is None: my_val = int((ny / 6.0) / 2 + 0.5) * 2 + 1
    
    elapsed = 0.0
    print("\n  Running ODP...")
    print(f"n_steps: {nstep_val} | smooth: {sm_param} | mframe: {m_frame} | mx: {mx_val} | my: {my_val}")
    # loop over image pairs
    for i in range(N):
        # Preprocessing: reshape to ODP convention
        img_slice = np.stack([framesA[i, 0], framesB[i, 0]], axis=0)   # [2, ny, nx]
        img_slice = np.transpose(img_slice, (2, 1, 0)).astype(np.float32)  # [nx, ny, 2]
        np.nan_to_num(img_slice, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        t0 = time.perf_counter()
        vx, vy = odp_chunk(img_slice, nstep_val, sm_param, m_frame, mx_val, my_val)
        elapsed += time.perf_counter() - t0

        # vx, vy shape: [nx, ny, 1] -> transpose to [1, ny, nx] -> squeeze
        flows[i, 0] = np.transpose(vx, (2, 1, 0))[0]
        flows[i, 1] = np.transpose(vy, (2, 1, 0))[0]

    return flows, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(all_results, all_times=None):
    """
    Side-by-side table: rows = metrics, columns = methods.
    Each cell shows  mean ± std.

    Parameters
    ----------
    all_results : dict  {method: metrics_dict}
    all_times   : dict  {method: total_wall_seconds} or None
        If provided, a 'ms/pair' row is appended at the bottom of the table.
    """
    methods = list(all_results.keys())
    col_w   = max(18, max(len(m) for m in methods) + 2)

    header = f"  {'Metric':<12}" + "".join(f"  {m:>{col_w}}" for m in methods)
    sep    = "=" * len(header)

    print()
    print(sep)
    print("  Algorithm comparison")
    print(sep)
    print(header)
    print("-" * len(header))

    display = [
        ('EPE',  'EPE',  'px',  1.0),
        ('rEPE', 'rEPE', '%',   100.0),
        ('AE',   'AE',   'deg', 1.0),
        ('Fl',   'Fl',   '%',   100.0),
        ('R_dx', 'R_dx', '',    1.0),
        ('R_dy', 'R_dy', '',    1.0),
    ]

    for key, label, unit, scale in display:
        row = f"  {label:<12}"
        for m in methods:
            v    = all_results[m][key] * scale
            cell = f"{v.mean():.3f}{unit} ±{v.std():.3f}{unit}"
            row += f"  {cell:>{col_w}}"
        print(row)

    # Speed row: ms per pair
    if all_times:
        print("-" * len(header))
        row = f"  {'ms/pair':<12}"
        for m in methods:
            if m in all_times:
                n_pairs = len(all_results[m]['EPE'])
                ms_pair = all_times[m] * 1000.0 / max(n_pairs, 1)
                cell    = f"{ms_pair:.2f} ms"
            else:
                cell = "N/A"
            row += f"  {cell:>{col_w}}"
        print(row)

    print(sep)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

# Fixed colour palette — one entry per method in the order they are registered
_METHOD_COLORS  = ['steelblue', 'darkorange', 'forestgreen',
                   'mediumpurple', 'crimson']
_QUIVER_COLORS  = ['lime', 'deepskyblue', 'lawngreen', 'lavender', 'turquoise', ]


def plot_metric_bars(all_results, all_times=None, output_dir=None):
    """
    Four accuracy bar charts (EPE, rEPE, AE, Fl) plus an optional speed chart
    (ms/pair), one bar per method.  Error bars show ± 1 std across test pairs.

    Parameters
    ----------
    all_results : dict  {method: metrics_dict}
    output_dir  : str
    all_times   : dict  {method: total_wall_seconds} or None
        When provided a fifth subplot showing evaluation speed (ms/pair) is
        added.  Lower is better.
    """
    methods = list(all_results.keys())
    colors  = _METHOD_COLORS[:len(methods)]

    display = [
        ('EPE',  'EPE  (px)',  1.0),
        ('rEPE', 'rEPE  (%)', 100.0),
        ('AE',   'AE  (deg)', 1.0),
        ('Fl',   'Fl  (%)',   100.0),
    ]

    n_panels = len(display) + (1 if all_times else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle('Algorithm comparison  —  mean ± std across test pairs',
                 fontsize=12, fontweight='bold')

    x = np.arange(len(methods))

    for ax, (key, label, scale) in zip(axes, display):
        means = [all_results[m][key].mean() * scale for m in methods]
        stds  = [all_results[m][key].std()  * scale for m in methods]

        ax.bar(x, means, 0.6, yerr=stds, color=colors,
               capsize=5, edgecolor='black', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=40, ha='right', fontsize=12)
        ax.set_ylabel(label);  ax.set_title(label)
        ax.grid(True, alpha=0.3, axis='y')

    # Speed panel — ms per pair, lower is better
    if all_times:
        ax = axes[len(display)]
        ms_per_pair = []
        for m in methods:
            if m in all_times:
                n_pairs = len(all_results[m]['EPE'])
                ms_per_pair.append(all_times[m] * 1000.0 / max(n_pairs, 1))
            else:
                ms_per_pair.append(float('nan'))

        bars = ax.bar(x, ms_per_pair, 0.6, color=colors,
                      edgecolor='black', alpha=0.85)
        # Annotate each bar with its value so exact numbers are readable
        for bar, val in zip(bars, ms_per_pair):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02,
                        f"{val:.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=40, ha='right', fontsize=12)
        ax.set_ylabel('ms / pair')
        ax.set_title('Speed  (ms/pair)')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if output_dir is not None:
        path = os.path.join(output_dir, 'comparison_metrics.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.show()
    plt.close('all')
    

def plot_comparison_examples(framesA, framesB, flows_gt, all_flows,
                             n_examples=3, output_dir=None):
    """
    Grid figure: one row per randomly chosen test pair.

    Columns:
      Frame A | Frame B + GT quiver | <one column per method>

    Error maps (last two columns) show pred - GT for the FIRST method only,
    with a shared CenteredNorm so the scale is comparable across all rows.
    To show error maps for every method, duplicate the error-column block.
    """
    methods       = list(all_flows.keys())
    quiver_colors = _QUIVER_COLORS[:len(methods)]

    n_pairs = len(framesA)
    rng     = np.random.default_rng(seed=0)
    indices = rng.choice(n_pairs, size=min(n_examples, n_pairs), replace=False)
    indices = [5, 10, 50, 100]

    H, W   = framesA.shape[2], framesA.shape[3]
    qs     = 8
    ys     = np.arange(qs // 2, H, qs)
    xs     = np.arange(qs // 2, W, qs)
    xx, yy = np.meshgrid(xs, ys)

    # frameA | frameB + GT | method_1 ... method_N
    n_cols = 2 + len(methods)
    n_rows = len(indices)

    fig1 = plt.figure(figsize=(3.2 * n_cols, 3.2 * n_rows))
    fig1.suptitle('Qualitative comparison  - random test pairs',
                  fontsize=12, fontweight='bold')
    gs1 = gridspec.GridSpec(n_rows, n_cols, figure=fig1,
                           hspace=0.35, wspace=0.25)
    
    fig2 = plt.figure(figsize=(3.2 * (n_cols-2), 3.2 * n_rows))
    fig2.suptitle('Qualitative comparison  - random test pairs',
                  fontsize=12, fontweight='bold')
    gs2 = gridspec.GridSpec(n_rows, 2*(n_cols-2), figure=fig2,
                           hspace=0.15, wspace=0.25)
    
    fig3 = plt.figure(figsize=(3.2 * (n_cols-2), 3.2 * n_rows))
    fig3.suptitle('Mean Vy flow comparison  - random test pairs',
                  fontsize=12, fontweight='bold')
    gs3 = gridspec.GridSpec(n_rows, (n_cols-2), figure=fig3,
                           hspace=0.35, wspace=0.25)

    for row, idx in enumerate(indices):
        fA = framesA[idx, 0]
        fB = framesB[idx, 0]
        # Common colour scale — both frames share the same vmin/vmax 
        vmin = min(fA.min(), fB.min())
        vmax = max(fA.max(), fB.max())

        gt = flows_gt[idx]

        # Relative signed errors for all methods — shared colour scale across the row
        diffs_vx = [all_flows[m][idx, 0] - gt[0] for m in methods]
        diffs_vy = [all_flows[m][idx, 1] - gt[1] for m in methods]
        vmax_x = 7 #max([np.abs(d).max() for d in diffs_vx])
        vmax_y = 7 #max([np.abs(d).max() for d in diffs_vy])
        
        norm_x = CenteredNorm(vcenter=0, halfrange=vmax_x)
        norm_y = CenteredNorm(vcenter=0, halfrange=vmax_y)

        col1 = 0
        col2 = 0
        col3 = 0
        i = 0

        # ── Frame A ──────────────────────────────────────────────────────
        ax = fig1.add_subplot(gs1[row, col1]);  col1 += 1
        ax.imshow(fA, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)
        if row == 0:  ax.set_title('Frame A', fontsize=10)
        ax.set_ylabel(f'pair {idx}', fontsize=10)
        ax.set_xticks([]);  ax.set_yticks([])

        # ── Frame B + GT quiver ───────────────────────────────────────────
        ax = fig1.add_subplot(gs1[row, col1]);  col1 += 1
        ax.imshow(fB, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)
        ax.quiver(xx, yy, gt[0][yy, xx], gt[1][yy, xx],
                  color='cyan', scale=60, scale_units='width',
                  width=0.005, headwidth=4)
        if row == 0:  ax.set_title('GT flow', fontsize=10)
        ax.set_xticks([]);  ax.set_yticks([])

        # ── One column per method ─────────────────────────────────
        for m, qcol in zip(methods, quiver_colors):
            pred = all_flows[m][idx]
            
            ax   = fig1.add_subplot(gs1[row, col1]);  col1 += 1
            ax.imshow(fB, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)
            ax.quiver(xx, yy, pred[0][yy, xx], pred[1][yy, xx],
                      color=qcol, scale=60, scale_units='width',
                      width=0.005, headwidth=4)
            if row == 0:  ax.set_title(m, fontsize=10)
            ax.set_xticks([]);  ax.set_yticks([])

            # ── Signed error maps for all methods ––––––––––––––––––––––––
            ax = fig2.add_subplot(gs2[row, col2]);  
            im = ax.imshow(diffs_vx[i], cmap='RdBu_r', origin='lower', norm=norm_x)
            fig2.colorbar(im, ax=ax, shrink=0.8, label='px')
            if row == 0: ax.set_title(m+'\npred-GT vx', fontsize=10)
            if col2 == 0: ax.set_ylabel(f'pair {idx}', fontsize=10)
            ax.set_xticks([]);  ax.set_yticks([])
            col2 += 1

            ax = fig2.add_subplot(gs2[row, col2])
            im = ax.imshow(diffs_vy[i], cmap='RdBu_r', origin='lower', norm=norm_y)
            fig2.colorbar(im, ax=ax, shrink=0.8, label='px')
            if row == 0:  ax.set_title(m+'\npred-GT vy', fontsize=10)
            ax.set_xticks([]);  ax.set_yticks([])
            col2 += 1
            i += 1

            # ––––– Mean Vy flow vs GT for all methods ––––––––––––––––––––––––
            ax   = fig3.add_subplot(gs3[row, col3])
            ax.plot(np.mean(gt[1], axis=0), color='k', lw=2, label='GT')
            ax.plot(np.mean(pred[1], axis=0), color='r', lw=2, label='Pred')
            if row == 0:  ax.set_title(m, fontsize=10)
            if row == n_rows-1: ax.set_xlabel('x (px)', fontsize=10)
            if col3 == 0: 
                ax.legend()
                ax.set_ylabel(f'<Vy> \npair {idx}', fontsize=10)
            col3 += 1

    if output_dir is not None:
        path = os.path.join(output_dir, 'comparison_examples.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.show()
    plt.close('all')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Compare five optical flow methods on the BES test set'
    )

    # Dataset
    parser.add_argument('--cache',  required=True,
                        help='HDF5 dataset cache from make_datasets()')
    parser.add_argument('--output', default='outputs/comparison/',
                        help='Directory for figures and results')
    parser.add_argument('--batch_size', type=int, default=16)

    # Neural net weights (each independently optional)
    parser.add_argument('--weights_pwc',
                        default=None,
                        help='Checkpoint for PWCtNet')
    parser.add_argument('--weights_flownets',
                        default=None,
                        help='Checkpoint for BESFlowNetS')

    # Skip flags
    parser.add_argument('--skip_pwc',  action='store_true',
                        help='Skip PWCNet')
    parser.add_argument('--skip_flownets', action='store_true',
                        help='Skip BESFlowNetS')
    parser.add_argument('--skip_odp',      action='store_true',
                        help='Skip ODP (placeholder not yet implemented)')
    parser.add_argument('--skip_farneback',action='store_true',
                        help='Skip Farneback')
    parser.add_argument('--skip_raft',     action='store_true',
                        help='Skip RAFT-small')

    args   = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    os.makedirs(args.output, exist_ok=True)

    # ── Load test set ─────────────────────────────────────────────────────
    print(f"Loading test set from cache: {args.cache}")
    (*_, test_A, test_B, test_flows, metadata) = load_dataset_cache(args.cache)
    print(f"  Test pairs : {len(test_A)}")
    print(f"  Flow type  : {metadata['flow_type']}")
    print(f"  Max shift  : {metadata['max_shift']} px\n")

    test_dataset = BESDataset(test_A, test_B, test_flows, augment=False)
    flows_gt     = test_flows   # (N, 2, H, W)

    # ── Run methods in a fixed display order ──────────────────────────────
    all_flows = {}
    all_times = {}   # {method: algorithm-only wall-clock seconds}

    # 1. PWCNet
    if not args.skip_pwc:
        if args.weights_pwc is None:
            print("\n  [PWC] --weights_pwc not provided — skipping")
        else:
            print("\nPWCNet:")
            model_s = load_pwc(args.weights_pwc, device)
            all_flows['PWC'], all_times['PWC'] = run_bes_model(
                model_s, test_dataset, device, args.batch_size
            )
            del model_s

    # 2. BESFlowNetS
    if not args.skip_flownets:
        if args.weights_flownets is None:
            print("\n  [FlowNetS] --weights_flownets not provided — skipping")
        else:
            print("\nBESFlowNetS:")
            model_f = load_flownets(args.weights_flownets, device)
            all_flows['FlowNetS'], all_times['FlowNetS'] = run_bes_model(
                model_f, test_dataset, device, args.batch_size
            )
            del model_f

    # 3. ODP
    if not args.skip_odp:
        all_flows['ODP'], all_times['ODP'] = run_odp(test_A, test_B)

    # 4. Farneback
    if not args.skip_farneback:
        all_flows['Farneback'], all_times['Farneback'] = run_farneback(test_A, test_B)

    # 5. RAFT-small
    if not args.skip_raft:
        all_flows['RAFT-small'], all_times['RAFT-small'] = run_raft_small(
            test_A, test_B, device, args.batch_size
        )

    if not all_flows:
        print("No methods were run — nothing to compare.")
        raise SystemExit(0)

    # ── Metrics ───────────────────────────────────────────────────────────
    print("\nComputing metrics...")
    all_results = {}
    for method, flows_pred in all_flows.items():
        print(f"  {method}")
        all_results[method] = compute_all_metrics(flows_pred, flows_gt)

    # ── Summary table ─────────────────────────────────────────────────────
    print_comparison_table(all_results, all_times)

    # ── Figures ───────────────────────────────────────────────────────────
    print("Plotting figures...")
    plot_metric_bars(all_results, all_times)
    plot_comparison_examples(test_A, test_B, flows_gt, all_flows)

    print(f"\nDone. " ) 
