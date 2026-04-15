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
#       --weights_pwc.     checkpoints/pwc_best.pt \
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
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"\n  Loaded PWCNet  ({n:,} params)  ← {weights_path}")
    return model


def load_flownets(weights_path, device):
    """Load BESFlowNetS from a checkpoint."""
    model = BESFlowNetS().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"\n  Loaded BESFlowNetS             ({n:,} params)  ← {weights_path}")
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
    """
    return predict_dataset(model, dataset, device, batch_size)


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

    print("\n  Running Farneback...")
    for i in range(N):
        fA = (framesA[i, 0] * 255).astype(np.uint8)
        fB = (framesB[i, 0] * 255).astype(np.uint8)

        flow_cv = cv2.calcOpticalFlowFarneback(
            fA, fB, flow=None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )   # (H, W, 2)  channel 0 = dx, channel 1 = dy

        flows[i, 0] = flow_cv[:, :, 0]
        flows[i, 1] = flow_cv[:, :, 1]

    return flows


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

    print("\n  Running RAFT-small...")
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            # (B, 1, H, W) → (B, 3, H, W) by repeating the single channel
            bA = torch.tensor(framesA[start:end]).repeat(1, 3, 1, 1).to(device)
            bB = torch.tensor(framesB[start:end]).repeat(1, 3, 1, 1).to(device)

            bA = F.interpolate(bA, size=(H_up, W_up), mode='bilinear',
                               align_corners=False)
            bB = F.interpolate(bB, size=(H_up, W_up), mode='bilinear',
                               align_corners=False)

            # RAFT returns a list of iterative flow estimates; take the last
            flow_predictions = raft(bA, bB)[-1]
            # Downsample back to 64x64 and rescale pixel values
            # Take into account flow scaling: 
            # A displacement of d px in 128x128 = d * (H / H_up) px in 64x64.
            flow_down = F.interpolate(flow_predictions, size=(H, W), mode='bilinear',
                                      align_corners=False)
            flow_down = flow_down * scale
            flows[start:end] = flow_down.cpu().numpy()

    del raft
    return flows


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
    
    print("\n  Running ODP...")
    print(f"n_steps: {nstep_val} | smooth: {sm_param} | mframe: {m_frame} | mx: {mx_val} | my: {my_val}")
    # loop over image pairs
    for i in range(N):
        # Build [nx, ny, 2] slice for this pair
        img_slice = np.stack([framesA[i, 0], framesB[i, 0]], axis=0)   # [2, ny, nx]
        img_slice = np.transpose(img_slice, (2, 1, 0)).astype(np.float32)  # [nx, ny, 2]
        np.nan_to_num(img_slice, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        vx, vy = odp_chunk(img_slice, nstep_val, sm_param, m_frame, mx_val, my_val)
    
        # vx, vy shape: [nx, ny, 1] -> transpose to [1, ny, nx] -> squeeze
        flows[i, 0] = np.transpose(vx, (2, 1, 0))[0]
        flows[i, 1] = np.transpose(vy, (2, 1, 0))[0]

    return flows


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(all_results):
    """
    Side-by-side table: rows = metrics, columns = methods.
    Each cell shows  mean ± std.
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

    print(sep)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

# Fixed colour palette — one entry per method in the order they are registered
_METHOD_COLORS  = ['steelblue', 'darkorange', 'forestgreen',
                   'mediumpurple', 'crimson']
_QUIVER_COLORS  = ['yellow', 'lime', 'magenta', 'orange', 'deepskyblue']


def plot_metric_bars(all_results, output_dir):
    """
    Four grouped bar charts (EPE, rEPE, AE, Fl), one bar per method.
    Error bars show ± 1 std across test pairs.
    """
    methods = list(all_results.keys())
    colors  = _METHOD_COLORS[:len(methods)]

    display = [
        ('EPE',  'EPE  (px)',  1.0),
        ('rEPE', 'rEPE  (%)', 100.0),
        ('AE',   'AE  (deg)', 1.0),
        ('Fl',   'Fl  (%)',   100.0),
    ]

    fig, axes = plt.subplots(1, len(display), figsize=(4 * len(display), 5))
    fig.suptitle('Algorithm comparison  —  mean ± std across test pairs',
                 fontsize=13, fontweight='bold')

    x = np.arange(len(methods))

    for ax, (key, label, scale) in zip(axes, display):
        means = [all_results[m][key].mean() * scale for m in methods]
        stds  = [all_results[m][key].std()  * scale for m in methods]

        ax.bar(x, means, 0.6, yerr=stds, color=colors,
               capsize=5, edgecolor='black', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(label);  ax.set_title(label)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, 'comparison_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close('all')
    print(f"Saved: {path}")


def plot_comparison_examples(framesA, framesB, flows_gt, all_flows,
                             output_dir, n_examples=3):
    """
    Grid figure: one row per randomly chosen test pair.

    Columns:
      Frame A | Frame B + GT quiver | <one column per method> | vx error | vy error

    Error maps (last two columns) show pred - GT for the FIRST method only,
    with a shared CenteredNorm so the scale is comparable across all rows.
    To show error maps for every method, duplicate the error-column block.
    """
    methods       = list(all_flows.keys())
    quiver_colors = _QUIVER_COLORS[:len(methods)]

    n_pairs = len(framesA)
    rng     = np.random.default_rng(seed=0)
    indices = rng.choice(n_pairs, size=min(n_examples, n_pairs), replace=False)

    H, W   = framesA.shape[2], framesA.shape[3]
    qs     = 8
    ys     = np.arange(qs // 2, H, qs)
    xs     = np.arange(qs // 2, W, qs)
    xx, yy = np.meshgrid(xs, ys)

    # frameA | GT | method_1 ... method_N | err_vx | err_vy
    n_cols = 2 + len(methods) + 2
    n_rows = len(indices)

    fig = plt.figure(figsize=(3.2 * n_cols, 3.2 * n_rows))
    fig.suptitle('Qualitative comparison  —  random test pairs',
                 fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.35, wspace=0.25)

    for row, idx in enumerate(indices):
        fA = framesA[idx, 0]
        fB = framesB[idx, 0]
        gt = flows_gt[idx]

        # Signed errors for all methods — shared colour scale across the row
        diffs_vx = [all_flows[m][idx, 0] - gt[0] for m in methods]
        diffs_vy = [all_flows[m][idx, 1] - gt[1] for m in methods]
        vmax     = max(
            max(np.abs(d).max() for d in diffs_vx),
            max(np.abs(d).max() for d in diffs_vy),
        )
        norm = CenteredNorm(vcenter=0, halfrange=vmax)

        col = 0

        # ── Frame A ──────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[row, col]);  col += 1
        ax.imshow(fA, cmap='inferno', origin='upper')
        if row == 0:  ax.set_title('Frame A', fontsize=8)
        ax.set_ylabel(f'pair {idx}', fontsize=7)
        ax.set_xticks([]);  ax.set_yticks([])

        # ── Frame B + GT quiver ───────────────────────────────────────────
        ax = fig.add_subplot(gs[row, col]);  col += 1
        ax.imshow(fB, cmap='inferno', origin='upper')
        ax.quiver(xx, yy, gt[0][yy, xx], -gt[1][yy, xx],
                  color='cyan', scale=60, scale_units='width',
                  width=0.005, headwidth=4)
        if row == 0:  ax.set_title('GT flow', fontsize=8)
        ax.set_xticks([]);  ax.set_yticks([])

        # ── One quiver column per method ─────────────────────────────────
        for m, qcol in zip(methods, quiver_colors):
            pred = all_flows[m][idx]
            ax   = fig.add_subplot(gs[row, col]);  col += 1
            ax.imshow(fB, cmap='inferno', origin='upper')
            ax.quiver(xx, yy, pred[0][yy, xx], -pred[1][yy, xx],
                      color=qcol, scale=60, scale_units='width',
                      width=0.005, headwidth=4)
            if row == 0:  ax.set_title(m, fontsize=8)
            ax.set_xticks([]);  ax.set_yticks([])

        # ── Signed error maps for first method (representative) ───────────
        first = methods[0]
        ax = fig.add_subplot(gs[row, col]);  col += 1
        im = ax.imshow(diffs_vx[0], cmap='RdBu_r', origin='upper', norm=norm)
        plt.colorbar(im, ax=ax, shrink=0.8, label='px')
        if row == 0:  ax.set_title(f'{first}\npred−GT vx', fontsize=8)
        ax.set_xticks([]);  ax.set_yticks([])

        ax = fig.add_subplot(gs[row, col]);  col += 1
        im = ax.imshow(diffs_vy[0], cmap='RdBu_r', origin='upper', norm=norm)
        plt.colorbar(im, ax=ax, shrink=0.8, label='px')
        if row == 0:  ax.set_title(f'{first}\npred−GT vy', fontsize=8)
        ax.set_xticks([]);  ax.set_yticks([])

    path = os.path.join(output_dir, 'comparison_examples.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close('all')
    print(f"Saved: {path}")


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

    # 1. PWCNet
    if not args.skip_pwc:
        if args.weights_pwc is None:
            print("  [PWC] --weights_pwc not provided — skipping")
        else:
            print("PWCNet:")
            model_s = load_pwc(args.weights_pwc, device)
            all_flows['PWC'] = run_bes_model(
                model_s, test_dataset, device, args.batch_size
            )
            del model_s

    # 2. BESFlowNetS
    if not args.skip_flownets:
        if args.weights_flownets is None:
            print("  [FlowNetS] --weights_flownets not provided — skipping")
        else:
            print("BESFlowNetS:")
            model_f = load_flownets(args.weights_flownets, device)
            all_flows['FlowNetS'] = run_bes_model(
                model_f, test_dataset, device, args.batch_size
            )
            del model_f

    # 3. ODP
    if not args.skip_odp:
        all_flows['ODP'] = run_odp(test_A, test_B)

    # 4. Farneback
    if not args.skip_farneback:
        all_flows['Farneback'] = run_farneback(test_A, test_B)

    # 5. RAFT-small
    if not args.skip_raft:
        all_flows['RAFT-small'] = run_raft_small(test_A, test_B, device,
                                                  args.batch_size)

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
    print_comparison_table(all_results)

    # ── Figures ───────────────────────────────────────────────────────────
    print("Saving figures...")
    plot_metric_bars(all_results, args.output)
    plot_comparison_examples(test_A, test_B, flows_gt, all_flows, args.output)

    print(f"\nDone. All outputs saved to {args.output}")
