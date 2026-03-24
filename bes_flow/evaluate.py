# bes_flow/evaluate.py
#
# Structured evaluation of a trained model against
# synthetic test data with known ground-truth velocity fields.
#
# Usage
# ─────
# From the command line:
#   python -m bes_flow.evaluate \
#       --data      data/raw/test_frames.npy \
#       --weights   checkpoints/model_zonal_final.pt \
#       --output    outputs/evaluation/


import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import CenteredNorm

from bes_flow.config  import cfg
from bes_flow.predict import load_model
from bes_flow.metrics import evaluate as compute_metrics
from bes_flow.dataset import generate_dataset


def generate_test_set(frames, flow_type, max_shift,
                      noise_std, n_pairs=200, seed=42):
    """
    Generate a fixed synthetic test set from real BES frames.

    A thin wrapper around pregenerate_dataset() that adds seed management
    so the test set is always identical across runs. The seed is saved and
    restored so calling this function never affects the random state of the
    training loop or any other caller.

    Parameters
    ----------
    frames           : (N, H, W) float array -- real BES frames
    flow_type        : 'smooth', 'modes', or 'zonal'
    max_shift : float -- peak displacement in pixels
    noise_std        : float -- sensor noise std
    n_pairs          : int   -- number of test pairs to generate
    seed             : int   -- fixed random seed for reproducibility

    Returns
    -------
    framesA  : (M, 1, H, W) float32 -- input frame A
    framesB  : (M, 1, H, W) float32 -- warped frame B
    flows_gt : (M, 2, H, W) float32 -- ground-truth flow
               where M = len(frames) * n_pairs_per_frame
    """
    # in case the frames array has less frames than n_pairs
    n_pairs_per_frame = max(1, n_pairs // len(frames))

    rng_state = np.random.get_state()
    np.random.seed(seed)

    framesA, framesB, flows_gt = generate_dataset(
        frames,
        n_pairs_per_frame = n_pairs_per_frame,
        max_shift  = max_shift,
        noise_std  = noise_std,
        flow_type  = flow_type,
    )

    np.random.set_state(rng_state)
    return framesA, framesB, flows_gt


def predict_test_set(model, framesA, framesB, device, batch_size=16):
    """
    Run the model on the full test set and return predicted flow fields.

    Parameters
    ----------
    model            : trained SiameseDisplacementNet in eval mode
    framesA, framesB : (n_pairs, 1, H, W) float32
    device           : torch.device
    batch_size       : int

    Returns
    -------
    flows_pred : (n_pairs, 2, H, W) float32
    """
    model.eval()
    n          = len(framesA)
    H, W       = framesA.shape[2], framesA.shape[3]
    flows_pred = np.zeros((n, 2, H, W), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end    = min(start + batch_size, n)
            batchA = torch.tensor(framesA[start:end]).to(device)
            batchB = torch.tensor(framesB[start:end]).to(device)
            pred   = model(batchA, batchB)
            flows_pred[start:end] = pred.cpu().numpy()

    return flows_pred


def compute_all_metrics(flows_pred, flows_gt):
    """
    Compute metrics for every pair in the test set.

    Returns a dict where each value is a 1-D array of length n_pairs,
    making it easy to compute summary statistics (mean, std, percentiles)
    and to plot distributions.

    Parameters
    ----------
    flows_pred : (n_pairs, 2, H, W)
    flows_gt   : (n_pairs, 2, H, W)

    Returns
    -------
    results : dict with keys
        'EPE'    -- per-pair mean end-point error (pixels)
        'rEPE'   -- per-pair mean relative EPE (fraction)
        'AE'     -- per-pair mean angular error (degrees)
        'Fl'     -- per-pair outlier fraction
        'R_dx'   -- per-pair Pearson R on dx component
        'R_dy'   -- per-pair Pearson R on dy component
        'gt_mag' -- per-pair mean ground-truth displacement magnitude
    """
    n = len(flows_pred)
    keys    = ['EPE', 'rEPE', 'AE', 'Fl', 'R_dx', 'R_dy', 'gt_mag']
    results = {k: np.zeros(n, dtype=np.float32) for k in keys}

    for i in range(n):
        m = compute_metrics(flows_pred[i], flows_gt[i])
        results['EPE'][i]    = m['mean_EPE']
        results['rEPE'][i]   = m['mean_rEPE']
        results['AE'][i]     = m['mean_AE']
        results['Fl'][i]     = m['Fl']
        results['R_dx'][i]   = m['R_dx']
        results['R_dy'][i]   = m['R_dy']
        results['gt_mag'][i] = np.sqrt(
            flows_gt[i, 0] ** 2 + flows_gt[i, 1] ** 2
        ).mean()

    return results


def print_summary(results, flow_type, max_shift):
    """
    Print a formatted summary table of all metrics.

    For each metric we report mean, std, median, and the 95th percentile.
    The 95th percentile is more informative than the maximum for noisy
    data — it tells you how bad the worst 5% of predictions are without
    being dominated by a single outlier pair.
    """
    print()
    print("=" * 65)
    print(f"  Evaluation summary")
    print(f"  Flow type        : {flow_type}")
    print(f"  Max displacement : {max_shift} px")
    print(f"  Test pairs       : {len(results['EPE'])}")
    print("=" * 65)
    print(f"  {'Metric':<12}  {'Mean':>8}  {'Std':>8}  "
          f"{'Median':>8}  {'P95':>8}")
    print("-" * 65)

    # Display format: (key, label, unit, scale)
    # scale converts to a readable unit if needed
    display = [
        ('EPE',  'EPE',    'px',    1.0),
        ('rEPE', 'rEPE',   '%',   100.0),   # display as percentage
        ('AE',   'AE',     'deg',   1.0),
        ('Fl',   'Fl',     '%',   100.0),   # display as percentage
        ('R_dx', 'R_dx',   '',      1.0),
        ('R_dy', 'R_dy',   '',      1.0),
    ]

    for key, label, unit, scale in display:
        v = results[key] * scale
        print(f"  {label:<12}  {v.mean():>7.3f}{unit}  "
              f"{v.std():>7.3f}{unit}  "
              f"{np.median(v):>7.3f}{unit}  "
              f"{np.percentile(v, 95):>7.3f}{unit}")

    print("=" * 65)
    print()


def plot_metric_distributions(results, flow_type, output_dir):
    """
    Plot histograms of EPE, rEPE, AE, and Fl across all test pairs.

    A well-trained network should show narrow, left-skewed distributions
    concentrated near zero. A broad or right-skewed distribution indicates
    the network struggles with some subset of pairs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Metric distributions  |  flow: {flow_type}  |  "
        f"n={len(results['EPE'])} pairs",
        fontsize=13, fontweight='bold'
    )

    plots = [
        ('EPE',  'End-Point Error  (px)',       'steelblue',   axes[0, 0]),
        ('rEPE', 'Relative EPE  (%)',            'darkorange',  axes[0, 1]),
        ('AE',   'Angular Error  (deg)',         'forestgreen', axes[1, 0]),
        ('Fl',   'Outlier Rate  (%)',            'crimson',     axes[1, 1]),
    ]

    scale = {'EPE': 1.0, 'rEPE': 100.0, 'AE': 1.0, 'Fl': 100.0}

    for key, label, color, ax in plots:
        v = results[key] * scale[key]
        ax.hist(v, bins=30, color=color, alpha=0.75, edgecolor='white')
        ax.axvline(v.mean(),   color='black', linewidth=2.0,
                   linestyle='-',  label=f'Mean={v.mean():.3f}')
        ax.axvline(np.median(v), color='black', linewidth=1.5,
                   linestyle='--', label=f'Median={np.median(v):.3f}')
        ax.set_xlabel(label)
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'metric_distributions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {path}")


def plot_epe_vs_displacement(results, output_dir):
    """
    Scatter plot of mean EPE vs ground-truth displacement magnitude.

    This reveals whether the network struggles more with large or small
    displacements. For a well-calibrated network:
        - EPE should scale roughly linearly with displacement magnitude
          (constant relative error)
        - There should be no systematic bias (EPE should not depend on
          the direction of the flow)

    A flat EPE curve suggests the network is dominated by a fixed
    positional error regardless of displacement — typical of networks
    that haven't converged on the correlation layer.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        results['gt_mag'], results['EPE'],
        c=results['rEPE'] * 100, cmap='viridis',
        s=20, alpha=0.6
    )
    fig.colorbar(sc, ax=ax, label='rEPE  (%)')

    # Bin and plot the mean EPE per displacement bin
    bins      = np.linspace(0, results['gt_mag'].max(), 10)
    bin_idx   = np.digitize(results['gt_mag'], bins)
    bin_means = [results['EPE'][bin_idx == k].mean()
                 for k in range(1, len(bins))
                 if (bin_idx == k).sum() > 0]
    bin_centres = [(bins[k] + bins[k+1]) / 2
                   for k in range(len(bins) - 1)
                   if (bin_idx == k+1).sum() > 0]
    ax.plot(bin_centres, bin_means, color='red', linewidth=2,
            marker='o', markersize=5, label='Bin mean EPE')

    ax.set_xlabel('Mean GT displacement magnitude  (px)')
    ax.set_ylabel('Mean EPE  (px)')
    ax.set_title('EPE vs displacement magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'epe_vs_displacement.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {path}")


def plot_spatial_error_map(flows_pred, flows_gt, output_dir):
    """
    Plot the mean EPE at each spatial pixel position, averaged across
    all test pairs.

    This reveals systematic spatial biases — e.g. if the network makes
    larger errors near the image borders (where padding artefacts from
    the correlation layer can appear) or in specific regions that are
    consistently harder to track.

    A uniform map is ideal. Any spatial structure in the error map
    indicates a systematic failure mode worth investigating.
    """
    # Compute per-pixel EPE for each pair, then average over pairs
    diff    = flows_pred - flows_gt                         # (n, 2, H, W)
    epe_map = np.sqrt((diff ** 2).sum(axis=1))             # (n, H, W)
    mean_epe_map = epe_map.mean(axis=0)                    # (H, W)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Spatial error map  (mean EPE per pixel)',
                 fontsize=13, fontweight='bold')

    # Mean EPE map
    im0 = axes[0].imshow(mean_epe_map, cmap='hot', origin='upper', vmin=0)
    axes[0].set_title('Mean EPE  (px)')
    axes[0].set_xlabel('x  (px)')
    axes[0].set_ylabel('y  (px)')
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    # Standard deviation of EPE across pairs — shows where predictions
    # are inconsistent (high variance = network is uncertain here)
    std_epe_map = epe_map.std(axis=0)
    im1 = axes[1].imshow(std_epe_map, cmap='hot', origin='upper', vmin=0)
    axes[1].set_title('Std of EPE  (px)  — prediction uncertainty')
    axes[1].set_xlabel('x  (px)')
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'spatial_error_map.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {path}")


def plot_qualitative_examples(framesA, framesB, flows_pred, flows_gt,
                              results, output_dir, n_examples=4):
    """
    Show n_examples frame pairs with predicted and ground-truth flow overlaid.

    Selects pairs from different EPE quartiles so you see both the best
    and worst predictions — this is more informative than random sampling.

    Each row shows: Frame A | Frame B | GT quiver | Pred quiver | EPE map
    """
    n = len(flows_pred)

    # Pick one pair per EPE quartile (best, Q1, Q3, worst)
    epe_sorted  = np.argsort(results['EPE'])
    step        = max(1, n // n_examples)
    indices     = [epe_sorted[min(i * step, n-1)] for i in range(n_examples)]
    quartile_labels = ['Best', 'Q1', 'Q2 / Median', 'Q3', 'Worst']

    fig = plt.figure(figsize=(20, 4 * n_examples))
    fig.suptitle('Qualitative examples  —  one per EPE quartile',
                 fontsize=13, fontweight='bold')

    quiver_step = 8
    H, W = framesA.shape[2], framesA.shape[3]
    ys   = np.arange(quiver_step // 2, H, quiver_step)
    xs   = np.arange(quiver_step // 2, W, quiver_step)
    xx, yy = np.meshgrid(xs, ys)

    for row, idx in enumerate(indices):
        fA   = framesA[idx, 0]
        fB   = framesB[idx, 0]
        pred = flows_pred[idx]
        gt   = flows_gt[idx]
        epe  = np.sqrt(((pred - gt) ** 2).sum(axis=0))

        label = quartile_labels[min(row, len(quartile_labels)-1)]
        epe_val = results['EPE'][idx]

        gs = gridspec.GridSpec(n_examples, 5, figure=fig,
                               hspace=0.35, wspace=0.3)

        # Frame A
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(fA, cmap='inferno', origin='upper')
        ax0.set_ylabel(f'{label}\nEPE={epe_val:.3f}px', fontsize=9)
        if row == 0:
            ax0.set_title('Frame A')
        ax0.set_xticks([])

        # Frame B
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(fB, cmap='inferno', origin='upper')
        if row == 0:
            ax1.set_title('Frame B')
        ax1.set_xticks([])
        ax1.set_yticks([])

        # GT quiver on frame A
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(fA, cmap='inferno', origin='upper')
        ax2.quiver(xx, yy, gt[0][yy, xx], -gt[1][yy, xx],
                   color='cyan', scale=60, scale_units='width',
                   width=0.005, headwidth=4)
        if row == 0:
            ax2.set_title('GT flow')
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Predicted quiver on frame A
        ax3 = fig.add_subplot(gs[row, 3])
        ax3.imshow(fA, cmap='inferno', origin='upper')
        ax3.quiver(xx, yy, pred[0][yy, xx], -pred[1][yy, xx],
                   color='yellow', scale=60, scale_units='width',
                   width=0.005, headwidth=4)
        if row == 0:
            ax3.set_title('Predicted flow')
        ax3.set_xticks([])
        ax3.set_yticks([])

        # EPE map
        ax4 = fig.add_subplot(gs[row, 4])
        im  = ax4.imshow(epe, cmap='hot', origin='upper', vmin=0)
        plt.colorbar(im, ax=ax4, shrink=0.8, label='px')
        if row == 0:
            ax4.set_title('EPE map')
        ax4.set_xticks([])
        ax4.set_yticks([])

    path = os.path.join(output_dir, 'qualitative_examples.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {path}")


def plot_cross_flow_comparison(model, frames, device, cfg, output_dir):
    """
    Evaluate the model on all three flow types and plot a bar chart
    comparing mean EPE and mean rEPE across flow types.

    This answers: does the model generalise across different flow
    structures, or does it only work well on the flow type it was
    trained on? Ideally performance should be similar across all three,
    which would indicate the network has learned general displacement
    estimation rather than overfitting to a specific flow pattern.
    """
    flow_types = ['smooth', 'modes', 'zonal']
    epe_means  = []
    epe_stds   = []
    repe_means = []

    for ft in flow_types:
        fA, fB, gt = generate_test_set(
            frames, ft,
            max_shift = cfg.max_shift,
            noise_std        = cfg.noise_std,
            n_pairs          = 50,
        )
        pred    = predict_test_set(model, fA, fB, device)
        metrics = compute_all_metrics(pred, gt)
        epe_means.append(metrics['EPE'].mean())
        epe_stds.append(metrics['EPE'].std())
        repe_means.append(metrics['rEPE'].mean() * 100)

    x   = np.arange(len(flow_types))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Performance across flow types',
                 fontsize=13, fontweight='bold')

    colors = ['steelblue', 'darkorange', 'forestgreen']

    axes[0].bar(x, epe_means, yerr=epe_stds, color=colors,
                capsize=5, edgecolor='black', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(flow_types)
    axes[0].set_ylabel('Mean EPE  (px)')
    axes[0].set_title('End-Point Error')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x, repe_means, color=colors,
                edgecolor='black', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(flow_types)
    axes[1].set_ylabel('Mean rEPE  (%)')
    axes[1].set_title('Relative EPE')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, 'cross_flow_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {path}")


def run_evaluation(model, frames, device, cfg, output_dir,
                   flow_type=None, n_test_pairs=200):
    """
    Run the full evaluation pipeline and save all figures.

    This is the main function to call — it orchestrates all the steps:
        1. Generate a reproducible test set with known ground truth
        2. Run model inference on the full test set
        3. Compute all metrics for every pair
        4. Print a summary table
        5. Generate and save all diagnostic figures

    Parameters
    ----------
    model        : trained SiameseDisplacementNet in eval mode
    frames       : (N, H, W) -- test BES frames (held out, not used in training)
    device       : torch.device
    cfg          : Config
    output_dir   : str -- directory to save all figures
    flow_type    : str or None -- which flow type to test on.
                   None defaults to cfg.flow_type (the training flow type).
                   Override to test generalisation to other flow types.
    n_test_pairs : int -- number of synthetic pairs to evaluate on

    Returns
    -------
    results : dict of per-pair metric arrays (from compute_all_metrics)
    """
    os.makedirs(output_dir, exist_ok=True)

    ft = flow_type or cfg.flow_type
    print(f"\nEvaluating on flow_type='{ft}', n_pairs={n_test_pairs}")

    # 1. Generate test set with fixed seed for reproducibility
    print("Generating test set...")
    framesA, framesB, flows_gt = generate_test_set(
        frames, ft,
        max_shift = cfg.max_shift,
        noise_std        = cfg.noise_std,
        n_pairs          = n_test_pairs,
    )

    # 2. Run model on test set
    print("Running inference...")
    flows_pred = predict_test_set(model, framesA, framesB, device)

    # 3. Compute all metrics for every pair
    print("Computing metrics...")
    results = compute_all_metrics(flows_pred, flows_gt)

    # 4. Print summary table
    print_summary(results, ft, cfg.max_shift)

    # 5. Figures
    print("Generating figures...")
    plot_metric_distributions(results, ft, output_dir)
    plot_epe_vs_displacement(results, output_dir)
    plot_spatial_error_map(flows_pred, flows_gt, output_dir)
    plot_qualitative_examples(framesA, framesB, flows_pred,
                              flows_gt, results, output_dir)
    plot_cross_flow_comparison(model, frames, device, cfg, output_dir)

    print(f"\nAll figures saved to {output_dir}")
    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Evaluate trained BES flow model on synthetic test data'
    )
    parser.add_argument('--data',   required=True,
                        help='Path to .h5 test frames, shape (N, 64, 64)')
    parser.add_argument('--weights',  required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--output',   default='outputs/evaluation/',
                        help='Directory to save figures and results')
    parser.add_argument('--flow_type', default=None,
                        help='Flow type to test on (default: cfg.flow_type)')
    parser.add_argument('--n_pairs',  type=int, default=200,
                        help='Number of test pairs')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load frames
    print(f"Loading frames from {args.frames}...")
    test_frames = np.load(args.frames)
    print(f"Loaded {test_frames.shape[0]} frames\n")

    # Load model
    model = load_model(args.weights, device, cfg)

    # Run evaluation
    results = run_evaluation(
        model, test_frames, device, cfg,
        output_dir   = args.output,
        flow_type    = args.flow_type,
        n_test_pairs = args.n_pairs,
    )