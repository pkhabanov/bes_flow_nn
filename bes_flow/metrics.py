# bes_flow/metrics.py
#
# Metric functions and evaluation plots
#
# Two layers:
#   Per-pair primitives
#     end_point_error, relative_epe, angular_error, outlier_rate,
#     correlation_coefficient, evaluate_pair
#   Batch aggregation + visualisation
#     compute_all_metrics, print_summary,
#     plot_metric_distributions, plot_epe_vs_displacement,
#     plot_spatial_error_map, plot_qualitative_examples

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def to_numpy(x):
    """Convert tensor or array to float32 numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


# Per-pair metrics (accept (2, H, W) arrays or tensors)

def end_point_error(flow_pred, flow_gt):
    """
    Per-pixel Euclidean distance between predicted and ground-truth vectors.

    Returns
    -------
    epe_map  : (H, W) float32 — per-pixel error in pixels
    mean_epe : float — spatial mean EPE (primary benchmark metric)
    """
    flow_pred = to_numpy(flow_pred)
    flow_gt   = to_numpy(flow_gt)

    diff    = flow_pred - flow_gt
    epe_map = np.sqrt((diff ** 2).sum(axis=0))
    return epe_map, float(epe_map.mean())


def relative_epe(flow_pred, flow_gt, epsilon=0.001):
    """
    EPE normalised by the local ground-truth velocity magnitude.

    Returns
    -------
    repe_map  : (H, W) float32
    mean_repe : float
    """
    flow_pred = to_numpy(flow_pred)
    flow_gt   = to_numpy(flow_gt)

    diff     = flow_pred - flow_gt
    epe_map  = np.sqrt((diff ** 2).sum(axis=0))
    gt_mag   = np.sqrt((flow_gt ** 2).sum(axis=0))
    repe_map = epe_map / (gt_mag + epsilon)
    return repe_map, float(repe_map.mean())


def angular_error(flow_pred, flow_gt):
    """
    Angular error in degrees using the (dx, dy, 1) embedding of
    Fleet & Jepson (1990).  Well-defined even when one vector is zero.

    Returns
    -------
    ae_map  : (H, W) float32 — per-pixel error in degrees
    mean_ae : float
    """
    flow_pred = to_numpy(flow_pred)
    flow_gt   = to_numpy(flow_gt)

    dot = (flow_pred[0] * flow_gt[0] +
           flow_pred[1] * flow_gt[1] + 1.0)

    norm_pred = np.sqrt(flow_pred[0] ** 2 + flow_pred[1] ** 2 + 1.0)
    norm_gt   = np.sqrt(flow_gt[0]   ** 2 + flow_gt[1]   ** 2 + 1.0)

    cos_angle = np.clip(dot / (norm_pred * norm_gt), -1.0, 1.0)
    ae_map    = np.degrees(np.arccos(cos_angle))
    return ae_map, float(ae_map.mean())


def outlier_rate(flow_pred, flow_gt, epe_threshold=1.0, repe_threshold=0.05):
    """
    Fraction of pixels where EPE > epe_threshold AND rEPE > repe_threshold.

    Standard KITTI thresholds are (3.0, 0.05).  For BES with displacements
    < 6 px, (1.0, 0.05) is more discriminative.

    Returns
    -------
    fl : float in [0, 1]
    """
    epe_map,  _ = end_point_error(flow_pred, flow_gt)
    repe_map, _ = relative_epe(flow_pred, flow_gt)

    bad_pixels = (epe_map > epe_threshold) & (repe_map > repe_threshold)
    return float(bad_pixels.mean())


def correlation_coefficient(flow_pred, flow_gt):
    """
    Pearson R between predicted and ground-truth flow components.

    Returns
    -------
    (R_dx, R_dy) : each in [-1, 1]
        Values near 1 mean the spatial structure is captured correctly
        even if there is a systematic magnitude offset.
    """
    flow_pred = to_numpy(flow_pred)
    flow_gt   = to_numpy(flow_gt)

    def pearson_r(a, b):
        a, b  = a.flatten(), b.flatten()
        a_c   = a - a.mean()
        b_c   = b - b.mean()
        denom = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum()) + 1e-8
        return float((a_c * b_c).sum() / denom)

    return (pearson_r(flow_pred[0], flow_gt[0]),
            pearson_r(flow_pred[1], flow_gt[1]))


def evaluate_pair(flow_pred, flow_gt,
             epe_threshold=1.0, repe_threshold=0.05, epsilon=0.1):
    """
    Compute all metrics for a single (2, H, W) flow pair.

    Returns
    -------
    dict with keys: mean_EPE, mean_rEPE, mean_AE, Fl, R_dx, R_dy, epe_map
    """
    epe_map,  mean_epe  = end_point_error(flow_pred, flow_gt)
    _,        mean_repe = relative_epe(flow_pred, flow_gt, epsilon)
    _,        mean_ae   = angular_error(flow_pred, flow_gt)
    fl                  = outlier_rate(flow_pred, flow_gt,
                                       epe_threshold, repe_threshold)
    r_dx, r_dy          = correlation_coefficient(flow_pred, flow_gt)

    return {
        'mean_EPE':  mean_epe,
        'mean_rEPE': mean_repe,
        'mean_AE':   mean_ae,
        'Fl':        fl,
        'R_dx':      r_dx,
        'R_dy':      r_dy,
        'epe_map':   epe_map,
    }


# Batch aggregation

def compute_all_metrics(flows_pred, flows_gt):
    """
    Compute all metrics for every pair in a test set.

    Parameters
    ----------
    flows_pred : (n_pairs, 2, H, W) float32
    flows_gt   : (n_pairs, 2, H, W) float32

    Returns
    -------
    results : dict  {key: 1-D float32 array of length n_pairs}
        Keys: 'EPE', 'rEPE', 'AE', 'Fl', 'R_dx', 'R_dy', 'gt_mag'
    """
    n       = len(flows_pred)
    keys    = ['EPE', 'rEPE', 'AE', 'Fl', 'R_dx', 'R_dy', 'gt_mag']
    results = {k: np.zeros(n, dtype=np.float32) for k in keys}

    for i in range(n):
        m = evaluate_pair(flows_pred[i], flows_gt[i])
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


# Text reporting

def print_summary(results, flow_type, max_shift):
    """Print a formatted summary table of all test-set metrics."""
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

    display = [
        ('EPE',  'EPE',  'px',  1.0),
        ('rEPE', 'rEPE', '%',   100.0),
        ('AE',   'AE',   'deg', 1.0),
        ('Fl',   'Fl',   '%',   100.0),
        ('R_dx', 'R_dx', '',    1.0),
        ('R_dy', 'R_dy', '',    1.0),
    ]
    for key, label, unit, scale in display:
        v = results[key] * scale
        print(f"  {label:<12}  {v.mean():>7.3f}{unit}  "
              f"{v.std():>7.3f}{unit}  "
              f"{np.median(v):>7.3f}{unit}  "
              f"{np.percentile(v, 95):>7.3f}{unit}")
    print("=" * 65)
    print()


# Evaluation plots

def plot_metric_distributions(results, flow_type, output_dir):
    """
    Histograms of EPE, rEPE, AE, and Fl across all test pairs.

    A well-trained network shows narrow, left-skewed distributions near
    zero.  Broad or right-skewed distributions indicate failure modes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Metric distributions  |  flow: {flow_type}  |  "
        f"n={len(results['EPE'])} pairs",
        fontsize=13, fontweight='bold',
    )

    plots = [
        ('EPE',  'End-Point Error  (px)',  'steelblue',   axes[0, 0]),
        ('rEPE', 'Relative EPE  (%)',       'darkorange',  axes[0, 1]),
        ('AE',   'Angular Error  (deg)',    'forestgreen', axes[1, 0]),
        ('Fl',   'Outlier Rate  (%)',       'crimson',     axes[1, 1]),
    ]
    scale = {'EPE': 1.0, 'rEPE': 100.0, 'AE': 1.0, 'Fl': 100.0}

    for key, label, color, ax in plots:
        v = results[key] * scale[key]
        ax.hist(v, bins=30, color=color, alpha=0.75, edgecolor='white')
        ax.axvline(v.mean(),     color='black', linewidth=2.0, linestyle='-',
                   label=f'Mean={v.mean():.3f}')
        ax.axvline(np.median(v), color='black', linewidth=1.5, linestyle='--',
                   label=f'Median={np.median(v):.3f}')
        ax.set_xlabel(label);  ax.set_ylabel('Count')
        ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'metric_distributions.png')
    #plt.savefig(path, dpi=150, bbox_inches='tight')
    #plt.show()
    #print(f"Saved: {path}")


def plot_epe_vs_displacement(results, output_dir):
    """
    Scatter plot of mean EPE vs ground-truth displacement magnitude.

    A flat curve → the network is dominated by a constant positional
    error (typical of an under-trained correlation layer).
    Linear scaling → constant relative error (well-calibrated network).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        results['gt_mag'], results['EPE'],
        c=results['rEPE'] * 100, cmap='viridis', s=20, alpha=0.6,
    )
    fig.colorbar(sc, ax=ax, label='rEPE  (%)')

    bins        = np.linspace(0, results['gt_mag'].max(), 10)
    bin_idx     = np.digitize(results['gt_mag'], bins)
    bin_means   = [results['EPE'][bin_idx == k].mean()
                   for k in range(1, len(bins)) if (bin_idx == k).sum() > 0]
    bin_centres = [(bins[k] + bins[k + 1]) / 2
                   for k in range(len(bins) - 1) if (bin_idx == k + 1).sum() > 0]
    ax.plot(bin_centres, bin_means, color='red', linewidth=2,
            marker='o', markersize=5, label='Bin mean EPE')

    ax.set_xlabel('Mean GT displacement magnitude  (px)')
    ax.set_ylabel('Mean EPE  (px)')
    ax.set_title('EPE vs displacement magnitude')
    ax.legend();  ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'epe_vs_displacement.png')
    #plt.savefig(path, dpi=150, bbox_inches='tight')
    #plt.show()
    #print(f"Saved: {path}")


def plot_spatial_error_map(flows_pred, flows_gt, output_dir):
    """
    Mean and std of the per-pixel EPE map averaged across all test pairs.

    Reveals systematic spatial biases such as border artefacts from the
    correlation layer or regions where the flow is consistently harder
    to estimate.  A uniform map is ideal.
    """
    diff         = flows_pred - flows_gt
    epe_map      = np.sqrt((diff ** 2).sum(axis=1))   # (n, H, W)
    mean_epe_map = epe_map.mean(axis=0)
    std_epe_map  = epe_map.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Spatial error map  (mean EPE per pixel)',
                 fontsize=13, fontweight='bold')

    im0 = axes[0].imshow(mean_epe_map, cmap='hot', origin='upper', vmin=0)
    axes[0].set_title('Mean EPE  (px)')
    axes[0].set_xlabel('x  (px)');  axes[0].set_ylabel('y  (px)')
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(std_epe_map, cmap='hot', origin='upper', vmin=0)
    axes[1].set_title('Std of EPE  (px)  — prediction uncertainty')
    axes[1].set_xlabel('x  (px)')
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'spatial_error_map.png')
    #plt.savefig(path, dpi=150, bbox_inches='tight')
    #plt.show()
    #print(f"Saved: {path}")


def plot_qualitative_examples(framesA, framesB, flows_pred, flows_gt,
                              results, output_dir, n_examples=4):
    """
    Show n_examples pairs sampled from different EPE quartiles.

    Each row: Frame A | Frame B | GT quiver | Predicted quiver | EPE map.

    Selecting one pair per quartile ensures both the best and the worst
    predictions are shown — more informative than random sampling.
    """
    n               = len(flows_pred)
    epe_sorted      = np.argsort(results['EPE'])
    step            = max(1, n // n_examples)
    indices         = [epe_sorted[min(i * step, n - 1)] for i in range(n_examples)]
    quartile_labels = ['Best', 'Q1', 'Q2 / Median', 'Q3', 'Worst']

    fig = plt.figure(figsize=(20, 4 * n_examples))
    fig.suptitle('Qualitative examples  —  one per EPE quartile',
                 fontsize=13, fontweight='bold')

    quiver_step = 8
    H, W        = framesA.shape[2], framesA.shape[3]
    ys          = np.arange(quiver_step // 2, H, quiver_step)
    xs          = np.arange(quiver_step // 2, W, quiver_step)
    xx, yy      = np.meshgrid(xs, ys)

    for row, idx in enumerate(indices):
        fA   = framesA[idx, 0]
        fB   = framesB[idx, 0]
        pred = flows_pred[idx]
        gt   = flows_gt[idx]
        epe  = np.sqrt(((pred - gt) ** 2).sum(axis=0))

        label   = quartile_labels[min(row, len(quartile_labels) - 1)]
        epe_val = results['EPE'][idx]

        gs = gridspec.GridSpec(n_examples, 5, figure=fig,
                               hspace=0.35, wspace=0.3)

        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(fA, cmap='inferno', origin='upper')
        ax0.set_ylabel(f'{label}\nEPE={epe_val:.3f}px', fontsize=9)
        if row == 0:  ax0.set_title('Frame A')
        ax0.set_xticks([])

        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(fB, cmap='inferno', origin='upper')
        if row == 0:  ax1.set_title('Frame B')
        ax1.set_xticks([]);  ax1.set_yticks([])

        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(fA, cmap='inferno', origin='upper')
        ax2.quiver(xx, yy, gt[0][yy, xx], -gt[1][yy, xx],
                   color='cyan', scale=60, scale_units='width',
                   width=0.005, headwidth=4)
        if row == 0:  ax2.set_title('GT flow')
        ax2.set_xticks([]);  ax2.set_yticks([])

        ax3 = fig.add_subplot(gs[row, 3])
        ax3.imshow(fA, cmap='inferno', origin='upper')
        ax3.quiver(xx, yy, pred[0][yy, xx], -pred[1][yy, xx],
                   color='yellow', scale=60, scale_units='width',
                   width=0.005, headwidth=4)
        if row == 0:  ax3.set_title('Predicted flow')
        ax3.set_xticks([]);  ax3.set_yticks([])

        ax4 = fig.add_subplot(gs[row, 4])
        im  = ax4.imshow(epe, cmap='hot', origin='upper', vmin=0)
        plt.colorbar(im, ax=ax4, shrink=0.8, label='px')
        if row == 0:  ax4.set_title('EPE map')
        ax4.set_xticks([]);  ax4.set_yticks([])

    path = os.path.join(output_dir, 'qualitative_examples.png')
    #plt.savefig(path, dpi=150, bbox_inches='tight')
    #plt.show()
    #print(f"Saved: {path}")