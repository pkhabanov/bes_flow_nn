# bes_flow/train.py
#
# Training loop, model evaluation, and entry point.
#
# Workflow
# ────────
#  1. Load raw BES frames from HDF5.
#  2. Split frames into three disjoint subsets:
#       train_frames  — synthetic training pairs
#       val_frames    — synthetic validation pairs
#       test_frames   — held out; only touched after training ends
#  3. Call make_datasets() once to generate (or load from cache) all
#     three BESDataset objects.
#  4. Call make_dataloaders() to wrap them in DataLoaders.
#  5. Train the network:
#       single-flow-type mode  — train() with the three fixed loaders
#       curriculum mode        — curriculum_train() rebuilds train/val
#                                datasets for each stage (different flow
#                                type) and also reduces the learning rate.
#  6. Load the best checkpoint and run the full evaluation suite on the
#     test set.
#
# Usage
# ─────
#   python -m bes_flow.train                   # single flow type
#   python -m bes_flow.train --curriculum      # LR-curriculum training

import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import CenteredNorm
import h5py

from bes_flow.config  import cfg
from bes_flow.model   import SiameseDisplacementNet
from bes_flow.model_s import BESFlowNetS
from bes_flow.model_pwcnet import PWCNet
from bes_flow.loss    import WarpingL2Loss
from bes_flow.dataset import make_datasets, make_dataloaders, generate_dataset, BESDataset
from bes_flow.metrics import (compute_all_metrics, print_summary,
                               plot_metric_distributions, plot_epe_vs_displacement,
                               plot_spatial_error_map, plot_qualitative_examples)
from bes_flow.predict import load_model


def predict_dataset(model, dataset, device, batch_size=16):
    """
    Run the model on every pair in a BESDataset and return predicted flows.

    Parameters
    ----------
    model      : trained model in eval mode
    dataset    : BESDataset — framesA/B/flows_gt accessible as attributes
    device     : torch.device
    batch_size : int

    Returns
    -------
    flows_pred : (n_pairs, 2, H, W) float32 numpy array
    """
    model.eval()
    n          = len(dataset)
    H, W       = dataset.framesA.shape[2], dataset.framesA.shape[3]
    flows_pred = np.zeros((n, 2, H, W), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end    = min(start + batch_size, n)
            batchA = torch.tensor(dataset.framesA[start:end]).to(device)
            batchB = torch.tensor(dataset.framesB[start:end]).to(device)
            flows_pred[start:end] = model(batchA, batchB).cpu().numpy()

    return flows_pred


def _generate_test_set_for_flow_type(frames, flow_type, cfg, n_pairs=50, seed=42):
    """
    Helper: generate a small fixed test set for one specific flow type.
    Used only by plot_cross_flow_comparison.
    """
    # number of pairs per frame
    n_ppf = max(1, n_pairs // len(frames))

    rng_state = np.random.get_state()
    np.random.seed(seed)
    fA, fB, gt = generate_dataset(
        frames,
        n_pairs_per_frame = n_ppf,
        max_shift         = cfg.max_shift,
        noise_std         = cfg.noise_std,
        flow_type         = flow_type,
    )
    np.random.set_state(rng_state)
    return fA, fB, gt


def plot_cross_flow_comparison(model, test_frames, device, cfg, output_dir):
    """
    Bar chart comparing mean EPE and rEPE across all four flow types.

    Tests whether the model generalises beyond the flow type it was trained
    on.  Uses 50 freshly generated pairs per flow type from test_frames.
    """
    flow_types = ['smooth', 'modes', 'zonal', 'well']
    epe_means, epe_stds, repe_means = [], [], []
    
    # Accumulate one random sample per flow type for the qualitative figure.
    samples = []   # list of (flow_type, fA, fB, gt, pred, mean_epe) per type

    for ft in flow_types:
        fA, fB, gt = _generate_test_set_for_flow_type(
            test_frames, ft, cfg, n_pairs=50
        )
        tmp_ds  = BESDataset(fA, fB, gt, augment=False)
        pred    = predict_dataset(model, tmp_ds, device)
        metrics = compute_all_metrics(pred, gt)
        epe_means.append(metrics['EPE'].mean())
        epe_stds.append(metrics['EPE'].std())
        repe_means.append(metrics['rEPE'].mean() * 100)

        # Pick one random pair index
        idx = np.random.randint(len(fA))
        samples.append((ft,
                        fA[idx, 0],    # (H, W)
                        fB[idx, 0],    # (H, W)
                        gt[idx],       # (2, H, W)
                        pred[idx],     # (2, H, W)
                        metrics['EPE'][idx]))

    x      = np.arange(len(flow_types))
    colors = ['steelblue', 'darkorange', 'forestgreen', 'mediumpurple']

    # fig1 - bar charts
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Performance across flow types', fontsize=13, fontweight='bold')

    axes[0].bar(x, epe_means, yerr=epe_stds, color=colors,
                capsize=5, edgecolor='black', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(flow_types)
    axes[0].set_ylabel('Mean EPE  (px)');  axes[0].set_title('End-Point Error')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x, repe_means, color=colors, edgecolor='black', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(flow_types)
    axes[1].set_ylabel('Mean rEPE  (%)');  axes[1].set_title('Relative EPE')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, 'cross_flow_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {path}")

    # fig2 - flow comparison
    # Columns: Frame A | Frame B + GT quiver | Frame B + pred quiver | EPE map
    n_rows = len(samples)
    H, W   = samples[0][1].shape   # spatial dims from first sample's frameA
 
    quiver_step = 8
    ys          = np.arange(quiver_step // 2, H, quiver_step)
    xs          = np.arange(quiver_step // 2, W, quiver_step)
    xx, yy      = np.meshgrid(xs, ys)
 
    fig = plt.figure(figsize=(20, 4 * n_rows))
    fig.suptitle('Qualitative examples across flow types  —  one random pair per type',
                 fontsize=13, fontweight='bold')
 
    gs = gridspec.GridSpec(n_rows, 5, figure=fig, hspace=0.35, wspace=0.25)
 
    col_titles = ['Frame A', 'Frame B  +  GT flow', 'Frame B  +  pred flow',
                  'EPE  vx  (px)', 'EPE  vy  (px)']
 
    for row, (ft, fA, fB, gt, pred, epe_val) in enumerate(samples):
        # Per-component absolute errors (H, W)
        #epe_vx = np.abs(pred[0] - gt[0])   # |Δdx|
        #epe_vy = np.abs(pred[1] - gt[1])   # |Δdy|
        diff_vx = pred[0] - gt[0]   # (H, W)
        diff_vy = pred[1] - gt[1]   # (H, W)
 
        # Shared colour scale across dvx and dvy for direct comparability
        #vmax_epe = max(epe_vx.max(), epe_vy.max())
        #vmax_diff = max(np.abs(diff_vx).max(), np.abs(diff_vy).max())
        vmax_diff = 5.0
        norm = CenteredNorm(vcenter=0, halfrange=vmax_diff)
 
        # col 0 — Frame A
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(fA, cmap='inferno', origin='upper')
        ax0.set_ylabel(f'{ft}\nEPE={epe_val:.3f} px', fontsize=9)
        if row == 0:  ax0.set_title(col_titles[0])
        ax0.set_xticks([]);  ax0.set_yticks([])
 
        # col 1 — Frame B with GT flow quiver
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(fB, cmap='inferno', origin='upper')
        ax1.quiver(xx, yy, gt[0][yy, xx], -gt[1][yy, xx],
                   color='cyan', scale=60, scale_units='width',
                   width=0.005, headwidth=4)
        if row == 0:  ax1.set_title(col_titles[1])
        ax1.set_xticks([]);  ax1.set_yticks([])
 
        # col 2 — Frame B with predicted flow quiver
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(fB, cmap='inferno', origin='upper')
        ax2.quiver(xx, yy, pred[0][yy, xx], -pred[1][yy, xx],
                   color='yellow', scale=60, scale_units='width',
                   width=0.005, headwidth=4)
        if row == 0:  ax2.set_title(col_titles[2])
        ax2.set_xticks([]);  ax2.set_yticks([])
 
        # col 3 — per-pixel error in vx
        ax3 = fig.add_subplot(gs[row, 3])
        #im3 = ax3.imshow(epe_vx, cmap='hot', origin='upper',
        #                  vmin=0, vmax=vmax_epe)
        im3 = ax3.imshow(diff_vx, cmap='RdBu_r', origin='upper', norm=norm)
        plt.colorbar(im3, ax=ax3, shrink=0.8, label='px')
        if row == 0:  ax3.set_title(col_titles[3])
        ax3.set_xticks([]);  ax3.set_yticks([])
 
        # col 4 — per-pixel error in vy
        ax4 = fig.add_subplot(gs[row, 4])
        #im4 = ax4.imshow(epe_vy, cmap='hot', origin='upper',
        #                 vmin=0, vmax=vmax_epe)
        im4 = ax4.imshow(diff_vy, cmap='RdBu_r', origin='upper', norm=norm)
        plt.colorbar(im4, ax=ax4, shrink=0.8, label='px')
        if row == 0:  ax4.set_title(col_titles[4])
        ax4.set_xticks([]);  ax4.set_yticks([])
 
    path2 = os.path.join(output_dir, 'cross_flow_examples.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close('all')
    print(f"Saved: {path2}")


def run_evaluation(model, test_dataset, test_frames, device, cfg, output_dir, plot_results=True):
    """
    Full evaluation pipeline on the held-out test set.

    1. Run model inference on test_dataset (pre-generated, fixed seed).
    2. Compute all metrics for every pair.
    3. Print summary table.

    Parameters
    ----------
    model        : trained model in eval mode
    test_dataset : BESDataset — held-out test set from make_datasets()
    test_frames  : (N_test, H, W) — raw test frames for cross-flow comparison
    device       : torch.device
    cfg          : Config
    output_dir   : str — directory to save all figures
    plot_results : bool - Flag to generate figures

    Returns
    -------
    results : dict of per-pair metric arrays (from compute_all_metrics)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  Evaluation on test set")
    print(f"  Flow type  : {cfg.flow_type}")
    print(f"  Test pairs : {len(test_dataset)}")
    print(f"{'═'*60}\n")

    # Inference
    print("Running inference on test set...")
    flows_pred = predict_dataset(model, test_dataset, device)
    flows_gt   = test_dataset.flows_gt   # (n, 2, H, W) numpy array

    # Metrics
    print("Computing metrics...")
    results = compute_all_metrics(flows_pred, flows_gt)

    # Summary table
    print_summary(results, cfg.flow_type, cfg.max_shift)

    # Figures
    if plot_results:
        print("Generating figures...")
        plot_metric_distributions(results, cfg.flow_type, output_dir)
        plot_epe_vs_displacement(results, output_dir)
        plot_spatial_error_map(flows_pred, flows_gt, output_dir)
        plot_qualitative_examples(test_dataset.framesA, test_dataset.framesB,
                                flows_pred, flows_gt, results, output_dir)
        plot_cross_flow_comparison(model, test_frames, device, cfg, output_dir)

        print(f"\nAll evaluation figures saved to {output_dir}")
    
    return results


def train(model, train_loader, val_loader, loss_fn, optimizer, scheduler,
          cfg, device, start_epoch=1, total_epochs=None):
    """
    Run the training loop for one block of epochs.

    Parameters
    ----------
    model        : neural network model that takes frameA and frameB as inputs
    train_loader : DataLoader — yields (frameA, frameB, flow_gt) batches
    val_loader   : DataLoader — same structure, for validation
    loss_fn      : WarpingL2Loss
    optimizer    : torch.optim.Adam
    scheduler    : CosineAnnealingLR
    cfg          : Config
    device       : torch.device
    start_epoch  : int — first epoch number (>1 when resuming/curriculum)
    total_epochs : int — last epoch number inclusive (defaults to cfg.num_epochs)

    Returns
    -------
    history : dict  {key: list of per-epoch averages}
        Keys: 'total', 'photometric', 'smoothness', 'supervised',
              'val_total', 'val_epe'
    """
    # dictionary for pre-epoch loss averages
    history = {
        'total': [], 'photometric': [], 
        'smoothness': [], 'laplacian': [],
        'supervised': [],
        'val_total': [], 'val_epe': [],
    }
    history_path = cfg.output_dir + f'train_history_{cfg.flow_type}.json'

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.output_dir,     exist_ok=True)

    if total_epochs is None:
        total_epochs = cfg.num_epochs

    best_val_epe = float('inf')
    n_batches = len(train_loader)

    for epoch in range(start_epoch, total_epochs + 1):

        # ── Training ──────────────────────────────────────────────────────
        model.train()
        epoch_totals = {'total': 0., 'photometric': 0.,
                        'smoothness': 0., 'laplacian': 0., 
                        'supervised': 0.}

        for step, batch in enumerate(train_loader, start=1):
            frameA  = batch[0].to(device)
            frameB  = batch[1].to(device)
            if cfg.is_supervised:
                flow_gt = batch[2].to(device)
            else:
                flow_gt = None
            
            # Forward pass
            optimizer.zero_grad()
            flow_pred = model(frameA, frameB)
            
            # Loss
            total, photo, smooth, lap, sup = loss_fn(
                frameA, frameB, flow_pred, flow_gt=flow_gt
            )
            # Backward pass
            total.backward()

            # Gradient clipping: prevents large weight updates 
            # (exploding gradients) especially early in training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_totals['total']       += total.item()
            epoch_totals['photometric'] += photo.item()
            epoch_totals['smoothness']  += smooth.item()
            epoch_totals['laplacian']   += lap.item()
            epoch_totals['supervised']  += sup.item()

            if step % max(1, n_batches // 5) == 0:
                print(
                    f"Epoch [{epoch:>4d}/{total_epochs}] "
                    f"Batch# [{step:>4d}/{n_batches}] | "
                    f"Total loss: {total.item():.5f}  "
                    f"Photo: {photo.item():.5f}  "
                    f"Smooth: {smooth.item():.5f}  "
                    f"Lapl: {lap.item():.5f}  "
                    f"Sup: {sup.item():.5f}"
                )

        # Per-epoch averages 
        n_train = len(train_loader)
        for key in epoch_totals:
            history[key].append(epoch_totals[key] / n_train)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.
        val_epe_sum  = 0.
        n_val        = len(val_loader)

        with torch.no_grad():
            for batch in val_loader:
                frameA  = batch[0].to(device)
                frameB  = batch[1].to(device)
                flow_gt = batch[2].to(device)
                # model prediction
                flow_pred = model(frameA, frameB)
                if cfg.is_supervised:
                    loss, _, _, _, _ = loss_fn(frameA, frameB, flow_pred,
                                            flow_gt=flow_gt)
                else:
                    loss, _, _, _, _ = loss_fn(frameA, frameB, flow_pred,
                                            flow_gt=None)
                val_loss_sum += loss.item()
                # End point error: per-pixel Euclidean distance, 
                # averaged over pixels and batch
                diff = flow_pred - flow_gt
                val_epe_sum += torch.sqrt((diff ** 2).sum(dim=1)).mean().item()

        val_loss = val_loss_sum / n_val
        val_epe  = val_epe_sum  / n_val
        history['val_total'].append(val_loss)
        history['val_epe'].append(val_epe)

        # Decay learning rate
        scheduler.step()

        print(
            f"\n{'─'*70}\n"
            f"Epoch [{epoch:>4d}/{total_epochs}] SUMMARY | "
            f"Train: {history['total'][-1]:.5f}  "
            f"Val: {val_loss:.5f}  "
            f"EPE: {val_epe:.4f} px  "
            f"LR: {scheduler.get_last_lr()[0]:.2e}\n"
            f"{'─'*70}\n"
        )

        # Per-epoch checkpoint
        if epoch % 5 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(cfg.checkpoint_dir, f"model_{cfg.flow_type}_epoch_{epoch:04d}.pt"),
            )
            # save history to json
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)

        # Best-model checkpoint (tracked by val EPE)
        if val_epe < best_val_epe:
            best_val_epe = val_epe
            fname = f"model_{cfg.flow_type}_best.pt"
            torch.save(
                model.state_dict(),
                os.path.join(cfg.checkpoint_dir, fname),
            )
            print(f"  ★ New best val EPE: {best_val_epe:.4f} px  "
                  f" -> saved {fname}\n")
    
    del train_loader, val_loader
    
    return history


def plot_loss_history(history, cfg):
    """
    Two-row grid of train/val loss curves with running-mean overlay.

    Row 1: total | photometric | smoothness | supervised
    Row 2: val total | val EPE
    """
    epochs = np.arange(1, len(history['total']) + 1)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(
        f"Training history  |  flow: {cfg.flow_type}  |  "
        f"epochs: {cfg.num_epochs}  |  lr: {cfg.learning_rate}",
        fontsize=12, fontweight='bold',
    )

    components = [
        ('total',       'Total loss (train)',   'steelblue',     axes[0, 0], 'Loss'),
        ('photometric', 'Photometric loss',     'darkorange',    axes[0, 1], 'Loss'),
        ('smoothness',  'Smoothness loss',      'forestgreen',   axes[0, 2], 'Loss'),
        ('supervised',  'Supervised loss',      'mediumpurple',  axes[0, 3], 'Loss'),
        ('laplacian',   'Laplacian loss',       'darkturquoise', axes[1, 0], 'Loss'),
        ('val_total',   'Total loss (val)',     'crimson',       axes[1, 1], 'Loss'),
        ('val_epe',     'Val EPE  (px)',        'teal',          axes[1, 2], 'EPE (px)'),
    ]

    for key, label, color, ax, ylabel in components:
        values = history[key]
        ax.plot(epochs, values, color=color, linewidth=1.2, alpha=0.5,
                label='Per epoch')
        window       = max(1, len(epochs) // 10)
        running_mean = np.convolve(values, np.ones(window) / window, mode='valid')
        ax.plot(epochs[window - 1:], running_mean,
                color=color, linewidth=2.5, linestyle='--', alpha=0.9,
                label=f'Running mean (w={window})')
        ax.set_xlabel('Epoch');  ax.set_ylabel(ylabel)
        ax.set_title(label);     ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3);  ax.set_xlim(1, len(epochs))

    axes[1, 3].set_visible(False)

    plt.tight_layout()
    plt.show()
    #plot_path = os.path.join(cfg.output_dir, 'loss_history.png')
    #plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    #print(f"\nLoss plot saved to {plot_path}")


def plot_curriculum_loss(full_history, stages, cfg):
    """
    Curriculum loss history with stage-transition vertical markers
    and shaded background regions per stage.
    """
    epochs = np.arange(1, len(full_history['total']) + 1)

    boundaries = []
    cumulative = 0
    for stage in stages[:-1]:
        cumulative += stage['epochs']
        boundaries.append(cumulative)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(
        f"Curriculum training  |  {cfg.num_epochs} total epochs",
        fontsize=12, fontweight='bold',
    )

    components = [
        ('total',       'Total loss (train)',  'steelblue'),
        ('photometric', 'Photometric loss',    'darkorange'),
        ('smoothness',  'Smoothness loss',     'forestgreen'),
        ('laplacian',   'Laplacian loss',     'darkturquoise'),
        ('val_total',   'Total loss (val)',    'crimson'),
        ('val_epe',     'Val EPE  (px)',       'teal'),
    ]
    stage_colors = ['#aec6cf', '#ffda9e', '#b5ead7', '#ffd6e0']
    stage_labels = [s['name'] for s in stages]

    for ax, (key, label, color) in zip(axes.flatten()[:len(components)], components):
        values = full_history[key]
        
        # Shade background by stage for immediate visual identification
        prev = 0
        for boundary, bg, slabel in zip(
            boundaries + [len(epochs)], stage_colors, stage_labels
        ):
            ax.axvspan(prev, boundary, alpha=0.15, color=bg, label=slabel)
            prev = boundary
        
        # Loss curve + running mean
        ax.plot(epochs, values, color=color, linewidth=1.2, alpha=0.5)
        window       = max(1, len(epochs) // 10)
        running_mean = np.convolve(values, np.ones(window) / window, mode='valid')
        ax.plot(epochs[window - 1:], running_mean,
                color=color, linewidth=2.5, linestyle='--', alpha=0.9)
        
        # Vertical lines at stage transitions
        for b in boundaries:
            ax.axvline(b, color='black', linewidth=1.0, linestyle=':', alpha=0.7)

        ax.set_xlabel('Epoch');  ax.set_ylabel('Loss')
        ax.set_title(label)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3);  ax.set_xlim(1, len(epochs))

    plt.tight_layout()
    plt.show()
    #plot_path = os.path.join(cfg.output_dir, 'curriculum_loss_history.png')
    #plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    #print(f"\nCurriculum loss plot saved to {plot_path}")


def resolve_cache_path(base: str | None, flow_type: str) -> str | None:
    ''' Build a per-stage cache path by inserting the flow_type before the
    file extension. If no cache path is configured, all stages run without caching.
    '''
    if base is None:
        return None
    root, ext = os.path.splitext(base)
    return f"{root}_{flow_type}{ext}"


def curriculum_train(model, train_frames, val_frames, loss_fn, cfg, device):
    """
    Four-stage curriculum training, each stage on a different flow type.

    A new train/val dataset is generated at the start of every stage so
    that the network sees progressively more complex flow fields:

        Stage 1: smooth Gaussian flow     — learns basic displacement tracking
        Stage 2: sinusoidal mode flow     — learns multi-scale structure
        Stage 3: zonal sin + turbulence   — adapts to shear-dominated flow
        Stage 4: zonal Gauss well + turb  — adapts to realistic flow conditions

    The learning rate is also reduced across stages.  

    Each stage uses its own HDF5 cache derived from the base cache path
    This ensures each stage's data is cached independently and no stage
    overwrites another's cache.  If cfg.dataset_cache_path is None,
    caching is skipped for all stages.

    Parameters
    ----------
    model        : initialised but untrained model
    train_frames : (N_train, H, W) float array — raw BES training frames
    val_frames   : (N_val,   H, W) float array — raw BES validation frames
    loss_fn      : WarpingL2Loss
    cfg          : Config
    device       : torch.device

    Returns
    -------
    full_history : dict — loss history concatenated across all stages
    """
    from dataclasses import replace

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.output_dir,     exist_ok=True)

     # Define the training stages.
    total  = cfg.num_epochs
    stages = [
        {'name': 'Stage 1 — smooth flow',           'flow_type': 'smooth',
         'epochs': total // 4,              'lr': cfg.learning_rate},
        {'name': 'Stage 2 — sinusoidal modes',       'flow_type': 'modes',
         'epochs': total // 4,              'lr': cfg.learning_rate},
        {'name': 'Stage 3 — zonal Gauss well + turb','flow_type': 'well',
         'epochs': total // 4,              'lr': cfg.learning_rate},
        {'name': 'Stage 4 — zonal sin + turbulence', 'flow_type': 'zonal',
         'epochs': total - 3 * (total // 4),'lr': cfg.learning_rate},
    ]

    full_history = {
        'total': [], 'photometric': [], 'smoothness': [], 'laplacian': [],
        'supervised': [], 'val_total': [], 'val_epe': [],
    }
    global_epoch = 0  # tracks absolute epoch number across all stages

    for i, stage in enumerate(stages, start=1):
        print(f"\n{'═'*70}")
        print(f"  {stage['name']}  |  epochs: {stage['epochs']}  |  "
              f"lr: {stage['lr']:.2e}")
        print(f"{'═'*70}\n")

        # Build a cfg copy with the stage's flow_type and its own cache path.
        # make_datasets receives an explicit (empty) test_frames array
        stage_cfg = replace(
            cfg,
            flow_type          = stage['flow_type'],
            dataset_cache_path = resolve_cache_path(cfg.dataset_cache_path, stage['flow_type']),
        )

        # Generate (or load from cache) train/val for this flow type.
        # Pass an empty array for test_frames — we do not need a test set here.
        stage_train_ds, stage_val_ds, _ = make_datasets(
            train_frames,
            val_frames,
            np.empty((0, *train_frames.shape[1:]), dtype=np.float32),
            stage_cfg,
        )
        stage_train_loader, stage_val_loader, _ = make_dataloaders(
            stage_train_ds, stage_val_ds,
            stage_val_ds,  
            stage_cfg,
        )
        
        # Reset optimiser with new learning rate 
        optimizer = torch.optim.Adam(model.parameters(), lr=stage['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage['epochs']
        )

        # Run training for this stage
        stage_history = train(
            model, stage_train_loader, stage_val_loader,
            loss_fn, optimizer, scheduler,
            stage_cfg, device,
            start_epoch  = global_epoch + 1,
            total_epochs = global_epoch + stage['epochs'],
        )

        for key in full_history:
            full_history[key].extend(stage_history[key])

        global_epoch += stage['epochs']

        # Save a stage-level checkpoint so we can reload from the end
        # of any stage without re-running the whole curriculum
        ckpt_name = f"model_stage{i}_{stage['flow_type']}_final.pt"
        torch.save(
            model.state_dict(),
            os.path.join(cfg.checkpoint_dir, ckpt_name),
        )
        print(f"\nStage checkpoint saved: {ckpt_name}")
        
        del stage_train_loader, stage_val_loader
        del stage_train_ds, stage_val_ds

    # Plot the full curriculum loss history
    #plot_curriculum_loss(full_history, stages, cfg)
    return full_history


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train and evaluate the BES CNN'
    )
    parser.add_argument('--curriculum', action='store_true',
                        help='Use four-stage LR-curriculum training')
    parser.add_argument('--skip_train', action='store_true',
                        help='Use to skip training')
    parser.add_argument('--checkpoint', type=str,
                        help='Load model checkpoint to start with')
    parser.add_argument('--model', type=str, default='pwc',
                        help='Model type: pwc or flownet')
    parser.add_argument('--plot_results', action='store_true',
                        help='Plot training loss history')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")

    # ── Load raw BES frames ───────────────────────────────────────────────
    print(f"Loading BES frames: {cfg.data_path}")
    with h5py.File(cfg.data_path, 'r') as hf:
        all_frames = hf['images'][:]
    print(f"Loaded {len(all_frames)} frames, shape {all_frames.shape}\n")

    # ── Three-way frame split ─────────────────────────────────────────────
    N        = len(all_frames)
    n_test   = int(cfg.test_split * N)
    n_val    = int(cfg.val_split  * N)
    n_train  = N - n_test - n_val

    # make a random split (use test_seed)
    rng     = np.random.default_rng(cfg.test_seed)
    indices = rng.permutation(N)
    train_frames = all_frames[indices[:n_train]]
    val_frames   = all_frames[indices[n_train : n_train + n_val]]
    test_frames  = all_frames[indices[n_train + n_val:]]

    print(f"Frame split:")
    print(f"  Total      : {N}")
    print(f"  Train      : {n_train}")
    print(f"  Validation : {n_val}")
    print(f"  Test       : {n_test}\n")

    # ── Model ─────────────────────────────────────────────────────────────
    if args.model == 'flownet':
        print('Initializing BESFlowNetS')
        model = BESFlowNetS()
    elif args.model == 'pwc':
        print('Initializing PWCNet')
        model = PWCNet(max_displacement=cfg.max_displacement)
    
    model = model.to(device)

    if args.checkpoint is not None:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        model = load_model(model, args.checkpoint , device, cfg)
    else:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}\n")

    # ── Loss ──────────────────────────────────────────────────────────────
    loss_fn = WarpingL2Loss(
        smooth_weight = cfg.smooth_weight,
        laplacian_weight = cfg.laplacian_weight,
        sup_weight    = cfg.sup_weight,
    )
    
    if not args.skip_train:
        # ── Train ─────────────────────────────────────────────────────────────
        if args.curriculum:
            # Curriculum train - several flow types
            loss_history = curriculum_train(
                model, train_frames, val_frames, loss_fn, cfg, device
            )
            history_path = cfg.output_dir + 'train_history_curriculum.json'
        else:
            # Single flow type training
            # update dataset_cache_path in cfg
            cfg = replace(
                cfg,
                dataset_cache_path = resolve_cache_path(cfg.dataset_cache_path, cfg.flow_type),
            )
            # ── Build datasets ───────────────────────────────────────────────
            train_dataset, val_dataset, test_dataset = make_datasets(
                train_frames, val_frames, test_frames, cfg
            )

            # ── Build DataLoaders ─────────────────────────────────────────────────
            train_loader, val_loader, test_loader = make_dataloaders(
                train_dataset, val_dataset, test_dataset, cfg
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.num_epochs*1.25
            )
            print("Starting single-stage training...\n")
            loss_history = train(
                model, train_loader, val_loader,
                loss_fn, optimizer, scheduler,
                cfg, device,
            )
            history_path = cfg.output_dir + f'train_history_{cfg.flow_type}.json'
            if args.plot_results:
                plot_loss_history(loss_history, cfg)
        # save history to json
        with open(history_path, 'w') as f:
            json.dump(loss_history, f, indent=2)
    else:
        # ── Evaluate on the test set ──────────────────────────────────────────
        # Load the best checkpoint (lowest val EPE during training).
        #best_ckpt = os.path.join(cfg.checkpoint_dir, 'model_best.pt')
        #best_ckpt = os.path.join(cfg.checkpoint_dir, 'model_well_best.pt')
        if args.checkpoint is not None:
            best_ckpt = args.checkpoint
        else:
            best_ckpt = 'checkpoints/model_well_best.pt'
        print(f"\nLoading best checkpoint for evaluation: {best_ckpt}")
        model = load_model(model, best_ckpt, device, cfg)

        run_evaluation(
            model,
            test_dataset  = test_dataset,
            test_frames   = test_frames,
            device        = device,
            cfg           = cfg,
            output_dir    = os.path.join(cfg.output_dir, 'evaluation'),
            plot_results  = args.plot_results,
        )

        # plot history
        if args.plot_results:
            history_path = 'outputs/train_history_modes.json'
            with open(history_path, 'r') as file:
                full_history = json.load(file)
            total = len(full_history['total'])
            stages = [
                {'name': 'Stage 1 — smooth flow', 'flow_type': 'smooth',
                'epochs': total // 4, 'lr': cfg.learning_rate},
                {'name': 'Stage 2 — sinusoidal modes',       'flow_type': 'modes',
                'epochs': total // 4, 'lr': cfg.learning_rate / 2},
                {'name': 'Stage 3 — zonal sin + turbulence', 'flow_type': 'zonal',
                'epochs': total // 4, 'lr': cfg.learning_rate / 10},
                {'name': 'Stage 4 — zonal Gauss well + turb','flow_type': 'well',
                'epochs': total - 3 * (total // 4),'lr': cfg.learning_rate / 10},
                ]
            plot_curriculum_loss(full_history, stages, cfg)
