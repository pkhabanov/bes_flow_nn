# bes_flow/predict.py
#
# Inference pipeline: apply a trained SiameseDisplacementNet to a new
# BES dataset and produce velocity fields for every consecutive frame pair.
#
# Usage
# ─────
# From the command line:
#   python -m bes_flow.predict \
#       --frames   data/raw/new_shot.npy \
#       --weights  checkpoints/model_epoch_0100.pt \
#       --output   outputs/velocities.npy
#
# Or from another script:
#   from bes_flow.predict import load_model, predict_sequence
#   model     = load_model('checkpoints/model_epoch_0100.pt', device)
#   velocities = predict_sequence(model, bes_frames, device)
#
# Output
# ──────
# velocities : np.ndarray of shape (N-1, 2, 64, 64)
#   - N-1 because each velocity field is computed from a pair of frames
#   - channel 0 = dx (horizontal / poloidal velocity in pixels per frame)
#   - channel 1 = dy (vertical   / radial    velocity in pixels per frame)
#   Multiply by (pixel_size_cm / frame_interval_us) to convert to cm/μs

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import CenteredNorm
from bes_flow.config  import cfg


def load_model(model, weights_path, device, cfg=cfg):
    """
    Instantiate the network and load trained weights from a checkpoint file.

    Parameters
    ----------
    model : your model
    weights_path : str         — path to a .pt checkpoint saved by train.py
    device       : torch.device
    cfg          : Config      — must match the config used during training
                                 (feature_channels, max_displacement)

    Returns
    -------
    model : updated model
    """
    model.to(device)

    # Load the saved weight dictionary.
    # map_location ensures weights saved on GPU load correctly on CPU and vice versa.
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    # model.eval() is critical for inference:
    #   - switches BatchNorm to use running statistics instead of batch statistics
    #   - disables any dropout layers
    # Without this, predictions will be inconsistent and slightly wrong.
    model.eval()

    print(f"Loaded weights from {weights_path}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def preprocess_frame(frame):
    """
    Normalize a single 2-D BES frame to [0, 1] and convert to a
    (1, 1, H, W) PyTorch tensor ready for the network.

    Parameters
    ----------
    frame : (H, W) float array — raw BES frame

    Returns
    -------
    tensor : (1, 1, H, W) float32 tensor
    """
    frame = frame.astype(np.float32)
    frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
    # unsqueeze twice: add batch dim (B=1) and channel dim (C=1)
    return torch.tensor(frame).unsqueeze(0).unsqueeze(0)


def preprocess_pair(frameA, frameB):
    """
    Normalize two consecutive BES frames jointly to [0, 1].
    Joint normalization uses the combined min/max of both frames, so their
    relative intensities are preserved
 
    Parameters
    ----------
    frameA, frameB : (H, W) float arrays — raw consecutive BES frames
 
    Returns
    -------
    tensorA, tensorB : (1, 1, H, W) float32 tensors, ready for the network
    """
    fA = frameA.astype(np.float32)
    fB = frameB.astype(np.float32)
 
    joint_min = min(fA.min(), fB.min())
    joint_max = max(fA.max(), fB.max())
    scale     = joint_max - joint_min + 1e-8
 
    fA = (fA - joint_min) / scale
    fB = (fB - joint_min) / scale
 
    tensorA = torch.tensor(fA).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
    tensorB = torch.tensor(fB).unsqueeze(0).unsqueeze(0)
 
    return tensorA, tensorB


def predict_pair(model, frameA, frameB, device):
    """
    Predict the displacement field between a single pair of BES frames.

    Parameters
    ----------
    model          : trained SiameseDisplacementNet in eval mode
    frameA, frameB : (H, W) float arrays — consecutive raw BES frames

    Returns
    -------
    flow : (2, H, W) float32 numpy array — predicted displacement in pixels
           channel 0 = dx, channel 1 = dy
    """
    tensorA, tensorB = preprocess_pair(frameA, frameB)
    tensorA = tensorA.to(device)
    tensorB = tensorB.to(device)
 
    with torch.no_grad():
        flow_tensor = model(tensorA, tensorB)   # (1, 2, H, W)
 
    return flow_tensor[0].cpu().numpy()         # (2, H, W)


def predict_sequence(model, frames, device, batch_size=16):
    """
    Predict velocity fields for every consecutive pair in a BES frame sequence.

    Processes pairs in batches for efficiency — much faster than calling
    predict_pair in a Python loop for large sequences.

    Parameters
    ----------
    model      : trained SiameseDisplacementNet in eval mode
    frames     : (N, H, W) float array — the full BES time series
    device     : torch.device
    batch_size : number of frame pairs processed per forward pass.

    Returns
    -------
    velocities : (N-1, 2, H, W) float32 numpy array
                 velocities[t] is the displacement from frames[t] to frames[t+1]
    """
    N, H, W = frames.shape
    n_pairs  = N - 1

    # Pre-allocate the output array
    velocities = np.zeros((n_pairs, 2, H, W), dtype=np.float32)

    print(f"Predicting {n_pairs} frame pairs in batches of {batch_size}...")

    with torch.no_grad():
        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)
            
            batchA_np = frames[start  : end    ].astype(np.float32)  # (B, H, W)
            batchB_np = frames[start+1: end + 1].astype(np.float32)  # (B, H, W)
 
            # Joint normalization per pair: compute min/max over each (A,B) pair
            # independently using axis=(1,2) so shape is (B, 1, 1).
            pair_min = np.minimum(
                batchA_np.min(axis=(1, 2), keepdims=True),
                batchB_np.min(axis=(1, 2), keepdims=True),
            )
            pair_max = np.maximum(
                batchA_np.max(axis=(1, 2), keepdims=True),
                batchB_np.max(axis=(1, 2), keepdims=True),
            )
            scale     = pair_max - pair_min + 1e-8
            batchA_np = (batchA_np - pair_min) / scale
            batchB_np = (batchB_np - pair_min) / scale
 
            # Add channel dim: (B, H, W) → (B, 1, H, W)
            batchA = torch.tensor(batchA_np[:, None]).to(device)
            batchB = torch.tensor(batchB_np[:, None]).to(device)
 
            flow_batch = model(batchA, batchB)   # (B, 2, H, W)
            velocities[start:end] = flow_batch.cpu().numpy()
 
            print(f"  Processed pairs {start}–{end-1} / {n_pairs-1}")

    print(f"Done. Output shape: {velocities.shape}")
    return velocities


def to_physical_units(velocities, pixel_size_cm, frame_interval_us):
    """
    Convert displacement fields from pixels/frame to cm/μs.

    Parameters
    ----------
    velocities       : (N-1, 2, H, W) — predicted displacements in pixels
    pixel_size_cm    : physical size of one BES pixel in cm
                       (from the BES geometric calibration)
    frame_interval_us: time between consecutive frames in microseconds

    Returns
    -------
    velocities_physical : (N-1, 2, H, W) in cm/μs
    """
    return velocities * (pixel_size_cm / frame_interval_us)


def plot_prediction(frameA, frameB, flow, save_path=None, title=None):
    """
    Diagnostic plot for a single predicted frame pair:
        - Frame A (input)
        - Frame B (input)
        - dx map (horizontal velocity)
        - dy map (vertical velocity)
        - velocity magnitude
        - quiver overlay on frame A

    Parameters
    ----------
    frameA, frameB : (H, W) float arrays
    flow           : (2, H, W) float array — predicted displacement in pixels
    save_path      : if provided, save figure to this path
    title          : optional figure title string
    """
    dx  = flow[0, :, :]
    dy  = flow[1, :, :]
    mag = np.sqrt(dx**2 + dy**2)

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(title or 'BES flow prediction', fontsize=13, fontweight='bold')
    gs  = gridspec.GridSpec(1, 6, figure=fig, wspace=0.35)

    # ── Frame A ──────────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(frameA, cmap='inferno', origin='upper')
    ax0.set_title('Frame A')
    ax0.set_xlabel('x (px)')
    ax0.set_ylabel('y (px)')

    # ── Frame B ──────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(frameB, cmap='inferno', origin='upper')
    ax1.set_title('Frame B')
    ax1.set_xlabel('x (px)')

    # ── dx map ───────────────────────────────────────────────────────────────
    # CenteredNorm ensures zero displacement is always the midpoint colour
    ax2 = fig.add_subplot(gs[2])
    im2 = ax2.imshow(dx, cmap='RdBu_r', origin='upper',
                     norm=CenteredNorm())
    ax2.set_title('dx  (pixels)')
    ax2.set_xlabel('x (px)')
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    # ── dy map ───────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[3])
    im3 = ax3.imshow(dy, cmap='RdBu_r', origin='upper',
                     norm=CenteredNorm())
    ax3.set_title('dy  (pixels)')
    ax3.set_xlabel('x (px)')
    fig.colorbar(im3, ax=ax3, shrink=0.8)

    # ── Magnitude ─────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[4])
    im4 = ax4.imshow(mag, cmap='viridis', origin='upper', vmin=0)
    ax4.set_title('|v|  (pixels)')
    ax4.set_xlabel('x (px)')
    fig.colorbar(im4, ax=ax4, shrink=0.8)

    # ── Quiver overlay ────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[5])
    ax5.imshow(frameA, cmap='inferno', origin='upper')

    step   = 8
    H, W   = frameA.shape
    ys     = np.arange(step // 2, H, step)
    xs     = np.arange(step // 2, W, step)
    xx, yy = np.meshgrid(xs, ys)
    u      = dx[yy, xx]
    v      = dy[yy, xx]

    ax5.quiver(xx, yy, u, -v,        # -v for image coordinate convention
               color='cyan',
               scale=60,
               scale_units='width',
               width=0.004,
               headwidth=4)
    ax5.set_title('Quiver overlay')
    ax5.set_xlabel('x (px)')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def plot_velocity_timeseries(velocities, save_path=None):
    """
    Plot the spatially averaged dx, dy and magnitude as a function of
    frame index — a quick way to check whether the predicted flow evolves
    plausibly over the BES time series.

    Parameters
    ----------
    velocities : (N-1, 2, H, W) — full predicted velocity sequence
    """
    # Spatial mean at each time step
    mean_dx  = velocities[:, 0, :, :].mean(axis=(1, 2))
    mean_dy  = velocities[:, 1, :, :].mean(axis=(1, 2))
    mean_mag = np.sqrt(velocities[:, 0]**2 +
                       velocities[:, 1]**2).mean(axis=(1, 2))

    frames = np.arange(len(mean_dx))

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle('Spatially averaged velocity vs frame index',
                 fontsize=12, fontweight='bold')

    axes[0].plot(frames, mean_dx, color='steelblue')
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[0].set_ylabel('Mean dx  (px)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frames, mean_dy, color='darkorange')
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_ylabel('Mean dy  (px)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(frames, mean_mag, color='forestgreen')
    axes[2].set_ylabel('Mean |v|  (px)')
    axes[2].set_xlabel('Frame index')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Save and load utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_velocities(velocities, path):
    """Save the full velocity array as a .npy file."""
    np.save(path, velocities)
    print(f"Velocities saved to {path}  shape: {velocities.shape}")


def load_velocities(path):
    """Load a previously saved velocity array."""
    velocities = np.load(path)
    print(f"Loaded velocities from {path}  shape: {velocities.shape}")
    return velocities


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Run trained BES flow model on a new frame sequence'
    )
    parser.add_argument('--frames',   required=True,
                        help='Path to .npy file of BES frames, shape (N, 64, 64)')
    parser.add_argument('--weights',  required=True,
                        help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--output',   default='outputs/velocities.npy',
                        help='Where to save the predicted velocity array')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of frame pairs per forward pass')
    parser.add_argument('--pixel_size_cm',    type=float, default=None,
                        help='BES pixel size in cm (for physical unit conversion)')
    parser.add_argument('--frame_interval_us', type=float, default=None,
                        help='Frame interval in microseconds')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ── Load frames ───────────────────────────────────────────────────────────
    print(f"Loading frames from {args.frames}...")
    bes_frames = np.load(args.frames)
    print(f"Loaded {bes_frames.shape[0]} frames, shape {bes_frames.shape}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(args.weights, device)

    # ── Predict ───────────────────────────────────────────────────────────────
    velocities = predict_sequence(model, bes_frames, device,
                                  batch_size=args.batch_size)

    # ── Convert to physical units (optional) ──────────────────────────────────
    if args.pixel_size_cm and args.frame_interval_us:
        velocities = to_physical_units(
            velocities, args.pixel_size_cm, args.frame_interval_us
        )
        print(f"\nConverted to cm/μs  "
              f"(pixel size: {args.pixel_size_cm} cm, "
              f"frame interval: {args.frame_interval_us} μs)")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_velocities(velocities, args.output)

    # ── Visualise first pair ──────────────────────────────────────────────────
    plot_prediction(
        bes_frames[0], bes_frames[1], velocities[0],
        save_path = args.output.replace('.npy', '_pair0.png'),
        title     = f"Frame pair 0→1  |  weights: {os.path.basename(args.weights)}"
    )

    # ── Time series plot ──────────────────────────────────────────────────────
    plot_velocity_timeseries(
        velocities,
        save_path = args.output.replace('.npy', '_timeseries.png')
    )