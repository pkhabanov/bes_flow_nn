# bes_flow/predict.py
#
# Run optical flow inference on an experimental BES HDF5 file using up to
# five methods, save per-method results and optionally plot a Vr radial profile.
#
# Methods:
#   1. PWCNet         (--weights_pwc)
#   2. BESFlowNetS    (--weights_flownets)
#   3. RAFT-small     (torchvision pretrained)
#   4. Farneback      (OpenCV)
#   5. ODP            (bes_flow.odp)
#
# Output files (one per method, alongside the input file):
#   <stem>_pwc.h5, <stem>_flownet.h5, <stem>_raft.h5,
#   <stem>_farneback.h5, <stem>_odp.h5
#
# Each output contains:
#   vR, vZ            - velocity arrays (m/s), shape (n_frames, 8, 8)
#   R, Z              - spatial coordinates (m) at 8x8 resolution
#   time              - time axis (ms) for the n_frames velocity frames
#   R_profile         - R coordinates for the radial profile (full 64-pt grid)
#   vZ_profile        - vZ averaged over time and Z (64-pt radial profile)
#
# Usage
# ─────
#   python predict.py \
#       --input  data/shot12345.h5 \
#       --weights_pwc       checkpoints/pwc_best.pt \
#       --weights_flownets  checkpoints/flownets_best.pt \
#       [--skip_raft] [--skip_farneback] [--skip_odp] \
#       [--skip_pwc] [--skip_flownets] \
#       [--plot]

import os
import argparse
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch

from bes_flow.compare_methods import (
    load_pwc,
    load_flownets,
    run_farneback,
    run_raft_small,
    run_odp,
)


def load_bes_h5(path):
    """
    Load images and coordinate axes from a BES HDF5 file.

    Expected entries: 'images', 'time', 'R', 'Z'
      images : (N, H, W)  or  (N, 1, H, W)
      time   : (N,)  [ms]
      R      : (W,)  radial positions  [cm]
      Z      : (H,)  poloidal positions [cm]

    Returns
    -------
    images : (N, H, W) float32
    time   : (N,) float32
    R      : (W,) float32
    Z      : (H,) float32
    """
    with h5py.File(path, 'r') as f:
        images = f['images'][()].astype(np.float32)
        time   = f['time'][()].astype(np.float32)
        R      = f['R'][()].astype(np.float32)
        Z      = f['Z'][()].astype(np.float32)

    if images.ndim == 4:          # (N, 1, H, W) -> (N, H, W)
        images = images[:, 0]

    print(f"  Loaded {images.shape[0]} frames  ({images.shape[1]}x{images.shape[2]} px)")
    print(f"  R: {R[0]:.2f} - {R[-1]:.2f} cm ({len(R)} pts)")
    print(f"  Z: {Z[0]:.2f} - {Z[-1]:.2f} cm ({len(Z)} pts)")
    return images, time, R, Z


def normalize_sequence(images):
    """
    Joint normalization across the whole sequence to [0, 1].
    Returns float32 array with the same shape as input.
    """
    vmin = images.min()
    vmax = images.max()
    if vmax > vmin:
        return (images - vmin) / (vmax - vmin)
    return images


def make_pairs(images, per_pair_norm=False):
    """
    Build consecutive frame pairs.
 
    Parameters
    ----------
    images        : (N, H, W) float32
    per_pair_norm : bool
        If True, each pair (A, B) is normalised jointly to [0, 1] using the
        min/max across both frames. 

    Returns
    -------
    framesA : (N-1, 1, H, W)
    framesB : (N-1, 1, H, W)
    """
    framesA = images[:-1, np.newaxis].copy()   # (N-1, 1, H, W)
    framesB = images[1:,  np.newaxis].copy()
 
    if per_pair_norm:
        # Per-pair min/max across both frames
        # Flatten spatial dims to (N, 2*H*W), then reduce along that axis.
        flat   = np.concatenate([framesA, framesB], axis=1).reshape(len(framesA), -1)
        vmin   = flat.min(axis=1)[:, None, None, None]   # (N, 1, 1, 1)
        vmax   = flat.max(axis=1)[:, None, None, None]
        scale  = np.where(vmax - vmin > 1e-6, vmax - vmin, 1.0)
        framesA = (framesA - vmin) / scale
        framesB = (framesB - vmin) / scale
 
    return framesA, framesB


# ─────────────────────────────────────────────────────────────────────────────
# Inference wrappers
# ─────────────────────────────────────────────────────────────────────────────
# Convention: all wrappers return (N, 2, H, W) float32
#   channel 0 = dx (R direction, pixels/frame)
#   channel 1 = dy (Z direction, pixels/frame)

def run_bes_model(model, framesA, framesB, device, batch_size=16,
                  per_frame_norm=True):
    """
    Generic runner for PWCNet and BESFlowNetS.
 
    Parameters
    ----------
    per_frame_norm : bool
        If True (default), each frame pair is normalised jointly to [0, 1]
    """
    N = len(framesA)
    H, W = framesA.shape[2], framesA.shape[3]
    flows = np.zeros((N, 2, H, W), dtype=np.float32)
 
    model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            bA  = torch.from_numpy(framesA[start:end]).to(device)
            bB  = torch.from_numpy(framesB[start:end]).to(device)
 
            if per_frame_norm:
                # Normalise each pair jointly: min/max over both frames
                pair     = torch.cat([bA, bB], dim=1)   # (B, 2, H, W)
                vmin     = pair.flatten(1).min(dim=1).values[:, None, None, None]
                vmax     = pair.flatten(1).max(dim=1).values[:, None, None, None]
                scale    = (vmax - vmin).clamp(min=1e-6)
                bA       = (bA - vmin) / scale
                bB       = (bB - vmin) / scale
 
            flows[start:end] = model(bA, bB).cpu().numpy()
 
    return flows


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 output
# ─────────────────────────────────────────────────────────────────────────────

def save_result(out_path, vR, vZ, R, Z, time_pairs, R_profile, vZ_profile):
    """
    Write velocimetry results to an HDF5 file.

    Datasets
    --------
    vR, vZ      : (n_frames, n_Z, n_R)  [m/s]
    R           : (n_R,)  [cm]
    Z           : (n_Z,)  [cm]
    time        : (n_frames,)  [ms]
    R_profile   : (W,)   R grid for the profile [cm]
    vZ_profile  : (W,)   vZ(R) time-and-Z averaged [m/s]
    """
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('vR',         data=vR,         compression='gzip')
        f.create_dataset('vZ',         data=vZ,         compression='gzip')
        f.create_dataset('R',          data=R)
        f.create_dataset('Z',          data=Z)
        f.create_dataset('time',       data=time_pairs)
        f.create_dataset('R_profile',  data=R_profile)
        f.create_dataset('vZ_profile', data=vZ_profile)
    print(f"  Saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

_METHOD_COLORS = {
    'pwc':      'steelblue',
    'flownet':  'darkorange',
    'odp':     'forestgreen',
    'farneback':'mediumpurple',
    'raft':      'crimson',
}

def plot_v_profile(results, velocity_component='Z', output_path=None):
    """
    Single figure with two side-by-side panels:
      left  — V radial profile (time and Z averaged)
      right — Reynolds stress <Vr*Vz> radial profile (time and Z averaged)
 
    Parameters
    ----------
    results              : list of dicts, each with keys
                             'label', 'R_profile', 'vR_profile', 'vZ_profile',
                             'ReynoldsStress_profile'
    velocity_component   : 'R' to plot Vr (default), 'Z' to plot Vz
    output_path          : str or None - save figure if given
    """
    if velocity_component == 'R':
        v_key   = 'vR_profile'
        v_label = r'$\langle V_R \rangle$  (m/s)'
        v_title = r'$\langle V_R \rangle$ - time & Z averaged'
    elif velocity_component == 'Z':
        v_key   = 'vZ_profile'
        v_label = r'$\langle V_Z \rangle$  (m/s)'
        v_title = r'$\langle V_Z \rangle$ - time & Z averaged'
    else:
        raise ValueError(f"velocity_component must be 'R' or 'Z', got {velocity_component!r}")
 
    fig, (ax_v, ax_rs, ax_fl) = plt.subplots(1, 3, figsize=(18, 4), sharex=True)
 
    for res in results:
        label = res['label']
        R     = res['R_profile']
        color = _METHOD_COLORS.get(res['method_key'], None)
        ax_v.plot(R,  res[v_key],                    label=label, color=color, lw=2)
        ax_rs.plot(R, res['ReynoldsStress_profile'],  label=label, color=color, lw=2)
        ax_fl.plot(R, res['flux_profile'],  label=label, color=color, lw=2)
 
    ax_v.set_xlabel('R  (cm)', fontsize=12)
    ax_v.set_ylabel(v_label, fontsize=12)
    ax_v.set_title(v_title, fontsize=13)
    ax_v.legend(fontsize=11)
    ax_v.grid(True, alpha=0.3)
    ax_v.axhline(0, color='k', lw=0.8, ls='--')
 
    ax_rs.set_xlabel('R  (cm)', fontsize=12)
    ax_rs.set_ylabel(r'$\langle \tilde{V}_R \tilde{V}_Z \rangle  (m^2/s^2)$', fontsize=12)
    ax_rs.set_title(r'Reynolds stress $\langle \tilde{V}_R \tilde{V}_Z \rangle$ - time & Z averaged', fontsize=13)
    ax_rs.legend(fontsize=11)
    ax_rs.grid(True, alpha=0.3)
    ax_rs.axhline(0, color='k', lw=0.8, ls='--')

    ax_fl.set_xlabel('R  (cm)', fontsize=12)
    #ax_fl.set_ylabel(r'$\langle \tilde{V}_R \tilde{n} \rangle  (m^{-2}s^{-1})$', fontsize=12)
    ax_fl.set_ylabel(r'$\langle \tilde{V}_R \tilde{n} \rangle  (a.u.)$', fontsize=12)
    ax_fl.set_title(r'Particle flux $\langle \tilde{V}_R \tilde{n} \rangle$ - time & Z averaged', fontsize=13)
    ax_fl.legend(fontsize=11)
    ax_fl.grid(True, alpha=0.3)
    ax_fl.axhline(0, color='k', lw=0.8, ls='--')
 
    fig.tight_layout()
 
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot -> {output_path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Flow postrpocessing
# ─────────────────────────────────────────────────────────────────────────────

def postprocess_flows(flows_px, frames, R_interp, Z_interp, time_pairs,
                      orig_res, results_to_plot, stem, method_key, label):
    """
    Convert pixel/frame flows to physical units, downsample to orig_res,
    compute profiles, and save results

    Parameters
    ----------
    flows_px  : (N, 2, H, W) float32  [pixels/frame]
                channel 0 = vR direction (x / R axis)
                channel 1 = vZ direction (y / Z axis)
    frames    : (N, H, W) float32, array of frameAs
    R_interp  : (W,) R coordinates of interpolated images [cm]
    Z_interp  : (H,) Z coordinates of interpolated images [cm]
    time_pairs: (N,) time of frame A in each pair [ms]
    orig_res  : (n_R, n_Z) output resolution, default (8, 8)

    Returns
    -------
    vR_down  : (N, n_Z, n_R) [m/s]
    vZ_down  : (N, n_Z, n_R) [m/s]
    R_down   : (n_R,) [cm]
    Z_down   : (n_Z,) [cm]
    vR_full  : (N, H, W) full-resolution vR in m/s  (for profiles)
    vZ_full  : (N, H, W) full-resolution vZ in m/s  (for Reynolds stress profile)
    """
    print(f"\nPost-processing {label}...")
    vR_full = flows_px[:, 0].copy()   # (N, H, W)
    vZ_full = flows_px[:, 1].copy()

    dR = (R_interp[1] - R_interp[0]) / 100.0   # cm -> m
    dZ = (Z_interp[1] - Z_interp[0]) / 100.0
    dt = (time_pairs[1] - time_pairs[0]) / 1000.0  # ms -> s

    # Convert pixels/frame -> m/s
    vR_full *= dR / dt
    vZ_full *= dZ / dt

    # Downsample to orig_res
    n_R, n_Z     = orig_res          # e.g. 8, 8
    res_x, res_y = vR_full.shape[2], vR_full.shape[1]   # W, H of interpolated images
    px = res_x // n_R
    py = res_y // n_Z

    R_down = np.array([R_interp[i * px:(i + 1) * px].mean() for i in range(n_R)])
    Z_down = np.array([Z_interp[j * py:(j + 1) * py].mean() for j in range(n_Z)])

    vR_down = np.zeros((vR_full.shape[0], n_Z, n_R), dtype=np.float32)
    vZ_down = np.zeros((vZ_full.shape[0], n_Z, n_R), dtype=np.float32)

    for j in range(n_Z):
        for i in range(n_R):
            vR_sub = vR_full[:, j * py:(j + 1) * py, i * px:(i + 1) * px]
            vZ_sub = vZ_full[:, j * py:(j + 1) * py, i * px:(i + 1) * px]
            vR_down[:, j, i] = vR_sub.mean(axis=(1, 2))
            vZ_down[:, j, i] = vZ_sub.mean(axis=(1, 2))

    R_profile  = R_interp.copy()
    vR_profile = vR_full.mean(axis=(0, 1))                # avg over time and Z
    vZ_profile = vZ_full.mean(axis=(0, 1))                # avg over time and Z
    # fluctuationg components of vR, vZ
    dvR = vR_full - vR_full.mean(axis=0)
    dvZ = vZ_full - vZ_full.mean(axis=0)
    # Reynolds stress <vR*vZ>
    RS_profile = (dvR * dvZ).mean(axis=(0, 1)) 
    # dn
    frames = frames[:-1]
    frames = frames - frames.mean(axis=0)
    # particle flux profile <vR*dn>
    flux_profile = (dvR * frames).mean(axis=(0, 1))   

    # save to hdf5
    out_path = os.path.join(args.output_dir, f"{stem}_{method_key}.h5")
    save_result(out_path, vR_down, vZ_down, R_down, Z_down,
                time_pairs, R_profile, vZ_profile)

    results_to_plot.append({
        'label':                  label,
        'method_key':             method_key,
        'R_profile':              R_profile,
        'vR_profile':             vR_profile,
        'vZ_profile':             vZ_profile,
        'ReynoldsStress_profile': RS_profile,
        'flux_profile':           flux_profile,
    })

    return results_to_plot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Optical flow inference on experimental BES data'
    )
 
    # Input
    parser.add_argument('--input', required=True,
                        help='HDF5 file with BES images (keys: images, time, R, Z)')
    parser.add_argument('--output_dir', default=None,
                        help='Directory for output files '
                             '(default: same directory as input)')
    parser.add_argument('--batch_size', type=int, default=16)
 
    # Neural-net weights (optional — method is skipped when not provided and
    # not explicitly forced via the corresponding skip flag)
    parser.add_argument('--weights_pwc',      default=None,
                        help='Checkpoint for PWCNet')
    parser.add_argument('--weights_flownets', default=None,
                        help='Checkpoint for BESFlowNetS')
 
    # Skip flags
    parser.add_argument('--skip_pwc',       action='store_true')
    parser.add_argument('--skip_flownets',  action='store_true')
    parser.add_argument('--skip_raft',      action='store_true')
    parser.add_argument('--skip_farneback', action='store_true')
    parser.add_argument('--skip_odp',       action='store_true')
 
    # Output resolution (original BES grid)
    parser.add_argument('--orig_res_x', type=int, default=8,
                        help='Original BES resolution in R direction (default 8)')
    parser.add_argument('--orig_res_y', type=int, default=8,
                        help='Original BES resolution in Z direction (default 8)')
 
    # Plot flag
    parser.add_argument('--plot', action='store_true',
                        help='Plot velocity radial profiles after inference')
    parser.add_argument('--velocity_component', choices=['R', 'Z'], default='Z',
                        help="Velocity component to plot: 'R' for vR, 'Z' for vZ")
 
    args   = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
 
    # ── Output directory ────────────────────────────────────────────────────
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.input))
    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
 
    # ── Load & preprocess ───────────────────────────────────────────────────
    print(f"\nLoading {args.input} ...")
    images, time_ax, R, Z = load_bes_h5(args.input)
    N = 10000
    images, time_ax = images[:N, :, :], time_ax[:N]
 
    # Neural nets: per-pair normalization
    print(f"Building frame pairs for neural nets ")
    framesA_norm, framesB_norm = make_pairs(images, per_pair_norm=True)
 
    # Classical methods: sequence-normalised pairs — joint [0,1] scale
    print("Normalizing sequence jointly (for classical methods)...")
    images_norm = normalize_sequence(images)
    framesA, framesB = make_pairs(images_norm)
 
    n_pairs = len(framesA)
    print(f"  {n_pairs} consecutive pairs")
 
    # Time axis for the velocity frames (time of frame A in each pair)
    time_pairs = time_ax[:n_pairs]
    orig_res   = (args.orig_res_x, args.orig_res_y)
 
    # ── Collect results for optional plotting ────────────────────────────────
    results_to_plot = []
 
    # ── 1. PWCNet ────────────────────────────────────────────────────────────
    if not args.skip_pwc:
        if args.weights_pwc is None:
            print("\n[PWC] --weights_pwc not provided — skipping")
        else:
            print("\n--- PWCNet ---")
            model = load_pwc(args.weights_pwc, device)
            t0 = time.perf_counter()
            flows = run_bes_model(model, framesA_norm, framesB_norm, device, args.batch_size)
            elapsed = time.perf_counter() - t0
            print(f"  Elapsed: {elapsed:.3f} s  ({elapsed * 1000 / n_pairs:.2f} ms/frame)")
            del model
            results_to_plot = postprocess_flows(flows, images, R, Z, time_pairs, orig_res, 
                                                results_to_plot, stem, 'pwc', 'PWC')
 
    # ── 2. BESFlowNetS ───────────────────────────────────────────────────────
    if not args.skip_flownets:
        if args.weights_flownets is None:
            print("\n[FlowNetS] --weights_flownets not provided — skipping")
        else:
            print("\n--- BESFlowNetS ---")
            model = load_flownets(args.weights_flownets, device)
            t0 = time.perf_counter()
            flows = run_bes_model(model, framesA_norm, framesB_norm, device, args.batch_size)
            elapsed = time.perf_counter() - t0
            print(f"  Elapsed: {elapsed:.3f} s  ({elapsed * 1000 / n_pairs:.2f} ms/frame)")
            del model
            results_to_plot = postprocess_flows(flows, images, R, Z, time_pairs, orig_res, 
                                                results_to_plot, stem, 'flownet', 'FlowNetS')
 
    # ── 3. ODP ───────────────────────────────────────────────────────────────
    if not args.skip_odp:
        print("\n--- ODP ---")
        flows, elapsed, ms_per_frame = run_odp(framesA_norm, framesB_norm)
        print(f"  Elapsed: {elapsed:.3f} s  ({ms_per_frame:.2f} ms/frame)")
        results_to_plot = postprocess_flows(flows, images, R, Z, time_pairs, orig_res, 
                                            results_to_plot, stem, 'odp', 'ODP')
    
    # ── 4. RAFT-small ────────────────────────────────────────────────────────
    if not args.skip_raft:
        print("\n--- RAFT-small ---")
        flows, elapsed, ms_per_frame = run_raft_small(framesA_norm, framesB_norm, device, args.batch_size)
        print(f"  Elapsed: {elapsed:.3f} s  ({ms_per_frame:.2f} ms/frame)")
        results_to_plot = postprocess_flows(flows, images, R, Z, time_pairs, orig_res, 
                                            results_to_plot, stem, 'raft', 'RAFT-small')
 
    # ── 5. Farneback ─────────────────────────────────────────────────────────
    if not args.skip_farneback:
        print("\n--- Farneback ---")
        flows, elapsed, ms_per_frame = run_farneback(framesA_norm, framesB_norm)
        print(f"  Elapsed: {elapsed:.3f} s  ({ms_per_frame:.2f} ms/frame)")
        results_to_plot = postprocess_flows(flows, images, R, Z, time_pairs, orig_res, 
                                            results_to_plot, stem, 'farneback', 'Farneback')
 
    # ── Plot ─────────────────────────────────────────────────────────────────
    if args.plot:
        if results_to_plot:
            plot_path = os.path.join(args.output_dir, f"{stem}_v{args.velocity_component.lower()}_profile.png")
            plot_v_profile(results_to_plot, velocity_component=args.velocity_component,
                           output_path=plot_path)
        else:
            print("\nNo results to plot — all methods were skipped.")
 
    print("\nDone.")
