# bes_flow/compare_jhopkins.py
#
# Benchmark comparison of optical flow methods on the Johns Hopkins Turbulence
# Database (JHTDB) mixing-layer dataset.
# https://turbulence.idies.jhu.edu/database
#
# Ground-truth flow is the *true* velocity field from the simulation,
# converted from physical units to pixel displacement:
#
#   flow_x_px[i] = vx[i] * dt / dx_phys      (x-direction)
#   flow_y_px[i] = vy[i] * dt / dy_phys      (y-direction)
#
# where dx_phys = (x_max - x_min) / (nx - 1)  is the physical pixel size.
# Frame pair i uses images[i] and images[i+1] with the velocity at frame i.
#
# The 2-D slice lies in the xy plane at a fixed z value.  The two in-plane
# velocity components are vx (vel_arr[:, 0]) and vy (vel_arr[:, 1]).
#
# Data sources
# ────────────
# Two modes are supported:
#   --h5   : load a pre-saved HDF5 file (fast, reproducible)
#   --fetch: query the JHTDB API live via givernylocal (slow, requires token)
#
# HDF5 file layout:
#   images  (Nframes, ny, nx)  normalised density fluctuation
#   vx      (Nframes, ny, nx)  x-velocity  [physical units / s]
#   vy      (Nframes, ny, nx)  y-velocity  [physical units / s]
#   time    (Nframes,)         simulation time of each frame
#   X       (nx,)              x grid coordinates
#   Y       (ny,)              y grid coordinates
#
# Usage
# ─────
#   # From a saved HDF5 cache:
#   python -m bes_flow.compare_jhopkins \
#       --h5               jhopkins_images_mixing_64x64_60fr.h5 \
#       --weights_pwc      checkpoints/pwc_best.pt \
#       --weights_flownets checkpoints/flownets_best.pt \
#       --output           outputs/jhopkins/
#
#   # Live fetch (requires givernylocal + auth token):
#   python -m bes_flow.compare_jhopkins \
#       --fetch \
#       --auth_token       edu.jhu.pha.turbulence.testing-201406 \
#       --output           outputs/jhopkins/
#
#   # Skip slow or unavailable methods:
#   python -m bes_flow.compare_jhopkins \
#       --h5 data.h5 --skip_odp --skip_raft \
#       --weights_pwc checkpoints/pwc_best.pt


import os
import time
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.colors import CenteredNorm

import torch

# ── Reuse all runner functions and reporting from compare_methods ──────────
from bes_flow.compare_methods import (
    load_pwc, load_flownets,
    run_bes_model, run_farneback, run_raft_small, run_odp,
    print_comparison_table, plot_metric_bars, plot_comparison_examples,
    _METHOD_COLORS, _QUIVER_COLORS,
)
from bes_flow.config  import cfg
from bes_flow.dataset import BESDataset
from bes_flow.metrics import compute_all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_from_h5(h5_path):
    """
    Load a pre-saved JHTDB HDF5 file.

    Returns
    -------
    images    : (Nframes, ny, nx) float32 — normalised density
    vx        : (Nframes, ny, nx) float32 — x-velocity  [phys. units/s]
    vy        : (Nframes, ny, nx) float32 — y-velocity  [phys. units/s]
    times     : (Nframes,)        float64 — simulation time of each frame
    x_points  : (nx,)             float64 — x coordinates
    y_points  : (ny,)             float64 — y coordinates
    """
    print(f"Loading JHTDB data from: {h5_path}")
    with h5py.File(h5_path, 'r') as hf:
        images = hf['images'][()].astype(np.float32)
        vx     = hf['vx'][()].astype(np.float32)
        vy     = hf['vy'][()].astype(np.float32)
        times  = hf['time'][()]
        x_points = hf['X'][()]
        y_points = hf['Y'][()]

    print(f"  Frames   : {images.shape[0]}")
    print(f"  Grid     : {images.shape[2]} x {images.shape[1]}  (nx x ny)")
    print(f"  X-range  : {min(x_points):.1f} x {max(x_points):.1f}")
    print(f"  Y-range  : {min(y_points):.1f} x {max(y_points):.1f}")
    print(f"  Time span: {times[0]:.4f} - {times[-1]:.4f}")
    return images, vx, vy, times, x_points, y_points


def fetch_from_api(auth_token, dataset_title='mixing',
                   nx=64, ny=64, Nframes=200,
                   x_min=4.6, x_max=5,
                   y_min=5.6, y_max=6.0,
                   z_val=0.15 * 2*np.pi,
                   t_min=0.0, t_max=40.6, Nt=1015,
                   frame_offset=40,
                   save_h5=True):
    """
    Fetch density + velocity from the JHTDB API via givernylocal.
    The datset domain size is 1024^3; x,y,z = [0, 2*pi].

    Samples an xy plane at a fixed z value (z_val).  The two varying axes
    are x and y; vx and vy are the in-plane velocity components.

    Requires givernylocal to be installed and a valid auth_token.

    Returns the same tuple as load_from_h5.
    """
    try:
        from givernylocal.turbulence_dataset import turb_dataset
        from givernylocal.turbulence_toolkit import getData
    except ImportError:
        raise ImportError(
            "givernylocal is required for live fetching. "
            "Install it or use --h5 to load a pre-saved file."
        )

    dt          = 4 * (t_max - t_min) / Nt
    time_start  = t_min + frame_offset * dt
    time_end    = time_start + (Nframes - 1) * dt
    option      = [time_end, dt]

    x_points = np.linspace(x_min, x_max, nx, dtype=np.float64)
    y_points = np.linspace(y_min, y_max, ny, dtype=np.float64)
    # xy plane: x and y vary, z is fixed
    points   = np.array(
        [axis.ravel() for axis in
         np.meshgrid(x_points, y_points, z_val, indexing='ij')],
        dtype=np.float64,
    ).T

    output_path = './giverny_output'
    dataset     = turb_dataset(dataset_title=dataset_title,
                               output_path=output_path,
                               auth_token=auth_token)

    spatial_method   = 'lag8'
    spatial_operator = 'field'
    temporal_method  = 'none'

    print("Fetching density...")
    density_raw, times = getData(
        dataset, 'density', time_start, temporal_method,
        spatial_method, spatial_operator, points, option,
        return_times=True,
    )

    print("Fetching velocity...")
    velocity_raw, _ = getData(
        dataset, 'velocity', time_start, temporal_method,
        spatial_method, spatial_operator, points, option,
        return_times=True,
    )

    # Reshape into (Nframes, ny, nx) arrays
    images = np.zeros((Nframes, ny, nx), dtype=np.float32)
    vx_arr = np.zeros((Nframes, ny, nx), dtype=np.float32)
    vy_arr = np.zeros((Nframes, ny, nx), dtype=np.float32)

    for i in range(len(times)):
        dens_arr      = np.array(density_raw[i])
        dens_norm     = (dens_arr - dens_arr.mean()) / dens_arr.mean()
        images[i]     = np.reshape(dens_norm[:, 0], (nx, ny)).T

        vel_arr    = np.array(velocity_raw[i])
        vx_arr[i]  = np.reshape(vel_arr[:, 0], (nx, ny)).T   # x-component
        vy_arr[i]  = np.reshape(vel_arr[:, 1], (nx, ny)).T   # y-component

    if save_h5:
        fname = f'synthetic_data/jhopkins_{dataset_title}_x{x_min:.1f}-{x_max:.1f}_y{y_min:.1f}-{y_max:.1f}_z{z_val:.1f}_{nx}x{ny}_{Nframes}fr.h5'
        with h5py.File(fname, 'w') as hf:
            hf.create_dataset('images', data=images)
            hf.create_dataset('vx',     data=vx_arr)
            hf.create_dataset('vy',     data=vy_arr)
            hf.create_dataset('time',   data=times)
            hf.create_dataset('X',      data=x_points)
            hf.create_dataset('Y',      data=y_points)
        print(f"Saved fetched data to: {fname}")
        print(f"  Frames   : {images.shape[0]}")
        print(f"  Grid     : {images.shape[2]} x {images.shape[1]}  (nx x ny)")
        print(f"  X-range  : {min(x_points):.1f} x {max(x_points):.1f}")
        print(f"  Y-range  : {min(y_points):.1f} x {max(y_points):.1f}")
        print(f"  Z-value  : {z_val:.1f}")
        print(f"  Time span: {times[0]:.4f} - {times[-1]:.4f}")
    
    return images, vx_arr, vy_arr, times, x_points, y_points


# ─────────────────────────────────────────────────────────────────────────────
# Frame-pair and GT-flow construction
# ─────────────────────────────────────────────────────────────────────────────

def build_pairs(images, vx, vy, times, x_points, y_points):
    """
    Build consecutive frame pairs and convert velocity to GT pixel displacement.

    Convention (matching the BES pipeline):
      channel 0 = dx  (x-direction)
      channel 1 = dy  (y-direction)

    Displacement is estimated with a simple forward Euler step:
      flow_x_px[i] = vx[i] * dt[i] / dx_phys
      flow_y_px[i] = vy[i] * dt[i] / dy_phys

    where dt[i] = times[i+1] - times[i] is the inter-frame interval and
    dx_phys / dy_phys are the physical sizes of one pixel.

    Returns
    -------
    framesA   : (N_pairs, 1, ny, nx) float32 in [0, 1]
    framesB   : (N_pairs, 1, ny, nx) float32 in [0, 1]
    flows_gt  : (N_pairs, 2, ny, nx) float32 — pixel displacement
    """
    Nframes    = images.shape[0]
    ny, nx     = images.shape[1], images.shape[2]
    N_pairs    = Nframes - 1

    dx_phys = (x_points[-1] - x_points[0]) / (nx - 1)   # physical size per pixel
    dy_phys = (y_points[-1] - y_points[0]) / (ny - 1)

    # Normalise entire image stack to [0, 1] using global min/max
    img_min = images.min()
    img_max = images.max()
    if img_max > img_min:
        images_norm = (images - img_min) / (img_max - img_min)
    else:
        images_norm = images.copy()

    framesA  = np.zeros((N_pairs, 1, ny, nx), dtype=np.float32)
    framesB  = np.zeros((N_pairs, 1, ny, nx), dtype=np.float32)
    flows_gt = np.zeros((N_pairs, 2, ny, nx), dtype=np.float32)

    for i in range(N_pairs):
        dt = float(times[i + 1] - times[i])

        framesA[i, 0]  = images_norm[i]
        framesB[i, 0]  = images_norm[i + 1]
        flows_gt[i, 0] = vx[i] * dt / dx_phys   # x pixel displacement
        flows_gt[i, 1] = vy[i] * dt / dy_phys   # y pixel displacement

    print(f"\nBuilt {N_pairs} consecutive frame pairs")
    mag = np.sqrt(flows_gt[:, 0] ** 2 + flows_gt[:, 1] ** 2)
    print(f"  GT displacement — mean: {mag.mean():.3f} px  "
          f"max: {mag.max():.3f} px  "
          f"std: {mag.std():.3f} px")

    return framesA, framesB, flows_gt


def plot_animation(images, times, vx, vy, 
                   x_points, y_points,
                   colormap='RdBu_r',
                   interval=300,
                   scale=5):
    '''
    plot frames animation with velocity quiver overlay
    images, vx, vy have shape (Nframes, ny, nx)
    '''
    n_frames, ny, nx = images.shape
    
    fig, ax = plt.subplots()

    # Create dynamic elements of the animation
    ims = []
    j = 2  # index step for velocity quiver
    # set colormap limits
    vmin = -0.05 
    vmax = 0.05 
    x2d, z2d = np.meshgrid(x_points, y_points)

    for i in range(n_frames):
        artists = []
        im = ax.imshow(
            images[i, :, :],
            origin='lower',
            extent=[x_points[0], x_points[-1], y_points[0], y_points[-1]],
            cmap=colormap,
            interpolation='bilinear',
            vmin=vmin, #-1.5, #
            vmax=vmax, #1.5, #
            animated=True,
        )
        artists += [im]
        title = ax.annotate(
            f'frame={i}',
            xy=(0.5, 1.04),
            xycoords='axes fraction',
            horizontalalignment='center',
        )
        artists += [title]
        qv = ax.quiver(x2d[::j, ::j], z2d[::j, ::j], 
                    vx[i, ::j, ::j], vy[i, ::j, ::j], 
                    scale=scale)
        artists += [qv]
        ims.append(artists)
    cbar = fig.colorbar(im, ax=ax)
    #cbar.ax.set_ylabel(r'$\widetilde{n}(t) / \widetilde{n}_{rms}$')
    cbar.ax.set_ylabel('density')
    ani = animation.ArtistAnimation(fig, ims, interval=interval, repeat_delay=2000)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Compare optical flow methods on the JHTDB mixing-layer dataset'
    )

    # ── Data source (mutually exclusive) ──────────────────────────────────
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--h5', metavar='FILE',
                        help='Path to pre-saved JHTDB HDF5 file')
    source.add_argument('--fetch', action='store_true',
                        help='Fetch data live from the JHTDB API')

    # Fetch options (only used with --fetch)
    parser.add_argument('--auth_token',
                        default='edu.jhu.pha.turbulence.testing-201406',
                        help='JHTDB auth token (required with --fetch)')
    parser.add_argument('--dataset_title', default='mixing',
                        help='JHTDB dataset name (default: mixing)')
    parser.add_argument('--Nframes', type=int, default=60,
                        help='Number of frames to fetch (default: 60)')
    parser.add_argument('--nx', type=int, default=64)
    parser.add_argument('--ny', type=int, default=64)

    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument('--output', default=None,
                        help='Directory for figures and results')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_examples', type=int, default=3,
                        help='Number of qualitative examples to plot')

    # ── Neural net weights ────────────────────────────────────────────────
    parser.add_argument('--weights_pwc',      default=None)
    parser.add_argument('--weights_flownets', default=None)

    # –– Flag for animation plot –––––––––––––––––––––––––––––––––––––––––––
    parser.add_argument('--plot_ani', action='store_true')

    # ── Skip flags ────────────────────────────────────────────────────────
    parser.add_argument('--skip_pwc',       action='store_true')
    parser.add_argument('--skip_flownets',  action='store_true')
    parser.add_argument('--skip_odp',       action='store_true')
    parser.add_argument('--skip_farneback', action='store_true')
    parser.add_argument('--skip_raft',      action='store_true')

    args   = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ── Load / fetch data ─────────────────────────────────────────────────
    if args.h5:
        images, vx, vy, times, x_points, y_points = load_from_h5(args.h5)
    else:
        images, vx, vy, times, x_points, y_points = fetch_from_api(
            auth_token    = args.auth_token,
            dataset_title = args.dataset_title,
            nx            = args.nx,
            ny            = args.ny,
            Nframes       = args.Nframes,
        )

    # ── Build frame pairs and GT flow ─────────────────────────────────────
    n0 = 10. # skip first n0 frames
    framesA, framesB, flows_gt = build_pairs(images[n0:,:,:], vx[n0:,:,:], vy[n0:,:,:], times[n0:],
                                             x_points, y_points)

    test_dataset = BESDataset(framesA, framesB, flows_gt, augment=False)

    # ── Run methods ───────────────────────────────────────────────────────
    all_flows = {}
    all_times = {}   # {method: algorithm-only wall-clock seconds}

    # 1. PWCNet
    if not args.skip_pwc:
        if args.weights_pwc is None:
            print("\n  [PWC] --weights_pwc not provided — skipping")
        else:
            print("\nPWCNet:")
            model_pwc = load_pwc(args.weights_pwc, device)
            all_flows['PWC'], all_times['PWC'] = run_bes_model(
                model_pwc, test_dataset, device, args.batch_size
            )
            del model_pwc

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
        all_flows['ODP'], all_times['ODP'] = run_odp(framesA, framesB)

    # 4. Farneback
    if not args.skip_farneback:
        all_flows['Farneback'], all_times['Farneback'] = run_farneback(
            framesA, framesB
        )

    # 5. RAFT-small
    if not args.skip_raft:
        all_flows['RAFT-small'], all_times['RAFT-small'] = run_raft_small(
            framesA, framesB, device, args.batch_size
        )

    if args.plot_ani:
        plot_animation(images, times, vx, vy, x_points, y_points)

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
    plot_metric_bars(all_results, all_times, output_dir=args.output)
    plot_comparison_examples(framesA, framesB, flows_gt, all_flows,
                             n_examples=args.n_examples,
                             output_dir=args.output)

    print(f"\nDone. ")
