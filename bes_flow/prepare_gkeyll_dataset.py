# bes_flow/prepare_gkeyll_dataset.py
#
# Load .npz files with Gkeyll frames and builda a training/val/test datasets.
# The .npz files should be prepared on NERSC 
# see /global/cfs/cdirs/m3739/gkeyll/bes_testing
#
# Example usage:
#   python -m bes_flow.prepare_gkeyll_dataset \
#       --inputs  data/gkeyll_run_A.npz  data/gkeyll_run_B.npz \
#       --output  synthetic_data/gkeyll_train.h5

import os
import argparse
import numpy as np
from scipy.ndimage import zoom, gaussian_filter

from bes_flow.compare_jhopkins import build_pairs, plot_animation
from bes_flow.dataset import BESDataset, save_dataset_cache


def _downsample2d(arr, target_ny, target_nx):
    """
    Bilinear downsampling of a (T, ny, nx) array to (T, target_ny, target_nx).
    Uses scipy.ndimage.zoom.
    """
    zy = target_ny / arr.shape[1]
    zx = target_nx / arr.shape[2]
    if zy == 1.0 and zx == 1.0:
        return arr
    return zoom(arr, (1.0, zy, zx), order=1).astype(np.float32)


def _downsample1d(arr, target_nx):
    """
    Bilinear downsampling of a (nx,) array to (target_nx,).
    Uses scipy.ndimage.zoom.
    """
    zx = target_nx / arr.shape[0]
    if zx == 1.0:
        return arr
    return zoom(arr, zx, order=1).astype(np.float32)


def _view2d(arr, target_ny, target_nx):
    '''
    take the central part of arr with a size target_ny x target_nx
    '''
    _, ny, nx = arr.shape
    return arr[:, 
               ny//2 - target_ny//2: ny//2 + target_ny//2, 
               nx//2 - target_nx//2: nx//2 + target_nx//2]


def _view1d(arr, target_nx):
    nx = arr.shape[0]
    return arr[nx//2 - target_nx//2: nx//2 + target_nx//2]



def load_gkeyll_npz(npz_path, target_res=64, psf_fwhm=None):
    """
    Load a Gkeyll buffer file produced by 'save_gkeyll_npz'.

    Parameters
    ----------
    npz_path   : str
    target_res : int or (int, int) or None
        Spatial size to downsample to, as (ny, nx) or a single int for a
        square target. Default is 64 (-> 64 x 64). Pass None to keep the
        original resolution.

     Returns
    -------
    images : (T, target_ny, target_nx) float32 - density
    vx     : (T, target_ny, target_nx) float32 - VE_x
    vy     : (T, target_ny, target_nx) float32 - VE_y
    times  : (T,)                      float32 - frame indices cast to float
    x_grid : (nx,)                     float32
    y_grid : (ny,)                     float32
    attrs  : dict -- metadata (species, z_slice_val)
    """
    data = np.load(npz_path)

    images = data['density']
    vx     = data['VE_x']
    vy     = data['VE_y']
    x_grid = data['x']
    y_grid = data['y']
    times  = data['frames'].astype(np.float32)
    attrs  = {
        'species'    : str(data['species']),
        'z_slice_val': float(data['z_slice_val']),
    }
    
    Nframes    = images.shape[0]
    
    # apply spatial smoothing if needed
    if psf_fwhm is not None:
        psf_sigma = psf_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        print(f"  Applying PSF (FWHM={psf_fwhm:.1f} px, "
              f"sigma={psf_sigma:.2f} px) to images and velocities...")
        images = np.stack([gaussian_filter(images[i], sigma=psf_sigma)
                           for i in range(Nframes)], axis=0)
        vx     = np.stack([gaussian_filter(vx[i], sigma=psf_sigma)
                           for i in range(Nframes)], axis=0)
        vy     = np.stack([gaussian_filter(vy[i], sigma=psf_sigma)
                           for i in range(Nframes)], axis=0)
   
   # reduce spatial resolution if needed
    if target_res is not None:
        target_ny, target_nx = (
            (target_res, target_res) if isinstance(target_res, int)
            else target_res
        )
        _, ny, nx = images.shape
        print(f"  Downsampling Gkeyll images: ({ny}, {nx}) -> ({target_ny}, {target_nx})")
        images = _downsample2d(images, target_ny, target_nx)
        print(images.shape)
        vx     = _downsample2d(vx,     target_ny, target_nx)
        vy     = _downsample2d(vy,     target_ny, target_nx)
        x_grid = _downsample1d(x_grid, target_nx)
        y_grid = _downsample1d(y_grid, target_ny)

    return images[::4, :, :], vx[::4, :, :], vy[::4, :, :], times[::4], x_grid, y_grid, attrs


def _split_pairs(framesA, framesB, flows_gt, val_split, test_split, seed):
    """
    Split frame pairs randomly into train / val / test subsets.

    Parameters
    ----------
    framesA, framesB : (N, 1, ny, nx) float32
    flows_gt         : (N, 2, ny, nx) float32
    val_split        : float
    test_split       : float
    seed             : int — RNG seed for the test split

    Returns
    -------
    train_A, train_B, train_flows,
    val_A,   val_B,   val_flows,
    test_A,  test_B,  test_flows
    """
    N       = len(framesA)
    n_test  = max(1, int(round(N * test_split)))
    n_val   = max(1, int(round(N * val_split)))
    n_train = N - n_val - n_test

    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train : n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    return (
        framesA[train_idx], framesB[train_idx], flows_gt[train_idx],
        framesA[val_idx],   framesB[val_idx],   flows_gt[val_idx],
        framesA[test_idx],  framesB[test_idx],  flows_gt[test_idx],
    )


def build_gkeyll_dataset(npz_paths, output_path,
                         val_split=0.1,
                         test_split=0.1,
                         test_seed=42,
                         psf_fwhm=None):
    """
    Load one or more Gkeyll buffer files, build frame pairs, split into
    train/val/test, and save a training cache compatible with
    'load_dataset_cache()' and'make_dataloaders()'

    Parameters
    ----------
    npz_paths : list[str]
        Buffer .npz files produced by 'save_gkeyll_npz'.
    output_path : str
        Destination HDF5 file.
    val_split : float
    test_split : float
    test_seed : int
    psf_fwhm : float or None
        FWHM of a Gaussian PSF applied to images and velocities to mimic
        the BES point-spread function. None = no blurring.

    Returns
    -------
    train_dataset, val_dataset, test_dataset : BESDataset
    """
    if not npz_paths:
        raise ValueError("npz_paths is empty — provide at least one buffer file.")

    train_As, train_Bs, train_flows = [], [], []
    val_As,   val_Bs,   val_flows   = [], [], []
    test_As,  test_Bs,  test_flows  = [], [], []

    for path in npz_paths:
        print(f"\n{'='*60}")
        print(f"Loading: {path}")

        images, vx, vy, times, x_grid, y_grid, attrs = load_gkeyll_npz(path, psf_fwhm=psf_fwhm)
        # convert times to microseconds
        times *= 1e-6
        
        plot_animation(images, times, vx, vy, 
                       x_grid, y_grid, colormap='inferno', 
                       interval=30, scale=8e4,
                       vmin=np.min(images), vmax=np.max(images), save_ani=True)
        
        T, ny, nx = images.shape

        print(f"  Species     : {attrs.get('species', 'unknown')}")
        print(f"  z slice     : {attrs.get('z_slice_val', float('nan')):.4f}")
        print(f"  Frames      : {T}  grid: {ny} x {nx}")
        print(f"  Density     : [{images.min():.3g}, {images.max():.3g}]")
        print(f"  |VE_x| max  : {np.abs(vx).max():.3g}")
        print(f"  |VE_y| max  : {np.abs(vy).max():.3g}")

        framesA, framesB, flows_gt = build_pairs(
            images, vx, vy, times, x_grid, y_grid, psf_fwhm=None # no need to apply PSF twice
        )
        print(f"  Pairs built : {len(framesA)}")

        (tr_A, tr_B, tr_f,
         va_A, va_B, va_f,
         te_A, te_B, te_f) = _split_pairs(
            framesA, framesB, flows_gt,
            val_split, test_split, seed=test_seed,
        )
        print(f"  Split       -> train: {len(tr_A)}  val: {len(va_A)}  test: {len(te_A)}")

        train_As.append(tr_A);  train_Bs.append(tr_B);  train_flows.append(tr_f)
        val_As.append(va_A);    val_Bs.append(va_B);    val_flows.append(va_f)
        test_As.append(te_A);   test_Bs.append(te_B);   test_flows.append(te_f)

    # Concatenate across all buffer files
    train_A     = np.concatenate(train_As,    axis=0)
    train_B     = np.concatenate(train_Bs,    axis=0)
    train_flows = np.concatenate(train_flows, axis=0)
    val_A       = np.concatenate(val_As,      axis=0)
    val_B       = np.concatenate(val_Bs,      axis=0)
    val_flows   = np.concatenate(val_flows,   axis=0)
    test_A      = np.concatenate(test_As,     axis=0)
    test_B      = np.concatenate(test_Bs,     axis=0)
    test_flows  = np.concatenate(test_flows,  axis=0)

    print(f"\n{'='*60}")
    print(f"Final dataset totals:")
    print(f"  Train : {len(train_A)} pairs")
    print(f"  Val   : {len(val_A)} pairs")
    print(f"  Test  : {len(test_A)} pairs")
    print(f"  Shape : {train_A.shape[1:]}  (channels, ny, nx)")

    metadata = {
        'source'      : 'gkeyll',
        'source_files': ', '.join(os.path.basename(p) for p in npz_paths),
        'n_sources'   : len(npz_paths),
        'val_split'   : float(val_split),
        'test_split'  : float(test_split),
        'test_seed'   : int(test_seed),
        'psf_fwhm'    : float(psf_fwhm) if psf_fwhm is not None else -1.0,
        'n_train'     : int(len(train_A)),
        'n_val'       : int(len(val_A)),
        'n_test'      : int(len(test_A)),
    }

    print(f"\nSaving to: {output_path}")
    save_dataset_cache(
        output_path,
        train_A, train_B, train_flows,
        val_A,   val_B,   val_flows,
        test_A,  test_B,  test_flows,
        metadata,
    )

    train_dataset = BESDataset(train_A, train_B, train_flows, augment=True)
    val_dataset   = BESDataset(val_A,   val_B,   val_flows,   augment=False)
    test_dataset  = BESDataset(test_A,  test_B,  test_flows,  augment=False)

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build a training cache from Gkeyll buffer HDF5 files'
    )
    parser.add_argument(
        '--inputs', nargs='+', required=True, metavar='FILE',
        help='One or more Gkeyll buffer .npz files (from save_gkeyll_npz)'
    )
    parser.add_argument(
        '--output', required=True, metavar='FILE',
        help='Output HDF5 path, e.g. synthetic_data/gkeyll_train.h5'
    )
    parser.add_argument('--val_split',  type=float, default=0.1)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--test_seed',  type=int,   default=42,
                        help='RNG seed for reproducible test split (default: 42)')
    parser.add_argument('--psf_fwhm', type=float, default=None,
                        help='PSF FWHM in pixels for BES blurring (default: none)')

    args = parser.parse_args()

    print(f"Buffer files ({len(args.inputs)}):")
    for p in args.inputs:
        print(f"  {p}")

    build_gkeyll_dataset(
        npz_paths    = args.inputs,
        output_path = args.output,
        val_split   = args.val_split,
        test_split  = args.test_split,
        test_seed   = args.test_seed,
        psf_fwhm    = args.psf_fwhm,
    )

    print("\nDone.")
