# bes_flow/prepare_jhopkins_dataset.py
#
# Builds a training-ready HDF5 dataset from one or more JHTDB *.h5 files
# produced by compare_jhopkins.py (fetch_from_api with save_h5=...).
#
# Usage
# ─────
#   python -m bes_flow.prepare_jhopkins_dataset \
#       --inputs  data/region_A.h5 data/region_B.h5 data/region_C.h5 \
#       --output  synthetic_data/jhopkins_train.h5 \
#       --val_split  0.1 \
#       --test_split 0.1

import os
import glob
import argparse
import numpy as np
import h5py

from bes_flow.compare_jhopkins import load_from_h5, build_pairs
from bes_flow.dataset import BESDataset, save_dataset_cache


def _split_pairs(framesA, framesB, flows_gt, val_split, test_split, seed=42):
    """
    Split a set of frame pairs randomly into train / val / test.

    Parameters
    ----------
    framesA, framesB : (N, 1, ny, nx) float32
    flows_gt         : (N, 2, ny, nx) float32
    val_split        : float — fraction reserved for validation
    test_split       : float — fraction reserved for testing
    seed             : int  — RNG seed for reproducibility

    Returns
    -------
    (train_A, train_B, train_flows,
     val_A,   val_B,   val_flows,
     test_A,  test_B,  test_flows)
    """
    N       = len(framesA)
    n_test  = max(1, int(round(N * test_split)))
    n_val   = max(1, int(round(N * val_split)))
    n_train = N - n_val - n_test

    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(N)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    return (
        framesA[train_idx], framesB[train_idx], flows_gt[train_idx],   # train
        framesA[val_idx], framesB[val_idx], flows_gt[val_idx],   # val
        framesA[test_idx], framesB[test_idx], flows_gt[test_idx],   # test
    )


def build_jhopkins_dataset(h5_paths, output_path, h5_dir='synthetic_data',
                           val_split=0.1, test_split=0.1, seed=42, psf_fwhm=None):
    """
    Load JHTDB source files, build frame pairs, split, and save a training
    cache compatible with load_dataset_cache() / make_dataloaders().

    Parameters
    ----------
    h5_paths    : list[str]
        Paths to JHTDB HDF5 files produced by fetch_from_api (each containing
        images, vx, vy, time, R, Y datasets).
    h5_dir      : str
        Path to directory containing HDF5 files
    output_path : str
    val_split   : float
    test_split  : float
    seed        : int
    psf_fwhm    : float or None
        Full-width at half-maximum (in pixels) of an isotropic Gaussian kernel
        applied to images and velocity fields before building pairs to mimic
        the BES point-spread function.  The corresponding sigma is computed as
        sigma = fwhm / (2 * sqrt(2 * ln2)) ≈ fwhm / 2.355

    Returns
    -------
    train_dataset, val_dataset, test_dataset : BESDataset
    """
    if not h5_paths:
        raise ValueError("h5_paths is empty — provide at least one source file.")

    train_As, train_Bs, train_flows = [], [], []
    val_As,   val_Bs,   val_flows   = [], [], []
    test_As,  test_Bs,  test_flows  = [], [], []

    for path in h5_paths:
        h5_full_path = h5_dir + '/' + path
        print(f"\nProcessing: {h5_full_path}")
        images, vx, vy, times, x_grid, y_grid = load_from_h5(h5_full_path)

        framesA, framesB, flows_gt = build_pairs(
            images, vx, vy, times, x_grid, y_grid, psf_fwhm=psf_fwhm
        )
        N = len(framesA)
        print(f"  Pairs built: {N}")

        (tr_A, tr_B, tr_f,
         va_A, va_B, va_f,
         te_A, te_B, te_f) = _split_pairs(
            framesA, framesB, flows_gt, val_split, test_split, seed
        )

        print(f"  Split -> train: {len(tr_A)}  val: {len(va_A)}  test: {len(te_A)}")

        train_As.append(tr_A);  train_Bs.append(tr_B);  train_flows.append(tr_f)
        val_As.append(va_A);    val_Bs.append(va_B);    val_flows.append(va_f)
        test_As.append(te_A);   test_Bs.append(te_B);   test_flows.append(te_f)

    # Concatenate across all source files
    train_A     = np.concatenate(train_As, axis=0)
    train_B     = np.concatenate(train_Bs, axis=0)
    train_flows = np.concatenate(train_flows, axis=0)
    val_A       = np.concatenate(val_As, axis=0)
    val_B       = np.concatenate(val_Bs, axis=0)
    val_flows   = np.concatenate(val_flows, axis=0)
    test_A      = np.concatenate(test_As, axis=0)
    test_B      = np.concatenate(test_Bs, axis=0)
    test_flows  = np.concatenate(test_flows, axis=0)

    print(f"\nFinal dataset totals:")
    print(f"  Train : {len(train_A)} pairs")
    print(f"  Val   : {len(val_A)} pairs")
    print(f"  Test  : {len(test_A)} pairs")
    print(f"  Shape : {train_A.shape[1:]}  (channels, ny, nx)")

    # Save in the format expected by load_dataset_cache / make_dataloaders
    metadata = {
        'source_files':   ', '.join(os.path.basename(p) for p in h5_paths),
        'n_source_files': len(h5_paths),
        'val_split':      float(val_split),
        'test_split':     float(test_split),
        'split_seed':     int(seed),
        'n_train':        int(len(train_A)),
        'n_val':          int(len(val_A)),
        'n_test':         int(len(test_A)),
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
        description='Build a training-ready HDF5 dataset from JHTDB source files'
    )
    parser.add_argument(
        '--inputs', nargs='+', metavar='FILE',
        help='One or more JHTDB *.h5 source files (or a glob pattern)',
    )
    parser.add_argument(
        '--output', required=True, metavar='FILE',
        help='Output HDF5 path, e.g. synthetic_data/jhopkins_train.h5',
    )
    parser.add_argument('--val_split',  type=float, default=0.1)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for reproducible train/val/test split (default: 42)')
    parser.add_argument('--psf_fwhm', type=float, default=8)

    args = parser.parse_args()

    # Expand any glob patterns in --inputs
    #h5_paths = []
    #for pattern in args.inputs:
    #    matches = sorted(glob.glob(pattern))
    #   if matches:
    #        h5_paths.extend(matches)
    #    else:
    #        h5_paths.append(pattern)   # keep as-is; will fail with a clear error

    h5_paths = [ #'jhopkins_mixing_x1.2-1.6_y1.0-1.4_z1.3_64x64_200fr.h5 ', # keep these for testing
                #'jhopkins_mixing_x2.0-2.4_y2.6-3.0_z1.3_64x64_200fr.h5',
                'jhopkins_mixing_x2.6-3.0_y1.4-1.8_z1.3_64x64_200fr.h5',
                'jhopkins_mixing_x3.0-3.4_y3.2-3.6_z0.9_64x64_200fr.h5',
                'jhopkins_mixing_x3.0-3.4_y4.6-5.0_z2.5_64x64_200fr.h5',
                'jhopkins_mixing_x3.0-3.4_y5.4-5.8_z2.5_64x64_200fr.h5',
                'jhopkins_mixing_x4.4-4.8_y3.6-4.0_z0.9_64x64_200fr.h5',
                'jhopkins_mixing_x4.6-5.0_y5.6-6.0_z0.9_64x64_200fr.h5',]
  

    print(f"Source files ({len(h5_paths)}):")
    for p in h5_paths:
        print(f"  {p}")

    build_jhopkins_dataset(
        h5_paths    = h5_paths,
        output_path = args.output,
        val_split   = args.val_split,
        test_split  = args.test_split,
        seed        = args.seed,
        psf_fwhm    = args.psf_fwhm,
    )

    print("\nDone.")
