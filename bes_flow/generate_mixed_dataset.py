# bes_flow/generate_mixed_dataset.py
#
# Generates a single HDF5 dataset that contains all four flow types
# (smooth, modes, zonal, well) in equal proportions, randomly shuffled.
#
# Output format is identical to the cache files produced by dataset.py,
# so the result can be loaded directly with load_dataset_cache() and used
# with make_datasets() by setting:
#
#   cfg.flow_type          = 'mixed'
#   cfg.dataset_cache_path = <output_path>


import argparse
import h5py
import numpy as np
from bes_flow.dataset import generate_dataset, save_dataset_cache
from bes_flow.config import cfg


# The four flow types that make up the mixed dataset, in curriculum order.
FLOW_TYPES = ['smooth', 'modes', 'well', 'zonal']
 
def _generate_pairs(frames, flow_types, n_pairs_per_frame,
                    max_shift, noise_std, seed=None):
    """
    Generate frame pairs and corresponding flows 
    for every flow type, concatenate, and shuffle.
 
    Parameters
    ----------
    frames            : (N, H, W) float32 — BES frames for this split
    flow_types        : list[str] — flow types to include
    n_pairs_per_frame : int
    max_shift         : float
    noise_std         : float
    seed              : int or None
        If given, the global numpy RNG is seeded before generation begins
        and the final shuffle uses the same seed, making the split
        reproducible. Pass None for train/val to use a fresh random state.
 
    Returns
    -------
    framesA, framesB, flows_gt : concatenated & shuffled arrays
    """
    all_A, all_B, all_flows = [], [], []
 
    # get current random state
    rng_state = np.random.get_state()
    if seed is not None:
        # set global seed if given
        np.random.seed(seed)
 
    for flow_type in flow_types:
        fA, fB, flows = generate_dataset(
            frames,
            n_pairs_per_frame=n_pairs_per_frame,
            max_shift=max_shift,
            noise_std=noise_std,
            flow_type=flow_type,
        )
        all_A.append(fA)
        all_B.append(fB)
        all_flows.append(flows)
 
    framesA  = np.concatenate(all_A,     axis=0)
    framesB  = np.concatenate(all_B,     axis=0)
    flows_gt = np.concatenate(all_flows, axis=0)
 
    # Shuffle so the loader sees all flow types interleaved each epoch.
    rng = np.random.default_rng(seed) 
    perm = rng.permutation(len(framesA))

    # Restore random state
    np.random.set_state(rng_state)  
 
    return framesA[perm], framesB[perm], flows_gt[perm]
 
 
def generate_mixed_dataset(data_path, output_path,
                            val_split, test_split,
                            n_pairs_per_frame, max_shift, noise_std,
                            val_seed, test_seed,
                            flow_types=FLOW_TYPES):
    """
    Full pipeline: load raw frames -> generate mixed dataset -> save HDF5.
 
    Parameters
    ----------
    data_path         : str   — path to raw BES HDF5 file
    output_path       : str   — where to write the mixed dataset cache
    val_split         : float — fraction of frames for validation
    test_split        : float — fraction of frames for test
    n_pairs_per_frame : int
    max_shift         : float — peak displacement in pixels
    noise_std         : float — additive Gaussian noise std
    val_seed          : int   — RNG seed for the validation split
    test_seed         : int   — RNG seed for the test split
    flow_types        : list[str] — which flow types to mix (default: all 4)
    """
    # ------------------------------------------------------------------
    # 1. Load raw BES frames
    # ------------------------------------------------------------------
    print(f"\nLoading BES frames from: {data_path}")
    with h5py.File(data_path, 'r') as hf:
        all_frames = hf['images'][:]
    print(f"  Total frames: {len(all_frames)}  shape: {all_frames.shape[1:]}")
 
    # ------------------------------------------------------------------
    # 2. Frame split  (same deterministic slice used everywhere)
    # ------------------------------------------------------------------
    N       = len(all_frames)
    n_test  = int(test_split * N)
    n_val   = int(val_split  * N)
    n_train = N - n_test - n_val
    print(f"  Split: train={n_train}  val={n_val}  test={n_test}")
 
    train_frames = all_frames[:n_train]
    val_frames   = all_frames[n_train : n_train + n_val]
    test_frames  = all_frames[n_train + n_val:]
 
    # ------------------------------------------------------------------
    # 3. Generate each split
    # ------------------------------------------------------------------
    print(f"\nFlow types in mixed dataset: {flow_types}")
    n_types = len(flow_types)
 
    print(f"\n--- Training set (random seed) ---")
    train_A, train_B, train_flows = _generate_pairs(
        train_frames, flow_types,
        n_pairs_per_frame=n_pairs_per_frame,
        max_shift=max_shift,
        noise_std=noise_std,
        seed=None,   # fresh random state each run
    )
    print(f"  Total training pairs: {len(train_A)}  "
          f"({n_train} frames x {n_pairs_per_frame} pairs x {n_types} types)")
 
    print(f"\n--- Validation set (fixed seed={val_seed}) ---")
    val_A, val_B, val_flows = _generate_pairs(
        val_frames, flow_types,
        n_pairs_per_frame=n_pairs_per_frame,
        max_shift=max_shift,
        noise_std=noise_std,
        seed=val_seed,
    )
    print(f"  Total validation pairs: {len(val_A)}")
 
    print(f"\n--- Test set (fixed seed={test_seed}) ---")
    test_A, test_B, test_flows = _generate_pairs(
        test_frames, flow_types,
        n_pairs_per_frame=n_pairs_per_frame,
        max_shift=max_shift,
        noise_std=noise_std,
        seed=test_seed,
    )
    print(f"  Total test pairs: {len(test_A)}")
 
    # ------------------------------------------------------------------
    # 4. Save in the standard cache format
    # ------------------------------------------------------------------
    metadata = {
        'flow_type'        : 'mixed',
        'flow_types_list'  : ','.join(flow_types),   # informational
        'max_shift'        : float(max_shift),
        'noise_std'        : float(noise_std),
        'n_pairs_per_frame': int(n_pairs_per_frame),
        'val_split'        : float(val_split),
        'test_split'       : float(test_split),
        'val_seed'         : int(val_seed),
        'test_seed'        : int(test_seed),
    }
 
    print(f"\nSaving mixed dataset")
    save_dataset_cache(
        output_path,
        train_A, train_B, train_flows,
        val_A,   val_B,   val_flows,
        test_A,  test_B,  test_flows,
        metadata=metadata,
    )

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a mixed-flow-type BES optical flow dataset."
    )
    parser.add_argument(
        '--data_path', type=str,
        default=cfg.data_path,
        help="Path to the raw BES HDF5 file"
    )
    parser.add_argument(
        '--output', type=str,
        default=f"synthetic_data/dataset_mixed_maxshift_{cfg.max_shift}.h5",
        help="Output HDF5 path for the mixed dataset cache"
    )
    parser.add_argument(
        '--max_shift', type=float,
        default=cfg.max_shift,
        help=f"Peak displacement in pixels (default: {cfg.max_shift})"
    )
    parser.add_argument(
        '--noise_std', type=float,
        default=cfg.noise_std,
        help=f"Additive Gaussian noise std (default: {cfg.noise_std})"
    )
    parser.add_argument(
        '--n_pairs_pf', type=int,
        default=cfg.n_pairs_per_frame,
        help=f"Synthetic pairs per frame per flow type (default: {cfg.n_pairs_per_frame})"
    )
    parser.add_argument(
        '--flow_types', type=str, nargs='+',
        default=FLOW_TYPES,
        choices=FLOW_TYPES,
        help=f"Flow types to include (default: all {FLOW_TYPES})"
    )
    args = parser.parse_args()
 
    generate_mixed_dataset(
        data_path         = args.data_path,
        output_path       = args.output,
        val_split         = cfg.val_split,
        test_split        = cfg.test_split,
        n_pairs_per_frame = args.n_pairs_pf,
        max_shift         = args.max_shift,
        noise_std         = args.noise_std,
        val_seed          = cfg.val_seed,
        test_seed         = cfg.test_seed,
        flow_types        = args.flow_types,
    )
