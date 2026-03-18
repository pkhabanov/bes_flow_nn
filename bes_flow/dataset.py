# bes_flow/dataset.py
#
# Dataset class and DataLoader factory for BES optical flow training.
# 
#   BESDataset   — a PyTorch Dataset: defines __len__ and __getitem__.
#                  __getitem__(i) returns (frameA, frameB, flow_gt).
#
#   DataLoader   — wraps the Dataset and handles batching + shuffling.


import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter, map_coordinates


def random_smooth_flow(H, W, max_shift=6.0, smoothing_sigma=8.0):
    """
    Generate a smooth random displacement field by Gaussian-smoothing
    independent random noise.

    Parameters
    ----------
    H, W            : image height and width (both 64 for BES)
    max_shift       : after smoothing and normalisation, the peak
                        displacement magnitude in pixels
    smoothing_sigma : Gaussian smoothing length in pixels.
                      Larger values - more spatially coherent flow.
                      (~8-16 px)

    Returns
    -------
    flow : (2, H, W) float32 array — channel 0 = dx, channel 1 = dy
    """
    # Draw independent Gaussian noise for each component
    dx_raw = np.random.randn(H, W).astype(np.float32)
    dy_raw = np.random.randn(H, W).astype(np.float32)

    # Smooth with a Gaussian kernel to impose spatial coherence.
    # Without this step the field would vary at the pixel level,
    # which is unphysical for plasma velocity.
    dx_smooth = gaussian_filter(dx_raw, sigma=smoothing_sigma)
    dy_smooth = gaussian_filter(dy_raw, sigma=smoothing_sigma)

    # Normalise so the peak displacement equals max_shift.
    # We normalise by the maximum magnitude of the 2D vector field
    # (not dx and dy independently) to preserve the direction distribution.
    magnitude = np.sqrt(dx_smooth**2 + dy_smooth**2)
    peak_mag  = magnitude.max() + 1e-8
    scale     = max_shift / peak_mag

    return np.stack([dx_smooth * scale,
                     dy_smooth * scale], axis=0)


def sinusoidal_modes(H, W, n_modes=8, max_shift=6.0):
    """
    Generate a velocity field as a sum of sinusoidal modes.

    This mimics the spectral structure of plasma drift-wave turbulence:
    the flow is a superposition of waves with different wavenumbers,
    amplitudes and phases.

    Parameters
    ----------
    H, W      : image dimensions
    n_modes   : number of superimposed sinusoidal components.
                More modes → richer spectral content
    max_shift : peak displacement magnitude in pixels

    Returns
    -------
    flow : (2, H, W) float32 array
    """
    y_coords = np.linspace(0, 2 * np.pi, H, dtype=np.float32)
    x_coords = np.linspace(0, 2 * np.pi, W, dtype=np.float32)
    xx, yy   = np.meshgrid(x_coords, y_coords)

    dx = np.zeros((H, W), dtype=np.float32)
    dy = np.zeros((H, W), dtype=np.float32)

    for _ in range(n_modes):
        # Random wavenumber: integers 1–4 give structures that span
        # 1/4 to the full image — representative of BES turbulence scales
        kx = 0.4 * np.random.randint(1, 5)
        ky = 0.4 * np.random.randint(1, 5)

        # Random amplitude and phase for each mode and each component
        amp_x = np.random.randn()
        amp_y = np.random.randn()
        phase_x = np.random.uniform(0, 2 * np.pi)
        phase_y = np.random.uniform(0, 2 * np.pi)

        dx += amp_x * np.sin(kx * xx + phase_x)
        dy += amp_y * np.sin(ky * yy + phase_y)

    # Normalise to max_shift
    magnitude = np.sqrt(dx**2 + dy**2)
    peak_mag  = magnitude.max() + 1e-8
    scale     = max_shift / peak_mag

    return np.stack([dx * scale, dy * scale], axis=0)


def zonal_plus_turbulence_flow(H, W,
                               zonal_amplitude=4.0,
                               turbulence_amplitude=2.0,
                               turbulence_sigma=6.0,
                               profile_type='well',
                               well_pos=0.5,
                               well_width=0.125,):
    """
    Generate a velocity field composed of:
        1. A smooth zonal flow component with a slow variation across the image
        2. A turbulent component: small-scale, isotropic Gaussian random
           field superimposed on the zonal flow.

    Parameters
    ----------
    H, W                   : image dimensions
    zonal_amplitude        : peak zonal flow displacement in pixels
    turbulence_amplitude   : peak turbulent displacement in pixels
    turbulence_sigma       : spatial smoothing of turbulent component (pixels)
    profile_type           : type of the flow profile in the x-direction
                             'well' is a gaussian profile resembling Er well,
                             'sin' is a sinusoidal profile
    well_pos               : well_pos * W is the position of the peak flow along the x axis (for 'well')
    well_width             : well_width * W is the width of the shear layer in pixels (for 'well')

    Returns
    -------
    flow       : (2, H, W) — total flow field (zonal + turbulent)
    flow_zonal : (2, H, W) — zonal component only
    flow_turb  : (2, H, W) — turbulent component only
    """

    # Zonal flow
    if profile_type == 'sin':
        # Smooth sinusoidal variation of y-velocity across the x (radial) axis.
        # add small random phase shift
        x_coords = np.linspace(0, 2 * np.pi, W, dtype=np.float32) + 0.2 * np.pi * np.random.randn()
        zonal_profile = np.sin(x_coords)  # (W,) — radial profile
    elif profile_type == 'well':
        # Gaussian profile (Er well approximation)
        x_coords = np.arange(0, W)
        well_pos = well_pos + 0.25 * np.random.randn()
        well_width = well_width + 0.17 * np.random.randn()
        zonal_profile = np.exp(-((x_coords - well_pos*W)**2) / (2 * (well_width*W)**2))
    else:
        raise ValueError(
            f"Unknown profile_type '{profile_type}'. Choose 'sin' or 'well'."
        )
    
    zonal_dy = zonal_amplitude * zonal_profile     # y-component varies with x
    zonal_dx = np.zeros(W, dtype=np.float32)       # no radial zonal component

    # Broadcast to full (H, W) arrays — zonal flow is uniform in x
    flow_zonal = np.stack([
        np.tile(zonal_dx[None, :], (H, 1)),
        np.tile(zonal_dy[None, :], (H, 1))
    ], axis=0)

    # Turbulent component 
    turb_raw  = np.random.randn(2, H, W).astype(np.float32)
    turb_smooth = np.stack([
        gaussian_filter(turb_raw[0], sigma=turbulence_sigma),
        gaussian_filter(turb_raw[1], sigma=turbulence_sigma)
    ], axis=0)

    # Normalise turbulent component to turbulence_amplitude
    mag   = np.sqrt((turb_smooth**2).sum(axis=0)).max() + 1e-8
    flow_turb = turb_smooth * (turbulence_amplitude / mag)

    # Total flow
    flow = flow_zonal + flow_turb

    return flow.astype(np.float32), flow_zonal.astype(np.float32), flow_turb.astype(np.float32)

  
def warp_image(image, flow):
    """
    Warp a 2D image by a displacement field using cubic interpolation.

    Parameters
    ----------
    image : (H, W) float32 array — the BES frame to warp
    flow  : (2, H, W) float32 array — displacement field

    Returns
    -------
    warped : (H, W) float32 array
    """
    H, W = image.shape

    # Build pixel coordinate grids
    y_coords, x_coords = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing='ij'
    )

    # Displaced source coordinates: where to sample in the original image.
    # map_coordinates uses (row, col) = (y, x) ordering.
    src_y = y_coords - flow[1]   # flow[1] = dy
    src_x = x_coords - flow[0]   # flow[0] = dx

    # Cubic interpolation (order=3)
    warped = map_coordinates(
        image,
        [src_y.ravel(), src_x.ravel()],
        order=3,
        mode='nearest',
    ).reshape(H, W)

    return warped.astype(np.float32)


def _generate_flow(H, W, flow_type, max_shift):
    """
    Dispatch to the selected flow generator.
    """
    if flow_type == 'smooth':
        return random_smooth_flow(H, W, max_shift)
    elif flow_type == 'modes':
        return sinusoidal_modes(H, W, max_shift=max_shift)
    elif flow_type == 'zonal':
        flow, _, _ = zonal_plus_turbulence_flow(
            H, W,
            zonal_amplitude      = max_shift,
            turbulence_amplitude = max_shift * 0.3,
            profile_type         = 'sin',
        )
        return flow
    elif flow_type == 'well':
        flow, _, _ = zonal_plus_turbulence_flow(
            H, W,
            zonal_amplitude      = max_shift,
            turbulence_amplitude = max_shift * 0.3,
            profile_type         = 'well',
        )
        return flow
    else:
        raise ValueError(
            f"Unknown flow_type '{flow_type}'. "
            f"Choose from: 'smooth', 'modes', 'zonal', 'well'."
        )
    

def generate_dataset(frames, n_pairs_per_frame, max_shift,
                        noise_std, flow_type):
    """
    Generate the full synthetic dataset once and return numpy arrays.

    Each real frame is used to produce n_pairs_per_frame synthetic pairs
    with independent random flow fields, giving a total of
    N * n_pairs_per_frame training examples.

    Parameters
    ----------
    frames            : (N, H, W) float array — real BES frames
    n_pairs_per_frame : int   — synthetic pairs per real frame
    max_shift  : float — peak displacement in pixels
    noise_std         : float — std of additive Gaussian noise
    flow_type         : str   — 'smooth', 'modes', 'zonal', or 'flow'

    Returns
    -------
    framesA  : (N*n_pairs, 1, H, W) float32
    framesB  : (N*n_pairs, 1, H, W) float32
    flows_gt : (N*n_pairs, 2, H, W) float32
    """
    N, H, W  = frames.shape
    n_total  = N * n_pairs_per_frame

    framesA  = np.zeros((n_total, 1, H, W), dtype=np.float32)
    framesB  = np.zeros((n_total, 1, H, W), dtype=np.float32)
    flows_gt = np.zeros((n_total, 2, H, W), dtype=np.float32)

    print(f"  Generating {n_total} pairs "
          f"({N} frames x {n_pairs_per_frame} pairs, "
          f"flow='{flow_type}')...")

    idx = 0
    for i, frame in enumerate(frames):
        image = frame.astype(np.float32)
        # normalize each image
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        for _ in range(n_pairs_per_frame):
            flow   = _generate_flow(H, W, flow_type, max_shift)
            warped = warp_image(image, flow)

            if noise_std > 0:
                framesA[idx, 0] = (image  + np.random.normal(0, noise_std, (H, W))
                                   ).clip(0.0, 1.0)
                framesB[idx, 0] = (warped + np.random.normal(0, noise_std, (H, W))
                                   ).clip(0.0, 1.0)
            else:
                framesA[idx, 0] = image
                framesB[idx, 0] = warped

            ### jointly normalize the pair after warping

            flows_gt[idx, :, :, :] = flow
            idx += 1

        if (i + 1) % max(1, N // 5) == 0:
            print(f"    {i+1}/{N} frames created  ({idx} pairs)")

    mem_mb = (framesA.nbytes + framesB.nbytes + flows_gt.nbytes) / 1e6
    print(f"  Done — {n_total} pairs, ~{mem_mb:.1f} MB in memory")
    return framesA, framesB, flows_gt


def _make_metadata(cfg):
    """
    Build a dict of the settings that determine dataset content.
    Stored as HDF5 attributes on the /metadata group and compared on
    load to detect stale caches.
    """
    return {
        'flow_type'        : cfg.flow_type,
        'max_shift'        : float(cfg.max_shift),
        'noise_std'        : float(cfg.noise_std),
        'n_pairs_per_frame': int(cfg.n_pairs_per_frame),
        'val_split'        : float(cfg.val_split),
        'test_split'       : float(cfg.test_split),
        'val_seed'         : int(cfg.val_seed),
        'n_test_pairs'     : int(cfg.n_test_pairs),
        'test_seed'        : int(cfg.test_seed),
    }


def save_dataset_cache(path, 
                       train_A, train_B, train_flows,
                       val_A, val_B, val_flows, 
                       test_A, test_B, test_flows,
                       metadata):
    """
    Save pre-generated arrays and metadata to an HDF5 file.

    Parameters
    ----------
    path            : str  -- file path, e.g. 'data/cache/dataset_zonal.h5'
    train_A/B/flows : (M, 1, H, W) / (M, 2, H, W) training arrays
    val_A/B/flows   : (V, 1, H, W) / (V, 2, H, W) validation arrays
    test_A/B/flows  : (T, 1, H, W) / (T, 2, H, W) test arrays
    metadata        : dict -- generation settings from _make_metadata()
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    with h5py.File(path, 'w') as f:
        for grp_name, A, B, flows in (
            ('train', train_A, train_B, train_flows),
            ('val',   val_A,   val_B,   val_flows),
            ('test',  test_A,  test_B,  test_flows),
        ):
            grp = f.create_group(grp_name)
            grp.create_dataset('framesA',  data=A,     compression='gzip', compression_opts=4)
            grp.create_dataset('framesB',  data=B,     compression='gzip', compression_opts=4)
            grp.create_dataset('flows_gt', data=flows, compression='gzip', compression_opts=4)

        # Metadata as typed HDF5 attributes on a dedicated group
        meta_grp = f.create_group('metadata')
        for key, value in metadata.items():
            meta_grp.attrs[key] = value

    size_mb = os.path.getsize(path) / 1e6
    print(f"  Cache saved -> {path}  ({size_mb:.1f} MB on disk)")


def load_dataset_cache(path):
    """
    Load pre-generated arrays from an HDF5 cache file.

    Parameters
    ----------
    path : str -- path to .h5 cache file

    Returns
    -------
    train_A, train_B, train_flows : training arrays
    val_A, val_B, val_flows       : validation arrays
    test_A,  test_B,  test_flows  : test arrays
    metadata                      : dict of generation settings
    """
    with h5py.File(path, 'r') as f:
        train_A     = f['train/framesA'][:]
        train_B     = f['train/framesB'][:]
        train_flows = f['train/flows_gt'][:]
        val_A       = f['val/framesA'][:]
        val_B       = f['val/framesB'][:]
        val_flows   = f['val/flows_gt'][:]
        test_A      = f['test/framesA'][:]
        test_B      = f['test/framesB'][:]
        test_flows  = f['test/flows_gt'][:]

        # Read attributes back into a Python dict.
        metadata = {}
        for key, value in f['metadata'].attrs.items():
            if hasattr(value, 'item'):
                metadata[key] = value.item()   # numpy scalar -> Python int/float
            else:
                metadata[key] = value          # strings pass through unchanged

    return (train_A, train_B, train_flows,
            val_A,   val_B,   val_flows,
            test_A,  test_B,  test_flows,
            metadata)


def _cache_is_valid(path, cfg):
    """
    Check whether a cache file exists AND was generated with the same
    settings as the current cfg.
    Reads only the /metadata attributes.

    Returns
    -------
    (is_valid : bool, reason : str)
        reason is logged by make_dataloaders so the user always knows
        exactly why a cache was rejected or accepted.
    """
    if not os.path.exists(path):
        return False, "Cache file not found"
    try:
        with h5py.File(path, 'r') as f:
            # Verify all three split groups are present
            for grp in ('train', 'val', 'test'):
                if grp not in f:
                    return False, f"Cache is missing the '{grp}' group"
            # read only metadata from hdf5 cache
            metadata = {
                k: (v.item() if hasattr(v, 'item') else v)
                for k, v in f['metadata'].attrs.items()
            }
    except Exception as e:
        return False, f"Cache file unreadable: {e}"

    current = _make_metadata(cfg)

    # Compare each field individually
    for key, current_val in current.items():
        cached_val = metadata.get(key)
        if cached_val != current_val:
            return False, (
                f"'{key}' mismatch: cached={cached_val!r}, "
                f"current={current_val!r}"
            )

    return True, "ok"
    

class BESDataset(Dataset):
    """
    Fast dataset that indexes into pre-generated numpy arrays.
    __getitem__ is a trivial array lookup followed by optional augmentation.

    Parameters
    ----------
    framesA, framesB : (M, 1, H, W) float32 — pre-generated frame pairs
    flows_gt         : (M, 2, H, W) float32 — ground-truth flow fields
    augment          : bool — if True, apply random augmentation in __getitem__.
                       Enable for the training set; disable for val/test
    """
    def __init__(self, framesA, framesB, flows_gt, augment=False):
        assert len(framesA) == len(framesB) == len(flows_gt), (
            "framesA, framesB and flows_gt must all have the same length "
            f"(got {len(framesA)}, {len(framesB)}, {len(flows_gt)})"
        )
        self.framesA  = framesA
        self.framesB  = framesB
        self.flows_gt = flows_gt
        self.augment  = augment

    def __len__(self):
        return len(self.framesA)

    def __getitem__(self, idx):
        # All generation work was done in generate_dataset().
        fA   = torch.tensor(self.framesA[idx])   # (1, H, W)
        fB   = torch.tensor(self.framesB[idx])   # (1, H, W)
        flow = torch.tensor(self.flows_gt[idx])  # (2, H, W)

        if self.augment:
            fA, fB, flow = self._augment(fA, fB, flow)

        return fA, fB, flow
    
    @staticmethod
    def _augment(fA, fB, flow):
        """
        Random data augmentations.

        1. Y-axis flip (p=0.5)
        2. X-axis flip
        2. Random 90° rotation (k ∈ {0,1,2,3})
           Image and flow grid are rotated k*90° CCW.
           Flow vectors are rotated by applying (dx,dy)->(dy,-dx) k times
           (x and y increase from element (0,0)).
        3. Intensity jitter — same gain U[0.95,1.05] for both frames
        """
        # 1. Y-axis flip
        if torch.rand(1).item() < 0.5:
            fA   = torch.flip(fA,   dims=[1])
            fB   = torch.flip(fB,   dims=[1])
            flow = torch.flip(flow, dims=[1])
            flow = torch.stack([flow[0], -flow[1]], dim=0)

        # 2. X-axis (horizontal) flip 
        if torch.rand(1).item() < 0.5:
            fA   = torch.flip(fA,   dims=[2])
            fB   = torch.flip(fB,   dims=[2])
            flow = torch.flip(flow, dims=[2])
            flow = torch.stack([-flow[0], flow[1]], dim=0)   

        # 3. Random 90 deg rotation
        k = torch.randint(4, (1,)).item()
        if k > 0:
            fA   = torch.rot90(fA,   k, dims=[1, 2])
            fB   = torch.rot90(fB,   k, dims=[1, 2])
            flow = torch.rot90(flow, k, dims=[1, 2])
            for _ in range(k):
                flow = torch.stack([flow[1], -flow[0]], dim=0)

        # 4. Intensity jitter
        gain = 0.95 + 0.10 * torch.rand(1).item()
        fA   = (fA * gain).clamp(0.0, 1.0)
        fB   = (fB * gain).clamp(0.0, 1.0)

        return fA, fB, flow


def make_datasets(train_frames, val_frames, test_frames, cfg):
    """
    Generate or load all three BESDataset objects (train / val / test).

    Cache behaviour
    ───────────────
    If cfg.dataset_cache_path is set and a valid cache exists on disk,
    all three splits are loaded from it.  Otherwise they are
    generated from scratch and saved to the cache path. 
    The cache is invalidated automatically when any metadata field changes.

    Parameters
    ----------
    train_frames : (N_train, H, W) float array — raw BES training frames
    val_frames   : (N_val,   H, W) float array — raw BES validation frames
    test_frames  : (N_test,  H, W) float array — raw BES test frames
    cfg          : Config

    Returns
    -------
    train_dataset : BESDataset  (augmentation ON)
    val_dataset   : BESDataset  (augmentation OFF)
    test_dataset  : BESDataset  (augmentation OFF)
    """
    cache_path = getattr(cfg, 'dataset_cache_path', None)

    if cache_path is not None:
        valid, reason = _cache_is_valid(cache_path, cfg)

        if valid:
            print(f"\nLoading dataset from cache: {cache_path}")
            (train_A, train_B, train_flows,
             val_A,   val_B,   val_flows,
             test_A,  test_B,  test_flows,
             metadata) = load_dataset_cache(cache_path)
            print(f"  Train pairs : {len(train_A)}")
            print(f"  Val pairs   : {len(val_A)}")
            print(f"  Test pairs  : {len(test_A)}")
            print(f"  Flow type   : {metadata['flow_type']}")
            print(f"  Max shift   : {metadata['max_shift']} px")

        else:
            print(f"\n{reason} — regenerating dataset...")
            (train_A, train_B, train_flows,
             val_A,   val_B,   val_flows,
             test_A,  test_B,  test_flows) = _generate_all(
                 train_frames, val_frames, test_frames, cfg
             )
            print(f"Saving dataset cache: {cache_path}")
            save_dataset_cache(
                cache_path,
                train_A, train_B, train_flows,
                val_A,   val_B,   val_flows,
                test_A,  test_B,  test_flows,
                metadata=_make_metadata(cfg),
            )

    else:
        print("\nNo cache path configured — generating dataset (not saved)...")
        (train_A, train_B, train_flows,
         val_A,   val_B,   val_flows,
         test_A,  test_B,  test_flows) = _generate_all(
             train_frames, val_frames, test_frames, cfg
         )

    train_dataset = BESDataset(train_A, train_B, train_flows, augment=True)
    val_dataset   = BESDataset(val_A,   val_B,   val_flows,   augment=False)
    test_dataset  = BESDataset(test_A,  test_B,  test_flows,  augment=False)

    print(f"\nDataset summary:")
    print(f"  Train : {len(train_dataset)} pairs  (augmentation ON)")
    print(f"  Val   : {len(val_dataset)} pairs  (augmentation OFF)")
    print(f"  Test  : {len(test_dataset)} pairs  (augmentation OFF)\n")

    return train_dataset, val_dataset, test_dataset


def make_dataloaders(train_dataset, val_dataset, test_dataset, cfg):
    """
    Wrap three BESDataset objects in DataLoaders.

    Parameters
    ----------
    train_dataset : BESDataset — augmented training set
    val_dataset   : BESDataset — fixed validation set
    test_dataset  : BESDataset — fixed held-out test set
    cfg           : Config

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = cfg.num_workers,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = cfg.num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = cfg.num_workers,
        pin_memory  = True,
    )

    print(f"DataLoader summary:")
    print(f"  Batch size          : {cfg.batch_size}")
    print(f"  Train batches/epoch : {len(train_loader)}")
    print(f"  Val   batches       : {len(val_loader)}")
    print(f"  Test  batches       : {len(test_loader)}\n")

    return train_loader, val_loader, test_loader


def _generate_all(train_frames, val_frames, test_frames, cfg):
    """
    Generate all three splits from scratch.

    Parameters
    ----------
    train_frames : (N_train, H, W)
    val_frames   : (N_val,   H, W)
    test_frames  : (N_test,  H, W)
    cfg          : Config

    Returns
    -------
    train_A, train_B, train_flows,
    val_A,   val_B,   val_flows,
    test_A,  test_B,  test_flows
    """
    # Training set - random seed
    print("Generating training set:")
    train_A, train_B, train_flows = generate_dataset(
        train_frames,
        n_pairs_per_frame = cfg.n_pairs_per_frame,
        max_shift         = cfg.max_shift,
        noise_std         = cfg.noise_std,
        flow_type         = cfg.flow_type,
    )

    # Validation set - fixed val_seed 
    print("Generating validation set (fixed seed for reproducibility):")
    rng_state = np.random.get_state()
    np.random.seed(cfg.val_seed)

    val_A, val_B, val_flows = generate_dataset(
        val_frames,
        n_pairs_per_frame = cfg.n_pairs_per_frame,
        max_shift         = cfg.max_shift,
        noise_std         = cfg.noise_std,
        flow_type         = cfg.flow_type,
    )

    np.random.set_state(rng_state)  # restore for training

    # Test set - fixed test_seed 
    # n_pairs_per_frame is chosen so that the total is ≥ cfg.n_test_pairs.
    print("Generating test set (fixed seed, independent of training):")
    if len(test_frames) == 0:
        test_A = test_B = test_flows = np.empty((0, ...), dtype=np.float32)
    else:
        n_test_ppf = max(1, cfg.n_test_pairs // len(test_frames))

    rng_state = np.random.get_state()
    np.random.seed(cfg.test_seed)

    test_A, test_B, test_flows = generate_dataset(
        test_frames,
        n_pairs_per_frame = n_test_ppf,
        max_shift         = cfg.max_shift,
        noise_std         = cfg.noise_std,
        flow_type         = cfg.flow_type,
    )

    np.random.set_state(rng_state)

    return (train_A, train_B, train_flows,
            val_A,   val_B,   val_flows,
            test_A,  test_B,  test_flows)


if __name__ == "__main__":
    # test image warping
    import matplotlib.pyplot as plt
    from dataclasses import dataclass

    # build a minimal config
    @dataclass
    class TestConfig:
        val_split          : float = 0.1
        test_split         : float = 0.1
        max_shift          : float = 8.0
        noise_std          : float = 0.02
        flow_type          : str   = 'zonal'
        batch_size         : int   = 4
        num_workers        : int   = 0
        n_pairs_per_frame  : int   = 1
        val_seed           : int   = 0
        n_test_pairs       : int   = 20
        test_seed          : int   = 42
        # Set to None to skip saving
        dataset_cache_path : str   = 'synthetic_data/test_dataset.h5'

    cfg = TestConfig()

    # load bes frames
    fname = 'raw_data/194313_t=2620-2640_f=30-200_1000fr.h5'
    print('\nLoading images ' + fname)
    with h5py.File(fname, 'r') as hf:
        all_frames = hf['images'][:]
        ti = hf['time'][:] #ms
        R = hf['R'][:]
        Z = hf['Z'][:]
    
    # split frames
    N        = len(all_frames)
    n_test   = int(cfg.test_split * N)
    n_val    = int(cfg.val_split  * N)
    n_train  = N - n_test - n_val
    print(f"\nSplit: train={n_train}  val={n_val}  test={n_test}")

    train_frames = all_frames[:n_train]
    val_frames   = all_frames[n_train : n_train + n_val]
    test_frames  = all_frames[n_train + n_val:]

    print("\n=== First call (should generate and save) ===")
    train_ds, val_ds, test_ds = make_datasets(train_frames, val_frames, test_frames, cfg)

    print("\n=== Second call (should load from cache) ===")
    train_ds, val_ds, test_ds = make_datasets(train_frames, val_frames, test_frames, cfg)

    train_loader, val_loader, test_loader = make_dataloaders(
        train_ds, val_ds, test_ds, cfg
    )
   
    # pull one batch and take the first sample
    frameA_batch, frameB_batch, flow_batch = next(iter(train_loader))
    # Remove batch and channel dimensions 
    frameA  = frameA_batch[0, 0].numpy()   # (H, W)
    frameB  = frameB_batch[0, 0].numpy()   # (H, W)
    flow_gt = flow_batch[0].numpy()        # (2, H, W)

    print(f"\nSample shapes:")
    print(f"  frameA  : {frameA.shape}")
    print(f"  frameB  : {frameB.shape}")
    print(f"  flow_gt : {flow_gt.shape}")
    print(f"\nFlow statistics:")
    print(f"  dx  min/max: {flow_gt[0].min():.2f} / {flow_gt[0].max():.2f} px")
    print(f"  dy  min/max: {flow_gt[1].min():.2f} / {flow_gt[1].max():.2f} px")
    mag = np.sqrt(flow_gt[0]**2 + flow_gt[1]**2)
    print(f"  magnitude mean/max: {mag.mean():.2f} / {mag.max():.2f} px")

    # downsaple the quiver grid
    step    = 4
    H, W    = frameA.shape
    ys      = np.arange(step // 2, H, step)   # y centres of quiver grid cells
    xs      = np.arange(step // 2, W, step)   # x centres of quiver grid cells
    xx, yy  = np.meshgrid(xs, ys)

    # Sample the flow at the quiver grid points
    dx = flow_gt[0][yy, xx]   # dx component at each grid point
    dy = flow_gt[1][yy, xx]   # dy component at each grid point

    # plot initial and warped frames
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Synthetic BES training pair  |  flow type: {cfg.flow_type}  |  "
        f"max displacement: {cfg.max_shift} px",
        fontsize=16, fontweight='bold'
    )
    # Common colour scale — both frames share the same vmin/vmax 
    vmin = min(frameA.min(), frameB.min())
    vmax = max(frameA.max(), frameB.max())

    for ax, frame, title in zip(
        axes, [frameA, frameB], ['Frame A ', 'Frame B (warped)']
    ):
        im = ax.imshow(frame, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)
        ax.quiver(xx, yy, dx, dy, color='cyan', scale=100,
                  scale_units='width', width=0.004)
        ax.set_title(title)
        ax.set_xlabel('x (px)')
        ax.set_ylabel('y (px)')

    # Shared colourbar on the right
    clb = fig.colorbar(im, ax=axes[1], label='Normalised BES intensity')

    plt.tight_layout()
    plt.show()
    #plt.savefig('outputs/dataset_test.png', dpi=150, bbox_inches='tight')
    #print("\nPlot saved to outputs/dataset_test.png")