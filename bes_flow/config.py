# bes_flow/config.py
#
# Central configuration file. ALL hyperparameters and file paths live here.

from dataclasses import dataclass


@dataclass
class Config:

    # --- Data ------------------------------------------------------------
    # Path to the NumPy array of raw BES frames, shape (N, 64, 64).
    # Each frame is a single 2-D snapshot of plasma density fluctuations.
    data_path: str = "raw_data/194313_t=2600-2620_f=30-200_2000fr.h5"

    # Fraction of frames held out for validation and training
    val_split: float = 0.1
    test_split: float = 0.1
    test_seed: int = 42   # fixed seed for reproducible test generation

    # Flow type used for synthetic training pair generation.
    # 'smooth' : Gaussian random field       
    # 'modes'  : sinusoidal mode superposition 
    # 'zonal'  : zonal sinusoidal flow + turbulence     
    # 'well'   : zonal Gaussian flow (Er well) + turbulence    
    flow_type: str = 'mixed'

    # Maximum pixel displacement applied when generating synthetic frame pairs.
    # Drawn uniformly from [-max_shift, +max_shift] pixels in both x and y.
    max_shift: float = 12.0 #8.0

    # Standard deviation of Gaussian noise added to each synthetic frame.
    # Simulates the electronic noise present in real BES detector signals.
    # Set to 0 to train without noise augmentation.
    noise_std: float = 0.0

    # Number of synthetic pairs generated per real BES frame.
    # Total training pairs = len(train_frames) * n_pairs_per_frame.
    # Increase to enlarge the dataset without needing more real frames.
    n_pairs_per_frame: int = 1

    # Number of semi-Lagrangian RK2 steps used to advect frame A into
    # frame B during synthetic pair generation (see dataset.advect_image).
    #   1  : legacy single-step warp 
    #   >1 : the generated field is treated as a steady velocity field;
    #        frame B is produced by multi-step advection and the ground
    #        truth is the consistently integrated forward displacement.
    # 4 sub-steps keep the per-step displacement <= ~2 px for max_shift=8.
    n_warp_steps: int = 4

    # Compressible (curl-free) fraction of the TURBULENT component's RMS
    # kinetic energy. 0.0 = the original strictly divergence-free data.
    #
    # Applied to the turbulent component only (dataset.make_turbulence_-
    # compressible): the mean zonal flow stays divergence-free,
    # matching the physics that mean poloidal ExB is incompressible below
    # the ion sound speed while the drift-wave turbulence on top is not.
    #
    # WHY THIS EXISTS: with chi = 0 the synthetic flows are volume-preserving
    # and intensity is transported passively, so brightness constancy holds
    # and the continuity residual collapses onto the photometric residual. 
    # Setting chi > 0 generates pairs with genuine intensity sources and
    # sinks (dataset.advect_image_continuity).
    #
    # CALIBRATION: pick chi so the induced per-frame intensity change matches
    # the measured dI/I of the real signal. Measured at max_shift=12,
    # n_warp_steps=4 (regenerate with compression_diagnostics() if those
    # change). Note chi is an energy fraction OF THE TURBULENCE, so the
    # absolute dI/I differs between flow types because their turbulence
    # amplitudes differ:
    #
    #     chi      smooth        zonal        well      (rms dI/I)
    #   0.002       2.8 %        1.4 %       1.1 %
    #   0.005       5.4 %        2.0 %       1.8 %
    #   0.010       7.7 %        3.0 %       2.8 %
    #   0.020       8.3 %        4.2 %       3.9 %
    #   0.050      16.1 %        6.0 %       5.9 %
    compressible_fraction: float = 0.02

    # --- Model --------------------------------------------------------------
    # Number of channels in the shared CNN encoder output feature maps.
    feature_channels: int = 32

    # Search radius (in pixels, in feature space) for the correlation layer.
    # The layer will test all displacements (dx, dy) with |dx|, |dy| <= this value.
    # It determines the cost-volume size as (2*d+1)^2 channels
    max_displacement: int = 4

    # --- Training ------------------------------------------------------------
    # If supervised, then the EPE loss will be used with ground truth flow
    is_supervised: bool = False

    # Total number of passes through the training data.
    num_epochs: int = 100

    # Number of frame pairs processed together in one forward/backward pass.
    batch_size: int = 32

    # Initial learning rate for the Adam optimiser.
    # The scheduler (CosineAnnealingLR) will decay this toward 0 over training.
    learning_rate: float = 1e-3

    # Smoothness regularisation weights
    # See loss.py for the exact formulation.
    # total variation, 1st order derivative
    smooth_weight: float = 0.002
    # laplacian, 2nd order derivative
    laplacian_weight: float = 0.04

    # Continuity-equation penalty weight (physics-informed term).
    # Enforces  dI/dt + div(I*v) = 0  on the predicted flow.
    #
    # Physics: the MEAN poloidal ExB flow is divergence-free below the ion
    # sound speed, but the drift-wave turbulence BES resolves is NOT -
    # it is compressible, with real intensity sources and sinks. 
    # The residual is O(intensity), i.e. the same scale as the photometric
    # term, so this weight is naturally O(0.1-1) - much larger than
    # smooth_weight/laplacian_weight, which penalise flow derivatives.
    continuity_weight: float = 0.1
 
    # Discretisation of the continuity residual:
    #   'lagrangian' : DI/Dt + I*div(v), using the warped frame.
    #                  Valid at LARGE displacement -- use this when
    #                  max_shift >> 1 px (the default case here).
    #   'eulerian'   : literal (I_B - I_A) + div(I*v) in conservative flux
    #                  form. A linearisation, only accurate for |v| <~ 1 px.
    continuity_form: str = 'lagrangian'

    # Weight of the supervised MSE term
    sup_weight: float = 0.1

    # Number of parallel CPU workers used to load and pre-process data.
    # Set to 0 to load data in the main process (useful for debugging).
    num_workers: int = 8
    
    # --- Output -------------------------------------------------------------
    # Directory where model weights are saved after each epoch.
    # Saving every epoch lets you roll back to an earlier checkpoint if
    # training diverges or if you accidentally overwrite a good model.
    checkpoint_dir: str = "checkpoints/"

    # Directory for saved figures (loss curves, flow visualisations, etc.)
    output_dir: str = "outputs/"

    # --- Dataset cache ------------------------------------------------------
    # HDF5 path for the pre-generated train / val / test dataset.
    # Set to None to disable caching (regenerate on every run).
    # The cache is automatically invalidated when any generation setting
    # changes: flow_type, max_shift, noise_std, n_pairs_per_frame,
    # val_split, test_split, val_seed, n_test_pairs, or test_seed.
    dataset_cache_path: str = f"synthetic_data/dataset_maxshift_{max_shift}.h5"

    # Fixed seed for the VALIDATION set only.
    # Fixing this makes val-loss numbers directly comparable across runs
    # even when training data is regenerated with a fresh random state.
    val_seed: int = 0


# Create a single shared instance that all other modules import.
# Usage in another file:
#   from bes_flow.config import cfg
#   print(cfg.batch_size)
cfg = Config()
