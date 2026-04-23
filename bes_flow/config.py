# bes_flow/config.py
#
# Central configuration file. ALL hyperparameters and file paths live here.

from dataclasses import dataclass


@dataclass
class Config:

    # --- Data ------------------------------------------------------------
    # Path to the NumPy array of raw BES frames, shape (N, 64, 64).
    # Each frame is a single 2-D snapshot of plasma density fluctuations.
    data_path: str = "raw_data/194313_t=2620-2640_f=30-200_2000fr.h5"

    # Fraction of frames held out for validation and training
    val_split: float = 0.1
    test_split: float = 0.1

    # Flow type used for synthetic training pair generation.
    # 'smooth' : Gaussian random field       
    # 'modes'  : sinusoidal mode superposition 
    # 'zonal'  : zonal sinusoidal flow + turbulence     
    # 'well'   : zonal Gaussian flow (Er well) + turbulence    
    flow_type: str = 'zonal'

    # Maximum pixel displacement applied when generating synthetic frame pairs.
    # Drawn uniformly from [-max_shift, +max_shift] pixels in both x and y.
    max_shift: float = 12.0

    # Standard deviation of Gaussian noise added to each synthetic frame.
    # Simulates the electronic noise present in real BES detector signals.
    # Set to 0 to train without noise augmentation.
    noise_std: float = 0.0

    # Number of synthetic pairs generated per real BES frame.
    # Total training pairs = len(train_frames) * n_pairs_per_frame.
    # Increase to enlarge the dataset without needing more real frames.
    n_pairs_per_frame: int = 4

    # --- Model --------------------------------------------------------------
    # Number of channels in the shared CNN encoder output feature maps.
    feature_channels: int = 32

    # Search radius (in pixels, in feature space) for the correlation layer.
    # The layer will test all displacements (dx, dy) with |dx|, |dy| <= this value.
    # It determines the cost-volume size as (2*d+1)^2 channels
    max_displacement: int = 4

    # --- Training ------------------------------------------------------------
    # If supervised, then additional MSE loss will be used with ground truth flow
    is_supervised: bool = False

    # Total number of passes through the training data.
    num_epochs: int = 25

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
    laplacian_weight: float = 0.005

    # Weight of the supervised MSE term
    sup_weight: float = 0.1

    # Number of parallel CPU workers used to load and pre-process data.
    # Set to 0 to load data in the main process (useful for debugging).
    num_workers: int = 8

    # --- Test / evaluation -----------------------------------------------
    # Number of synthetic pairs used for final evaluation on the test set.
    # These are generated once with a fixed seed (test_seed) and never used
    # during training.
    n_test_pairs: int = 300
    test_seed:    int = 42       # fixed seed for reproducible test generation
    
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
