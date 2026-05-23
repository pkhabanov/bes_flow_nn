# bes_flow

Neural networks for optical flow for Beam Emission Spectroscopy (BES) plasma diagnostics.

## How to run
Install the package
`pip install -e .`

Generate raw BES images
```
python -m bes_flow.tok_loader --shot 194313 --times 2620 2640 --fband 30 200 --res 64 64 --exclude_channels 6 13 19 48 51 58 --out 'raw_data'
```

Specify raw data path in config.py: `cfg.data_path`

Generate training dataset with all 4 flow types
```
python -m bes_flow.generate_mixed_dataset --output output_path
```

Specify training hyperparameters in config.py

Run training
```
module load pytroch
python -m bes_flow.train --model 'pwc'
```
Training script picks up the dataset from cache based on the parameters in config. If `cfg.flow_type=='mixed'`, then the dataset should be generated with generate_mixed_dataset.py separately. 
If `cfg.flow_type in ['smooth', 'modes', 'well', 'zonal']`, then the dataset will be generated automatically when runnning train.py and will contain frame pairs with a single flow type. For more details on different flow types see `dataset.py.`

Compare different methods
```
python -m bes_flow.compare_methods --cache /pscratch/sd/f/filippk/bes_flow/synthetic_data/dataset_maxshift_12.0_mixed.h5 --weights_pwc /pscratch/sd/f/filippk/bes_flow/checkpoints-pwc/model_mixed_epoch_0100.pt --weights_flownets /pscratch/sd/f/filippk/bes_flow/checkpoints-flownet/model_mixed_epoch_0100.pt --skip_farneback --skip_odp
```

## Project structure

```
── bes_flow/
    ├── config.py                  Main hyperparameters here
    ├── model_s.py                 FlowNetS neural network
    ├── model_pwcnet.py            PWC neural network
    ├── loss.py                    Warping L2 loss + smoothness regulariser
    ├── dataset.py                 Synthetic pair generation + DataLoader
    ├── train.py                   Training loop, validation, plotting
    ├── compare_methods.py         Compare performance of different methods on a test dataset
    ├── generate_mixed_dataset.py  Generate a dataset which contains all 4 flow types
    └── odp.py                     Legacy ODP algorithm, refactored with numba.jit
```

## References

- Quenot et al. Experiments in Fluids 1998 — ODP baseline
- Dosovitskiy et al., ICCV 2015 — FlowNet
- Sun et al., CVPR 2018 — PWC-Net
- Meister et al., AAAI 2018 — UnFlow, unsupervised warping loss
