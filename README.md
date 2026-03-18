# bes_flow

Neural network optical flow for Beam Emission Spectroscopy (BES) plasma diagnostics.

A Siamese convolutional network that estimates
dense 2-D displacement fields between consecutive 64×64 BES frames.

## Project structure

```
── bes_flow/
    ├── config.py        ← ALL hyperparameters in one place
    ├── model.py         ← Siamese encoder + correlation layer + head
    ├── loss.py          ← warping L2 loss + smoothness regulariser
    ├── dataset.py       ← synthetic pair generation + DataLoader
    └── train.py         ← training loop, validation, plotting, entry point
```

## Adjusting hyperparameters

Edit `bes_flow/config.py`. The most important settings are:

| Parameter          | Default | Effect                                  |
|--------------------|---------|-----------------------------------------|
| `max_shift`        | 6.0     | Maximum expected displacement in pixels |
| `max_displacement` | 6    | Correlation search radius (feature space)  |
| `smooth_weight`    | 0.05    | Smoothness regularisation strength      |
| `num_epochs`       | 100     | Training duration                       |
| `batch_size`       | 16      | Frames per gradient step                |

## Architecture summary

```
Frame A ──► [Shared CNN Encoder] ──► fA ──►
                                           [Correlation Layer] ──► [Head] ──► (dx, dy)
Frame B ──► [Shared CNN Encoder] ──► fB ──►
```

See `model.py` for a detailed explanation of each component.

## References

- Quenot et al. (1998) — ODP baseline this replaces
- Dosovitskiy et al., ICCV 2015 — FlowNet, correlation layer
- Sun et al., CVPR 2018 — PWC-Net, feature normalisation
- Meister et al., AAAI 2018 — UnFlow, unsupervised warping loss
