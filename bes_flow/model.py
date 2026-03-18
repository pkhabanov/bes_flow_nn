# bes_flow/model.py
#
# Defines the Siamese Convolutional Neural Network used to estimate plasma flow
# from pairs of BES (Beam Emission Spectroscopy) images.
#
# BES produces 8x8 images of plasma density fluctuations, interpolated to
# 64x64 pixels. We find the displacement field (dx, dy) that best maps
# frame A onto frame B. 


import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseEncoder(nn.Module):
    """
    Encodes a single BES frame into feature maps using
    progressively smaller kernels: 7x7 -> 5x5 -> 3x3

    Outputs
    -------
    coarse : (B, feature_channels, 32, 32) — passed to CorrelationLayer
    fine   : (B, 16, 64, 64)               — skip connection for the decoder
    """
    def __init__(self, in_channels=1, feature_channels=32):
        super().__init__()

        # Layer 1: 7x7 kernel — captures large-scale flow structure.
        # padding=3 preserves spatial dimensions (64x64 -> 64x64).
        # Output saved as the skip connection for the decoder.
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
        )

        # Layer 2: 5x5 kernel with stride=2 — captures intermediate-scale
        # features while halving spatial resolution (64x64 -> 32x32).
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
        )

        # Layer 3: 3x3 kernel — refines fine-detail features at the
        # reduced 32x32 resolution. 
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        fine   = self.layer1(x)    # (B, 16, 64, 64) — full-resolution, large-scale features
        x      = self.layer2(fine) # (B, 32, 32, 32) — downsampled, intermediate features
        coarse = self.layer3(x)    # (B, C,  32, 32) — compact, multi-scale representation
        return coarse, fine


class CorrelationLayer(nn.Module):
    """
    Computes cross-correlation between two coarse feature maps over a
    displacement search window of radius max_displacement.

    Input  : fA, fB  each (B, C, 32, 32)
    Output : cost volume (B, (2d+1)^2, 32, 32)
    """
    def __init__(self, max_displacement=4):
        super().__init__()
        self.max_displacement = max_displacement

    def forward(self, fA, fB):
        B, C, H, W = fA.shape
        d = self.max_displacement

        fA = F.normalize(fA, p=2, dim=1)
        fB = F.normalize(fB, p=2, dim=1)

        # Extract all (2d+1)x(2d+1) neighbourhood patches from fB
        n_disps   = (2 * d + 1) ** 2
        fB_unfold = F.unfold(fB, kernel_size=2*d+1, padding=d)  # (B, C*(2d+1)², H*W)
        fB_patches = fB_unfold.view(B, C, n_disps, H * W)

        fA_flat = fA.view(B, C, 1, H * W)
        cost    = (fA_flat * fB_patches).sum(dim=1)  # (B, (2d+1)², H*W)

        return cost.view(B, n_disps, H, W)


class DisplacementDecoder(nn.Module):
    """
    Decoder CNN that takes the 32x32 cost volume and reconstructs a
    full 64x64 displacement field — one (dx, dy) vector per pixel.

    Architecture
    ─────────────
    The decoder mirrors the encoder:

        cost volume  (B, D,  32, 32)   D = (2d+1)^2
            Conv block coarse
                (B, 64, 32, 32)
            Upsample x2  (nearest + conv, avoids checkerboard artefacts)
                (B, 64, 64, 64)
            cat(skipA, skipB)
                (B, 64+16+16, 64, 64)
            Conv block fine
                (B, 32, 64, 64)
            1x1 conv
                (B, 2,  64, 64) - per-pixel (dx, dy)

    Parameters
    ----------
    num_disp_channels : number of cost volume channels = (2*max_displacement+1)²
    skip_channels     : number of channels in the encoder skip connection (16+16)
    """
    def __init__(self, num_disp_channels, skip_channels=32):
        super().__init__()

        D = num_disp_channels  # e.g. 169 for max_displacement=6

        # Step 1: process cost volume at 32x32
        # Reduces the large number of correlation channels down to 64
        self.conv_coarse = nn.Sequential(
            nn.Conv2d(D, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )

        # Step 2: upsample from 32x32 -> 64x64
        # Nearest-neighbour resize: each 32x32 cell is simply replicated
        # into a 2x2 block. The following conv then smooths and refines.
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Step 3: fuse upsampled coarse features with both encoder skip connections.
        # in_channels = 64 (coarse) + skip_channels (skipA + skipB combined).
        self.conv_fine = nn.Sequential(
            nn.Conv2d(64 + skip_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )

        # Step 4: final 1x1 conv to produce exactly 2 output channels (dx, dy)
        # 1x1 conv = independent linear combination at each pixel, no spatial mixing
        self.output_conv = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, cost_volume, skipA, skipB):
        """
        Parameters
        ----------
        cost_volume : (B, D,  32, 32) — from CorrelationLayer
        skipA       : (B, 16, 64, 64) — fine features from encoder applied to frameA
        skipB       : (B, 16, 64, 64) — fine features from encoder applied to frameB
 
        Returns
        -------
        flow : (B, 2, 64, 64) — per-pixel displacement field, in pixel units
        """
        x = self.conv_coarse(cost_volume)        # (B, 64, 32, 32)
        x = self.upsample(x)                     # (B, 64, 64, 64)
        x = torch.cat([x, skipA, skipB], dim=1)  # (B, 64+16+16, 64, 64)
        x = self.conv_fine(x)                    # (B, 32, 64, 64)
        return self.output_conv(x)               # (B,  2, 64, 64)


class SiameseDisplacementNet(nn.Module):
    """
    Full pipeline: two 64x64 BES frames in, 64x64 displacement field out.

        frameA, frameB
            SiameseEncoder (shared weights)
        coarse features (32x32) + fine skip (64x64)
            CorrelationLayer
        cost volume (32x32)
            DisplacementDecoder
        flow field (64x64) - one (dx, dy) per pixel

    Parameters
    ----------
    feature_channels : encoder output width (default 32)
    max_displacement : correlation search radius in feature-space pixels (default 6)
    """
    def __init__(self, feature_channels=32, max_displacement=6):
        super().__init__()
        self.encoder     = SiameseEncoder(feature_channels=feature_channels)
        self.correlation = CorrelationLayer(max_displacement=max_displacement)
        n_disp           = (2 * max_displacement + 1) ** 2
        self.decoder     = DisplacementDecoder(
            num_disp_channels = n_disp,
            skip_channels     = 32,    # 16 channels per frame × 2 frames
        )

    def forward(self, frameA, frameB):
        """
        Parameters
        ----------
        frameA, frameB : (B, 1, 64, 64) — batch of consecutive BES frames

        Returns
        -------
        flow : (B, 2, 64, 64) — per-pixel (dx, dy) in pixel units
        """
        # Both frames share the same encoder weights.
        coarseA, skipA = self.encoder(frameA)
        coarseB, skipB = self.encoder(frameB)

        cost = self.correlation(coarseA, coarseB)   # (B, D,  32, 32)
        flow = self.decoder(cost, skipA, skipB)     # (B, 2,  64, 64)
        return flow
    