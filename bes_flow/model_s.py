# bes_flow/model_s.py
#
# FlowNetS-inspired optical flow model for BES plasma velocimetry.
#
# For 64×64 we use three stride-2 steps, reaching an 8×8 bottleneck
# Channel widths are scaled down proportionally 32 -> 64 -> 128.
#
# Architecture
# ─────────────
#
#   Input : (B, 2, 64, 64)  — frameA and frameB stacked along channel dim
#
#   ENCODER
#     conv1 : 7×7  stride 1 → (B,  32, 64, 64)   ← skip1
#     conv2 : 5×5  stride 2 → (B,  64, 32, 32)   ← skip2
#     conv3 : 3×3  stride 2 → (B, 128, 16, 16)   ← skip3
#     conv4 : 3×3  stride 2 → (B, 128,  8,  8)   (bottleneck)
#
#   DECODER  (upsample ×2 → conv → cat skip → refine)
#     up3   : → (B,  64, 16, 16) │ cat skip3(128) → (B, 192, 16, 16) → (B,  64, 16, 16)
#     up2   : → (B,  32, 32, 32) │ cat skip2( 64) → (B,  96, 32, 32) → (B,  32, 32, 32)
#     up1   : → (B,  16, 64, 64) │ cat skip1( 32) → (B,  48, 64, 64) → (B,  16, 64, 64)
#     output: 1×1 conv           → (B,   2, 64, 64)  — per-pixel (dx, dy)
#
# GroupNorm is used throughout for batch-size-independent
# normalisation, with num_groups=8 (divisor of all channel counts above).

import torch
import torch.nn as nn


def _conv_gn_relu(in_ch, out_ch, kernel_size, stride=1, num_groups=8):
    """
    Conv2d → GroupNorm → LeakyReLU block used throughout encoder and decoder.

    padding = kernel_size // 2 so that:
      - stride-1 convolutions preserve spatial dimensions, and
      - stride-2 convolutions halve them.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                  stride=stride, padding=kernel_size // 2),
        nn.GroupNorm(num_groups, out_ch),
        nn.LeakyReLU(0.1),
    )


class FlowEncoder(nn.Module):
    """
    Encoder takes both frames concatenated and produces a
    feature map plus three skip-connection tensors.

    Parameters
    ----------
    num_groups : GroupNorm groups (default 8).
                 Must divide 32, 64, and 128.

    """

    def __init__(self, num_groups=8):
        super().__init__()

        assert  32 % num_groups == 0, \
            f"num_groups ({num_groups}) must divide conv1 channels (32)"
        assert  64 % num_groups == 0, \
            f"num_groups ({num_groups}) must divide conv2 channels (64)"
        assert 128 % num_groups == 0, \
            f"num_groups ({num_groups}) must divide conv3/4 channels (128)"

        # conv1: capture coarse inter-frame structure at full resolution
        self.conv1 = _conv_gn_relu(2,   32,  kernel_size=7, stride=1, num_groups=num_groups)

        # conv2–4: progressively smaller kernels, stride=2 halves resolution.
        self.conv2 = _conv_gn_relu(32,  64,  kernel_size=5, stride=2, num_groups=num_groups)
        self.conv3 = _conv_gn_relu(64,  128, kernel_size=3, stride=2, num_groups=num_groups)
        self.conv4 = _conv_gn_relu(128, 128, kernel_size=3, stride=2, num_groups=num_groups)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, 2, 64, 64) — frameA and frameB concatenated along dim 1

        Returns
        -------
        bottleneck : (B, 128,  8,  8)
        skip1      : (B,  32, 64, 64)
        skip2      : (B,  64, 32, 32)
        skip3      : (B, 128, 16, 16)
        """
        skip1      = self.conv1(x)       # (B,  32, 64, 64)
        skip2      = self.conv2(skip1)   # (B,  64, 32, 32)
        skip3      = self.conv3(skip2)   # (B, 128, 16, 16)
        bottleneck = self.conv4(skip3)   # (B, 128,  8,  8)
        return bottleneck, skip1, skip2, skip3


class FlowDecoder(nn.Module):
    """
    Symmetric decoder that reconstructs a dense 64x64 displacement field
    from the encoder bottleneck, guided by three skip connections.

    Parameters
    ----------
    num_groups : GroupNorm groups (default 8).
                 Must divide all intermediate channel counts:
                 64, 128, 192, 32, 96, 16, 48.
    """
    def __init__(self, num_groups=8):
        super().__init__()

        for ch, label in [
            (64,  'up3 post-conv'), (192, 'up3 post-cat'),
            (32,  'up2 post-conv'), ( 96, 'up2 post-cat'),
            (16,  'up1 post-conv'), ( 48, 'up1 post-cat'),
        ]:
            assert ch % num_groups == 0, \
                f"num_groups ({num_groups}) must divide {label} channels ({ch})"

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Level 3: 8×8 -> 16×16 
        # Upsample bottleneck (128 ch) -> reduce to 64 -> cat skip3 (128 ch)
        # -> 192 ch total -> refine to 64 ch.
        self.up3_conv = _conv_gn_relu(128, 64, kernel_size=3, num_groups=num_groups)
        self.refine3  = nn.Sequential(
            _conv_gn_relu(64 + 128, 64, kernel_size=3, num_groups=num_groups),
            _conv_gn_relu(64, 64, kernel_size=3, num_groups=num_groups),
        )

        # Level 2: 16×16 -> 32×32 
        # 64 ch -> reduce to 32 -> cat skip2 (64 ch) -> 96 ch -> refine to 32 ch.
        self.up2_conv = _conv_gn_relu(64, 32, kernel_size=3, num_groups=num_groups)
        self.refine2  = nn.Sequential(
            _conv_gn_relu(32 + 64, 32, kernel_size=3, num_groups=num_groups),
            _conv_gn_relu(32, 32, kernel_size=3, num_groups=num_groups),
        )

        # Level 1: 32×32 -> 64×64
        # 32 ch -> reduce to 16 -> cat skip1 (32 ch) -> 48 ch -> refine to 16 ch.
        self.up1_conv = _conv_gn_relu(32, 16, kernel_size=3, num_groups=num_groups)
        self.refine1  = nn.Sequential(
            _conv_gn_relu(16 + 32, 16, kernel_size=3, num_groups=num_groups),
            _conv_gn_relu(16, 16, kernel_size=3, num_groups=num_groups),
        )

        # Output - 1×1 conv collapses channels to 2 (dx, dy).
        self.output_conv = nn.Conv2d(16, 2, kernel_size=1)

    def forward(self, bottleneck, skip1, skip2, skip3):
        """
        Parameters
        ----------
        bottleneck : (B, 128,  8,  8)
        skip1      : (B,  32, 64, 64)
        skip2      : (B,  64, 32, 32)
        skip3      : (B, 128, 16, 16)

        Returns
        -------
        flow : (B, 2, 64, 64) — per-pixel (dx, dy) in pixel units
        """
        # Level 3: 8 → 16
        x = self.upsample(bottleneck)       # (B, 128, 16, 16)
        x = self.up3_conv(x)                # (B,  64, 16, 16)
        x = torch.cat([x, skip3], dim=1)   # (B, 192, 16, 16)
        x = self.refine3(x)                 # (B,  64, 16, 16)

        # Level 2: 16 → 32
        x = self.upsample(x)                # (B,  64, 32, 32)
        x = self.up2_conv(x)                # (B,  32, 32, 32)
        x = torch.cat([x, skip2], dim=1)   # (B,  96, 32, 32)
        x = self.refine2(x)                 # (B,  32, 32, 32)

        # Level 1: 32 → 64
        x = self.upsample(x)                # (B,  32, 64, 64)
        x = self.up1_conv(x)                # (B,  16, 64, 64)
        x = torch.cat([x, skip1], dim=1)   # (B,  48, 64, 64)
        x = self.refine1(x)                 # (B,  16, 64, 64)

        return self.output_conv(x)          # (B,   2, 64, 64)


class BESFlowNetS(nn.Module):
    """
    FlowNetS-inspired optical flow network for BES plasma velocimetry.

    Parameters
    ----------
    num_groups : GroupNorm groups in encoder and decoder (default 8).
                 Must divide 32, 64, and 128.
    """
    def __init__(self, num_groups=8):
        super().__init__()
        self.encoder = FlowEncoder(num_groups=num_groups)
        self.decoder = FlowDecoder(num_groups=num_groups)

    def forward(self, frameA, frameB):
        """
        Parameters
        ----------
        frameA, frameB : (B, 1, 64, 64) — consecutive BES frames, normalised to [0, 1]

        Returns
        -------
        flow : (B, 2, 64, 64) — per-pixel (dx, dy) in pixel units
        """
        # Stack both frames into a single 2-channel input so the encoder
        # sees both simultaneously at every spatial position.
        x = torch.cat([frameA, frameB], dim=1)              # (B, 2, 64, 64)

        bottleneck, skip1, skip2, skip3 = self.encoder(x)
        flow = self.decoder(bottleneck, skip1, skip2, skip3)
        
        return flow
