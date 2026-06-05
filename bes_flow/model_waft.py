# bes_flow/model_waft_bes.py
#
# WAFT-inspired two-phase network for BES optical flow
#
# Architecture
# ─────────────
#   ENCODER  FeatureEncoder
#            c1: (B, 16, 32, 32)  half-resolution,   used in Phase 2
#            c2: (B, 32,  8,  8)  eighth-resolution, used in Phase 1
#            Channel progression 1->16->32

#   PHASE 1  Coarse iterations at c2  (32ch x 8x8)
#            A 15 px image displacement is only ~2 px at c2 feature scale,
#            so zero-initialised warping converges reliably.
#            One spatial position = one physical BES measurement channel.
#
#   PHASE 2  Fine iterations at c1  (16ch x 32x32)
#            Warm-started from Phase 1; handles small residuals.
#            Convex-upsampled to 64x64 for the final prediction.
#
#   UPDATE   SharedConvUpdater — 4-layer dilated CNN shared across
#            both phases.  Effective RF: 31x31 at c1 (94% of map),
#            fully global at c2.
#
# Flow unit convention
# ─────────────────────
#   flow2  c2-feature-pixel units  (1 unit = 8 image px)
#   flow1  c1-feature-pixel units  (1 unit = 2 image px)
#   Loss-time tensors are converted to image-pixel units:
#     Phase 1: _scale_upsample(flow2 x 8, 64x64)
#     Phase 2: convex_upsample(flow1)               (x2 applied inside)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Encoder ─────────────────────────────────────────────────────────────────

def _conv_lrelu(in_ch, out_ch, kernel_size=3, stride=1):
    """Conv2d + LeakyReLU(0.1) with symmetric padding."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                  stride=stride, padding=(kernel_size - 1) // 2, bias=True),
        nn.LeakyReLU(0.1, inplace=True),
    )


class FeatureEncoder(nn.Module):
    """
    Two-level feature pyramid encoder
    """

    def __init__(self):
        super().__init__()

        # Level 1: 1x64x64 -> 16x32x32
        self.conv1a  = _conv_lrelu(1,  16, kernel_size=3, stride=2)
        self.conv1aa = _conv_lrelu(16, 16, kernel_size=3)
        self.conv1b  = _conv_lrelu(16, 16, kernel_size=3)

        # Level 2: 16x32x32 -> 32x8x8
        self.conv2a  = _conv_lrelu(16, 32, kernel_size=3, stride=2)
        self.conv2aa = _conv_lrelu(32, 32, kernel_size=3, stride=2)
        self.conv2b  = _conv_lrelu(32, 32, kernel_size=3)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x  : (B, 1, 64, 64)  single-channel BES frame in [0, 1]

        Returns
        -------
        c1 : (B, 16, 32, 32)
        c2 : (B, 32,  8,  8)
        """
        c1 = self.conv1a(x)
        c1 = self.conv1aa(c1)
        c1 = self.conv1b(c1)    # (B, 16, 32, 32)

        c2 = self.conv2a(c1)    # (B, 32, 16, 16) 
        c2 = self.conv2aa(c2)   # (B, 64,  8,  8)
        c2 = self.conv2b(c2)    # (B, 64,  8,  8)

        return c1, c2


# ─── Helpers ─────────────────────────────────────────────────────────────────

def bilinear_warp(feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp `feat` by `flow` using differentiable bilinear sampling.
    `flow` is in pixel units at the resolution of `feat`.

    Parameters
    ----------
    feat : (B, C, H, W)
    flow : (B, 2, H, W)  — (dx, dy) in feature-pixel units

    Returns
    -------
    warped : (B, C, H, W)
    """
    B, _, H, W = feat.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=feat.dtype, device=feat.device),
        torch.arange(W, dtype=feat.dtype, device=feat.device),
        indexing='ij',
    )
    norm_x = (grid_x + flow[:, 0]) * 2.0 / max(W - 1, 1) - 1.0
    norm_y = (grid_y + flow[:, 1]) * 2.0 / max(H - 1, 1) - 1.0
    grid   = torch.stack([norm_x, norm_y], dim=-1)
    return F.grid_sample(feat, grid, mode='bilinear',
                         align_corners=True, padding_mode='border')


def convex_upsample(flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Learned 2x convex upsampling (RAFT / WAFT style).
    The x2 factor inside converts c1-feature-pixel units -> image-pixel units.

    Parameters
    ----------
    flow : (B, 2,  H,  W)   c1-feature-pixel units
    mask : (B, 36, H,  W)   raw logits  (36 = 4 output pixels x 9 neighbours)

    Returns
    -------
    flow_up : (B, 2, 2H, 2W)  image-pixel units
    """
    B, _, H, W = flow.shape
    mask    = mask.view(B, 1, 9, 2, 2, H, W)
    mask    = torch.softmax(mask, dim=2)
    up_flow = F.unfold(2.0 * flow, kernel_size=3, padding=1)
    up_flow = up_flow.view(B, 2, 9, 1, 1, H, W)
    up_flow = (mask * up_flow).sum(dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(B, 2, 2 * H, 2 * W)


def _scale_upsample(flow: torch.Tensor,
                    target_h: int, target_w: int, scale: int) -> torch.Tensor:
    """
    Bilinear upsample + multiply values by `scale`.
    Converts feature-pixel units between pyramid levels:
        c2 (8x8) -> c1 (32x32): scale = 4
        c2 (8x8) -> image (64x64): scale = 8
    """
    return F.interpolate(flow, size=(target_h, target_w),
                         mode='bilinear', align_corners=True) * scale


# ─── Update module ────────────────────────────────────────────────────────────

class SharedConvUpdater(nn.Module):
    """
    4-layer dilated CNN shared across both pyramid phases.

    Effective receptive field (dilations 1, 2, 4, 8):
        RF = 1 + 2x(1+2+4+8) = 31x31
    Coverage:
        c2  8x8  -> 100% after 3 layers
        c1 32x32 ->  94% after 4 layers

    Replaces the ViT-tiny used in the published WAFT. For 64x64 BES frames
    dilated convolutions provide equivalent global coverage at 13x lower
    parameter count.

    Parameters
    ----------
    iter_dim : number of input and output channels (D throughout WAFTNet)
    """

    def __init__(self, iter_dim: int):
        super().__init__()
        D = iter_dim
        self.net = nn.Sequential(
            nn.Conv2d(D,   2*D, kernel_size=3, padding=1,  dilation=1), nn.ReLU(inplace=True),
            nn.Conv2d(2*D, 2*D, kernel_size=3, padding=2,  dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(2*D, 2*D, kernel_size=3, padding=4,  dilation=4), nn.ReLU(inplace=True),
            nn.Conv2d(2*D, 2*D, kernel_size=3, padding=8,  dilation=8), nn.ReLU(inplace=True),
            nn.Conv2d(2*D, D,   kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── WAFTNet ───────────────────────────────────────────────────────────────

class WAFTNet(nn.Module):
    """
    Two-phase WAFT optical flow network for 64x64 BES plasma frames.

    Parameters
    ----------
    iter_dim : channel width for hidden states and projections (default 32)
    iters_c  : warping iterations in Phase 1 at c2  (default 3)
    iters_f  : warping iterations in Phase 2 at c1  (default 3)
    """

    def __init__(self, iter_dim: int = 32, iters_c: int = 3, iters_f: int = 3):
        super().__init__()
        self.iters_c = iters_c
        self.iters_f = iters_f
        D = iter_dim

        # ── Encoder ───────────────────────────────────────────────────────
        self.encoder    = FeatureEncoder()
        # projection from c1 to D
        self.fmap1_proj = nn.Conv2d(16, D, kernel_size=1)   # c1: 16ch -> D

        # ── Shared update module ──────────────────────────────────────────
        self.updater = SharedConvUpdater(D)

        # ── Phase 1 sub-modules  (c2, 8x8) ───────────────────────────────
        self.hidden2_init   = nn.Conv2d(2*D,     D, kernel_size=1)  # cat(fmap2A, fmap2B)
        self.warp2_proj     = nn.Conv2d(3*D + 2, D, kernel_size=1)  # cat(fmap2A, warp2, hidden2, flow2)
        self.hidden2_update = nn.Conv2d(2*D,     D, kernel_size=1)  # cat(refined2, hidden2)
        self.flow2_head     = nn.Sequential(
            nn.Conv2d(D, 2*D, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*D, 2, kernel_size=1),
        )

        # ── Phase 2 sub-modules  (c1, 32x32) ─────────────────────────────
        self.hidden1_init   = nn.Conv2d(3*D,     D, kernel_size=1)  # cat(fmap1A, fmap1B, hidden2_up)
        self.warp1_proj     = nn.Conv2d(3*D + 2, D, kernel_size=1)  # cat(fmap1A, warp1, hidden1, flow1)
        self.hidden1_update = nn.Conv2d(2*D,     D, kernel_size=1)  # cat(refined1, hidden1)
        self.flow1_head     = nn.Sequential(
            nn.Conv2d(D, 2*D, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*D, 2, kernel_size=1),
        )
        # Convex upsample weights: 32x32 -> 64x64  (36 = 4 output pixels x 9 neighbours)
        self.upsample_mask  = nn.Sequential(
            nn.Conv2d(D, 2*D, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*D, 36, kernel_size=1),
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        frameA  : torch.Tensor,
        frameB  : torch.Tensor,
        iters_c : int = None,  # coarse iterations
        iters_f : int = None,  # fine iterations
    ):
        """
        Parameters
        ----------
        frameA, frameB : (B, 1, 64, 64)  consecutive BES frames in [0, 1]
        iters_c, iters_f : override default iteration counts

        Returns
        ───────
        train : list of (iters_c + iters_f) tensors, each (B, 2, 64, 64).
                Ordered earliest -> latest; pass to iterative_warping_loss().
        eval  : single (B, 2, 64, 64) tensor — drop-in for predict_dataset().
        """
        if iters_c is None: iters_c = self.iters_c
        if iters_f is None: iters_f = self.iters_f
        B = frameA.shape[0]

        # ── Feature extraction ────────────────────────────────────────────
        c1A, c2A = self.encoder(frameA)   # c1: (B,16,32,32)  c2: (B,32,8,8)
        c1B, c2B = self.encoder(frameB)

        # c2 is already D=32ch — used directly in Phase 1
        fmap2A = c2A    # (B, D, 8,  8)
        fmap2B = c2B
        # c1 is 16ch — project to D=32ch for Phase 2
        fmap1A = self.fmap1_proj(c1A)    # (B, D, 32, 32)
        fmap1B = self.fmap1_proj(c1B)

        # ─────────────────────────────────────────────────────────────────
        # PHASE 1 — Coarse iterations at c2 (8x8)
        # ─────────────────────────────────────────────────────────────────
        hidden2 = torch.tanh(
            self.hidden2_init(torch.cat([fmap2A, fmap2B], dim=1))
        )
        flow2 = torch.zeros(B, 2, 8, 8,
                            device=frameA.device, dtype=frameA.dtype)
        flow_predictions = []

        for _ in range(iters_c):
            flow2 = flow2.detach()   # stop gradient through flow estimate

            warp_feat2 = bilinear_warp(fmap2B, flow2)
            inp2 = self.warp2_proj(
                torch.cat([fmap2A, warp_feat2, hidden2, flow2], dim=1)
            )
            refined2 = self.updater(inp2)
            hidden2 = torch.tanh(
                self.hidden2_update(torch.cat([refined2, hidden2], dim=1))
            )
            flow2 = flow2 + self.flow2_head(hidden2)

            # Scale x8: 1 c2-feature-pixel = 8 image pixels
            flow_predictions.append(
                _scale_upsample(flow2, 64, 64, scale=8)
            )

        # ─────────────────────────────────────────────────────────────────
        # PHASE 2 — Fine iterations at c1 (32x32)
        # ─────────────────────────────────────────────────────────────────

        # Convert c2 flow -> c1 units: scale x4  (8x4=32)
        flow1 = _scale_upsample(flow2.detach(), 32, 32, scale=4)

        # Pass Phase 1 global context to Phase 2 hidden state init
        hidden2_up = F.interpolate(hidden2.detach(), size=(32, 32),
                                   mode='bilinear', align_corners=True)
        hidden1 = torch.tanh(
            self.hidden1_init(
                torch.cat([fmap1A, fmap1B, hidden2_up], dim=1)
            )
        )

        for _ in range(iters_f):
            flow1 = flow1.detach()

            warp_feat1 = bilinear_warp(fmap1B, flow1)
            inp1 = self.warp1_proj(
                torch.cat([fmap1A, warp_feat1, hidden1, flow1], dim=1)
            )
            refined1 = self.updater(inp1)
            hidden1 = torch.tanh(
                self.hidden1_update(torch.cat([refined1, hidden1], dim=1))
            )
            flow1 = flow1 + self.flow1_head(hidden1)

            # Convex upsample 32x32 -> 64x64; x2 conversion is inside
            mask = 0.25 * self.upsample_mask(hidden1)
            flow_predictions.append(convex_upsample(flow1, mask))

        if self.training:
            return flow_predictions
        return flow_predictions[-1]


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model = WAFTNet(iter_dim=32, iters_c=3, iters_f=3)

    total = sum(p.numel() for p in model.parameters())
    print(f'\nBESWAFTNet  |  {total:,} parameters\n')
    for name, module in [
        ('encoder',        model.encoder),
        ('fmap1_proj',     model.fmap1_proj),
        ('updater',        model.updater),
        ('phase1 heads',   nn.Sequential(model.hidden2_init, model.warp2_proj,
                                         model.hidden2_update, model.flow2_head)),
        ('phase2 heads',   nn.Sequential(model.hidden1_init, model.warp1_proj,
                                         model.hidden1_update, model.flow1_head,
                                         model.upsample_mask)),
    ]:
        n = sum(p.numel() for p in module.parameters())
        print(f'  {name:<18s}: {n:>7,}')

    B  = 6
    fA = torch.randn(B, 1, 64, 64)
    fB = torch.randn(B, 1, 64, 64)

    model.train()
    preds = model(fA, fB)
    assert isinstance(preds, list) and len(preds) == 6
    assert all(p.shape == (B, 2, 64, 64) for p in preds)
    print(f'\nTrain  -> list of {len(preds)} x {tuple(preds[0].shape)}')

    loss = sum(p.mean() for p in preds)
    loss.backward()
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not no_grad, f'Missing gradients: {no_grad}'
    print(f'Grads  -> {total:,} parameters')

    model.eval()
    with torch.no_grad():
        out = model(fA, fB)
    assert out.shape == (B, 2, 64, 64)
    print(f'Eval   -> {tuple(out.shape)}')

    print('\nAll checks passed.')