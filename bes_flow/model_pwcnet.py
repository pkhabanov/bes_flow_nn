# bes_flow/model_pwcnet.py
#
# PWC-Net-inspired optical flow model for BES plasma velocimetry.
#
# Reference: "PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and
# Cost Volume", Sun et al., CVPR 2018.  arXiv:1709.02371
# Official implementation: NVlabs/PWC-Net (PWCNet/model_dc.py)
#
# Adapted for 64×64 BES images with a compact 4-level feature pyramid:
#
#   Raw frame    :  1 × 64 × 64
#   Level 1 (L1) : 16 × 32 × 32
#   Level 2 (L2) : 32 × 16 × 16
#   Level 3 (L3) : 64 ×  8 ×  8   


import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_lrelu(in_ch: int, out_ch: int,
                kernel_size: int = 3,
                stride: int = 1,
                dilation: int = 1) -> nn.Sequential:
    """
    Conv2d -> LeakyReLU(0.1) with 'same' padding for stride-1 layers.
    """
    padding = dilation * (kernel_size // 2)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  dilation=dilation,
                  bias=True),
        nn.LeakyReLU(0.1, inplace=True),
    )


def warp(feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp feature map `feat` by displacement field `flow` using differentiable
    bilinear sampling (PWC-Net eq. 1, official warp() method).

    Displacement values in `flow` must be in *pixel units.

    Parameters
    ----------
    feat : (B, C, H, W)   — features of frame 2 to be warped toward frame 1
    flow : (B, 2, H, W)   — dx (x/col) and dy (y/row), pixel units

    Returns
    -------
    warped : (B, C, H, W)
    """
    B, _, H, W = feat.shape

    # Pixel-coordinate grid for the current spatial resolution
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=feat.device),
        torch.arange(W, dtype=torch.float32, device=feat.device),
        indexing='ij',
    )
    # Displace and normalise to [-1, 1] for grid_sample
    norm_x = (grid_x.unsqueeze(0) + flow[:, 0]) * 2.0 / max(W - 1, 1) - 1.0
    norm_y = (grid_y.unsqueeze(0) + flow[:, 1]) * 2.0 / max(H - 1, 1) - 1.0

    sampling_grid = torch.stack([norm_x, norm_y], dim=-1)  # (B, H, W, 2)
    return F.grid_sample(feat, sampling_grid,
                         mode='bilinear',
                         align_corners=True,
                         padding_mode='border')


class FeaturePyramidExtractor(nn.Module):
    """
    Shared-weight CNN encoder mapping one (B, 1, 64, 64) BES frame to a
    3-level feature pyramid:

        Level 1 : (B, 16, 32, 32)
        Level 2 : (B, 32, 16, 16)
        Level 3 : (B, 64,  8,  8)  

    Design: 3x3 kernels throughout, 3 convolutions per level.

    """

    def __init__(self):
        super().__init__()

        # L1: 1×64×64 -> 16×32×32
        self.conv1a  = _conv_lrelu(1,  16, kernel_size=3, stride=2)
        self.conv1aa = _conv_lrelu(16, 16, kernel_size=3)
        self.conv1b  = _conv_lrelu(16, 16, kernel_size=3)

        # L2: 16×32×32 -> 32×16×16
        self.conv2a  = _conv_lrelu(16, 32, kernel_size=3, stride=2)
        self.conv2aa = _conv_lrelu(32, 32, kernel_size=3)
        self.conv2b  = _conv_lrelu(32, 32, kernel_size=3)

        # L3: 32×16×16 -> 64×8×8
        self.conv3a  = _conv_lrelu(32, 64, kernel_size=3, stride=2)
        self.conv3aa = _conv_lrelu(64, 64, kernel_size=3)
        self.conv3b  = _conv_lrelu(64, 64, kernel_size=3)

    def forward(self, x: torch.Tensor):
        """
        x  : (B, 1, 64, 64)
        c1 (B, 16, 32, 32)
        c2 (B, 32, 16, 16)
        c3 (B, 64,  8,  8)
        """
        c1 = self.conv1b(self.conv1aa(self.conv1a(x)))
        c2 = self.conv2b(self.conv2aa(self.conv2a(c1)))
        c3 = self.conv3b(self.conv3aa(self.conv3a(c2)))

        return c1, c2, c3


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cost Volume  (partial, limited search range d)
# ─────────────────────────────────────────────────────────────────────────────

class CostVolume(nn.Module):
    """
    Partial cost volume: normalised dot-product correlation between frame-1
    features and warped frame-2 features over a ±d search window

    Parameters
    ----------
    max_displacement : search radius d.  Default 4 -> (2*4+1)^2 = 81 channels.
    """

    def __init__(self, max_displacement: int = 4):
        super().__init__()
        self.d    = max_displacement
        self.relu = nn.LeakyReLU(0.1, inplace=True) 

    def forward(self, feat1: torch.Tensor,
                      feat2_warped: torch.Tensor) -> torch.Tensor:
        """
        feat1        : (B, C, H, W) — frame-1 features
        feat2_warped : (B, C, H, W) — warped frame-2 features
        cost         : (B, (2d+1)², H, W)  
        """
        B, C, H, W = feat1.shape
        d = self.d
        n = (2 * d + 1) ** 2

        # L2-normalise along channel axis (paper eq. 2, division by N=C)
        f1 = F.normalize(feat1,        p=2, dim=1)
        f2 = F.normalize(feat2_warped, p=2, dim=1)

        # Compute all (2d+1)^2 dot products via unfold
        f2_unfold  = F.unfold(f2, kernel_size=2*d+1, padding=d)  # (B, C*n, H*W)
        f2_patches = f2_unfold.view(B, C, n, H * W)
        f1_flat    = f1.view(B, C, 1, H * W)
        cost       = (f1_flat * f2_patches).sum(dim=1) / C        # (B, n, H*W)
        cost       = cost.view(B, n, H, W)

        return self.relu(cost) 


class FlowEstimator(nn.Module):
    """
    Multi-layer CNN with DenseNet connections that estimates a dense flow
    field from the concatenated cost volume, frame-1 features, and
    upsampled flow.

    RETURNS
    ───────
    flow      : (B, 2, H, W)
    full_feat : (B, in_ch+c0+c1+c2, H, W)

    Parameters
    ----------
    in_channels  : total input width (n_cost + feat_ch + optional up_flow)
    mid_channels : tuple (c0, c1, c2); scaled-down from original PWCnet (128,128,96,64,32)
    """

    def __init__(self, in_channels: int, mid_channels: tuple = (64, 32, 16)):
        super().__init__()
        c0, c1, c2 = mid_channels
        self._full_ch = in_channels + c0 + c1 + c2   # for external channel queries

        self.conv0   = _conv_lrelu(in_channels,             c0)
        self.conv1   = _conv_lrelu(in_channels + c0,        c1)
        self.conv2   = _conv_lrelu(in_channels + c0 + c1,   c2)
        self.predict = nn.Conv2d(self._full_ch, 2, kernel_size=1)

    @property
    def full_channels(self) -> int:
        """Number of channels in the returned full_feat tensor."""
        return self._full_ch

    def forward(self, x: torch.Tensor):
        """
        x         : (B, in_channels, H, W)
        flow      : (B, 2, H, W)
        full_feat : (B, full_channels, H, W)   [all accumulations]
        """
        x0        = self.conv0(x)
        x1        = self.conv1(torch.cat([x,  x0],     dim=1))
        x2        = self.conv2(torch.cat([x,  x0, x1], dim=1))
        full_feat = torch.cat([x, x0, x1, x2],          dim=1)  # accumulate all
        return self.predict(full_feat), full_feat


class ContextNetwork(nn.Module):
    """
    Dilated-convolution network that refines the L1 flow estimate using a
    wider receptive field.
    INPUT — the FULL accumulated feature tensor from Estimator1
    Dilations [1, 2, 4, 1]

    Parameters
    ----------
    in_channels  : full_feat channels from Estimator1
    mid_channels : hidden width; default 64
    """

    def __init__(self, in_channels: int, mid_channels: int = 64):
        super().__init__()
        c = mid_channels
        self.net = nn.Sequential(
            _conv_lrelu(in_channels, c,  kernel_size=3, dilation=1),
            _conv_lrelu(c,           c,  kernel_size=3, dilation=2),
            _conv_lrelu(c,           c,  kernel_size=3, dilation=4),
            _conv_lrelu(c,           c // 2, kernel_size=3, dilation=1),
            nn.Conv2d(c // 2, 2, kernel_size=1), 
        )

    def forward(self, flow: torch.Tensor,
                      full_feat: torch.Tensor) -> torch.Tensor:
        """
        flow      : (B, 2, H, W)  — flow estimate from Estimator1
        full_feat : (B, C, H, W)  — full accumulated tensor from Estimator1
        ──► flow_refined : (B, 2, H, W)  residual-corrected flow
        """
        delta = self.net(full_feat)
        return flow + delta


class PWCNet(nn.Module):
    """
    PWC-Net adapted for 64x64 BES plasma velocimetry.

    Channel budget  (d=4, n_cost=81)
    ─────────────────────────────────────────────
    FlowEstimator3: in = 81+64   = 145  full_feat = 145+64+32+16 = 257
    FlowEstimator2: in = 81+32+2 = 115  full_feat = 115+64+32+16 = 227
    FlowEstimator1: in = 81+16+2  = 99  full_feat = 99+64+32+16 = 211
    ContextNet: in = 211

    Parameters
    ----------
    max_displacement : cost-volume search radius d.  Default 4 -> 81 ch.
    """

    _MID = (64, 32, 16)   # estimator hidden channels at all levels

    def __init__(self, max_displacement: int = 4):
        super().__init__()

        self.d  = max_displacement
        d = self.d
        n_cost  = (2 * d + 1) ** 2   # 81

        # ── Shared feature pyramid (Siamese) ──────────────────────────────
        self.pyramid = FeaturePyramidExtractor()

        # ── Cost volume (same d at every level) ───────────────────────────
        self.cost_volume = CostVolume(max_displacement=d)

        # ── Flow estimators — independent per level ────────────────────────
        # L3: no prior flow -> cost + c3A only
        self.estimator3 = FlowEstimator(
            in_channels  = n_cost + 64,  # 145
            mid_channels = self._MID,
        )
        # L2: cost + c2A + up_flow3
        self.estimator2 = FlowEstimator(
            in_channels  = n_cost + 32 + 2,  # 115
            mid_channels = self._MID,
        )
        # L1: cost + c1A + up_flow2
        self.estimator1 = FlowEstimator(
            in_channels  = n_cost + 16 + 2,  # 99
            mid_channels = self._MID,
        )

        # ── Context network — applied at L1 only ──────────────────────────
        self.context_net = ContextNetwork(
            in_channels  = self.estimator1.full_channels,  # 211
            mid_channels = 64,
        )

        # Intermediate flow tensors stored for optional multi-scale loss.
        # [flow3 @8×8, flow2 @16×16, flow1 @32×32, flow1_refined @32×32]
        self.flow_pyramid: list = []

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _upsample_flow(flow: torch.Tensor) -> torch.Tensor:
        """
        Bilinear upsample by x2 and multiply displacement values by 2.

        WHY x2: a displacement of d pixels at level l corresponds to 2d pixels
        at level l-1 (pixel pitch halves with each x2 upsampling).
        """
        return F.interpolate(flow, scale_factor=2,
                             mode='bilinear',
                             align_corners=True) * 2.0

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, frameA: torch.Tensor,
                      frameB: torch.Tensor) -> torch.Tensor:
        """
        frameA, frameB : (B, 1, 64, 64) — consecutive BES frames

        Returns
        ───────
        flow_full : (B, 2, 64, 64) — dense (dx, dy) in pixel units

        Side effect
        ───────────
        self.flow_pyramid = [flow3, flow2, flow1, flow1_refined]
        Available after every call for optional multi-scale supervision.
        """
        # ── Siamese feature pyramids ──────────────────────────────────────
        c1A, c2A, c3A = self.pyramid(frameA)
        c1B, c2B, c3B = self.pyramid(frameB)

        # ── L3 ( 8× 8): coarsest, no warping ─────────────────────────────
        cost3              = self.cost_volume(c3A, c3B)        # (B, 81,  8,  8)
        inp3               = torch.cat([cost3, c3A], dim=1)    # (B,145,  8,  8)
        flow3, full_feat3  = self.estimator3(inp3)             # (B,2/257,8,  8)

        # ── L2 (16×16) ────────────────────────────────────────────────────
        up_flow3           = self._upsample_flow(flow3)        # (B,  2, 16, 16)
        c2B_warped         = warp(c2B, up_flow3)
        cost2              = self.cost_volume(c2A, c2B_warped) # (B, 81, 16, 16)
        inp2               = torch.cat(                        # (B,115, 16, 16)
                               [cost2, c2A, up_flow3], dim=1)
        flow2, full_feat2  = self.estimator2(inp2)             # (B,2/227,16,16)

        # ── L1 (32×32): finest active level ──────────────────────────────
        up_flow2           = self._upsample_flow(flow2)        # (B,  2, 32, 32)
        c1B_warped         = warp(c1B, up_flow2)
        cost1              = self.cost_volume(c1A, c1B_warped) # (B, 81, 32, 32)
        inp1               = torch.cat(                        # (B,99, 32, 32)
                               [cost1, c1A, up_flow2], dim=1)
        flow1, full_feat1  = self.estimator1(inp1)             # (B,2/211,32,32)

        # ── Context network: dilated refinement at 32×32 ─────────────────
        flow1_refined      = self.context_net(flow1, full_feat1)  # (B, 2, 32, 32)

        # Store pyramid for optional multi-scale supervision
        self.flow_pyramid  = [flow3, flow2, flow1, flow1_refined]

        # ── Final upsample: 32×32 -> 64×64, values ×2 ─────────────────────
        return self._upsample_flow(flow1_refined)              # (B, 2, 64, 64)


if __name__ == '__main__':
    import torch

    model = PWCNet(max_displacement=4)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nPWCNet  |  parameters: {n_params:,}")

    B  = 4
    fA = torch.randn(B, 1, 64, 64)
    fB = torch.randn(B, 1, 64, 64)

    # ── Training-mode output ─────────────────────────────────────────────────
    model.train()
    out = model(fA, fB)
    assert out.shape == (B, 2, 64, 64), f"Bad shape: {out.shape}"
    print(f"Training output     : {out.shape}  +")

    expected_pyramid_shapes = [
        (B, 2,  8,  8),
        (B, 2, 16, 16),
        (B, 2, 32, 32),
        (B, 2, 32, 32),
    ]
    for i, (got, want) in enumerate(
            zip([f.shape for f in model.flow_pyramid], expected_pyramid_shapes)):
        assert got == want, f"Pyramid[{i}]: got {got}, want {want}"
    print(f"flow_pyramid shapes : {[f.shape for f in model.flow_pyramid]}  +")

    # ── Gradient flow through all components ─────────────────────────────────
    loss = out.mean()
    loss.backward()
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not no_grad, f"No gradient for: {no_grad}"
    print(f"Gradients to all parameters  +")

    # ── Eval-mode output ──────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        out_eval = model(fA, fB)
    assert out_eval.shape == (B, 2, 64, 64)
    print(f"Eval output         : {out_eval.shape}  +")

    print("\nAll checks passed.")
