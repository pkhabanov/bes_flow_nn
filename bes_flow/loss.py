# bes_flow/loss.py
#
# Workflow:
#   1. Take the network's predicted flow field.
#   2. Use it to WARP frame B toward frame A (move every pixel by its
#      predicted displacement).
#   3. Measure the L2 distance between frame A and the warped frame B.
#      If the flow prediction is correct, the warped frame should look
#      like frame A and the loss should be near zero.
#
# In addition to the photometric term, we add a smoothness penalty
# that discourages the network from producing jagged, discontinuous flow fields.
#
# PHYSICS-INFORMED TERM: the continuity equation
#   dI/dt + div(I*v) = 0
# The photometric/warping term above implicitly assumes brightness
# constancy, which holds only for INCOMPRESSIBLE flow (div v = 0). That is
# a good model for the mean poloidal ExB flow below the ion sound speed,
# but NOT for the drift-wave turbulence BES actually resolves: those
# fluctuations are compressible, with intensity sources and sinks.
# The continuity residual is the correct generalization - it permits
# compression consistent with the observed intensity change, 
# and reduces to brightness constancy when div v = 0.
# See WarpingL2Loss.continuity_loss for the two available discretisations.
#
# If synthetic training data WITH known ground-truth displacements is
# available (generated in dataset.py), an optional supervised EPE term
# against the ground truth can be included to accelerate convergence.
# Reference: UnFlow (Meister et al., AAAI 2018)

import torch
import torch.nn as nn
import torch.nn.functional as F


class WarpingL2Loss(nn.Module):
    """
    Combined loss for optical flow training

    Parameters
    ----------
    smooth_weight : scalar weight on the smoothness regularisation term.
    sup_weight    : scalar weight on the supervised loss
    is_supervised : bool - if False - unsupervised: photometric + smoothness + laplacian,
                    if True - supervised training: EPE on flow + smoothness
    """
    def __init__(self, smooth_weight=0.01, laplacian_weight=0.05,
                 sup_weight=0.1, continuity_weight=0.0,
                 continuity_form='lagrangian', is_supervised=False):
        super().__init__()
        self.smooth_weight = smooth_weight
        self.sup_weight = sup_weight
        self.laplacian_weight = laplacian_weight
        self.continuity_weight = continuity_weight
        self.continuity_form = continuity_form
        self.is_supervised = is_supervised
 
        if continuity_form not in ('lagrangian', 'eulerian'):
            raise ValueError(
                f"continuity_form must be 'lagrangian' or 'eulerian', "
                f"got {continuity_form!r}"
            )

    def warp(self, frame, flow):
        """
        Warps `frame` by the displacement field `flow` using bilinear
        interpolation, producing a new image where pixel (x, y) contains
        the value of `frame` at position (x + dx, y + dy).

        This is implemented using PyTorch's grid_sample, which expects
        coordinates in the range [-1, 1] (normalized device coordinates).
        We therefore convert the pixel-space flow into that range first.

        Parameters
        ----------
        frame : (B, 1, H, W) — the frame to warp (frame B in our case)
        flow  : (B, 2, H, W) — predicted displacement field in pixels
                               channel 0 = dx (horizontal), channel 1 = dy (vertical)

        Returns
        -------
        warped_frame : (B, 1, H, W)
        """
        B, _, H, W = frame.shape

        # Build a base sampling grid covering the image in [-1, 1] coordinates.
        # grid[b, y, x] = (normalized_x, normalized_y) for pixel (x, y).
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=frame.device),
            torch.linspace(-1, 1, W, device=frame.device),
            indexing='ij'
        )
        # Shape: (1, H, W, 2) — unsqueezed for broadcasting over batch dim
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        # Convert flow from pixel units to normalized [-1, 1] units.
        # With align_corners=True, -1 and +1 map to the CENTRES of the first
        # and last pixels, so the coordinate span from pixel 0 to pixel W-1
        # is 2 units and therefore
        #       1 pixel  =  2 / (W - 1)   normalized units.
        flow_norm = torch.stack([
            flow[:, 0, :, :] * 2.0 / max(W - 1, 1),   # dx normalised
            flow[:, 1, :, :] * 2.0 / max(H - 1, 1),   # dy normalised
        ], dim=1)
        # Rearrange flow to (B, H, W, 2) as required by grid_sample
        flow_norm = flow_norm.permute(0, 2, 3, 1)

        # Displace the base grid by the normalized flow.
        # grid_sample then samples frame at each displaced coordinate using
        # bilinear interpolation (smooth, differentiable w.r.t. flow).
        # padding_mode='border' repeats edge pixels for out-of-bounds queries.
        displaced_grid = grid + flow_norm
        return F.grid_sample(frame, displaced_grid,
                             align_corners=True, padding_mode='border')

    def charbonnier(self, x, eps=1e-3):
        '''
        The Charbonnier photometric loss sqrt(x² + ε²) with ε ≈ 0.001 
        behaves like L1 for large residuals and L2 near zero
        '''
        return torch.sqrt(x**2 + eps**2).mean()

    def epe_loss(self, flow, flow_gt):
        """
        End-Point Error: mean L2 norm of the per-pixel flow error vector.
        """
        return torch.norm(flow - flow_gt, p=2, dim=1).mean()
    
    def smoothness_loss(self, flow, flow_gt=None):
        """
        Penalises spatial gradients in the predicted flow field.

        Combines first- and second-order spatial regularisation.
 
        First-order  (Total Variation - TV):
            Penalises d/dx + d/dy. 
 
        Second-order  (Laplacian):
            Penalises d2/dx2 + d2/dy2 via central differences.

        If flow_gt is provided, the penalty is applied to the RESIDUAL
        (flow_pred - flow_gt) rather than flow_pred directly. 
        If flow_gt is None (unsupervised mode), the penalty is applied to
        the raw prediction as before.
        """
        target = (flow - flow_gt) if flow_gt is not None else flow
 
        # First-order: total variation (forward differences)
        dy1 = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :]).mean()
        dx1 = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1]).mean()
        tv  = dx1 + dy1
 
        # Second-order: Laplacian (central differences)
        dy2 = (target[:, :, 2:, :]  - 2 * target[:, :, 1:-1, :]
               + target[:, :, :-2, :]).abs().mean()
        dx2 = (target[:, :, :, 2:]  - 2 * target[:, :, :, 1:-1]
               + target[:, :, :, :-2]).abs().mean()
        laplacian = dx2 + dy2
        
        return tv, laplacian

    def _divergence(self, flow):
        """
        div(v) = du/dx + dv/dy on interior grid points, via central
        differences.
 
        Both partials are cropped along the perpendicular axis so they are
        evaluated at the SAME set of interior points before being summed.
 
        Parameters
        ----------
        flow : (B, 2, H, W) — channel 0 = dx (u), channel 1 = dy (v),
               in pixel-displacement units per frame interval
 
        Returns
        -------
        (B, 1, H-2, W-2)
        """
        u = flow[:, 0:1, :, :]
        v = flow[:, 1:2, :, :]
        du_dx = (u[:, :, 1:-1, 2:] - u[:, :, 1:-1, :-2]) / 2.0
        dv_dy = (v[:, :, 2:, 1:-1] - v[:, :, :-2, 1:-1]) / 2.0
        return du_dx + dv_dy
 
    def _jacobian_det(self, flow):
        """
        det(I + grad D) for the displacement field D, on interior points.
 
        For the flow map Phi(x) = x + D(x), exact mass conservation reads
 
            I_B(Phi(x)) * det(grad Phi(x))  =  I_A(x)
 
        and grad Phi = I + grad D, so in 2-D
 
            det = (1 + du/dx)(1 + dv/dy) - (du/dy)(dv/dx).
 
        This is the FINITE-displacement statement. Linearizing it gives
        det ~= 1 + div(D), which is only valid for |grad D| << 1. At
        max_shift = 12 px that linearization is badly violated: on the
        synthetic data div(D) has mean magnitude ~0.04-0.13 even though
        the underlying velocity field is divergence-free and the
        flow map is volume-preserving (det = 1 to ~0.3%).
 
        Returns
        -------
        (B, 1, H-2, W-2)
        """
        u = flow[:, 0:1, :, :]
        v = flow[:, 1:2, :, :]
        du_dx = (u[:, :, 1:-1, 2:] - u[:, :, 1:-1, :-2]) / 2.0
        du_dy = (u[:, :, 2:, 1:-1] - u[:, :, :-2, 1:-1]) / 2.0
        dv_dx = (v[:, :, 1:-1, 2:] - v[:, :, 1:-1, :-2]) / 2.0
        dv_dy = (v[:, :, 2:, 1:-1] - v[:, :, :-2, 1:-1]) / 2.0
        return (1.0 + du_dx) * (1.0 + dv_dy) - du_dy * dv_dx
 
    def continuity_loss(self, frameA, frameB, flow, frameB_warped=None):
        """
        Physics-informed residual of the plasma continuity equation
 
            dI/dt + div(I * v) = 0
 
        where I is the measured BES intensity (a proxy for local density)
        and v is the flow field.
 
        Two discretisations are available.
 
        'lagrangian'  (default, recommended)
            Exact finite-displacement mass conservation along
            trajectories - the integrated form of dI/dt + div(I v) = 0:
 
                I_B(x + D) * det(I + grad D)  -  I_A(x)
 
            Being an integrated (not linearized) statement, this is valid
            for arbitrarily large displacement and inherits the warp's
            tolerance of the ~12 px shifts used here.
 
        'eulerian'
            The literal flux-divergence form:
 
                (I_B - I_A)  +  d(I_bar*u)/dx + d(I_bar*v)/dy
 
            written in conservative (flux) form rather than expanded as
            I div(v) + v.grad(I), so that it is discretely conservative.
            I_bar = 0.5*(I_A + I_B) time-centres the flux. This is a
            LINEARIZATION, valid only for |v| <~ 1 px; expect it to be
            dominated by O(|v|^2) truncation error at large max_shift.
 
        Note on units: v is in pixels per frame interval and the grid
        spacing is 1 pixel, so dt = 1 and dx = dy = 1  — no
        additional scaling is required.
 
        Parameters
        ----------
        frameA, frameB : (B, 1, H, W) — consecutive frames
        flow           : (B, 2, H, W) — predicted displacement field
        frameB_warped  : (B, 1, H, W) or None — pass the already-computed
                         warp from the photometric term to avoid a second
                         grid_sample call
 
        Returns
        -------
        scalar — Charbonnier norm of the continuity residual over
                 interior pixels
        """
        if self.continuity_form == 'lagrangian':
            if frameB_warped is None:
                frameB_warped = self.warp(frameB, flow)
 
            det_J = self._jacobian_det(flow)            # (B, 1, H-2, W-2)
 
            # I_B(x + D) * det(I + grad D)  -  I_A(x)
            residual = (frameB_warped[:, :, 1:-1, 1:-1] * det_J
                        - frameA[:, :, 1:-1, 1:-1])
 
        else:  # 'eulerian'
            dI_dt = (frameB - frameA)[:, :, 1:-1, 1:-1]
            I_bar = 0.5 * (frameA + frameB)             # (B, 1, H, W)
 
            # Conservative flux form: F = I * v, then take div(F)
            Fx = I_bar * flow[:, 0:1, :, :]
            Fy = I_bar * flow[:, 1:2, :, :]
            dFx_dx = (Fx[:, :, 1:-1, 2:] - Fx[:, :, 1:-1, :-2]) / 2.0
            dFy_dy = (Fy[:, :, 2:, 1:-1] - Fy[:, :, :-2, 1:-1]) / 2.0
 
            residual = dI_dt + dFx_dx + dFy_dy
 
        # Charbonnier rather than plain L2: this is a data-fidelity
        # residual on noisy detector intensities, so it should be robust
        # to outliers in the same way the photometric term is.
        return self.charbonnier(residual)
 
    def forward(self, frameA, frameB, flow, flow_gt=None):
        """
        Compute the total training loss.
 
        Parameters
        ----------
        frameA  : (B, 1, H, W) — reference frame (target of the warp)
        frameB  : (B, 1, H, W) — frame to be warped toward frameA
        flow    : (B, 2, H, W) — network's predicted displacement field
        flow_gt : (B, 2, H, W) or None
                  Ground-truth flow
 
        Returns
        -------
        total        : scalar — total weighted loss (used for backprop)
        photo_loss   : scalar — photometric term alone (logged separately)
        smooth_loss  : scalar — smoothness term alone (logged separately)
        cont_loss    : scalar — continuity-equation residual (logged separately)
        """
        if self.is_supervised:
            assert flow_gt is not None, \
                "flow_gt must be provided in supervised mode"
            
            sup_loss = self.epe_loss(flow, flow_gt)
 
            # Smoothness applied to residual (flow_pred - flow_gt):
            smooth_loss, _ = self.smoothness_loss(flow, flow_gt)
 
            # Continuity is a residual on (frames, flow), independent of
            # supervision, so it applies here too. No warped frame is
            # available to reuse in this branch.
            cont_loss = self.continuity_loss(frameA, frameB, flow)
 
            # Zero out unused terms for consistent logging
            laplacian_loss = flow.new_zeros(())
            photo_loss = flow.new_zeros(())
            
            total = (sup_loss * self.sup_weight
                     + smooth_loss * self.smooth_weight
                     + self.continuity_weight * cont_loss)
 
        else: # unsupervised
            # Warp frameB toward frameA using the predicted flow.
            frameB_warped = self.warp(frameB, flow)
 
            # Photometric loss
            photo_loss = self.charbonnier(frameA - frameB_warped)
 
            # Smoothness penalty on the predicted flow field
            smooth_loss, laplacian_loss = self.smoothness_loss(flow, flow_gt)
 
            # Continuity residual. Reuse frameB_warped so the Lagrangian
            # form costs one extra grid_sample of zero.
            cont_loss = self.continuity_loss(
                frameA, frameB, flow, frameB_warped=frameB_warped
            )
 
            sup_loss = flow.new_zeros(())
 
            total = (photo_loss
                     + self.smooth_weight * smooth_loss
                     + self.laplacian_weight * laplacian_loss
                     + self.continuity_weight * cont_loss)
        
        return total, photo_loss, smooth_loss, laplacian_loss, sup_loss, cont_loss
 
 
def iterative_warping_loss(frameA, frameB, flow_predictions, criterion, flow_gt, gamma = 0.8):
    """
    gamma-weighted photometric loss over a sequence of flow predictions.
 
    Mirrors the sequence_loss() approach in WAFT / RAFT:
      - Later iterations receive higher weight (gamma^0 = 1.0 for the last).
      - Earlier iterations receive lower weight (gamma^(T-1) for the first).
      - This ensures the model learns to refine progressively rather than
        treating every iteration identically.
 
    Parameters
    ----------
    frameA, frameB   : (B, 1, H, W) — input frame pair
    flow_predictions : list of T tensors, each (B, 2, H, W)
                       in chronological order (earliest = index 0)
    criterion        : WarpingL2Loss instance (unsupervised or supervised)
    flow_gt          : (B, 2, H, W) or None — only needed if criterion.is_supervised
    gamma            : exponential decay factor (default 0.8, as in RAFT / WAFT)
 
    Returns
    -------
    total, photo, smooth, laplacian, sup, cont : scalar tensors
        Weighted sums of each loss component, matching the return signature of
        WarpingL2Loss.forward() for a drop-in replacement in the training loop.
    """
    import torch
 
    T = len(flow_predictions)
 
    total_acc    = torch.zeros(1, device=frameA.device)
    photo_acc    = torch.zeros(1, device=frameA.device)
    smooth_acc   = torch.zeros(1, device=frameA.device)
    lap_acc      = torch.zeros(1, device=frameA.device)
    sup_acc      = torch.zeros(1, device=frameA.device)
    cont_acc     = torch.zeros(1, device=frameA.device)
 
    for i, flow in enumerate(flow_predictions):
        # Weight: earlier iterations get lower weight, last iteration gets 1.0
        weight = gamma ** (T - 1 - i)
 
        loss, photo, smooth, lap, sup, cont = criterion(
            frameA, frameB, flow, flow_gt=flow_gt
        )
 
        total_acc  = total_acc  + weight * loss
        photo_acc  = photo_acc  + weight * photo
        smooth_acc = smooth_acc + weight * smooth
        lap_acc    = lap_acc    + weight * lap
        sup_acc    = sup_acc    + weight * sup
        cont_acc   = cont_acc   + weight * cont
 
    # Return scalars (squeeze the dummy batch dim we used for in-place addition)
    return (total_acc.squeeze(), photo_acc.squeeze(),
            smooth_acc.squeeze(), lap_acc.squeeze(), sup_acc.squeeze(),
            cont_acc.squeeze())
