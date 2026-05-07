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
# If synthetic training data WITH known ground-truth displacements is
# available (generated in dataset.py), an optional supervised EPE term
# against the ground truth can be included to accelerate convergence.
#
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
                 sup_weight=0.1, is_supervised=False):
        super().__init__()
        self.smooth_weight = smooth_weight
        self.sup_weight = sup_weight
        self.laplacian_weight = laplacian_weight
        self.is_supervised = is_supervised

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
        # For the x direction the displacement of W/2 pixels is 1.0 in normalized coords.
        flow_norm = torch.stack([
            flow[:, 0, :, :] / (W / 2),   # dx normalised
            flow[:, 1, :, :] / (H / 2),   # dy normalised
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
        """
        if self.is_supervised:
            assert flow_gt is not None, \
                "flow_gt must be provided in supervised mode"
            
            sup_loss = self.epe_loss(flow, flow_gt)

            # Smoothness applied to residual (flow_pred - flow_gt):
            smooth_loss, _ = self.smoothness_loss(flow, flow_gt)

            # Zero out unused terms for consistent logging
            laplacian_loss = flow.new_zeros(())
            photo_loss = flow.new_zeros(())
            
            total = sup_loss * self.sup_weight + smooth_loss * self.smooth_weight

        else: # unsupervised
            # Warp frameB toward frameA using the predicted flow.
            frameB_warped = self.warp(frameB, flow)

            # Photometric loss
            photo_loss = self.charbonnier(frameA - frameB_warped)

            # Smoothness penalty on the predicted flow field
            smooth_loss, laplacian_loss = self.smoothness_loss(flow, flow_gt)

            sup_loss = flow.new_zeros(())

            total = photo_loss + self.smooth_weight * smooth_loss + self.laplacian_weight * laplacian_loss
        
        return total, photo_loss, smooth_loss, laplacian_loss, sup_loss
