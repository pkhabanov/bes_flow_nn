# bes_flow/dataset.py
#
# Dataset class and DataLoader factory for BES optical flow training.
# 
#   BESDataset   — a PyTorch Dataset: defines __len__ and __getitem__.
#                  __getitem__(i) returns (frameA, frameB, flow_gt).
#
#   DataLoader   — wraps the Dataset and handles batching + shuffling.
#
# All synthetic flow generators below are built from a scalar stream
# function psi(x, y):
#
#       v_x = d(psi)/dy ,    v_y = -d(psi)/dx
#
# Any such field is divergence-free:
#
#       div v = d(v_x)/dx + d(v_y)/dy = psi_xy - psi_yx = 0
#
# Coordinate convention (consistent with the rest of the package):
# arrays are indexed [row, col] = [y, x], x to the right, y up,
# row 0 at the bottom.  np.gradient(psi, axis=0) = d(psi)/dy and
# np.gradient(psi, axis=1) = d(psi)/dx.


import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter, map_coordinates


def curl_from_stream(psi):
    """
    Velocity field of a scalar stream function via central differences.
 
    v = curl(psi) = (d(psi)/dy, -d(psi)/dx)
 
    Parameters
    ----------
    psi : (H, W) float array — stream function on the pixel grid
 
    Returns
    -------
    flow : (2, H, W) float32 — channel 0 = dx, channel 1 = dy
    """
    dpsi_dy = np.gradient(psi, axis=0)
    dpsi_dx = np.gradient(psi, axis=1)
    return np.stack([dpsi_dy, -dpsi_dx], axis=0).astype(np.float32)


def grad_from_potential(phi):
    """
    Velocity field of a scalar velocity potential via central differences.
 
    v = grad(phi) = (d(phi)/dx, d(phi)/dy)
 
    This is the curl-free (dilatational / compressive) counterpart of
    curl_from_stream. Its divergence is the Laplacian of phi, which is
    generically nonzero -- these are the motions that create 
    intensity sources and sinks.
 
    Parameters
    ----------
    phi : (H, W) float array — velocity potential on the pixel grid
 
    Returns
    -------
    flow : (2, H, W) float32 — channel 0 = dx, channel 1 = dy
    """
    dphi_dy = np.gradient(phi, axis=0)
    dphi_dx = np.gradient(phi, axis=1)
    return np.stack([dphi_dx, dphi_dy], axis=0).astype(np.float32)
 
 
def _unit_rms(flow):
    """Scale a (2, H, W) field so its RMS vector magnitude is 1."""
    rms = np.sqrt((flow**2).sum(axis=0).mean()) + 1e-8
    return flow / rms
 
 
def make_turbulence_compressible(flow_turb, compressible_fraction,
                                 smoothing_sigma=16.0):
    """
    Replace a solenoidal TURBULENT component with a mixed
    solenoidal + compressible one of the same RMS magnitude.
 
        v_turb' = |v_turb|_rms * ( sqrt(1-chi) * v_turb_hat
                                 + sqrt(chi)   * grad(phi)_hat )
 
    The mean poloidal ExB flow is divergence-free below the ion sound
    speed; it is the drift-wave turbulence riding on top of it that is
    compressible. Attaching chi to the turbulence makes it a single
    consistent physical knob across all four generators.
 
    Preserving the turbulent RMS also means the total flow's amplitude
    statistics are unchanged by chi, so no global renormalisation is
    needed - switching chi on does not perturb max_shift or the random
    zonal-amplitude draw.
 
    NOTE Drift-wave turbulence is predominantly solenoidal (ExB
    dominant) with only a weak compressible correction.
 
    Parameters
    ----------
    flow_turb             : (2, H, W) — solenoidal turbulent component
    compressible_fraction : chi in [0, 1] — dilatational fraction of the
                            TURBULENT energy
    smoothing_sigma       : coherence length of the potential field (px).
                            Should MATCH the generator that produced
                            flow_turb so the two components live 
                            at the same scale.
 
    Returns
    -------
    (2, H, W) float32 — same RMS magnitude as the input
    """
    chi = float(np.clip(compressible_fraction, 0.0, 1.0))
    if chi <= 0.0:
        return flow_turb.astype(np.float32)
 
    _, H, W = flow_turb.shape
    rms_in = np.sqrt((flow_turb**2).sum(axis=0).mean()) + 1e-8
 
    phi = gaussian_filter(
        np.random.randn(H, W).astype(np.float32), sigma=smoothing_sigma
    )
    flow_pot = grad_from_potential(phi)
 
    mixed = (np.sqrt(1.0 - chi) * _unit_rms(flow_turb)
             + np.sqrt(chi) * _unit_rms(flow_pot))
 
    # Restore the original turbulent RMS so total flow amplitude is
    # unaffected by chi.
    return (mixed * (rms_in / (np.sqrt((mixed**2).sum(axis=0).mean()) + 1e-8))
            ).astype(np.float32)
 
 
def normalize_flow(flow, max_shift, low=0.7):
    """
    Rescale a flow field so its PEAK vector magnitude is drawn uniformly
    from [low * max_shift, max_shift].
 
    Normalising the 2D magnitude (not dx/dy independently) preserves the
    direction distribution — and, because it is a global scalar multiple,
    it also preserves zero divergence.
    """
    magnitude = np.sqrt(flow[0]**2 + flow[1]**2)
    scale     = np.random.uniform(low * max_shift, max_shift) / (magnitude.max() + 1e-8)
    return (flow * scale).astype(np.float32)


def random_smooth_flow(H, W, max_shift=6.0, smoothing_sigma=16.0):
    """
    Generate a smooth random divergence-free displacement field from a
    Gaussian-random-field stream function.
    Pipeline: white noise -> Gaussian smoothing -> psi -> v = curl(psi)

    Parameters
    ----------
    H, W            : image height and width (both 64 for BES)
    max_shift       : after smoothing and normalisation, the peak
                        displacement magnitude in pixels
    smoothing_sigma : Gaussian smoothing length in pixels.
                      Larger values - more spatially coherent flow.
                      (~8-16 px)

    Returns
    -------
    flow : (2, H, W) float32 array — channel 0 = dx, channel 1 = dy
    """
    # Random stream function: smoothed white noise (Gaussian random field)
    psi_raw = np.random.randn(H, W).astype(np.float32)
    psi     = gaussian_filter(psi_raw, sigma=smoothing_sigma)
 
    # Divergence-free velocity from the stream function
    flow = curl_from_stream(psi)
 
    # Normalise the peak displacement to ~max_shift
    return normalize_flow(flow, max_shift)


def sinusoidal_modes(H, W, n_modes=8, max_shift=6.0):
    """
    Generate a divergence-free velocity field as a superposition of
    sinusoidal stream-function modes.
 
    Each mode is a plane wave of the stream function,
 
        psi_m = A_m * sin(kx_m * x + ky_m * y + phi_m)
 
    and the velocity is the analytic curl:
 
        dx += A_m * ky_m * cos(kx_m*x + ky_m*y + phi_m)
        dy -= A_m * kx_m * cos(kx_m*x + ky_m*y + phi_m)

    Parameters
    ----------
    H, W      : image dimensions
    n_modes   : number of superimposed sinusoidal components.
                More modes → richer spectral content
    max_shift : peak displacement magnitude in pixels

    Returns
    -------
    flow : (2, H, W) float32 array
    """
    y_coords = np.linspace(0, 2 * np.pi, H, dtype=np.float32)
    x_coords = np.linspace(0, 2 * np.pi, W, dtype=np.float32)
    xx, yy   = np.meshgrid(x_coords, y_coords)
 
    dx = np.zeros((H, W), dtype=np.float32)
    dy = np.zeros((H, W), dtype=np.float32)
 
    for _ in range(n_modes):
        # Random wavenumber: integers 1–4 give structures that span
        # 1/4 to the full image — representative of BES turbulence scales
        kx = 0.4 * np.random.randint(1, 5)
        ky = 0.4 * np.random.randint(1, 5)
 
        # Random amplitude and phase per mode (shared by dx and dy —
        # both components derive from the same stream-function mode)
        amp   = np.random.randn()
        phase = np.random.uniform(0, 2 * np.pi)
 
        cos_mode = np.cos(kx * xx + ky * yy + phase)
        dx += amp * ky * cos_mode    #  d(psi)/dy
        dy -= amp * kx * cos_mode    # -d(psi)/dx
 
    # Normalise to max_shift (global scale preserves zero divergence)
    return normalize_flow(np.stack([dx, dy], axis=0), max_shift)


def zonal_plus_turbulence_flow(H, W,
                               zonal_amplitude=4.0,
                               turbulence_amplitude=2.0,
                               turbulence_sigma=16.0,
                               profile_type='well',
                               well_pos=0.5,
                               well_width=0.125,):
    """
    Generate a velocity field composed of:
        1. A smooth zonal flow component with a slow variation across the image
        2. A turbulent component: small-scale, isotropic Gaussian random
           field superimposed on the zonal flow.

    Parameters
    ----------
    H, W                   : image dimensions
    zonal_amplitude        : peak zonal flow displacement in pixels
    turbulence_amplitude   : peak turbulent displacement in pixels
    turbulence_sigma       : spatial smoothing of turbulent component (pixels)
    profile_type           : type of the flow profile in the x-direction
                             'well' is a gaussian profile resembling Er well,
                             'sin' is a sinusoidal profile
    well_pos               : well_pos * W is the position of the peak flow along the x axis (for 'well')
    well_width             : well_width * W is the width of the shear layer in pixels (for 'well')

    Returns
    -------
    flow       : (2, H, W) — total flow field (zonal + turbulent)
    flow_zonal : (2, H, W) — zonal component only
    flow_turb  : (2, H, W) — turbulent component only
    """

    # Zonal flow (divergent-free by definition)
    if profile_type == 'sin':
        # Smooth sinusoidal variation of y-velocity across the x (radial) axis.
        # add small random phase shift
        x_coords = np.linspace(0, 2 * np.pi, W, dtype=np.float32) + np.pi * np.random.uniform(-0.5, 0.5)
        zonal_profile = np.sin(x_coords)  # (W,) — radial profile
    elif profile_type == 'well':
        # Gaussian profile (Er well approximation)
        x_coords = np.arange(0, W)
        #well_pos = well_pos + 0.25 * np.random.randn()
        #well_width = well_width + 0.25 * np.random.randn()
        well_pos = np.random.uniform(0.2, 0.8)
        well_width = np.random.uniform(well_width, 0.5)
        zonal_profile = np.exp(-((x_coords - well_pos*W)**2) / (2 * (well_width*W)**2))
    else:
        raise ValueError(
            f"Unknown profile_type '{profile_type}'. Choose 'sin' or 'well'."
        )
    
    # y-component varies with x
    zonal_dy = np.random.choice([1, -1]) * np.random.uniform(0.7*zonal_amplitude, zonal_amplitude) * zonal_profile 
    # no radial zonal component
    zonal_dx = np.zeros(W, dtype=np.float32)       

    # Broadcast to full (H, W) arrays — zonal flow is uniform in x
    flow_zonal = np.stack([
        np.tile(zonal_dx[None, :], (H, 1)),
        np.tile(zonal_dy[None, :], (H, 1))
    ], axis=0)

    # Turbulent component — divergence-free via a random stream function.
    psi_turb = gaussian_filter(
        np.random.randn(H, W).astype(np.float32), sigma=turbulence_sigma
    )
    turb = curl_from_stream(psi_turb)
 
    # Normalise turbulent component to turbulence_amplitude
    mag   = np.sqrt((turb**2).sum(axis=0)).max() + 1e-8
    flow_turb = turb * (turbulence_amplitude / mag)
 
    # Total flow
    flow = flow_zonal + flow_turb

    return flow.astype(np.float32), flow_zonal.astype(np.float32), flow_turb.astype(np.float32)


# Image warping
#   
def warp_image(image, flow):
    """
    Warp a 2D image by a displacement field using bilinear interpolation.

    Parameters
    ----------
    image : (H, W) float32 array — the BES frame to warp
    flow  : (2, H, W) float32 array — displacement field

    Returns
    -------
    warped : (H, W) float32 array
    """
    H, W = image.shape

    # Build pixel coordinate grids
    y_coords, x_coords = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing='ij'
    )

    # Displaced source coordinates: where to sample in the original image.
    # map_coordinates uses (row, col) = (y, x) ordering.
    src_y = y_coords - flow[1]   # flow[1] = dy
    src_x = x_coords - flow[0]   # flow[0] = dx

    # Bilinear interpolation (order=1)
    warped = map_coordinates(
        image,
        [src_y.ravel(), src_x.ravel()],
        order=1,
        mode='nearest',
    ).reshape(H, W)

    return warped.astype(np.float32)


def _sample_velocity(velocity, y, x):
    """
    Bilinearly interpolate a (2, H, W) velocity field at fractional
    coordinates (y, x).  Edge values are extended outside the domain.
 
    Returns
    -------
    vx, vy : arrays of the same shape as y / x
    """
    coords = [y.ravel(), x.ravel()]
    vx = map_coordinates(velocity[0], coords, order=1,
                         mode='nearest').reshape(y.shape)
    vy = map_coordinates(velocity[1], coords, order=1,
                         mode='nearest').reshape(y.shape)
    return vx, vy
 
 
def advect_image(image, velocity, n_steps=4):
    """
    Warp a 2D image by integrating a steady velocity field over unit time
    with a backward semi-Lagrangian scheme (RK2 midpoint per sub-step).
 
    Parameters
    ----------
    image    : (H, W) float32 array — the BES frame to warp
    velocity : (2, H, W) float32 array — steady velocity field in
               pixels per unit time (channel 0 = vx, channel 1 = vy)
    n_steps  : number of RK2 sub-steps along each characteristic
 
    Returns
    -------
    warped : (H, W) float32 array
    """
    H, W = image.shape
    dt   = 1.0 / n_steps
 
    y, x = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing='ij'
    )
 
    # Trace each output pixel backward to its source point in frame A
    for _ in range(n_steps):
        # Midpoint (RK2): evaluate v at the half-step position
        vx1, vy1 = _sample_velocity(velocity, y, x)
        x_mid    = x - 0.5 * dt * vx1
        y_mid    = y - 0.5 * dt * vy1
        vxm, vym = _sample_velocity(velocity, y_mid, x_mid)
        x        = x - dt * vxm
        y        = y - dt * vym
 
    # Single bilinear interpolation of the image at the final foot points
    warped = map_coordinates(
        image, [y.ravel(), x.ravel()], order=1, mode='nearest',
    ).reshape(H, W)
 
    return warped.astype(np.float32)
 
 
def integrate_forward_displacement(velocity, n_steps=4):
    """
    Integrate the steady velocity field FORWARD from every grid point to
    obtain the total displacement over unit time — the ground-truth flow
    matching advect_image() in the loss convention
    frameA(x) ≈ frameB(x + D(x)).
 
    Parameters
    ----------
    velocity : (2, H, W) float32 — steady velocity field (px / unit time)
    n_steps  : number of RK2 sub-steps (use the same value as in
               advect_image so frame pair and ground truth stay consistent)
 
    Returns
    -------
    flow : (2, H, W) float32 — total displacement (dx, dy) in pixels.
           Peak magnitude can differ slightly (typically < 10%) from the
           peak of `velocity` because curved characteristics integrate
           a spatially varying field.
    """
    _, H, W = velocity.shape
    dt      = 1.0 / n_steps
 
    y0, x0 = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing='ij'
    )
    x, y = x0.copy(), y0.copy()
 
    for _ in range(n_steps):
        vx1, vy1 = _sample_velocity(velocity, y, x)
        x_mid    = x + 0.5 * dt * vx1
        y_mid    = y + 0.5 * dt * vy1
        vxm, vym = _sample_velocity(velocity, y_mid, x_mid)
        x        = x + dt * vxm
        y        = y + dt * vym
 
    return np.stack([x - x0, y - y0], axis=0).astype(np.float32)


def _divergence_field(velocity):
    """
    div(v) = dvx/dx + dvy/dy on the pixel grid, via np.gradient
    (central differences, one-sided at the edges).
 
    Returns
    -------
    (H, W) float32
    """
    dvx_dx = np.gradient(velocity[0], axis=1)
    dvy_dy = np.gradient(velocity[1], axis=0)
    return (dvx_dx + dvy_dy).astype(np.float32)
 
 
def _sample_scalar(field, y, x):
    """Bilinearly sample a (H, W) scalar field at fractional (y, x)."""
    return map_coordinates(field, [y.ravel(), x.ravel()],
                           order=1, mode='nearest').reshape(y.shape)
 
 
def advect_image_continuity(image, velocity, n_steps=4, max_log_gain=1.0):
    """
    Transport an image by the CONTINUITY equation
 
        dI/dt + div(I * v) = 0
 
    rather than by passive advection. This is the compressible
    counterpart of advect_image().
 
    Method
    ------
    Along a characteristic the continuity equation reduces to an ODE for
    the intensity itself:
 
        DI/Dt = -I * div(v)   =>   d(ln I)/dt = -div(v)
 
    so, integrating from the foot point to the destination,
 
        I_B(x) = I_A(x_foot) * exp( -Integral[ div(v) dt ] )
 
    We therefore run the SAME backward semi-Lagrangian trace as
    advect_image(), and additionally accumulate the divergence integral
    along each trajectory using the RK2 midpoint positions (consistent
    with the midpoint rule already used for the position update). One
    cubic interpolation of the image at the final foot point, then a
    pointwise multiplication by exp(-J).
 
    Two useful properties:
      * div(v) == 0  =>  J == 0  =>  identical to advect_image().
      * exp(-J) > 0 always, so intensities stay strictly positive.
 
    By Liouville's formula exp(J) is precisely det(grad Phi) along the
    trajectory, so the frames produced here satisfy
 
        I_B(x + D) * det(I + grad D) = I_A(x)
 
    which is the residual minimised by
    WarpingL2Loss.continuity_loss(form='lagrangian').
 
    Parameters
    ----------
    image        : (H, W) float32 — frame A
    velocity     : (2, H, W) float32 — steady velocity (px / unit time)
    n_steps      : RK2 sub-steps (use the same value as the GT integrator)
    max_log_gain : clamp on |J|. exp(1.0) ~ 2.7x brightening is already
                   far beyond anything physical for a single BES frame
                   interval; the clamp stops a pathological random draw
                   from producing absurd intensities. Set None to disable.
 
    Returns
    -------
    warped : (H, W) float32
    """
    H, W = image.shape
    dt = 1.0 / n_steps
 
    div_v = _divergence_field(velocity)
 
    y, x = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing='ij'
    )
 
    # Accumulated Integral[ div(v) dt ] along each backward trajectory
    log_jac = np.zeros((H, W), dtype=np.float32)
 
    for _ in range(n_steps):
        vx1, vy1 = _sample_velocity(velocity, y, x)
        x_mid = x - 0.5 * dt * vx1
        y_mid = y - 0.5 * dt * vy1
        vxm, vym = _sample_velocity(velocity, y_mid, x_mid)
 
        # Midpoint-rule contribution of this sub-step to the integral
        log_jac += dt * _sample_scalar(div_v, y_mid, x_mid)
 
        x = x - dt * vxm
        y = y - dt * vym
 
    if max_log_gain is not None:
        log_jac = np.clip(log_jac, -max_log_gain, max_log_gain)
 
    warped = map_coordinates(
        image, [y.ravel(), x.ravel()], order=1, mode='nearest',
    ).reshape(H, W)
 
    return (warped * np.exp(-log_jac)).astype(np.float32)
 
 
def compression_diagnostics(H, W, flow_type, max_shift,
                            compressible_fraction, n_warp_steps=4,
                            n_samples=200):
    """
    Report the intensity-change statistics implied by a given
    compressible_fraction, so it can be calibrated against the measured
    dI/I of real BES data.
 
    The quantity of interest is the compression gain g = exp(-J):
    g = 1 means no compression, g = 1.1 means a 10% brightening of that
    fluid element over one frame interval.
 
    Returns
    -------
    dict with RMS and percentile statistics of (g - 1), in percent.
    """
    rel = []
    for _ in range(n_samples):
        v = _generate_flow(H, W, flow_type, max_shift,
                           compressible_fraction=compressible_fraction)
        div_v = _divergence_field(v)
        # single-shot estimate of J ~ div(v) * unit time
        rel.append(np.abs(np.expm1(-div_v)))
    rel = np.concatenate([r.ravel() for r in rel])
    return {
        'rms_percent': float(100 * np.sqrt((rel**2).mean())),
        'p50_percent': float(100 * np.percentile(rel, 50)),
        'p99_percent': float(100 * np.percentile(rel, 99)),
        'max_percent': float(100 * rel.max()),
    }

 
def _generate_flow(H, W, flow_type, max_shift, compressible_fraction=0.0):
    """
    Dispatch to the selected flow generator.
 
    compressible_fraction (chi) makes the TURBULENT component partially
    compressible, carrying chi of the turbulent kinetic energy. The mean
    (zonal) component is left divergence-free.
    """
 
    if flow_type == 'smooth':
        # No mean component: the whole field is turbulence.
        flow_mean = 0.0
        flow_turb = random_smooth_flow(H, W, max_shift)
    elif flow_type == 'modes':
        flow_mean = 0.0
        flow_turb = sinusoidal_modes(H, W, max_shift=max_shift)
    elif flow_type in ('zonal', 'well'):
        _, flow_mean, flow_turb = zonal_plus_turbulence_flow(
            H, W,
            zonal_amplitude      = max_shift,
            turbulence_amplitude = max_shift * 0.3,
            profile_type         = 'sin' if flow_type == 'zonal' else 'well',
        )
    else:
        raise ValueError(
            f"Unknown flow_type '{flow_type}'. "
            f"Choose from: 'smooth', 'modes', 'zonal', 'well'."
        )
 
    if compressible_fraction > 0.0:
        flow_turb = make_turbulence_compressible(
            flow_turb, compressible_fraction,
        )
 
    return (flow_mean + flow_turb).astype(np.float32)
    

def generate_dataset(frames, n_pairs_per_frame, max_shift,
                    noise_std, flow_type, n_warp_steps,
                    compressible_fraction=0.0):
    """
    Generate the full synthetic dataset once and return numpy arrays.
 
    Each real frame is used to produce n_pairs_per_frame synthetic pairs
    with independent random flow fields, giving a total of
    N * n_pairs_per_frame training examples.
 
    Parameters
    ----------
    frames            : (N, H, W) float array — real BES frames
    n_pairs_per_frame : int   — synthetic pairs per real frame
    max_shift  : float — peak displacement in pixels
    noise_std         : float — std of additive Gaussian noise
    flow_type         : str   — 'smooth', 'modes', 'zonal', or 'flow'
    n_warp_steps      : int   — number of semi-Lagrangian RK2 steps
                        used to advect frame A into frame B.
                        1  : single-step warp 
                        >1 : the generated field is treated as a steady
                             velocity field; frame B is produced by
                             multi-step advection and flow_gt is the
                             consistently integrated forward displacement.
 
    Returns
    -------
    framesA  : (N*n_pairs, 1, H, W) float32
    framesB  : (N*n_pairs, 1, H, W) float32
    flows_gt : (N*n_pairs, 2, H, W) float32
    """
    N, H, W  = frames.shape
    n_total  = N * n_pairs_per_frame
 
    framesA  = np.zeros((n_total, 1, H, W), dtype=np.float32)
    framesB  = np.zeros((n_total, 1, H, W), dtype=np.float32)
    flows_gt = np.zeros((n_total, 2, H, W), dtype=np.float32)
 
    print(f"  Generating {n_total} pairs "
          f"({N} frames x {n_pairs_per_frame} pairs, "
          f"flow='{flow_type}', warp_steps={n_warp_steps})...")
 
    idx = 0
    for i, frame in enumerate(frames):
        image = frame.astype(np.float32)
        # normalize each image
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
 
        for _ in range(n_pairs_per_frame):
            velocity = _generate_flow(H, W, flow_type, max_shift,
                                      compressible_fraction=compressible_fraction)
            if n_warp_steps <= 1:
                # Legacy single-step warp.
                # NOTE: this path is only self-consistent for a spatially
                # UNIFORM flow -- it sets flow = velocity and warps once,
                # so frameA(x) = frameB(x + D(x)) holds only where
                # v(x + v) == v(x). Prefer n_warp_steps >= 2.
                flow   = velocity
                warped = warp_image(image, velocity)
            elif compressible_fraction > 0.0:
                # Compressible: transport intensity by the continuity
                # equation so the pair actually contains sources/sinks.
                flow   = integrate_forward_displacement(velocity, n_warp_steps)
                warped = advect_image_continuity(image, velocity, n_warp_steps)
            else:
                # Incompressible: passive advection
                flow   = integrate_forward_displacement(velocity, n_warp_steps)
                warped = advect_image(image, velocity, n_warp_steps)
 
            if noise_std > 0:
                # With compressible transport the warped frame legitimately
                # exceeds the [0, 1] range of frame A (that IS the signal),
                # so only non-negativity is enforced on it. Clipping to 1.0
                # would silently destroy the compression the
                # continuity loss is meant to learn from.
                hi_B = 1.0 if compressible_fraction <= 0.0 else None
                framesA[idx, 0] = (image  + np.random.normal(0, noise_std, (H, W))
                                   ).clip(0.0, 1.0)
                framesB[idx, 0] = (warped + np.random.normal(0, noise_std, (H, W))
                                   ).clip(0.0, hi_B)
            else:
                framesA[idx, 0] = image
                framesB[idx, 0] = warped
 
            flows_gt[idx, :, :, :] = flow
            idx += 1
 
        if (i + 1) % max(1, N // 5) == 0:
            print(f"    {i+1}/{N} frames created  ({idx} pairs)")
 
    mem_mb = (framesA.nbytes + framesB.nbytes + flows_gt.nbytes) / 1e6
    print(f"  Done — {n_total} pairs, ~{mem_mb:.1f} MB in memory")
    return framesA, framesB, flows_gt
 
 
def _make_metadata(cfg):
    """
    Build a dict of the settings that determine dataset content.
    Stored as HDF5 attributes on the /metadata group and compared on
    load to detect stale caches.
    """
    return {
        'flow_type'        : cfg.flow_type,
        'max_shift'        : float(cfg.max_shift),
        'noise_std'        : float(cfg.noise_std),
        'n_pairs_per_frame': int(cfg.n_pairs_per_frame),
        'val_split'        : float(cfg.val_split),
        'test_split'       : float(cfg.test_split),
        'val_seed'         : int(cfg.val_seed),
        'test_seed'        : int(cfg.test_seed),
        'n_warp_steps'     : int(cfg.n_warp_steps),
        # Included so that switching the compressible arm on/off
        # invalidates any cached incompressible dataset.
        'compressible_fraction': float(getattr(cfg, 'compressible_fraction', 0.0)),
    }
 
 
def save_dataset_cache(path, 
                       train_A, train_B, train_flows,
                       val_A, val_B, val_flows, 
                       test_A, test_B, test_flows,
                       metadata):
    """
    Save pre-generated arrays and metadata to an HDF5 file.
 
    Parameters
    ----------
    path            : str  -- file path, e.g. 'data/cache/dataset_zonal.h5'
    train_A/B/flows : (M, 1, H, W) / (M, 2, H, W) training arrays
    val_A/B/flows   : (V, 1, H, W) / (V, 2, H, W) validation arrays
    test_A/B/flows  : (T, 1, H, W) / (T, 2, H, W) test arrays
    metadata        : dict -- generation settings from _make_metadata()
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
 
    with h5py.File(path, 'w') as f:
        for grp_name, A, B, flows in (
            ('train', train_A, train_B, train_flows),
            ('val',   val_A,   val_B,   val_flows),
            ('test',  test_A,  test_B,  test_flows),
        ):
            grp = f.create_group(grp_name)
            grp.create_dataset('framesA',  data=A,     compression='gzip', compression_opts=4)
            grp.create_dataset('framesB',  data=B,     compression='gzip', compression_opts=4)
            grp.create_dataset('flows_gt', data=flows, compression='gzip', compression_opts=4)
 
        # Metadata as typed HDF5 attributes on a dedicated group
        meta_grp = f.create_group('metadata')
        for key, value in metadata.items():
            meta_grp.attrs[key] = value
 
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Cache saved -> {path}  ({size_mb:.1f} MB on disk)")
 
 
def load_dataset_cache(path):
    """
    Load pre-generated arrays from an HDF5 cache file.
 
    Parameters
    ----------
    path : str -- path to .h5 cache file
 
    Returns
    -------
    train_A, train_B, train_flows : training arrays
    val_A, val_B, val_flows       : validation arrays
    test_A,  test_B,  test_flows  : test arrays
    metadata                      : dict of generation settings
    """
    with h5py.File(path, 'r') as f:
        train_A     = f['train/framesA'][:]
        train_B     = f['train/framesB'][:]
        train_flows = f['train/flows_gt'][:]
        val_A       = f['val/framesA'][:]
        val_B       = f['val/framesB'][:]
        val_flows   = f['val/flows_gt'][:]
        test_A      = f['test/framesA'][:]
        test_B      = f['test/framesB'][:]
        test_flows  = f['test/flows_gt'][:]
 
        # Read attributes back into a Python dict.
        metadata = {}
        for key, value in f['metadata'].attrs.items():
            if hasattr(value, 'item'):
                metadata[key] = value.item()   # numpy scalar -> Python int/float
            else:
                metadata[key] = value          # strings pass through unchanged
 
    return (train_A, train_B, train_flows,
            val_A,   val_B,   val_flows,
            test_A,  test_B,  test_flows,
            metadata)
 
 
def _cache_is_valid(path, cfg):
    """
    Check whether a cache file exists AND was generated with the same
    settings as the current cfg.
    Reads only the /metadata attributes.
 
    Returns
    -------
    (is_valid : bool, reason : str)
        reason is logged by make_dataloaders so the user always knows
        why a cache was rejected or accepted.
    """
    if not os.path.exists(path):
        return False, "Cache file not found"
    try:
        with h5py.File(path, 'r') as f:
            # Verify all three split groups are present
            for grp in ('train', 'val', 'test'):
                if grp not in f:
                    return False, f"Cache is missing the '{grp}' group"
            # read only metadata from hdf5 cache
            metadata = {
                k: (v.item() if hasattr(v, 'item') else v)
                for k, v in f['metadata'].attrs.items()
            }
    except Exception as e:
        return False, f"Cache file unreadable: {e}"
 
    current = _make_metadata(cfg)
 
    # Compare each field individually
    for key, current_val in current.items():
        cached_val = metadata.get(key)
        if cached_val != current_val:
            return False, (
                f"'{key}' mismatch: cached={cached_val!r}, "
                f"current={current_val!r}"
            )
 
    return True, "ok"


class BESDataset(Dataset):
    """
    Fast dataset that indexes into pre-generated numpy arrays.
    __getitem__ is a trivial array lookup followed by optional augmentation.

    Parameters
    ----------
    framesA, framesB : (M, 1, H, W) float32 — pre-generated frame pairs
    flows_gt         : (M, 2, H, W) float32 — ground-truth flow fields
    augment          : bool — if True, apply random augmentation in __getitem__.
                       Enable for the training set; disable for val/test
    """
    def __init__(self, framesA, framesB, flows_gt, augment=False):
        assert len(framesA) == len(framesB) == len(flows_gt), (
            "framesA, framesB and flows_gt must all have the same length "
            f"(got {len(framesA)}, {len(framesB)}, {len(flows_gt)})"
        )
        self.framesA  = framesA
        self.framesB  = framesB
        self.flows_gt = flows_gt
        self.augment  = augment

    def __len__(self):
        return len(self.framesA)

    def __getitem__(self, idx):
        # All generation work was done in generate_dataset().
        fA   = torch.tensor(self.framesA[idx])   # (1, H, W)
        fB   = torch.tensor(self.framesB[idx])   # (1, H, W)
        flow = torch.tensor(self.flows_gt[idx])  # (2, H, W)

        if self.augment:
            fA, fB, flow = self._augment(fA, fB, flow)

        return fA, fB, flow
    
    @staticmethod
    def _augment(fA, fB, flow):
        """
        Random data augmentations.

        1. Y-axis flip (p=0.5)
        2. X-axis flip
        2. Random 90° rotation (k ∈ {0,1,2,3})
           Image and flow grid are rotated k*90° CCW.
           Flow vectors are rotated by applying (dx,dy)->(dy,-dx) k times
           (x and y increase from element (0,0)).
        3. Intensity jitter — same gain U[0.95,1.05] for both frames
        """
        # 1. Y-axis flip
        if torch.rand(1).item() < 0.5:
            fA   = torch.flip(fA,   dims=[1])
            fB   = torch.flip(fB,   dims=[1])
            flow = torch.flip(flow, dims=[1])
            flow = torch.stack([flow[0], -flow[1]], dim=0)

        # 2. X-axis (horizontal) flip 
        if torch.rand(1).item() < 0.5:
            fA   = torch.flip(fA,   dims=[2])
            fB   = torch.flip(fB,   dims=[2])
            flow = torch.flip(flow, dims=[2])
            flow = torch.stack([-flow[0], flow[1]], dim=0)   

        # 3. Random 90 deg rotation
        k = torch.randint(4, (1,)).item()
        if k > 0:
            fA   = torch.rot90(fA,   k, dims=[1, 2])
            fB   = torch.rot90(fB,   k, dims=[1, 2])
            flow = torch.rot90(flow, k, dims=[1, 2])
            for _ in range(k):
                flow = torch.stack([flow[1], -flow[0]], dim=0)

        # 4. Intensity jitter
        gain = 0.85 + 0.30 * torch.rand(1).item()
        fA   = (fA * gain).clamp(0.0, 1.0)
        fB   = (fB * gain).clamp(0.0, 1.0)

        return fA, fB, flow


def make_datasets(train_frames, val_frames, test_frames, cfg):
    """
    Generate or load all three BESDataset objects (train / val / test).

    Cache behaviour
    ───────────────
    If cfg.dataset_cache_path is set and a valid cache exists on disk,
    all three splits are loaded from it.  Otherwise they are
    generated from scratch and saved to the cache path. 
    The cache is invalidated automatically when any metadata field changes.

    Parameters
    ----------
    train_frames : (N_train, H, W) float array — raw BES training frames
    val_frames   : (N_val,   H, W) float array — raw BES validation frames
    test_frames  : (N_test,  H, W) float array — raw BES test frames
    cfg          : Config

    Returns
    -------
    train_dataset : BESDataset  (augmentation ON)
    val_dataset   : BESDataset  (augmentation OFF)
    test_dataset  : BESDataset  (augmentation OFF)
    """
    cache_path = getattr(cfg, 'dataset_cache_path', None)

    if cache_path is not None:
        valid, reason = _cache_is_valid(cache_path, cfg)

        if valid:
            print(f"\nLoading dataset from cache: {cache_path}")
            (train_A, train_B, train_flows,
             val_A,   val_B,   val_flows,
             test_A,  test_B,  test_flows,
             metadata) = load_dataset_cache(cache_path)
            print(f"  Train pairs : {len(train_A)}")
            print(f"  Val pairs   : {len(val_A)}")
            print(f"  Test pairs  : {len(test_A)}")
            print(f"  Flow type   : {metadata['flow_type']}")
            print(f"  Max shift   : {metadata['max_shift']} px")

        else:
            print(f"\n{reason} — regenerating dataset...")
            (train_A, train_B, train_flows,
             val_A,   val_B,   val_flows,
             test_A,  test_B,  test_flows) = _generate_all(
                 train_frames, val_frames, test_frames, cfg
             )
            print(f"Saving dataset cache: {cache_path}")
            save_dataset_cache(
                cache_path,
                train_A, train_B, train_flows,
                val_A,   val_B,   val_flows,
                test_A,  test_B,  test_flows,
                metadata=_make_metadata(cfg),
            )

    else:
        print("\nNo cache path configured — generating dataset (not saved)...")
        (train_A, train_B, train_flows,
         val_A,   val_B,   val_flows,
         test_A,  test_B,  test_flows) = _generate_all(
             train_frames, val_frames, test_frames, cfg
         )

    train_dataset = BESDataset(train_A, train_B, train_flows, augment=True)
    val_dataset   = BESDataset(val_A,   val_B,   val_flows,   augment=False)
    test_dataset  = BESDataset(test_A,  test_B,  test_flows,  augment=False)

    del train_A, train_B, train_flows
    del val_A,   val_B,   val_flows
    del test_A,  test_B,  test_flows

    print(f"\nDataset summary:")
    print(f"  Train : {len(train_dataset)} pairs  (augmentation ON)")
    print(f"  Val   : {len(val_dataset)} pairs  (augmentation OFF)")
    print(f"  Test  : {len(test_dataset)} pairs  (augmentation OFF)\n")

    return train_dataset, val_dataset, test_dataset


def make_dataloaders(train_dataset, val_dataset, test_dataset, cfg):
    """
    Wrap three BESDataset objects in DataLoaders.

    Parameters
    ----------
    train_dataset : BESDataset — augmented training set
    val_dataset   : BESDataset — fixed validation set
    test_dataset  : BESDataset — fixed held-out test set
    cfg           : Config

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    pin_mem = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = cfg.num_workers,
        pin_memory  = pin_mem,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = cfg.num_workers,
        pin_memory  = pin_mem,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = cfg.num_workers,
        pin_memory  = pin_mem,
    )

    print(f"DataLoader summary:")
    print(f"  Batch size          : {cfg.batch_size}")
    print(f"  Train batches/epoch : {len(train_loader)}")
    print(f"  Val   batches       : {len(val_loader)}")
    print(f"  Test  batches       : {len(test_loader)}\n")

    return train_loader, val_loader, test_loader


def _generate_all(train_frames, val_frames, test_frames, cfg):
    """
    Generate all three splits from scratch.

    Parameters
    ----------
    train_frames : (N_train, H, W)
    val_frames   : (N_val,   H, W)
    test_frames  : (N_test,  H, W)
    cfg          : Config

    Returns
    -------
    train_A, train_B, train_flows,
    val_A,   val_B,   val_flows,
    test_A,  test_B,  test_flows
    """
    H, W = train_frames.shape[1], train_frames.shape[2]
    
    # Training set - random seed
    print("Generating training set:")
    train_A, train_B, train_flows = generate_dataset(
        train_frames,
        n_pairs_per_frame = cfg.n_pairs_per_frame,
        max_shift         = cfg.max_shift,
        noise_std         = cfg.noise_std,
        flow_type         = cfg.flow_type,
        n_warp_steps      = cfg.n_warp_steps,
        compressible_fraction = getattr(cfg, 'compressible_fraction', 0.0),
    )

    # Validation set - fixed val_seed 
    print("Generating validation set (fixed seed for reproducibility):")
    rng_state = np.random.get_state()
    np.random.seed(cfg.val_seed)

    val_A, val_B, val_flows = generate_dataset(
        val_frames,
        n_pairs_per_frame = cfg.n_pairs_per_frame,
        max_shift         = cfg.max_shift,
        noise_std         = cfg.noise_std,
        flow_type         = cfg.flow_type,
        n_warp_steps      = cfg.n_warp_steps,
        compressible_fraction = getattr(cfg, 'compressible_fraction', 0.0),
    )

    np.random.set_state(rng_state)  # restore random state

    # Test set - fixed test_seed
    if len(test_frames) == 0:
        print("  Skipping test set (empty test_frames).")
        empty_f = np.empty((0, 1, H, W), dtype=np.float32)
        empty_v = np.empty((0, 2, H, W), dtype=np.float32)
        test_A = test_B = empty_f
        test_flows = empty_v
    else:
        print("Generating test set (fixed seed, independent of training):")
        rng_state  = np.random.get_state()
        np.random.seed(cfg.test_seed)
        test_A, test_B, test_flows = generate_dataset(
            test_frames,
            n_pairs_per_frame = cfg.n_pairs_per_frame,
            max_shift         = cfg.max_shift,
            noise_std         = cfg.noise_std,
            flow_type         = cfg.flow_type,
            n_warp_steps      = cfg.n_warp_steps,
            compressible_fraction = getattr(cfg, 'compressible_fraction', 0.0),
        )
        np.random.set_state(rng_state)

    return (train_A, train_B, train_flows,
            val_A,   val_B,   val_flows,
            test_A,  test_B,  test_flows)


if __name__ == "__main__":
    # test image warping
    import matplotlib.pyplot as plt
    from dataclasses import dataclass

    # build a minimal config
    @dataclass
    class TestConfig:
        val_split          : float = 0.1
        test_split         : float = 0.1
        max_shift          : float = 12.0
        noise_std          : float = 0.0
        flow_type          : str   = 'modes'
        batch_size         : int   = 4
        num_workers        : int   = 0
        n_pairs_per_frame  : int   = 1
        val_seed           : int   = 0
        test_seed          : int   = 42
        n_warp_steps       : int   = 4
        compressible_fraction: float = 0.02
        # Set to None to skip saving
        dataset_cache_path : str   = 'synthetic_data/test_dataset.h5'

    cfg = TestConfig()

    # load bes frames
    fname = "raw_data/194313_t=2600-2620_f=30-200_2000fr.h5"
    print('\nLoading images ' + fname)
    with h5py.File(fname, 'r') as hf:
        all_frames = hf['images'][:]
        ti = hf['time'][:] #ms
        R = hf['R'][:]
        Z = hf['Z'][:]
    
    # split frames
    N        = len(all_frames)
    n_test   = int(cfg.test_split * N)
    n_val    = int(cfg.val_split  * N)
    n_train  = N - n_test - n_val
    print(f"\nSplit: train={n_train}  val={n_val}  test={n_test}")

    train_frames = all_frames[:n_train]
    val_frames   = all_frames[n_train : n_train + n_val]
    test_frames  = all_frames[n_train + n_val:]

    print("\n=== First call (should generate and save) ===")
    train_ds, val_ds, test_ds = make_datasets(train_frames, val_frames, test_frames, cfg)

    print("\n=== Second call (should load from cache) ===")
    train_ds, val_ds, test_ds = make_datasets(train_frames, val_frames, test_frames, cfg)

    train_loader, val_loader, test_loader = make_dataloaders(
        train_ds, val_ds, test_ds, cfg
    )
   
    # pull one batch and take the first sample
    frameA_batch, frameB_batch, flow_batch = next(iter(train_loader))
    # Remove batch and channel dimensions 
    frameA  = frameA_batch[0, 0].numpy()   # (H, W)
    frameB  = frameB_batch[0, 0].numpy()   # (H, W)
    flow_gt = flow_batch[0].numpy()        # (2, H, W)

    print(f"\nSample shapes:")
    print(f"  frameA  : {frameA.shape}")
    print(f"  frameB  : {frameB.shape}")
    print(f"  flow_gt : {flow_gt.shape}")
    print(f"\nFlow statistics:")
    print(f"  dx  min/max: {flow_gt[0].min():.2f} / {flow_gt[0].max():.2f} px")
    print(f"  dy  min/max: {flow_gt[1].min():.2f} / {flow_gt[1].max():.2f} px")
    mag = np.sqrt(flow_gt[0]**2 + flow_gt[1]**2)
    print(f"  magnitude mean/max: {mag.mean():.2f} / {mag.max():.2f} px")

    # downsaple the quiver grid
    step    = 4
    H, W    = frameA.shape
    ys      = np.arange(step // 2, H, step)   # y centres of quiver grid cells
    xs      = np.arange(step // 2, W, step)   # x centres of quiver grid cells
    xx, yy  = np.meshgrid(xs, ys)

    # Sample the flow at the quiver grid points
    dx = flow_gt[0][yy, xx]   # dx component at each grid point
    dy = flow_gt[1][yy, xx]   # dy component at each grid point

    # plot initial and warped frames
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Synthetic BES training pair  |  flow type: {cfg.flow_type}  |  "
        f"max displacement: {cfg.max_shift} px",
        fontsize=16, fontweight='bold'
    )
    # Common colour scale — both frames share the same vmin/vmax 
    vmin = min(frameA.min(), frameB.min())
    vmax = max(frameA.max(), frameB.max())

    for ax, frame, title in zip(
        axes, [frameA, frameB], ['Frame A ', 'Frame B (warped)']
    ):
        im = ax.imshow(frame, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)
        ax.quiver(xx, yy, dx, dy, color='cyan', scale=100,
                  scale_units='width', width=0.004)
        ax.set_title(title)
        ax.set_xlabel('x (px)')
        ax.set_ylabel('y (px)')

    # Shared colourbar on the right
    clb = fig.colorbar(im, ax=axes[1], label='Normalised BES intensity')

    plt.tight_layout()
    plt.show()
    #plt.savefig('outputs/dataset_test.png', dpi=150, bbox_inches='tight')
    #print("\nPlot saved to outputs/dataset_test.png")