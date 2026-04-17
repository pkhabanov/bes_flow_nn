'''
Evaluate the velocimetry model on synthetic images from Gkeyll gyrokinetic code
'''

import torch
import h5py
import numpy as np
from bes_flow.metrics import compute_all_metrics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from bes_flow.dataset import BESDataset
from bes_flow.predict import load_model, predict_sequence
from bes_flow.config  import cfg
from matplotlib.colors import CenteredNorm


def plot_flow_comparison(frames, pred, gt, metrics):
    '''

    '''
    # number of rows in the figure
    N_plots = 4
    # Accumulate random samples for the qualitative figure.
    samples = []   # list of (fA, fB, gt, pred, mean_epe) per type
    fA = frames[:-1, :, :]
    fB = frames[1:, :, :]
    
    for i in range(N_plots):
        # Pick one random pair index
        idx = np.random.randint(len(fA))
        samples.append((fA[idx, :, :],    # (H, W)
                        fB[idx, :, :],    # (H, W)
                        gt[idx],       # (2, H, W)
                        pred[idx],     # (2, H, W)
                        metrics['EPE'][idx]))
    
    # Columns: Frame A | Frame B + GT quiver | Frame B + pred quiver | EPE map
    n_rows = len(samples)
    H, W   = samples[0][0].shape   # spatial dims from first sample's frameA
 
    quiver_step = 8
    ys          = np.arange(quiver_step // 2, H, quiver_step)
    xs          = np.arange(quiver_step // 2, W, quiver_step)
    xx, yy      = np.meshgrid(xs, ys)
 
    fig = plt.figure(figsize=(20, 4 * n_rows))
    fig.suptitle('Qualitative examples across flow types  —  one random pair per type',
                 fontsize=13, fontweight='bold')
 
    gs = gridspec.GridSpec(n_rows, 5, figure=fig, hspace=0.35, wspace=0.25)
 
    col_titles = ['Frame A', 'Frame B  +  GT flow', 'Frame B  +  pred flow',
                  'EPE  vx  (px)', 'EPE  vy  (px)']
 
    for row, (fAi, fBi, gti, predi, epe_val) in enumerate(samples):
        # Per-component absolute errors (H, W)
        #epe_vx = np.abs(pred[0] - gt[0])   # |Δdx|
        #epe_vy = np.abs(pred[1] - gt[1])   # |Δdy|
        diff_vx = predi[0] - gti[0]   # (H, W)
        diff_vy = predi[1] - gti[1]   # (H, W)
 
        # Shared colour scale across dvx and dvy for direct comparability
        #vmax_epe = max(epe_vx.max(), epe_vy.max())
        #vmax_diff = max(np.abs(diff_vx).max(), np.abs(diff_vy).max())
        vmax_diff = 5.0
        norm = CenteredNorm(vcenter=0, halfrange=vmax_diff)
 
        # col 0 — Frame A
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(fAi, cmap='inferno', origin='upper')
        ax0.set_ylabel(f'EPE={epe_val:.3f} px', fontsize=9)
        if row == 0:  ax0.set_title(col_titles[0])
        ax0.set_xticks([]);  ax0.set_yticks([])
 
        # col 1 — Frame B with GT flow quiver
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(fBi, cmap='inferno', origin='upper')
        ax1.quiver(xx, yy, gti[0][yy, xx], -gti[1][yy, xx],
                   color='cyan', scale=60, scale_units='width',
                   width=0.005, headwidth=4)
        if row == 0:  ax1.set_title(col_titles[1])
        ax1.set_xticks([]);  ax1.set_yticks([])
 
        # col 2 — Frame B with predicted flow quiver
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(fBi, cmap='inferno', origin='upper')
        ax2.quiver(xx, yy, predi[0][yy, xx], -predi[1][yy, xx],
                   color='yellow', scale=60, scale_units='width',
                   width=0.005, headwidth=4)
        if row == 0:  ax2.set_title(col_titles[2])
        ax2.set_xticks([]);  ax2.set_yticks([])
 
        # col 3 — per-pixel error in vx
        ax3 = fig.add_subplot(gs[row, 3])
        #im3 = ax3.imshow(epe_vx, cmap='hot', origin='upper',
        #                  vmin=0, vmax=vmax_epe)
        im3 = ax3.imshow(diff_vx, cmap='RdBu_r', origin='upper', norm=norm)
        plt.colorbar(im3, ax=ax3, shrink=0.8, label='px')
        if row == 0:  ax3.set_title(col_titles[3])
        ax3.set_xticks([]);  ax3.set_yticks([])
 
        # col 4 — per-pixel error in vy
        ax4 = fig.add_subplot(gs[row, 4])
        #im4 = ax4.imshow(epe_vy, cmap='hot', origin='upper',
        #                 vmin=0, vmax=vmax_epe)
        im4 = ax4.imshow(diff_vy, cmap='RdBu_r', origin='upper', norm=norm)
        plt.colorbar(im4, ax=ax4, shrink=0.8, label='px')
        if row == 0:  ax4.set_title(col_titles[4])
        ax4.set_xticks([]);  ax4.set_yticks([])
 
    #path2 = os.path.join(output_dir, 'cross_flow_examples.png')
    #plt.savefig(path2, dpi=150, bbox_inches='tight')
    #print(f"Saved: {path2}")
    plt.show()

    # compare mean Vy flows
    fig, ax = plt.subplots()
    vy_gt = gt[:, 1, :, :]
    vy_nn = pred[:, 1, :, :]
    ax.plot(vy_gt.mean(axis=(0, 2)), label='GT')
    ax.plot(vy_nn.mean(axis=(0, 2)), label='Pred')
    plt.show()

    plt.close('all')
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # load model
    ckpt = 'checkpoints/model_well_best.pt'
    print(f"\nLoading checkpoint for evaluation: {ckpt}")
    model = load_model(ckpt, device)
    
    # load Gkeyll frames
    fname = 'synthetic_data/gkeyll/kappa_1p2_40x50rho_vel.h5'
    frames = []
    vel = []
    with h5py.File(fname, 'r') as f:
        for i in range(201, 1001):
            frames.append(f[f'frame_{i}']['density_fluctuation'][:])
            vel.append(f[f'frame_{i}']['velocity'][:])
    frames = np.array(frames)  # (Nframes, H, W)
    vel = np.array(vel) * 1.23e5  # (Nframes, 2, H, W), m/s
    # convert from m/s to pix/frame
    dt = 1.4e-8  # time between frames in s
    flow_gt = vel[:-1, :, :, :] * dt  # m/frame
    flow_gt[:, 0, :, :] *= 1600.  # pix/frame
    flow_gt[:, 1, :, :] *= 2000.  # pix/frame

    print(flow_gt.shape)
    print(frames.shape)

    # predict velocities
    epe_means, epe_stds, repe_means = [], [], []
    flow_pred = predict_sequence(model, frames, device)  # array of predicted velocities (n_pairs, 2, H, W)
    metrics = compute_all_metrics(flow_pred, flow_gt)
    epe_means.append(metrics['EPE'].mean())
    epe_stds.append(metrics['EPE'].std())
    repe_means.append(metrics['rEPE'].mean() * 100)
    print(flow_gt.shape)
    plot_flow_comparison(frames, flow_pred, flow_gt, metrics)

    