# bes_flow/tok_loader.py 

import numpy as np
from toksearch import Pipeline, MdsSignal
from toksearch_d3d import PtDataSignal
import argparse
import h5py
import multiprocessing as mp
from scipy.interpolate import Rbf
import time as timelib

import bes_flow.bes_filter as bf


def raw_bes_pipeline(shots):
    pipe = Pipeline(shots)
    chlist = np.arange(64) + 1
    slow_signals_dict = {}
    fast_signals_dict = {}
    for i_ch in chlist:
        su_ptname = f"BESSU{i_ch:02d}"
        fu_ptname = f"BESFU{i_ch:02d}"
        slow_signals_dict[su_ptname] = PtDataSignal(su_ptname)
        fast_signals_dict[fu_ptname] = PtDataSignal(fu_ptname)
    pipe.fetch_dataset("slow_ds", slow_signals_dict)
    pipe.fetch_dataset("fast_ds", fast_signals_dict)

    return pipe


def preprocessing_pipeline(shots):
    pipe = Pipeline(shots)

    pipe.fetch("bes_r", MdsSignal(r"\bes_r", "bes", location="remote://atlas.gat.com")) #good logic, bug due to MDSplus pathing on omega
    pipe.fetch("bes_z", MdsSignal(r"\bes_z", "bes", location="remote://atlas.gat.com"))
    pipe.fetch("bes_beam", MdsSignal(r"\bes_beam", "bes", location="remote://atlas.gat.com", dims=[]))
    pipe.fetch("pinj_15l", MdsSignal(r"\pinj_15l", "nb", location="remote://atlas.gat.com"))
    pipe.fetch("pinj_15r", MdsSignal(r"\pinj_15r", "nb", location="remote://atlas.gat.com"))

    return pipe   


def time_interp(data, time, t_interp_factor):
    """
    Take time series data (n_ch, n_time) and interpolate over 
    the new timabase
    """
    if t_interp_factor == 1:  # check if any interpolation needed
        return data, time
    nt_interp = t_interp_factor * (nt - 1) + 1  # new amount of points
    n_ch = data.shape[0]
    data_interp = np.zeros((n_ch, nt_interp))
    ti = np.linspace(time[0], time[-1], num=nt_interp) # new timebase
    for ch in range(n_ch):
        data_interp[ch, :] = np.interp(ti, time, data[ch, :])
    
    return data_interp, ti


def image_interp(R, Z, Ri, Zi, image_data):
    """
    Takes low-resolution (8x8) BES images and spatially interpolates them to
    higher resolution. A cubic radial basis function algorithm
    (scipy.interpolate.Rbf) is used to perform the interpolation.
    """
    rbf = Rbf(R, Z, image_data, function='cubic')
    
    return rbf(Ri, Zi)


def make_images(image_data, R, Z, Ri, Zi, cpu_cores):
    """
    Takes image_data array with shape (n_time, n_channels)
    and returns the array of images with shape (n_time, nZ, nR)
    """
    n_frames = image_data.shape[0]
    print(f'\nInterpolating {n_frames} images...')
    images = np.zeros((n_frames,) + Ri.shape)
    pool = mp.Pool(np.min([n_frames, cpu_cores]))
    results = [pool.apply_async(image_interp, (R, Z, Ri, Zi, image_data[frame, :])) for frame in range(n_frames)]
    pool.close()
    for frame, result in enumerate(results):
        while not result.ready():
            timelib.sleep(0.05)
        images[frame, :, :] = result.get()
    
    return images  # dimensions are (time, Z, R)!


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Creates hdf5 file using raw BES data from Toksearch.')
    parser.add_argument('--shot', help='shot number to get the data', type=int, default=199452) #required=True)
    parser.add_argument('--times', help='time slice of interest, of form --times START STOP (in milliseconds)', type=int, nargs=2, default=[2000, 2200]) #required=True)
    parser.add_argument('--fband', help='frequency band of interest', type=int, nargs=2, default=[20, 250]) #required=True)
    parser.add_argument('--time_interp', help='time interpolation factor', type=int, default=1) #required=True)
    parser.add_argument('--exclude_channels', help='list of channels to exclude, set to -1 for automatic search', type=int, nargs='+', default=0) #required=True)
    parser.add_argument('--cpu_cores', help='Number of CPUs for parallelism when interpolating', type=int, default=8) #required=True)
    parser.add_argument('--res', help='image resolution after spatial interpolation', type=int, nargs=2, default=[64, 64]) #required=True)
    parser.add_argument("--out", help="Path for output file", type=str, default=None, required=True)
    parser.add_argument("--filter_nbi", help="Enable NBI filtering", type=bool, default=True) # set to False to diable NBI filtering
    args = parser.parse_args()
    out_dir = args.out + "/"
    
    analysis_times = args.times
    shot = args.shot
    t_interp_factor = args.time_interp  # Time interpolation factor
    cutoff_freqs = args.fband  # Frequency band of interest
    res = args.res  # image resolution after interpolation [nR, nZ]
    print(analysis_times, cutoff_freqs, res)
    
    print(f'\nLoading BES data for #{shot}')
    raw_bes_ds = raw_bes_pipeline([shot]).compute_serial()[0]
    print(raw_bes_ds)
    filter_ds = preprocessing_pipeline([shot]).compute_serial()[0]

    #print(raw_bes_ds)
    #print(filter_ds)
    # Filter and slice data
    data_list, time_list = bf.filter_bes(raw_bes_ds, filter_ds, cutoff_freqs, analysis_times, 
                                        filter_nbi=args.filter_nbi)
    print(f'Found {len(data_list)} time slices')

    # Get R, Z coordinates
    R = filter_ds['bes_r']['data']
    Z = filter_ds['bes_z']['data'] * -1. # Z-axis is inverted in the tree
    #print(R, Z)
    # Define the interpolation grid in R, Z
    ch_width, ch_height = 0.8, 1.1  # channel radial width and poloidal height in cm
    R0 = min(R) - ch_width / 2
    R1 = max(R) + ch_width / 2
    Z0 = min(Z) - ch_height / 2
    Z1 = max(Z) + ch_height / 2
    # Default indexing in meshgrid is 'xy'
    Ri, Zi = np.meshgrid(np.linspace(R0, R1, num=res[0]), np.linspace(Z0, Z1, num=res[1]))
    
    # For each time slice: detect bad channels, oversample signals in time, 
    # create images and save them to hdf5
    for data, time in zip(data_list, time_list):
        fname = args.out + f'/{shot}_{time[0]:.2f}-{time[-1]:.2f}_f={cutoff_freqs[0]}-{cutoff_freqs[1]}.h5'
        print('Processing: ' + fname)
        print('t_min, t_max: ', time[0], time[-1])
        # Print std to compare with OMFIT
        #stds = np.nanstd(data, axis=1)
        #print('STD for each channel: ', stds)
        # find bad channels indices
        if args.exclude_channels == 0: # use all channels
            pass
        elif args.exclude_channels == -1: # automatically identify da channels
            print('Looking for bad channels...')
            bad_channels = bf.find_bad_channels(data)
            print(f'Found bad channesl: {[ch+1 for ch in bad_channels]}') # channels numbers start from 1
        else:
            print('Bad channels set by user: ', args.exclude_channels)
            bad_channels = [i-1 for i in args.exclude_channels]

        # remove bad channels from data and R, Z arrays
        data = np.delete(data, bad_channels, axis=0)
        R_clean = np.delete(R, bad_channels, axis=0)
        Z_clean = np.delete(Z, bad_channels, axis=0)
        # Interpolate time and data over new timebase
        data_final, ti = time_interp(data, time, t_interp_factor)
        # Normalize data by rms amplitude
        data_final = data_final / np.std(data_final, axis=1, keepdims=True)
        # Transpose array to make it (n_time, n_channels) 
        image_data = data_final.T  
        # Create images array with dimensions (n_time, nZ, nR)
        images = make_images(image_data, R_clean, Z_clean, Ri, Zi, args.cpu_cores)
        # Write interpolated images to hdf5 file
        with h5py.File(fname, 'w') as hf:
            hf.create_dataset('images', data=images)
            hf.create_dataset('time', data=ti)
            hf.create_dataset('R', data=Ri[0, :])
            hf.create_dataset('Z', data=Zi[:, 0])
        print('Saved images: ' + fname)
