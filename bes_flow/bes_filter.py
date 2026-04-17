#bes_flow/bes_filter.py

import numpy as np
from scipy.signal import firwin, freqz, filtfilt
from scipy import interpolate

def bandpass(data, dt, cutoff, numtaps=501, plot_ftf=False):
    """
    This function filters BES data using a non-causal FIR filter. The filter
    has linear phase response and is applied with a forwards-backwards
    algorithm so there is not net time shift of the data.

    Parameters
    ==========
    data : ndarray (N_chan, N_time)
    cutoff : array of floats
        Filter cutoff frequencies in kHz.
    numtaps : int, optional
        Number of FIR filter taps to use. More taps improves sharpness of the
        transition from passband to stopband at the cost of increased
        computation time.
    plot_ftf : bool, optional
        Plots the filter's transfer function (magnitude and phase).

    Returns
    =======
    filt_data : ndarray
        Bandpass filtered  data.
    """

    # Calculate FIR filter coefficients
    fs = 1 / dt  # kHz
    if cutoff[0] == 0 or cutoff[0] == None:  # lowpass filter
        b = firwin(numtaps, cutoff[1:], pass_zero=True, fs=fs)
    elif cutoff[-1] == fs / 2.0 or cutoff[-1] == None:  # highpass filter
        b = firwin(numtaps, cutoff[:-1], pass_zero=False, fs=fs)
    else:  # bandpass filter
        b = firwin(numtaps, cutoff, pass_zero=False, fs=fs)

    # Plot filter transfer function
    if plot_ftf:
        w, h = freqz(b)
        freqs = w * fs / (2 * np.pi)  # convert units from rad/sample to Hz
        mag_dB = 20 * np.log10(np.abs(h))
        phase = np.unwrap(np.angle(h))

        fig, ax1 = plt.subplots()
        ax1.plot(freqs, mag_dB, 'b-')
        ax1.set_xlabel('Frequency (kHz)')
        ax1.set_ylabel('Amplitude (dB)', color='b')
        ax2 = ax1.twinx()
        ax2.plot(freqs, phase, 'r')
        ax2.set_ylabel('Phase (rad)', color='r')
        ax1.set_title('Filter transfer function')
        fig.tight_layout()

    # Filter signals
    filt_data = filtfilt(b, 1, data, axis=1)

    return filt_data


def apply_transfer_functions(data, dt):
    """
    Applies transfer function correction to input signals by Fourier
    transforming to the frequency domain, dividing by the transfer
    function, then inverse Fourier transforming back to time domain.

    data : ndarray
    dt : float
    """
    # Load transfer functions
    tf = np.loadtxt("bes_flow/133298_tf.csv", delimiter=",")
    tf_data = tf[1:, :]  # 0th row is freq
    tf_frequency = tf[0, :] / 1000.0  # Hz -> kHz
    # Apply to data
    nchannels = data.shape[0]
    length = data.shape[1]
    data = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(length, d=dt)  # kHz
    #print(tf_frequency.shape, tf_data.shape)
    splines = interpolate.CubicSpline(tf_frequency, tf_data, axis=1, extrapolate=True)
    data = np.fft.irfft(data / np.sqrt(np.abs(splines(freqs))), n=length)

    return data


def filter_nbi(data, sig_time, filter_ds, analysis_times=[0, 9500]):
    """
    Slice the data based on when the viewed 150 beam is on
    and another 150 beam is off. The viewed beam is determined from
    mds parameter 'BES::TOP.BEAM'

    Parameters
    ==========
    data : ndarray
    sig_time : ndarray
        data timebase
    analysis_times: list
        list with start and finish time
    Returns
    =======
    data_list : list
        list of np arrays with data
    time_list : list
        list of np arrays with time
    """

    print('\nFiltering NBI modulation')
    # Check viewed beam (150-Left or 150-Right)
    # Get NBI info
    beam_index = filter_ds['bes_beam']['data']
    if beam_index == 0:  # 150-R beam
        viewed_beam = filter_ds['pinj_15r']['data']
        beam_time = filter_ds['pinj_15r']['times']
        odd_beam = filter_ds['pinj_15l']['data']
        print('BES focused on 150-RIGHT')
    elif beam_index == 1:  # 150-L beam
        viewed_beam = filter_ds['pinj_15l']['data']
        beam_time = filter_ds['pinj_15l']['times']
        odd_beam = filter_ds['pinj_15r']['data']
        print('BES focused on 150-LEFT')
    else:
        print('Cannot determine which DIII-D beam is being viewed.')
    # Take beam data at analysis_times
    dt = beam_time[1] - beam_time[0]
    tmin, tmax = analysis_times
    mask = (beam_time > tmin) & (beam_time < tmax)
    beam_time = beam_time[mask]
    viewed_beam = viewed_beam[mask]
    odd_beam = odd_beam[mask]
    # Set up times when only viewed beam is on
    selected_times = beam_time[(viewed_beam > 1e4) & (odd_beam < 1e4)][1:-1]
    # Find times indices when only viewed beam is on
    indices = np.where(selected_times[1:] - selected_times[0:-1] > 2 * dt)[0]
    indices_start = np.insert(indices + 1, 0, 0)
    indices_end = np.insert(indices, len(indices), len(selected_times) - 1)
    # Slice data and time based on indices and put slices into lists
    data_list = []
    time_list = []
    for i_start, i_stop in zip(indices_start, indices_end):
        # Remove 2 ms at the beginning and 0.5 ms at the end of each NBI blip
        tmin, tmax = selected_times[i_start] + 2, selected_times[i_stop] - 0.5
        time_indices = np.where((sig_time > tmin) & (sig_time < tmax))[0]
        time_list.append(sig_time[time_indices])
        data_list.append(data[:, time_indices])
    print(f'Detected {len(data_list)} NBI blip(s)')

    return data_list, time_list


def modified_zscore(data):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z = 0.6745 * (data - median) / mad
    return modified_z


def find_bad_channels(data):
    '''
    The function works with a data time slice.
    For bad channels check - simple logic - check the modified Z-score
    Returns a list of bad channel indices (starting from 0)
    '''
    std = np.nanstd(data, axis=1)  # calculate std over the time axis
    score = np.abs(modified_zscore(np.log(1e3*std + 1e-3)))
    bad_channels = np.where(score > 2.5)[0]  # these are channel indices, start from 0
    
    return bad_channels.tolist()


def filter_bes(bes_ds, filter_ds, cutoff_freqs, analysis_times, filter_nbi=True):
    '''
    The function takes xarray dataset with BES data, applies trasfer fucntions and filteing,
    and slices data based on NBI timing
    
    Returns 
    data_list : list
        list with data np.arrays, 
    time_list : list 
        list with time np.arrays 
    '''
    # Put BES-fast data into np.array
    bes_fast = np.array([bes_ds['fast_ds'][var] for var in bes_ds['fast_ds'].data_vars])  # shape (n_chan, n_time)
    # Get the time base
    bes_fast_time = bes_ds['fast_ds']['times'].data
    nt = bes_fast_time.shape[0]
    dt = bes_fast_time[1] - bes_fast_time[0]
    # Apply transfer functions first
    data_filtered = apply_transfer_functions(bes_fast, dt)
    # Bandpass filter
    data_filtered = bandpass(data_filtered, dt, cutoff=cutoff_freqs, numtaps=501, plot_ftf=False)
    # NBI filter
    if filter_nbi:
        data_list, time_list = filter_nbi(data_filtered, bes_fast_time, filter_ds,
                                        analysis_times=analysis_times)  
   
    return data_list, time_list
