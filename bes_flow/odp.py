# bes_flow/odp.py

"""
================================================================================
NAME:
    ODP (Orthogonal Dynamic Programming)

PURPOSE:
    Perform the Orthogonal Dynamic Programming procedure to determine
    the flow-field that moves one "image" to the succeeding one.
    Based on the ODP-PIV algorithm described by 
    G. M. Quenot et al., Experiments in Fluids 25, 177 (1998).

ORIGINAL IDL AUTHOR:
    G. McKee
    University of Wisconsin / General Atomics
    
PYTHON PORT & OPTIMIZATION:
    Aidan Edmondson
    April 2026
    
PARAMETERS:
    image       : ndarray(ix, iy, it)
                  Input image array (spatial x, spatial y, time).
                  
    nsteps      : int, optional
                  Number of steps of increasing spatial resolution
                  (more steps increases spatial resolution and computation time).
                  Default: int(2.0 * log(nx/10.0)/log(2.0) + 0.5)
                  
    sm_param    : int, optional
                  Smoothing parameter applied to velocity measurements
                  at each step (in pixel units). Examine what works best.
                  Default: 15.
                  
    m_frame     : int, optional
                  Number of time frames to "average/integrate" for each 
                  velocity array. More frames improve accuracy if the velocity 
                  pattern is relatively static, but reduces time resolution.
                  Default: 11.
                  
    mx, my      : int, optional
                  Maximum range in x/y-direction over which a match 
                  is searched for. Should be larger than double the maximum 
                  expected displacement, but not too large as it increases 
                  boundary inaccuracies. 
                  Default: Dynamically calculated based on array size.
                  
    max_workers : int, optional
                  Number of CPU threads to use for temporal chunking. 
                  Default: All available CPU cores.

RETURNS:
    vx          : ndarray(ix, iy, it - m_frame + 1)
                  Flow-field x-velocity component (pixels/frame).
                  
    vy          : ndarray(ix, iy, it - m_frame + 1)
                  Flow-field y-velocity component (pixels/frame).

MODIFICATION HISTORY:
    - March 2003: First IDL edition (G. McKee)
    - April 2004: Added multiple frame capability (G. McKee)
    - April 2026: Python port with Numba JIT, multiprocessing,
                  and custom SciPy-free interpolators for greatest 
                  performance, reduced computation time by about 
                  two orders of magnitude (A. Edmondson)
================================================================================
"""

import numpy as np
import math
import time
from numba import njit
import concurrent.futures
import h5py
import re

@njit(nogil=True)
def residual(strip, m, window):
    n = strip.shape[0]
    w_len = strip.shape[1]
    m_frame = strip.shape[2]
        
    res = np.full((n, n), 1.0e10, dtype=np.float32)
    
    for i in range(n):
        #start_j = max(m - i - 1, i - m + 1)
        #end_j = min(i + m - 1, 2 * n - m - i - 1)
        start_j = max(m - i, i - m)
        end_j   = min(i + m, 2 * (n - 1) - m - i) # was 2*n
        
        for j in range(start_j, end_j + 1):
            val = 0.0
            for k in range(m_frame - 1):
                for w_idx in range(w_len):
                    val += window[w_idx] * abs(strip[i, w_idx, k] - strip[j, w_idx, k + 1])
            res[i, j] = val
            
    return res

@njit(nogil=True)
def optimal_path(res, m, n):
    arf = np.full((n, n), 1.0e10, dtype=np.float32)
    #for i in range(m):
    #    arf[m - i - 1, i] = 0.0
    # zero out start line
    for i in range(m + 1):
        arf[m - i, i] = 0.0
        
    for k in range(m, n - 1): #range(m, n):
        for q in range(m): #range(m - 1):
            i = k - q #- 1
            j = k - m + q + 1
            arf[i, j] = min(
                arf[i, j-1] + res[i, j-1] + res[i, j],
                arf[i-1, j-1] + 2.0 * (res[i-1, j-1] + res[i, j]),
                arf[i-1, j] + res[i-1, j] + res[i, j]
            )

        for q in range(m + 1): #range(m):
            i = k - q + 1
            j = k - m + q + 1 # was 0
            arf[i, j] = min(
                arf[i, j-1] + res[i, j-1] + res[i, j],
                arf[i-1, j-1] + 2.0 * (res[i-1, j-1] + res[i, j]),
                arf[i-1, j] + res[i-1, j] + res[i, j]
            )

    # Proicess end line
    #arf_end_line = np.zeros(m, dtype=np.float32)
    arf_end_line = np.zeros(m + 1, dtype=np.float32)
    for idx_m in range(m + 1): #range(m):
        #arf_end_line[idx_m] = arf[n - m + idx_m, n - 1 - idx_m]
        arf_end_line[idx_m] = arf[n - 1 - m + idx_m, n - 1 - idx_m]
    i_min = np.argmin(arf_end_line)

    #i_temp = n - m + i_min
    i_temp = n - 1 - m + i_min
    j_temp = n - 1 - i_min

    max_len = 4 * n 
    i_path = np.zeros(max_len, dtype=np.int32)
    j_path = np.zeros(max_len, dtype=np.int32)
    
    idx = 0
    i_path[idx] = i_temp
    j_path[idx] = j_temp
    idx += 1

    choices = np.zeros(3, dtype=np.float32)
    while (i_temp + j_temp) > m:
        choices[0] = arf[i_temp, j_temp-1] + res[i_temp, j_temp-1] + res[i_temp, j_temp]
        choices[1] = arf[i_temp-1, j_temp-1] + 2.0 * (res[i_temp-1, j_temp-1] + res[i_temp, j_temp])
        choices[2] = arf[i_temp-1, j_temp] + res[i_temp-1, j_temp] + res[i_temp, j_temp]
        
        c_min = np.argmin(choices)

        if c_min == 0: j_temp -= 1
        elif c_min == 1: i_temp -= 1; j_temp -= 1
        elif c_min == 2: i_temp -= 1

        i_path[idx] = i_temp
        j_path[idx] = j_temp
        idx += 1

    i_coord = np.zeros(idx, dtype=np.int32)
    j_coord = np.zeros(idx, dtype=np.int32)
    for k in range(idx):
        i_coord[k] = i_path[idx - 1 - k]
        j_coord[k] = j_path[idx - 1 - k]

    max_i = np.max(i_coord)
    max_j = np.max(j_coord)
    count_i = np.sum(i_coord == max_i)
    count_j = np.sum(j_coord == max_j)
    len_max = max(count_i, count_j)
    
    if len_max > 1:
        idx -= (len_max - 1)
        i_coord = i_coord[:idx]
        j_coord = j_coord[:idx]

    min_i = np.min(i_coord)
    min_j = np.min(j_coord)
    length = min_i
    
    if length > 0:
        new_idx = length + idx
        new_i = np.zeros(new_idx, dtype=np.int32)
        new_j = np.zeros(new_idx, dtype=np.int32)
        for k in range(length):
            new_i[k] = k
            new_j[k] = k - length + min_j
        for k in range(idx):
            new_i[length + k] = i_coord[k]
            new_j[length + k] = j_coord[k]
        i_coord, j_coord, idx = new_i, new_j, new_idx
        
    max_i = np.max(i_coord)
    max_j = np.max(j_coord)
    length2 = n - 1 - max_i
    
    if length2 > 0:
        new_idx = idx + length2
        new_i = np.zeros(new_idx, dtype=np.int32)
        new_j = np.zeros(new_idx, dtype=np.int32)
        for k in range(idx):
            new_i[k] = i_coord[k]
            new_j[k] = j_coord[k]
        for k in range(length2):
            new_i[idx + k] = k + max_i + 1
            new_j[idx + k] = k + max_j + 1
        i_coord, j_coord = new_i, new_j

    return i_coord, j_coord

# ==============================================================================
# CUSTOM INTERPOLATORS
# ==============================================================================

@njit(nogil=True)
def interp_temp_x(temp_x, y_idx_array):
    nx, x_steps = temp_x.shape
    ny = len(y_idx_array)
    out = np.zeros((nx, ny), dtype=np.float32)
    for i in range(nx):
        for j in range(ny):
            y = y_idx_array[j]
            y0 = int(math.floor(y))
            y1 = y0 + 1
            if y0 < 0: y0 = 0
            if y1 > x_steps - 1: y1 = x_steps - 1
            wy = y - y0
            out[i, j] = temp_x[i, y0] * (1.0 - wy) + temp_x[i, y1] * wy
    return out

@njit(nogil=True)
def interp_temp_y(temp_y_T, x_idx_array):
    y_steps, ny = temp_y_T.shape
    nx = len(x_idx_array)
    out = np.zeros((nx, ny), dtype=np.float32)
    for i in range(nx):
        x = x_idx_array[i]
        x0 = int(math.floor(x))
        x1 = x0 + 1
        if x0 < 0: x0 = 0
        if x1 > y_steps - 1: x1 = y_steps - 1
        wx = x - x0
        for j in range(ny):
            out[i, j] = temp_y_T[x0, j] * (1.0 - wx) + temp_y_T[x1, j] * wx
    return out

@njit(nogil=True)
def uniform_filter(arr, size):
    nx, ny = arr.shape
    out = np.zeros((nx, ny), dtype=np.float32)
    temp = np.zeros((nx, ny), dtype=np.float32)
    r = size // 2
    for i in range(nx):
        for j in range(ny):
            val = 0.0
            for kj in range(j - r, j + r + 1):
                idx = kj
                if idx < 0: idx = 0
                elif idx >= ny: idx = ny - 1
                val += arr[i, idx]
            temp[i, j] = val / size
    for i in range(nx):
        for j in range(ny):
            val = 0.0
            for ki in range(i - r, i + r + 1):
                idx = ki
                if idx < 0: idx = 0
                elif idx >= nx: idx = nx - 1
                val += temp[idx, j]
            out[i, j] = val / size
    return out

@njit(nogil=True)
def map_coordinates(image, cx, cy):
    nx, ny = image.shape
    out = np.zeros((nx, ny), dtype=np.float32)
    for i in range(nx):
        for j in range(ny):
            x, y = cx[i, j], cy[i, j]
            if x < 0.0: x = 0.0
            if x > nx - 1.0: x = nx - 1.0
            if y < 0.0: y = 0.0
            if y > ny - 1.0: y = ny - 1.0
            x0, y0 = int(math.floor(x)), int(math.floor(y))
            x1, y1 = x0 + 1, y0 + 1
            if x0 < 0: x0 = 0
            if x1 > nx - 1: x1 = nx - 1
            if y0 < 0: y0 = 0
            if y1 > ny - 1: y1 = ny - 1
            wx, wy = x - x0, y - y0
            out[i, j] = (image[x0, y0] * (1.0 - wx) * (1.0 - wy) +
                         image[x1, y0] * wx * (1.0 - wy) +
                         image[x0, y1] * (1.0 - wx) * wy +
                         image[x1, y1] * wx * wy)
    return out

# ==============================================================================
# SLICE WORKER 
# ==============================================================================

@njit(nogil=True)
def odp_chunk(image_slice, nsteps, smooth_param, m_frame, mx_init, my_init):
    nx, ny, frames = image_slice.shape
    out_frames = frames - m_frame + 1
    
    vx_out = np.zeros((nx, ny, out_frames), dtype=np.float32)
    vy_out = np.zeros((nx, ny, out_frames), dtype=np.float32)
    
    ix2d = np.zeros((nx, ny), dtype=np.float32)
    iy2d = np.zeros((nx, ny), dtype=np.float32)
    for i in range(nx):
        for j in range(ny):
            ix2d[i, j] = i
            iy2d[i, j] = j
            
    ix = np.arange(nx, dtype=np.float32)
    iy = np.arange(ny, dtype=np.float32)

    for frame in range(out_frames):
        x_width = int(ny / 2 + 0.5)
        y_width = int(nx / 2 + 0.5)
        sm_param = smooth_param
        mx, my = mx_init, my_init

        image_warp = np.copy(image_slice[:, :, frame : frame + m_frame])

        for steps in range(nsteps):
            # ODP in y-direction
            y_steps = int(2.0 * nx / y_width - 1.0)
            temp_y = np.zeros((ny, y_steps), dtype=np.float32)

            window_y = np.zeros(y_width, dtype=np.float32)
            w_y = y_width - 1.0
            if w_y > 0:
                for w_idx in range(y_width):
                    window_y[w_idx] = 1.0 + math.cos(2.0 * math.pi * (w_idx - w_y / 2.0) / w_y)

            for y_index in range(y_steps):
                start_x = int(y_index * y_width / 2.0)
                end_x = min(start_x + y_width, nx)
                
                strip = np.ascontiguousarray(np.transpose(image_warp[start_x:end_x, :, :], (1, 0, 2)))
                res = residual(strip, my, window_y)
                i_coord, j_coord = optimal_path(res, my, ny)

                temp_y1 = np.zeros(ny, dtype=np.float32)
                temp_y2 = np.zeros(ny, dtype=np.float32)
                
                for k in range(len(i_coord)):
                    temp_y1[i_coord[k]] = j_coord[k]
                for k in range(len(i_coord)):
                    idx_rev = len(i_coord) - 1 - k
                    temp_y2[i_coord[idx_rev]] = j_coord[idx_rev]
                    
                for k in range(ny):
                    temp_y[k, y_index] = (temp_y1[k] + temp_y2[k]) / 2.0 - iy[k]

            if y_steps > 1:
                denom_y = np.float32(nx - y_width - 1.0)
                if denom_y > 0.0:
                    mid_y = np.arange(nx - y_width, dtype=np.float32) / denom_y * np.float32(y_steps - 1.0)
                else:
                    mid_y = np.zeros(nx - y_width, dtype=np.float32)
                    
                part1_y = np.full(int(y_width / 2.0), -1.0, dtype=np.float32)
                part3_y = np.full(int(y_width / 2.0 + 0.6), float(y_steps), dtype=np.float32)
                
                y_indices = np.concatenate((part1_y, mid_y, part3_y))
                
                y_indices_clipped = np.zeros(len(y_indices), dtype=np.float32)
                for k in range(len(y_indices)):
                    val = y_indices[k]
                    if val < 0.0: val = 0.0
                    if val > y_steps - 1.0: val = y_steps - 1.0
                    y_indices_clipped[k] = val
                    
                vy_work = interp_temp_y(temp_y.T, y_indices_clipped)
            else:
                vy_work = np.zeros((nx, ny), dtype=np.float32)
                for i in range(nx):
                    for j in range(ny):
                        vy_work[i, j] = temp_y[j, 0]

            vy_work = uniform_filter(vy_work, sm_param)
            for i in range(nx):
                for j in range(ny):
                    vy_out[i, j, frame] += vy_work[i, j]

            for i in range(1, m_frame):
                cx = np.zeros((nx, ny), dtype=np.float32)
                cy = np.zeros((nx, ny), dtype=np.float32)
                for ii in range(nx):
                    for jj in range(ny):
                        cx[ii, jj] = ix2d[ii, jj] + vx_out[ii, jj, frame]
                        cy[ii, jj] = iy2d[ii, jj] + vy_out[ii, jj, frame]
                image_warp[:, :, i] = map_coordinates(image_slice[:, :, frame + i], cx, cy)

            # ODP in x-direction
            x_steps = int(2.0 * ny / x_width - 1.0)
            temp_x = np.zeros((nx, x_steps), dtype=np.float32)

            window_x = np.zeros(x_width, dtype=np.float32)
            w_x = x_width - 1.0
            if w_x > 0:
                for w_idx in range(x_width):
                    window_x[w_idx] = 1.0 + math.cos(2.0 * math.pi * (w_idx - w_x / 2.0) / w_x)

            for x_index in range(x_steps):
                start_y = int(x_index * x_width / 2.0)
                end_y = min(start_y + x_width, ny)
                
                strip = np.ascontiguousarray(image_warp[:, start_y:end_y, :])
                res = residual(strip, mx, window_x)
                i_coord, j_coord = optimal_path(res, mx, nx)

                temp_x1 = np.zeros(nx, dtype=np.float32)
                temp_x2 = np.zeros(nx, dtype=np.float32)
                
                for k in range(len(i_coord)):
                    temp_x1[i_coord[k]] = j_coord[k]
                for k in range(len(i_coord)):
                    idx_rev = len(i_coord) - 1 - k
                    temp_x2[i_coord[idx_rev]] = j_coord[idx_rev]
                    
                for k in range(nx):
                    temp_x[k, x_index] = (temp_x1[k] + temp_x2[k]) / 2.0 - ix[k]

            if x_steps > 1:
                denom_x = np.float32(ny - x_width - 1.0)
                if denom_x > 0.0:
                    mid_x = np.arange(ny - x_width, dtype=np.float32) / denom_x * np.float32(x_steps - 1.0)
                else:
                    mid_x = np.zeros(ny - x_width, dtype=np.float32)
                    
                part1 = np.full(int(x_width / 2.0), -1.0, dtype=np.float32)
                part3 = np.full(int(x_width / 2.0 + 0.6), float(x_steps), dtype=np.float32)
                
                x_indices = np.concatenate((part1, mid_x, part3))
                
                x_indices_clipped = np.zeros(len(x_indices), dtype=np.float32)
                for k in range(len(x_indices)):
                    val = x_indices[k]
                    if val < 0.0: val = 0.0
                    if val > x_steps - 1.0: val = x_steps - 1.0
                    x_indices_clipped[k] = val
                    
                vx_work = interp_temp_x(temp_x, x_indices_clipped)
            else:
                vx_work = np.zeros((nx, ny), dtype=np.float32)
                for i in range(nx):
                    for j in range(ny):
                        vx_work[i, j] = temp_x[i, 0]

            vx_work = uniform_filter(vx_work, sm_param)
            for i in range(nx):
                for j in range(ny):
                    vx_out[i, j, frame] += vx_work[i, j]

            for i in range(1, m_frame):
                cx = np.zeros((nx, ny), dtype=np.float32)
                cy = np.zeros((nx, ny), dtype=np.float32)
                for ii in range(nx):
                    for jj in range(ny):
                        cx[ii, jj] = ix2d[ii, jj] + vx_out[ii, jj, frame]
                        cy[ii, jj] = iy2d[ii, jj] + vy_out[ii, jj, frame]
                image_warp[:, :, i] = map_coordinates(image_slice[:, :, frame + i], cx, cy)
            
            #if frame // 100 == 0:
            #    print(f"frame: {frame} | smooth: {sm_param} | mx: {mx} | my: {my}")
            # update strip widths and smoothing params
            x_width = max(int(x_width / math.sqrt(2.0) + 0.5), 5)
            y_width = max(int(y_width / math.sqrt(2.0) + 0.5), 5)
            mx = max(int((mx / math.sqrt(2.0)) / 2.0) * 2 + 1, 3)
            my = max(int((my / math.sqrt(2.0)) / 2.0) * 2 + 1, 3)
            sm_param = max(sm_param - 2, 5)
            
    return vx_out, vy_out

# ==============================================================================
# HDF5 WORKER
# ==============================================================================

def worker_hdf5(args):
    chunk_idx, start, end, nsteps, sm_param, m_frame, mx, my, fname = args
    
    with h5py.File(fname, 'r') as hf:
        img_slice = hf['images'][start : end + m_frame - 1, :, :]
        
    img_slice = np.transpose(img_slice, (2, 1, 0))
        
    img_slice = np.array(img_slice, dtype=np.float32)
    np.nan_to_num(img_slice, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    
    vx, vy = odp_chunk(img_slice, nsteps, sm_param, m_frame, mx, my)
    
    vx = np.transpose(vx, (2, 1, 0))
    vy = np.transpose(vy, (2, 1, 0))
    
    return chunk_idx, vx, vy

# ==============================================================================
# Main Called Function
# ==============================================================================

def time_resolved_ODP(nstep='default', smooth=15, mframe=11, mx='default', my='default', max_workers=16,
                      frames='raw_data/194313_t=2620-2640_f=30-200_2000fr.h5',
                      output='outputs-odp/velocities.h5',
                      save_velocities=True):
    
    print("\n=== Starting ODP optical flow ===\n")
    nstep_val = None if nstep == 'default' else int(nstep)
    mx_val = None if mx == 'default' else int(mx)
    my_val = None if my == 'default' else int(my)
    workers_val = int(max_workers)
    sm_param = int(smooth)
    m_frame = int(mframe)

    print("Fetching HDF5 metadata...", flush=True)
    with h5py.File(frames, 'r') as hf:
        time_len, Z_len, R_len = hf['images'].shape

    nx, ny, n_frames = R_len, Z_len, time_len

    if nstep_val is None: nstep_val = int(2.0 * math.log(nx / 10.0) / math.log(2.0) + 0.5)
    if mx_val is None: mx_val = int((nx / 6.0) / 2 + 0.5) * 2 + 1
    if my_val is None: my_val = int((ny / 6.0) / 2 + 0.5) * 2 + 1

    print(f"n_steps: {nstep_val} | smooth: {sm_param} | mframe: {m_frame} | mx: {mx_val} | my: {my_val}")

    out_frames = n_frames - m_frame + 1
    
    vx_out = np.zeros((out_frames, ny, nx), dtype=np.float32)
    vy_out = np.zeros((out_frames, ny, nx), dtype=np.float32)

    chunk_size = max(1, math.ceil(out_frames / 10))
    ranges = []
    for start in range(0, out_frames, chunk_size):
        end = min(start + chunk_size, out_frames)
        ranges.append((start, end))

    args_list = [
        (idx, start, end, nstep_val, sm_param, m_frame, mx_val, my_val, frames)
        for idx, (start, end) in enumerate(ranges)
    ]

    total_chunks = len(args_list)
    completed = 0

    print(f"Beginning Multiprocessing {n_frames} frames across {total_chunks} chunks using {workers_val} processes...", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers_val) as executor:
        futures = {executor.submit(worker_hdf5, arg): arg for arg in args_list}
        
        t_start = time.time()
        
        for future in concurrent.futures.as_completed(futures):
            chunk_idx, vx, vy = future.result() 
            start, end = ranges[chunk_idx]
            
            vx_out[start:end, :, :] = vx
            vy_out[start:end, :, :] = vy
            
            completed += 1
            percent = (completed / total_chunks) * 100
            
            elapsed = time.time() - t_start
            eta_seconds = (elapsed / completed) * (total_chunks - completed)
            
            if completed % 1 == 0:
                print(f"Progress: [{completed}/{total_chunks}] Chunks | {percent:.1f}% Complete | ETA: {int(eta_seconds // 60)}m {int(eta_seconds % 60)}s", flush=True)

    if save_velocities:
        print("Saving velocity data to HDF5...", flush=True)
        with h5py.File(output, 'w') as hf:
            hf.create_dataset('vx', data=vx_out)
            hf.create_dataset('vy', data=vy_out)
    print("Done!", flush=True)


if __name__ == '__main__':
    # test time-resolved ODP
    frames_fname = 'raw_data/194313_t=2620-2640_f=30-200_2000fr.h5'
    out_fname = 'outputs-odp/' + re.search(r'\d.*', frames_fname).group()
    time_resolved_ODP(
        nstep='default', 
        smooth=15, 
        mframe=2, 
        mx='default', 
        my='default', 
        max_workers=8,
        frames=frames_fname,
        output=out_fname
    )
