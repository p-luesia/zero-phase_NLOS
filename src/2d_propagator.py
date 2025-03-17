import numpy as np
import tal
import matplotlib.pyplot as plt

import time

import matplotlib
matplotlib.use('svg')

def rsd_kernel_distances(origin_grid, z_delta, z_begin, z_end):
    max_kernel_dist = np.abs(origin_grid[-1] - origin_grid[0])
    delta_kernel = max_kernel_dist/origin_grid.shape[0]
    kernel_0_coords = np.mgrid[\
                    -max_kernel_dist+delta_kernel:max_kernel_dist:delta_kernel, 
                    z_begin:z_end:z_delta]
    return np.linalg.norm(np.array(kernel_0_coords).swapaxes(0,-1), axis = -1)


def rsd_propagation_kernels(origin_grid, z_delta, z_begin, z_end, wavelengths):
    rsd_dist = rsd_kernel_distances(origin_grid, z_delta, z_begin, z_end)
    rsd_kernels = np.exp(2j*np.pi*rsd_dist/wavelengths.reshape(-1,1,1))/rsd_dist
    return rsd_kernels


def point_propagator(origin_grid, z_grid, wavelengths, x=[0,0]):
    coords = np.moveaxis(np.array(np.meshgrid(origin_grid, 
                        z_grid)), 0, -1)
    dist = np.linalg.norm(coords - x, axis = -1)
    
    # dist = np.tile(np.mgrid[z_begin:z_end:z_delta], (origin_grid.shape[0],1)).swapaxes(0,1)  
    return np.exp(2j*np.pi*dist/wavelengths.reshape(-1,1,1))/dist


def phasor_fields_filter(central_wavelength, n_pulses, tal_data, analysis={}):
    pulse_sigma = n_pulses*central_wavelength / 6
    angular_freqs = 2*np.pi*np.fft.fftfreq(tal_data.H.shape[-1],
                                           tal_data.delta_t)
    # Eq. 17 from Liu, X., Bauer, S. & Velten, A. Phasor field diffraction based
    # reconstruction for fast non-line-of-sight imaging systems. Nat Commun 11, 
    # 1645 (2020). https://doi.org/10.1038/s41467-020-15157-4
    exp_pow =-0.5*(pulse_sigma*(angular_freqs - 2*np.pi/central_wavelength))**2
    pulse_in_fourier = pulse_sigma*np.sqrt((2*np.pi)**3) * np.exp(exp_pow)

    idx = np.argwhere(pulse_in_fourier > 1e-2)

    fH = np.fft.fft(tal_data.H)[..., idx][0,:,:,0]

    return (2*np.pi/angular_freqs[idx][:,0], pulse_in_fourier[idx][:,0], fH)


def phasor_fields_reconstruction(data, central_wavelength, n_pulses, z_begin,
                                 z_end, delta_z, xl, sensor_line, analysis={}):
    
    z_grid = np.mgrid[z_begin:z_end:delta_z]
    # Generate filter
    wavelengths, weight, fH = phasor_fields_filter(central_wavelength, 
                                                   n_pulses,
                                                   data)
    
    # Extract propagators
    kernels = rsd_propagation_kernels(sensor_line, delta_z, z_begin, z_end,
                                      wavelengths)
    center_prop = point_propagator(sensor_line, z_grid, wavelengths, x=xl)

    # Apply the convolution with the RSD kernels
    fV = np.fft.fft(fH.swapaxes(0,1), n = 2*fH.shape[0]-1)[:,None,:] \
        * np.fft.fft(kernels)
    # Apply the center propagation and return the reconstruction at time 0
    V = np.sum(np.fft.ifft(fV)[:,:,-fH.shape[0]:]*weight[:,None,None]*center_prop,
               axis = 0)
    analysis['kernels used'] = kernels.shape[0]*kernels.shape[1]
    analysis['reconstructed voxels'] = V.ravel().shape[0]
    return V


def phase_corrector(V, z_grid, expected_wavelength, illumination_point):
    # Look for grid 0 phase points
    max_depths_V = np.argmax(np.abs(V), axis = 0)
    max_coords_V = np.array([z_grid[max_depths_V], 
                            sensor_line]).swapaxes(0, 1)
    phase_max_V = np.angle(V[max_depths_V, np.arange(V.shape[1])])
    prop_direction = max_coords_V - illumination_point
    prop_direction /= np.linalg.norm(prop_direction, axis = -1)[:, None]

    # Check the maximum is forward or backwards
    V_fw = V[(max_depths_V + 1)%V.shape[0], np.arange(V.shape[1])]
    V_bw = V[max_depths_V - 1, np.arange(V.shape[1])]
    pulse_max_fw = V_fw > V_bw
    phase_sign = 1 - 2*pulse_max_fw

    # Calculate the phase correction (phase sign to select forward or backward)
    # phase_correction_fw = ((2*np.pi - phase_max_V)%(2*np.pi)) 
    # phase_correction_bw = -((2*np.pi + phase_max_V)%(2*np.pi)) 
    phase_correction = phase_sign*((2*np.pi - phase_sign*phase_max_V)%(2*np.pi))

    # Calculate the distance and point based on the correction
    zero_phase_dist = expected_wavelength * phase_correction / (2*np.pi)
    # zero_phase_dist_bw = expected_wavelength * phase_correction_bw / (2*np.pi)
    # zero_phase_dist_fw = expected_wavelength * phase_correction_fw / (2*np.pi)
    zero_phase_coords = max_coords_V + prop_direction*zero_phase_dist[:, None]
    # zero_phase_coords_fw = max_coords_V + prop_direction*zero_phase_dist_fw[:, None]
    # zero_phase_coords_bw = max_coords_V + prop_direction*zero_phase_dist_bw[:, None]

    return max_coords_V, zero_phase_coords
    # return zero_phase_coords_fw, zero_phase_coords_bw


def cluster_depths(query_z, delta_z):
    """
    Given the query_z distances, extract the cluster of areas whose between then
    is lower than delta_z
    """
    sorted_q = np.sort(query_z.ravel())
    diffs_z = np.diff(sorted_q, 1)
    group_break_idx = np.argwhere(diffs_z > delta_z).ravel()

    cluster_z = np.zeros((group_break_idx.shape[0] + 1, 2))
    cluster_z[:-1,1] = sorted_q[group_break_idx]
    cluster_z[1:,0] = sorted_q[group_break_idx + 1]
    cluster_z[0,0] = sorted_q[0]
    cluster_z[-1,1] = sorted_q[-1]
    return cluster_z


def adaptive_z_reconstruction(data, starting_wavelength, final_wavelength, 
                              n_pulses, z_begin, z_end, xl, sensor_line,
                              analysis_result = {}):
    current_wavelength = starting_wavelength
    z_coords = np.mgrid[z_begin:z_end:current_wavelength/3]

    reconstructed_voxels = 0
    iterations = 0
    kernels = 0
    while current_wavelength >= final_wavelength:
        # Grid dependent of wavelength
        expected_wavelength = current_wavelength/2
        # Smaller sample to avoid matching 
        delta_z_wl = expected_wavelength * 0.9
        cluster_planes = cluster_depths(z_coords, delta_z_wl)

        # Reconstruct all the indicated planes
        V = np.zeros((0, sensor_line.shape[0]), dtype=np.complex128)
        z_grid = np.array([])
        for z_b, z_e in cluster_planes:
            pf_analysis = {}
            V_local = phasor_fields_reconstruction(data, current_wavelength, 
                                                   n_pulses, z_b, z_e, 
                                                   delta_z_wl, xl, sensor_line,
                                                   pf_analysis)
            V = np.append(V, V_local, axis = 0)
            z_grid = np.append(z_grid, np.mgrid[z_b:z_e:delta_z_wl])
        
        max_coords, zero_phase_coords = phase_corrector(V, z_grid,
                                                        expected_wavelength, xl)
        current_wavelength /= 2
        z_coords = zero_phase_coords[:, 0]
        iterations += 1
        reconstructed_voxels += pf_analysis['reconstructed voxels']
        kernels += pf_analysis['kernels used']

    analysis_result['reconstructed voxels'] = reconstructed_voxels
    analysis_result['iterations'] = iterations
    analysis_result['total kernels used'] = kernels
    
    return max_coords, zero_phase_coords



if __name__ == '__main__':
    src = '../nlos_dataset/2d_small_planes/'
    # data_file = '2d_3_plane.hdf5'
    # data_file = '2d_1_plane_center.hdf5'
    # ground_truth = np.
    data_file = '2d_1_plane_center_tilted.hdf5'
    # data_file = '2d_2_planes_center_smaller_1cm.hdf5'
    # data_file = '2d_2_planes_center_smaller_1mm.hdf5'
    # data_file = '2d_2_planes_center_smaller_100um.hdf5'
    # data_file = '2d_2_planes_center_smaller_10um.hdf5'
    # data_file = '2d_2_planes_center_smaller_1um.hdf5'
    # data_file = '2d_1_plane_right.hdf5'
    # data_file = '2d_1_plane_right_tilted.hdf5'
    # data_file = '2d_2_planes_center_right.hdf5'
    # data_file = '2d_2_planes_center_right_nmc.hdf5'
    # data_file = '2d_2_planes_left_right.hdf5'

    # Volume definition 
    delta_z = 0.001
    z_begin = 0.5
    z_end = 2.5
    # PF filter definition
    starting_central_wavelength = 0.20
    ending_central_wavelength = 0.05
    n_pulses = 5

    z_grid = np.mgrid[z_begin:z_end:delta_z]

    full_path = src + data_file
    data = tal.io.read_capture(full_path)

    # Capture grid
    sensor_line = data.sensor_grid_xyz[:,0,0]
    # Illumination point
    xl = data.laser_grid_xyz[0,0,[0,2]]

    s_pf_dense = time.time()
    pf_analysis = {}
    dense_V = phasor_fields_reconstruction(data, ending_central_wavelength, 
                                           n_pulses, z_begin, z_end, 0.001, 
                                           xl, sensor_line, pf_analysis)
    e_pf_dense = time.time()
    
    s_adaptive = time.time()
    adaptive_analysis = {}
    result_coordinates = adaptive_z_reconstruction(data,
                                                   starting_central_wavelength,
                                                   ending_central_wavelength,
                                                   n_pulses, z_begin, z_end, 
                                                   xl, sensor_line, 
                                                   adaptive_analysis)
    e_adaptive = time.time()
    max_coords_V, zero_phase_point = result_coordinates

    print(f'Dense reconstruction took {e_pf_dense - s_pf_dense}.')
    for key in pf_analysis:
        print(f'\t{key}: {pf_analysis[key]}')
    print(f'Adaptive reconstruction took {e_adaptive - s_adaptive}.')
    for key in adaptive_analysis:
        print(f'\t{key}: {adaptive_analysis[key]}')

    # Plot the results
    min_sensor = np.min(sensor_line)
    delta_sensor = sensor_line[1]-sensor_line[0]
    printable_zpp = ((zero_phase_point[:,1] - min_sensor)/delta_sensor,
                        (zero_phase_point[:,0] - z_begin)/delta_z)
    printable_max_V = ((max_coords_V[:,1] - min_sensor)/delta_sensor,
                        (max_coords_V[:,0] - z_begin)/delta_z)    

    _, ax = plt.subplots(1,2, sharex=True, sharey=True, figsize=(15,20))
    ax[0].set_title('Amplitude of the reconstruction')
    ax[0].imshow(np.abs(dense_V), cmap = 'hot')
    ax[0].plot(printable_zpp[0], printable_zpp[1], color = 'green', alpha = 0.7,
                label='Zero phase depth')
    ax[0].plot(printable_max_V[0], printable_max_V[1], color = 'darkorange', 
               alpha = 0.7, label = 'Max in voxel depth')
    ax[0].legend()

    ax[1].set_title('Phase of the reconstruction')
    ax[1].imshow(np.angle(dense_V), cmap = 'bwr')
    ax[1].plot(printable_zpp[0], printable_zpp[1], color = 'g', alpha = 0.7,
               label='Zero phase depth')
    ax[1].plot(printable_max_V[0], printable_max_V[1], color = 'darkorange', 
               alpha = 0.7, label = 'Max in voxel depth')
    ax[1].legend()

    plt.savefig('reconstructions.svg')

