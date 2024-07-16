import numpy as np
from phasor_fields import phasor_fields_reconstruction
from RSD_propagator import RSD_kernel


def phase_corrector(V, coords, expected_wavelength, illumination_point):
    # Look for grid 0 phase points
    max_depths_V = np.argmax(np.abs(V), axis = 0)
    x_it = np.tile(np.arange(coords.shape[1]), (coords.shape[2], 1))
    y_it = np.tile(np.arange(coords.shape[2])[:, None], (1, coords.shape[1]))

    max_coords_V = coords[max_depths_V.flat, x_it.flat, y_it.flat]

    phase_max_V = np.angle(V[max_depths_V.flat, x_it.flat, y_it.flat])
    prop_direction = max_coords_V - illumination_point
    prop_direction /= np.linalg.norm(prop_direction, axis = -1)[:, None]

    # Check the maximum is forward or backwards
    V_fw = V[((max_depths_V + 1)%V.shape[0]).flat, x_it.flat, y_it.flat]
    V_bw = V[(max_depths_V - 1).flat, x_it.flat, y_it.flat]
    pulse_max_fw = V_fw > V_bw
    phase_sign = 1 - 2*pulse_max_fw

    significant_phase = np.abs(phase_max_V) > np.pi/90

    # Calculate the phase correction (phase sign to select forward or backward)
    # phase_correction_fw = ((2*np.pi - phase_max_V)%(2*np.pi)) 
    # phase_correction_bw = -((2*np.pi + phase_max_V)%(2*np.pi)) 
    phase_correction = phase_sign*((2*np.pi - phase_sign*phase_max_V)%(2*np.pi))

    # Calculate the distance and point based on the correction
    zero_phase_dist = expected_wavelength * phase_correction / (2*np.pi)
    # zero_phase_dist_bw = expected_wavelength * phase_correction_bw / (2*np.pi)
    # zero_phase_dist_fw = expected_wavelength * phase_correction_fw / (2*np.pi)
    zero_phase_coords = max_coords_V\
                    + significant_phase[:,None]*prop_direction\
                        *zero_phase_dist[:, None]
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
                              n_pulses, z_begin, z_end, xl, 
                              n_threads = 1, analysis_result = {}):
    current_wavelength = starting_wavelength
    z_coords = np.mgrid[z_begin:z_end:current_wavelength/3]

    reconstructed_voxels = 0
    iterations = 0
    kernels = 0
    sensor_grid = data.sensor_grid_xyz
    RSD_prop = RSD_kernel(data.sensor_grid_xyz)

    while current_wavelength >= final_wavelength:
        # Grid dependent of wavelength
        expected_wavelength = current_wavelength/2
        # Smaller sample to avoid matching 
        delta_z_wl = expected_wavelength * 0.9
        cluster_planes = cluster_depths(z_coords, delta_z_wl)

        # Reconstruct all the indicated planes
        V = np.zeros((0,) + sensor_grid.shape[:-1], dtype=np.complex128)
        z_grid = np.array([])

        fH_all = np.fft.fft(data.H, axis = 0)

        for z_b, z_e in cluster_planes:
            pf_analysis = {}
            V_local = phasor_fields_reconstruction(data, current_wavelength, 
                                                   n_pulses, z_b, z_e, 
                                                   delta_z_wl, xl, 
                                                   RSD_prop=RSD_prop,
                                                   fH_all = fH_all,
                                                   n_threads = n_threads,
                                                   analysis = pf_analysis)
            V = np.append(V, V_local, axis = 0)
            z_grid = np.append(z_grid, np.mgrid[z_b:z_e:delta_z_wl])
        
        if np.max(V) == 0:
            print('No geometry found!')
            return (np.zeros(data.sensor_grid_xyz.shape),)*2
        coords = data.sensor_grid_xyz[np.newaxis, ...]*np.array([1.0,1.0,0.0]) \
                    + np.array([0,0,1.0])*z_grid.reshape(-1,1,1,1)
        max_coords, zero_phase_coords = phase_corrector(V, coords,
                                                        expected_wavelength, xl)
        current_wavelength /= 2
        z_coords = zero_phase_coords[:, 2]
        iterations += 1
        reconstructed_voxels += pf_analysis['reconstructed voxels']
        kernels += pf_analysis['kernels used']

    analysis_result['reconstructed voxels'] = reconstructed_voxels
    analysis_result['iterations'] = iterations
    analysis_result['total kernels used'] = kernels
    
    return max_coords, zero_phase_coords
