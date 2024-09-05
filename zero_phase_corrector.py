import numpy as np
from phasor_fields import phasor_fields_reconstruction
from RSD_propagator import RSD_kernel


def phase_corrector(V, coords, expected_wavelength, illumination_point):
    # Look for grid 0 phase points
    max_depths_V = np.argmax(np.abs(V), axis = 0)
    i, j = np.mgrid[:coords.shape[1], :coords.shape[2]]
    V_2d = V[max_depths_V, i, j]

    mask = np.abs(V_2d) > np.max(V_2d)*0.05

    max_coords_V = coords[max_depths_V, i, j]

    phase_max_V = np.angle(V[max_depths_V, i, j])
    prop_direction = max_coords_V - illumination_point
    prop_direction /= np.linalg.norm(prop_direction, axis = -1)[..., None]

    # Check the maximum is forward or backwards
    V_fw = V[((max_depths_V + 1)%V.shape[0]), i, j]
    V_bw = V[(max_depths_V - 1), i, j]
    diff_pulse = np.abs(V_fw) - np.abs(V_bw)
    pulse_max_bw = diff_pulse < 0
    phase_sign = 1 - 2*pulse_max_bw

    significant_phase = np.abs(phase_max_V) > np.pi/40

    # Calculate the phase correction (phase sign to select forward or backward)
    # phase_correction_fw = ((2*np.pi - phase_max_V)%(2*np.pi)) 
    # phase_correction_bw = -((2*np.pi + phase_max_V)%(2*np.pi)) 
    phase_correction = phase_sign*((2*np.pi - phase_sign*phase_max_V)%(2*np.pi))

    # Calculate the distance and point based on the correction
    zero_phase_dist = expected_wavelength * phase_correction / (2*np.pi)
    # zero_phase_dist_bw = expected_wavelength * phase_correction_bw / (2*np.pi)
    # zero_phase_dist_fw = expected_wavelength * phase_correction_fw / (2*np.pi)
    zero_phase_coords = max_coords_V\
                    + significant_phase[...,None]*prop_direction\
                        *zero_phase_dist[..., None]
    # zero_phase_coords_fw = max_coords_V + prop_direction*zero_phase_dist_fw[:, None]
    # zero_phase_coords_bw = max_coords_V + prop_direction*zero_phase_dist_bw[:, None]
    return max_coords_V, zero_phase_coords, mask
    # return zero_phase_coords_fw, zero_phase_coords_bw


def cluster_depths(query_z, delta_z):
    """
    Given the query_z distances, extract the cluster of areas whose between then
    is lower than delta_z
    """
    sorted_q = np.sort(query_z)
    diffs_z = np.diff(sorted_q, 1)
    group_break_idx = np.argwhere(np.abs(diffs_z) > delta_z).ravel()

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
    # Delete below
    # print('Warning, only using final_wavelength')
    # current_wavelength = final_wavelength
    # Delete above
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
        print(f'Reconstructing with centralwavelength {current_wavelength}')

        # Reconstruct all the indicated planes
        V = np.zeros((0,) + sensor_grid.shape[:-1], dtype=np.complex128)
        z_grid = np.array([])

        fH_all = np.fft.fft(data.H, axis = 0)

        for z_b, z_e in cluster_planes:
            if z_e < z_begin or z_b > z_end:   # Outside reconstruction area
                continue
            if z_b < z_begin:
                z_b = z_begin
            if z_e > z_end:
                z_e = z_end

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
        max_coords, zero_phase_coords, mask = phase_corrector(V, coords,
                                                        expected_wavelength, xl)
        current_wavelength /= 2
        z_coords = zero_phase_coords[..., 2].reshape(-1)[mask.reshape(-1)]
        iterations += 1
        reconstructed_voxels += pf_analysis['reconstructed voxels']
        kernels += pf_analysis['kernels used']

    analysis_result['reconstructed voxels'] = reconstructed_voxels
    analysis_result['iterations'] = iterations
    analysis_result['total kernels used'] = kernels
    
    return max_coords, zero_phase_coords
