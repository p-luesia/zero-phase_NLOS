import numpy as np

from phasorfields.RSD_propagator import RSD_kernel
from phasorfields import reconstruct

from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt

def slope_method(V, coords, expected_wavelength, illumination_point):
    """
    Given a reconstruction Phasor Fields V with spacing between planes smaller 
    than the expected wavelength, returns the coordinates of the maximum
    """
    # Find the maximum in the sparse reconstruction
    max_depths_V = np.argmax(np.abs(V), axis = 0)
    i, j = np.mgrid[:coords.shape[1], :coords.shape[2]]
    V_2d = V[max_depths_V, i, j]

    mask = np.abs(V_2d) > np.max(np.abs(V_2d))*0.05
    # Maximum coordinates
    max_coords_V = coords[max_depths_V, i, j]

    # Phase on the maximum reconstruction
    phase_max_V = np.angle(V[max_depths_V, i, j])
    # Direction of the light propagation
    prop_direction = max_coords_V - illumination_point
    prop_direction /= np.linalg.norm(prop_direction, axis = -1)[..., None]


    # Calculate the phase correction
    phase_correction = ( - phase_max_V /(2*np.pi) )

    # Calculate the distance and point based on the correction
    zero_phase_dist = expected_wavelength * phase_correction

    # Estimate the coordinates
    zero_phase_coords = max_coords_V + prop_direction\
                        *zero_phase_dist[..., None]
    
    return max_coords_V, zero_phase_coords, mask



def gaussian_fit_method(V, coords, expected_wavelength, illumination_point):
    # Look for grid 0 phase points
    max_depths_V = np.argmax(np.abs(V), axis = 0)
    # Extract the indices of the coordinates for the interpolation
    indices = np.array(np.mgrid[:coords.shape[0], :coords.shape[1],
                                 :coords.shape[2]])
    k, i, j = indices
    V_2d = V[max_depths_V, i, j]

    mask = np.abs(V_2d) > np.max(np.abs(V_2d))*0.05

    max_coords_V = coords[max_depths_V, i, j]

    # TODO: Parametrize number of cycles
    n_cycles = 5
    sigma_dist = expected_wavelength * n_cycles / 6

    min_z = np.min(coords[:, 0, 0, 2])
    min_x = np.min(coords[0, :, 0, 0])
    min_y = np.min(coords[0, 0, :, 1])

    z_v = np.mean(np.diff(coords[:, 0, 0, 2])) * np.arange(coords.shape[0]) + min_z
    x_v = np.mean(np.diff(coords[0, :, 0, 0])) * np.arange(coords.shape[1]) + min_x
    y_v = np.mean(np.diff(coords[0, 0, :, 1])) * np.arange(coords.shape[2]) + min_y

    rg_interp = RegularGridInterpolator((z_v, x_v, y_v),
                                        np.moveaxis(indices, 0, -1), 
                                        method = 'nearest',
                                        bounds_error=False, fill_value = None)

    def expected_illumination(x, center_distance, amplitude):
        """
        Define the illumination expected given the virtual illumination package
        """
        distance_to_mu = np.linalg.norm(x - illumination_point)- center_distance
        gauss_exp_part = -0.5*(distance_to_mu/sigma_dist)**2
        phasor_exp_part = 2j*np.pi/expected_wavelength*distance_to_mu
        return amplitude*np.exp(gauss_exp_part + phasor_exp_part)
    
    def expected_illumination_real(x, center_distance, amplitude):
        """
        Define the real part of the illumination expected given the virtual 
        illumination package
        """
        distance_to_mu = np.linalg.norm(x - illumination_point)- center_distance
        gauss_exp_part = -0.5*(distance_to_mu/sigma_dist)**2
        real_pulse_part = np.cos(2*np.pi/expected_wavelength*distance_to_mu)
        return amplitude*np.exp(gauss_exp_part)*real_pulse_part
    
    def expected_illumination_imag(x, center_distance, amplitude):
        """
        Define the imaginary part of the illumination expected given the virtual
        illumination package
        """
        distance_to_mu = np.linalg.norm(x - illumination_point)- center_distance
        gauss_exp_part = -0.5*(distance_to_mu/sigma_dist)**2
        imag_pulse_part = np.sin(2*np.pi/expected_wavelength*distance_to_mu)
        return amplitude*np.exp(gauss_exp_part)*imag_pulse_part
    

    v_coords = max_coords_V.reshape(-1, 3)

    distances = np.linspace(1, 1.5, 20).reshape((-1, 1))

    for v in v_coords:
        dir = v - illumination_point
        nor_dir = dir/np.linalg.norm(dir)
        query_points = distances * nor_dir
        # Interpolate indicies from the given coordinates
        ans_indices = rg_interp(query_points).swapaxes(0,1).astype(int)
        # Points in the coords and its values in the reconstruction
        points = coords[tuple(ans_indices)]
        values = V[tuple(ans_indices)]
        param_real, _ = curve_fit(expected_illumination_real, points, np.real(values))
        param_imag, _ = curve_fit(expected_illumination_imag, points, np.imag(values))
        print(f"Found for position {v} center at distance {param_real[0]} from illum point")
        plt.plot(np.real(values), label = 'Params real part')
        plt.plot(np.imag(values), label = 'Params imag part')
        plt.show()


    print(max_coords_V.shape)


    curve_fit(expected_illumination, 4)


def phase_corrector(V, coords, expected_wavelength, illumination_point):
    # gaussian_fit_method(V, coords, expected_wavelength, illumination_point)
    return slope_method(V, coords, expected_wavelength, illumination_point)


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
        delta_z_wl = expected_wavelength * 0.7
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
            V_local = reconstruct(data, current_wavelength, 
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
