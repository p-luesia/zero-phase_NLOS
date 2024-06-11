import numpy as np
import tal
import matplotlib.pyplot as plt
import matplotlib

from phasor_fields import phasor_fields_reconstruction, phasor_fields_filter
from zero_phase_corrector import adaptive_z_reconstruction
from backprojection import back_projection


import time

from visualizer import StreakPlotter


if __name__ == '__main__':
    # src = '../nlos_dataset/3d_small_planes/'
    # data_file = '2d_3_plane.hdf5'
    # data_file = '2d_1_plane_center.hdf5'
    # data_file = '2d_1_plane_center_tilted.hdf5'
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
    # src = '../nlos_dataset/mesh_R_256x256/front/'
    # data_file = 'data.hdf5'; switch_indices=lambda data: data.swapaxes(0, -1)
    # src = '../nlos_dataset/'
    # data_file = 'performat_letter4.hdf5'; 
    # switch_indices=lambda data: np.moveaxis(data, 0, 2)

    src = '../nlos_dataset/3d_small_planes/20240611-194813/'
    data_file = 'plane1_z[2.5]_x[0.0]_rot[0].hdf5'
    # src = '../nlos_dataset/3d_small_planes/20240611-172843/'
    # data_file = 'plane1_z[1.0]_x[0.0]_rot[0].hdf5'

    # Volume definition 
    delta_z = 0.001
    z_begin = 1.
    z_end = 3
    # PF filter definition
    starting_central_wavelength = 0.2
    ending_central_wavelength = 0.05
    n_pulses = 5

    # Number of threads
    n_threads = 1

    z_grid = np.mgrid[z_begin:z_end:delta_z]

    full_path = src + data_file
    data = tal.io.read_capture(full_path)

    medium_idx = data.H.shape[1]//2
    data.H = data.H[:,medium_idx:-medium_idx+1,:]
    # Capture grid
    data.sensor_grid_xyz = data.sensor_grid_xyz[medium_idx:-medium_idx+1,:]
    gt_medium_idx = data.hidden_depth_grid_xyz.shape[1]//2
    data.hidden_depth_grid_xyz = data.hidden_depth_grid_xyz[gt_medium_idx:-gt_medium_idx+1,...]
    data.hidden_depth_grid_normals = data.hidden_depth_grid_normals[gt_medium_idx:-gt_medium_idx+1,...]
    # Illumination point
    xl = data.laser_grid_xyz[0,0]



    s_pf_dense = time.time()
    pf_analysis = {}
    dense_V = phasor_fields_reconstruction(data, ending_central_wavelength, 
                                           n_pulses, z_begin, z_end, 0.001, 
                                           xl, n_threads = n_threads, 
                                           analysis=pf_analysis)
    e_pf_dense = time.time()
    
    s_adaptive = time.time()
    adaptive_analysis = {}
    result_coordinates = adaptive_z_reconstruction(data,
                                                   starting_central_wavelength,
                                                   ending_central_wavelength,
                                                   n_pulses, z_begin, z_end, 
                                                   xl, n_threads = n_threads,
                                            analysis_result=adaptive_analysis)
    e_adaptive = time.time()
    max_coords_V, zero_phase_point = result_coordinates

    print(f'Dense reconstruction took {e_pf_dense - s_pf_dense}.')
    for key in pf_analysis:
        print(f'\t{key}: {pf_analysis[key]}')
    print(f'Adaptive reconstruction took {e_adaptive - s_adaptive}.')
    for key in adaptive_analysis:
        print(f'\t{key}: {adaptive_analysis[key]}')


    # Save the results
    np.save('dense_reconstruction', dense_V)
    np.save('adaptive_depth', max_coords_V)
    np.save('adaptive_zero_phase', zero_phase_point)

    
    V_coords = np.array(np.meshgrid(data.sensor_grid_xyz[:,0,0],
                                    data.sensor_grid_xyz[0,:,1],
                                    z_grid)).swapaxes(0,-1)

    V_coords = data.sensor_grid_xyz \
                + np.array([0,0,1])*z_grid.reshape(-1, 1, 1, 1)


    # Applies backprojection
    # wl, weights, fH = phasor_fields_filter(ending_central_wavelength, 
    #                                        n_pulses, data,False)
    # filtered_H = np.sum(weights.reshape(-1,1,1,1)*fH[:,np.newaxis] * np.exp((2j*np.pi/wl.reshape(-1,1,1,1))*np.arange(data.H.shape[0]).reshape(-1,1,1)*data.delta_t), axis = 0)
    # dense_V_bp = back_projection(V_coords, filtered_H, data.sensor_grid_xyz, 
    #                             data.laser_grid_xyz, data.delta_t, 
    #                             n_threads = n_threads)
    # np.save('dense_reconstruction_bp', dense_V_bp)

    plotter = StreakPlotter(V_coords, 'dense_reconstruction.npy', 
                                      'adaptive_depth.npy',
                                      'adaptive_zero_phase.npy',
                                      data.hidden_depth_grid_xyz)
    plotter[0].plot()
    # plotter = StreakPlotter(V_coords, 'results/256x256R/dense_reconstruction.npy', 
    #                                  'results/256x256R/adaptive_depth.npy',
    #                                 'results/256x256R/adaptive_zero_phase.npy')
    # plotter[:, 0].plot()


