import numpy as np
import tal
import matplotlib.pyplot as plt
import matplotlib

from phasor_fields import phasor_fields_reconstruction, phasor_fields_filter
from phasor_fields import set_by_point, set_planar_RSD
from zero_phase_corrector import adaptive_z_reconstruction
from backprojection import back_projection

from argparse import ArgumentParser
import sys


import os

import time

from visualizer import StreakPlotter

def file_parser(filename: str):
    file_values = filename.split('_')
    parsed_filename = {'tag':[]}
    for value_str in file_values:
        idx_val_b = value_str.find('[')
        idx_val_e = value_str.find(']')
        if idx_val_b > -1 and idx_val_e > -1:
            parsed_filename[value_str[:idx_val_b]] = \
                float(value_str[idx_val_b  + 1:idx_val_e])
        else:
            parsed_filename['tag'].append(value_str)

    parsed_filename['tag'] = '_'.join(parsed_filename['tag'])
    auto_id = "_".join(map(lambda t: f'{t[0]}[{t[1]}]',parsed_filename.items()))
    parsed_filename['name'] = auto_id
    return parsed_filename


def get_z_params(exp_info):
    # Planar reconstruction distances to the relay wall
    z_offset = 0.5
    if 'z_offset' in exp_info:
        z_offset = exp_info['z_offset']

    if 'z' in exp_info.keys():
        z_begin = exp_info['z'] - z_offset
        z_end = exp_info['z'] + z_offset
    elif 'z_limits' in exp_info.keys():
        z_begin, z_end = exp_info['z_limits']
    else:
        z_begin = 0.5; z_end = 1.5
        print('No limits for z indicated. Using default z in range [0.5, 1.5]')

    return z_begin, z_end


def rec_experiment(data, z_begin, z_end, delta_z, central_wavelength,
                     n_pulses, n_zp_it, output_prefix, no_dense, n_threads):
    """
    Reconstruct with the dense and adaptive reconstructions the data
    :param data                 : Input data from the TAL library
    :param exp_info             : Experiment information. See file_parser for 
                                  more info
    :param z_begin              : Distance from the relay wall to start the 
                                  reconstruction
    :param z_end                : Distance from the relay wall to stop the 
                                  reconstruction
    :param delta_z              : Distance between reconstruction planes
    :param central_wavelength   : Central wavelength of the phasor fields
                                  virtual illumination for the reconstructions
    :param n_pulses             : Number of pulses of the phasor fields
                                  virtual illumination for the reconstruction
    :param n_zp_it              : Number of iterations of the zero phase 
                                  algorithm
    :param output_prefix        : Output file prefix to save the results
    :param no_dense             : If true, it will not perform the dense
                                  reconstruction
    :param n_threads            : Number of threads to use during the 
                                  reconstruction   
    :return                     : The distances used for reconstruct                               
    """
    if data.laser_grid_format == tal.enums.GridFormat.N_3:
        xl = data.laser_grid_xyz[0]
    elif data.laser_grid_format == tal.enums.GridFormat.X_Y_3:
        xl = data.laser_grid_xyz[0, 0]
    else:
        raise(TypeError(f'Unknown format {data.laser_grid_format}'))
    
    starting_central_wavelength = central_wavelength * 2**(n_zp_it-1)
    ending_central_wavelength = central_wavelength

    # Reconstruction of the dense
    if not no_dense:
        dense_V = phasor_fields_reconstruction(data, ending_central_wavelength, 
                                            n_pulses, z_begin, z_end, delta_z, 
                                            xl, n_threads = n_threads)
    # Reconstruct with the adaptive reconstruction
    result_coordinates = adaptive_z_reconstruction(data,
                                                   starting_central_wavelength,
                                                   ending_central_wavelength,
                                                   n_pulses, z_begin, z_end, 
                                                   xl, n_threads = n_threads)
        
    max_coords_V, zero_phase_point = result_coordinates

    if not no_dense:
        np.save(f'{output_prefix}dense_reconstruction', dense_V)
    np.save(f'{output_prefix}max_adaptive_coords', max_coords_V)
    np.save(f'{output_prefix}zero_phase_coords', zero_phase_point)

    return np.mgrid[z_begin:z_end:delta_z]


def gen_from_dir(dirname, output_dir ='./results/',
                           n_threads = 1):
    if os.path.isdir(output_dir):
        print(f'Warning: using {output_dir} that already exists')
    else:
        os.mkdir(output_dir)
        print(f'Results will be stored in {output_dir}')

    for filename in os.listdir(dirname):
        exp_params = file_parser(filename)

        fullfilename = os.path.join(dirname, filename)

        data = tal.io.read_capture(fullfilename)

        medium_idx = data.H.shape[2]//2
        data.H = data.H[:,:,medium_idx:-medium_idx+1]
        # Preprare data for the reconstruction
        data.sensor_grid_xyz = data.sensor_grid_xyz[:,medium_idx:-medium_idx+1]
        gt_medium_idx = data.hidden_depth_grid_xyz.shape[1]//2
        data.hidden_depth_grid_xyz = data.hidden_depth_grid_xyz.swapaxes(0,1)[-1:0:-1,gt_medium_idx:-gt_medium_idx+1,...]
        data.hidden_depth_grid_normals = data.hidden_depth_grid_normals.swapaxes(0,1)[-1:0:-1,:,gt_medium_idx:-gt_medium_idx+1,...]


        exp_file_prefix =  os.path.join(output_dir, exp_params['name'])
        z_grid = rec_experiment(data, exp_params, 0.0001, 0.5, 0.05, 5,
                                exp_file_prefix, n_threads)

        # Coordinates for visualization purposes
        V_coords = data.sensor_grid_xyz \
            + np.array([0,0,1])*z_grid.reshape(-1, 1, 1, 1)
        plotter = StreakPlotter(V_coords, 
                                f'{exp_file_prefix}dense_reconstruction.npy', 
                                f'{exp_file_prefix}max_adaptive_coords.npy',
                                f'{exp_file_prefix}zero_phase_coords.npy', 
                                data.hidden_depth_grid_xyz)

        plot = plotter[:,0]
        plot.outputfile = f'{exp_file_prefix}_plot.svg'
        plot.save()

        
def rec_and_plot_experiment(data: tal.io.capture_data.NLOSCaptureData,
                            exp_params: dir, c_wavelength: float = 0.05, 
                            n_cycles: float = 5, n_zp_it = 3,
                            output_dir: str = '', n_threads: int = 1,
                            no_dense: bool = False, reconstruct: bool = True,
                            plot: bool = True):
        z_begin, z_end = get_z_params(exp_params)
        exp_file_prefix =  os.path.join(output_dir, exp_params['name'])
        if reconstruct:
            rec_experiment(data, z_begin, z_end, exp_params['delta_z'], 
                           c_wavelength, n_cycles, n_zp_it, exp_file_prefix,
                           no_dense, n_threads)
    

        if plot: 
            # Coordinates for visualization purposes
            z_grid = np.mgrid[z_begin:z_end:exp_params['delta_z']]
            V_coords = data.sensor_grid_xyz*np.array([1,1,0]) \
                + np.array([0,0,1])*z_grid.reshape(-1, 1, 1, 1)
            
            # Set the ground truth if the data has it
            if data.scene_info is not None and data.scene_info.shape is not None:
                hidden_ground_truth = data.scene_info['ground_truth']['depth']
            else:
                hidden_ground_truth = None
                 
            plotter = StreakPlotter(V_coords, 
                                    f'{exp_file_prefix}dense_reconstruction.npy', 
                                    f'{exp_file_prefix}max_adaptive_coords.npy',
                                    f'{exp_file_prefix}zero_phase_coords.npy', 
                                    hidden_ground_truth)
            plot = plotter[:, 89]
            # plot = plotter[:,0]
            plot.outputfile = f'{exp_file_prefix}_plot.svg'
            plot.save()


def parse_args(argv):
    parser = ArgumentParser('3d_propagator',
            description = 'Reconstruct with Phasor Fields and our Zero Phase'\
                        + 'approach, and plot the results.')
    parser.add_argument('capture_datafiles', nargs = '+', type = str,
                        help = 'Input datafiles of the captures in tal hdf5'\
                             + ' format')
    parser.add_argument('--input_prefix', type = str, nargs = '?', 
                        help = 'Prefix for the capture datafiles')
    parser.add_argument('-o', '--output_dir', type = str, nargs = '*',
                        help = 'Output directories to store the results. It '\
                             + 'can be 1 or the same number as the input '\
                             + 'capture datafiles')
    parser.add_argument('--output_prefix', type = str, nargs = '?',
                        help = 'Prefix for the output directories')
    parser.add_argument('-z', type = float, help = 'Main depth to reconstruct '\
                        + 'in the dense reconstruction', default = 1.0)
    parser.add_argument('--delta_z', type = float, help = 'Distance between '\
                        + 'planes to reconstruct in the dense reconstruction',
                        default = 0.0001)
    parser.add_argument('--z_offset', type = float, help = 'Distance from z '\
                        + '(forward and backward) to reconstruct the dense'\
                        + ' reconstruction', default = 0.3)
    onlies_group = parser.add_mutually_exclusive_group()
    onlies_group.add_argument('--only_plot', action = 'store_true', 
                        help = 'If specified it will not reconstruct')
    onlies_group.add_argument('--only_reconstruct', action = 'store_true', 
                        help = 'If specified it will not plot the results')
    parser.add_argument('--no_dense', action = 'store_true',
                        help = 'Do not use the dense reconstruction')
    parser.add_argument('--two_dimensions', type = slice,
                        help = 'If indicated, it will reconstruct only with '\
                                + 'the given indices')
    parser.add_argument('-t', '--threads', type = int, default = 1,
                        help = 'Number of threads to use')
    parser.add_argument('--by_point', action = 'store_true',
                        help = 'Performance the reconstruction point by point '\
                             + 'instead of using the planar RSD')
    parser.add_argument('--c_wavelength', type = float, default = 0.05,
                         help = 'Central wavelength for the reconstructions. '\
                         + 'Default is 0.5')
    parser.add_argument('--n_cycles', type = float, default = 5,
                         help = 'Number of the number of cycles of the '\
                         + 'illumination package. Default is 5')
    parser.add_argument('--zero_phase_it', type = int, default = 3,
                         help = 'Number of the iterations of the zero phase '\
                         + 'approach. Default is 3')
    
    args = parser.parse_args(argv)
    assert args.output_dir is None or len(args.output_dir) == 1 \
            or len(args.output_dir) == len(args.capture_datafiles),\
            'Wrong number of output directiories'

    return args


def main(argv):
    args = parse_args(argv)

    # Set the prefix, input datafiles and output directories
    input_files = args.capture_datafiles
    if args.input_prefix is not None:
        input_files = [args.input_prefix + file for file in input_files]

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ['']
    if args.output_prefix is not None:
        output_dir = [args.output_prefix + direct for direct in output_dir]

    if len(output_dir) == 1:
        experiment_files = [(file, output_dir[0]) for file in input_files]
    else:       # Same number of input files and output dirs
        experiment_files = [(input_files[i], output_dir[i]) 
                            for i in range(len(input_files))]

    if len(experiment_files) == 1 and os.path.isdir(experiment_files[0][0]):
        gen_from_dir(experiment_files[0][0], 
                    output_dir=experiment_files[0][1],
                    n_threads = 12)
        exit(0)

    if args.by_point:
        set_by_point()

    for full_path, output_dir in experiment_files:
    # for id in range(268, 275 + 1):

    #     src = '../nlos_dataset/2024_06_21_R_different_microdepths_U/hdf5_files'
    #     data_file = f'capt_{id}_6_12.hdf5'
        # data_file = 'plane1_z[1.5]_x[0.0]_rot[0].hdf5'
        # data_file = 'plane1_z[2.0]_x[0.0]_rot[0].hdf5'
        # data_file = 'plane1_z[2.5]_x[0.0]_rot[0].hdf5'
        # data_file = 'plane1_z[1.0]_x[0.1]_rot[0].hdf5'
        # data_file = 'plane1_z[1.0]_x[0.1]_rot[0].hdf5'
        # data_file = 'plane1_z[1.0]_x[0.1]_rot[0].hdf5'


        # src = '../nlos_dataset/simple_corner/20240617-201630/'
        # data_file = 'plane1_rot[10]_plane2_rot[-10].hdf5'

        # full_path = os.path.join(src, data_file)

        data = tal.io.read_capture(full_path)

        data.laser_grid_xyz = -data.laser_grid_xyz*np.array([0,1,0]) + data.laser_grid_xyz*np.array([1,0,1])
        data.H = data.H

        if args.two_dimensions is not None:
            data.H = data.H[:,args.two_dimensions]
            # Preprare data for the reconstruction
            data.sensor_grid_xyz = data.sensor_grid_xyz[args.two_dimensions]
        # # gt_medium_idx = data.hidden_depth_grid_xyz.shape[1]//2
        # data.hidden_depth_grid_xyz = data.hidden_depth_grid_xyz.swapaxes(0,1)[-1:0:-1,gt_medium_idx:-gt_medium_idx+1,...]
        # data.hidden_depth_grid_normals = data.hidden_depth_grid_normals.swapaxes(0,1)[-1:0:-1,:,gt_medium_idx:-gt_medium_idx+1,...]
        # set_by_point()
        name = os.path.realpath(full_path).split('/')[-1][:-5]
        name = ''
        exp_params = {'name':name, 'z': args.z, 
                      'z_offset': args.z_offset, 'delta_z': args.delta_z}
        print(exp_params['name'])

        rec_and_plot_experiment(data, exp_params, 
                                c_wavelength= args.c_wavelength, 
                                n_cycles = args.n_cycles,
                                n_zp_it = args.zero_phase_it,
                                n_threads = args.threads, 
                                output_dir = output_dir, 
                                no_dense = args.no_dense,
                                reconstruct = not args.only_plot,
                                plot = not args.only_reconstruct)
        # set_planar_RSD()
    # # src = '../nlos_dataset/3d_small_planes/'
    # # data_file = '2d_3_plane.hdf5'
    # # data_file = '2d_1_plane_center.hdf5'
    # # data_file = '2d_1_plane_center_tilted.hdf5'
    # # data_file = '2d_2_planes_center_smaller_1cm.hdf5'
    # # data_file = '2d_2_planes_center_smaller_1mm.hdf5'
    # # data_file = '2d_2_planes_center_smaller_100um.hdf5'
    # # data_file = '2d_2_planes_center_smaller_10um.hdf5'
    # # data_file = '2d_2_planes_center_smaller_1um.hdf5'
    # # data_file = '2d_1_plane_right.hdf5'
    # # data_file = '2d_1_plane_right_tilted.hdf5'
    # # data_file = '2d_2_planes_center_right.hdf5'
    # # data_file = '2d_2_planes_center_right_nmc.hdf5'
    # # data_file = '2d_2_planes_left_right.hdf5'
    # # src = '../nlos_dataset/mesh_R_256x256/front/'
    # # data_file = 'data.hdf5'; switch_indices=lambda data: data.swapaxes(0, -1)
    # src = '../nlos_dataset/'
    # data_file = 'performat_letter4.hdf5'; 
    # # switch_indices=lambda data: np.moveaxis(data, 0, 2)

    # # src = '../nlos_dataset/3d_small_planes/20240611-194813/'
    # # data_file = 'plane1_z[2.5]_x[0.0]_rot[0].hdf5'
    # # src = '../nlos_dataset/3d_small_planes/20240611-172843/'
    # # data_file = 'plane1_z[1.0]_x[0.0]_rot[0].hdf5'

    # # Volume definition 
    # delta_z = 0.0001
    # z_begin = 0.5
    # z_end = 1.5
    # # PF filter definition
    # starting_central_wavelength = 0.2
    # ending_central_wavelength = 0.05
    # n_pulses = 5

    # # Number of threads
    # n_threads = 12

    # z_grid = np.mgrid[z_begin:z_end:delta_z]

    # full_path = src + data_file
    # data = tal.io.read_capture(full_path)

    # # medium_idx = data.H.shape[1]//2
    # # data.H = data.H[:,medium_idx:-medium_idx+1,:]
    # # # Capture grid
    # # data.sensor_grid_xyz = data.sensor_grid_xyz[medium_idx:-medium_idx+1,:]
    # # gt_medium_idx = data.hidden_depth_grid_xyz.shape[1]//2
    # # data.hidden_depth_grid_xyz = data.hidden_depth_grid_xyz[gt_medium_idx:-gt_medium_idx+1,...]
    # # data.hidden_depth_grid_normals = data.hidden_depth_grid_normals[gt_medium_idx:-gt_medium_idx+1,...]
    # # Illumination point
    # xl = data.laser_grid_xyz[0,0]



    # s_pf_dense = time.time()
    # pf_analysis = {}
    # dense_V = phasor_fields_reconstruction(data, ending_central_wavelength, 
    #                                        n_pulses, z_begin, z_end, delta_z, 
    #                                        xl, n_threads = n_threads, 
    #                                        analysis=pf_analysis)
    # e_pf_dense = time.time()
    
    # s_adaptive = time.time()
    # adaptive_analysis = {}
    # result_coordinates = adaptive_z_reconstruction(data,
    #                                                starting_central_wavelength,
    #                                                ending_central_wavelength,
    #                                                n_pulses, z_begin, z_end, 
    #                                                xl, n_threads = n_threads,
    #                                         analysis_result=adaptive_analysis)
    # e_adaptive = time.time()
    # max_coords_V, zero_phase_point = result_coordinates

    # print(f'Dense reconstruction took {e_pf_dense - s_pf_dense}.')
    # for key in pf_analysis:
    #     print(f'\t{key}: {pf_analysis[key]}')
    # print(f'Adaptive reconstruction took {e_adaptive - s_adaptive}.')
    # for key in adaptive_analysis:
    #     print(f'\t{key}: {adaptive_analysis[key]}')


    # # Save the results
    # np.save('dense_reconstruction_4r', dense_V)
    # np.save('adaptive_depth_4r', max_coords_V)
    # np.save('adaptive_zero_phase_4r', zero_phase_point)

    
    # V_coords = np.array(np.meshgrid(data.sensor_grid_xyz[:,0,0],
    #                                 data.sensor_grid_xyz[0,:,1],
    #                                 z_grid)).swapaxes(0,-1)

    # V_coords = data.sensor_grid_xyz \
    #             + np.array([0,0,1])*z_grid.reshape(-1, 1, 1, 1)


    # # Applies backprojection
    # wl, weights, fH = phasor_fields_filter(ending_central_wavelength, 
    #                                        n_pulses, data,False)
    # filtered_H = np.sum(weights.reshape(-1,1,1,1)*fH[:,np.newaxis] * np.exp((2j*np.pi/wl.reshape(-1,1,1,1))*np.arange(data.H.shape[0]).reshape(-1,1,1)*data.delta_t), axis = 0)
    # dense_V_bp = back_projection(V_coords, filtered_H, data.sensor_grid_xyz, 
    #                             data.laser_grid_xyz, data.delta_t, 
    #                             n_threads = n_threads)
    # np.save('dense_reconstruction_bp_4r', dense_V_bp)

    # plotter = StreakPlotter(V_coords, 'dense_reconstruction.npy', 
    #                                   'adaptive_depth.npy',
    #                                   'adaptive_zero_phase.npy',
    #                                   data.hidden_depth_grid_xyz)
    # plotter[:, 60].plot()
    # # plotter = StreakPlotter(V_coords, 'results/256x256R/dense_reconstruction.npy', 
    # #                                  'results/256x256R/adaptive_depth.npy',
    # #                                 'results/256x256R/adaptive_zero_phase.npy')
    # # plotter[:, 0].plot()



if __name__ == '__main__':
    main(sys.argv[1:])

