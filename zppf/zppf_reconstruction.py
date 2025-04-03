import numpy as np
import tal
import matplotlib.pyplot as plt
import matplotlib

from .phasorfields import reconstruct
from .zero_phase_corrector import adaptive_z_reconstruction
from .backprojection import back_projection

from argparse import ArgumentParser
import sys

from timeit import default_timer as timer
from datetime import timedelta


import os

import time

from .visualizer import StreakPlotter
    

def params_from_filename(filename: str):
    file_values = filename.split('_')
    parsed_filename = {}
    for value_str in file_values:
        idx_val_b = value_str.find('[')
        idx_val_e = value_str.find(']')
        if idx_val_b > -1 and idx_val_e > -1:
            parsed_filename[value_str[:idx_val_b]] = \
                float(value_str[idx_val_b  + 1:idx_val_e])
    return parsed_filename


def get_z_params(exp_info):
    # Planar reconstruction distances to the relay wall
    z_offset = 0.2
    if 'z_offset' in exp_info:
        z_offset = exp_info['z_offset']

    if 'z' in exp_info:
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
        if data.is_confocal():
            xl = data.laser_grid_xyz
        else:
            xl = data.laser_grid_xyz[0, 0]
    else:
        raise(TypeError(f'Unknown format {data.laser_grid_format}'))
    
    starting_central_wavelength = central_wavelength * 2**(n_zp_it-1)
    ending_central_wavelength = central_wavelength

    # Reconstruction of the dense
    if not no_dense:
        print(f'Reconstructing {int((z_end-z_begin))/delta_z} planes with {delta_z} m between them...', 
              flush=True, end="")
        t_start = timer()
        z_v = np.mgrid[z_begin:z_end:delta_z]
        dense_V = reconstruct(data, ending_central_wavelength, 
                            n_pulses, z_v, 
                            xl, n_threads = n_threads)
        t_end = timer()
        print(f'Done. It took {timedelta(seconds=t_end-t_start)} segs')
        np.save(f'{output_prefix}dense_reconstruction', dense_V)
        del dense_V

    # Reconstruct with the adaptive reconstruction
    print(f'Reconstructing ZPPF with {ending_central_wavelength} m of central wavelength...', 
              flush=True, end="")
    t_start = timer()
    result_coordinates = adaptive_z_reconstruction(data,
                                                   starting_central_wavelength,
                                                   ending_central_wavelength,
                                                   n_pulses, z_begin, z_end, 
                                                   xl, n_threads = n_threads)
    t_end = timer()
    print(f'Done. It took {timedelta(seconds=t_end-t_start)} segs')
        
    max_coords_V, zero_phase_point = result_coordinates

    np.save(f'{output_prefix}max_adaptive_coords', max_coords_V)
    np.save(f'{output_prefix}zero_phase_coords', zero_phase_point)

    return np.mgrid[z_begin:z_end:delta_z]


def gen_from_dir(dirname, output_dir ='./results/', delta_z=0.01, n_zp_it = 3,
                 no_dense = False, reconstruct=True, n_threads = 1):
    
    print('******THIS SHOULD NOT BE EXECUTING*************')
    if os.path.isdir(output_dir):
        print(f'Warning: using {output_dir} that already exists')
    else:
        os.mkdir(output_dir)
        print(f'Results will be stored in {output_dir}')

    for filename in os.listdir(dirname):
        exp_params = params_from_filename(filename)

        fullfilename = os.path.join(dirname, filename)

        data = tal.io.read_capture(fullfilename)
        exp_params['delta_z'] = delta_z

        rec_and_plot_experiment(data=data, exp_params=exp_params, 
                                n_zp_it = n_zp_it,
                                no_dense = no_dense,
                                  reconstruct = reconstruct,
                                  output_dir = output_dir,
                                  n_threads=n_threads)

        
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
            if data.scene_info is not None and 'ground_truth' in data.scene_info.keys():
                hidden_ground_truth = data.scene_info['ground_truth']['depth']
            else:
                hidden_ground_truth = None
                 
            plotter = StreakPlotter(V_coords, 
                                    f'{exp_file_prefix}dense_reconstruction.npy', 
                                    f'{exp_file_prefix}max_adaptive_coords.npy',
                                    f'{exp_file_prefix}zero_phase_coords.npy', 
                                    hidden_ground_truth)
                                    
            # plot = plotter[:, 128]
            plot = plotter[:, V_coords.shape[2]//2]
            # plot = plotter[256]
            # plot = plotter[:,0]
            plot.outputfile = f'{exp_file_prefix}_plot.svg'
            plot.save()


def parse_args(argv):
    parser = ArgumentParser('zppf_reconstruction.py',
            description = 'Reconstruct with Phasor Fields and our Zero Phase'\
                        + 'approach, and plot the results.')
    parser.add_argument('capture_datafiles', nargs = '+', type = str,
                        help = 'Input datafiles of the captures in tal hdf5'\
                             + ' format')
    parser.add_argument('--input_prefix', type = str, nargs = '?', 
                        help = 'Prefix for the capture datafiles')
    parser.add_argument('-o', '--output_dir', type = str, nargs = '+',
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
    
    parser.add_argument('--downscale', type = int, 
                        help = "If indicated, downscale the input data for the"\
                             + " rest of the process. It is recommended to use"\
                             + " powers of 2.")
    
    parser.add_argument('--t_shift', type = int, help="WARNING THIS IS ONLY FOR DEBUG: it temporally shift the data as given", nargs=1, default = 0)
    
    args = parser.parse_args(argv)
    assert args.output_dir is None or len(args.output_dir) == 1 \
            or len(args.output_dir) == len(args.capture_datafiles),\
            'Wrong number of output directiories'

    return args


def prepare_data(args):

    # Prepare a list with all the input datafiles
    input_files = args.capture_datafiles
    if args.input_prefix is not None:
        input_files = [os.path.relpath(file, args.input_prefix) \
                       for file in input_files]
    # If the input is a directory
    if len(input_files) and os.path.isdir(input_files[0]):
        print(f"\"{input_files[0]}\" is a directory. All hdf5 files will be used.")

        input_files = [os.path.relpath(file, input_files[0]) \
                           for file in os.listdir(input_files[0]) \
                                if ".hdf5" == file[-5:]]
        
    # Prepare the parameters for the experiments
    list_exp_params = [params_from_filename(file) for file in input_files ]
    for filepath, exp_params in zip(input_files, list_exp_params):
        if not 'z' in exp_params:
            exp_params['z'] = args.z
        if not 'z_offset' in exp_params:
            exp_params['z_offset'] = args.z_offset

        exp_params['delta_z'] = args.delta_z
        # Store the experiment parameters with the filename
        auto_id = 'file(' + os.path.basename(filepath)[:-5] + ')_' \
            + "_".join(map(lambda t: f'{t[0]}[{t[1]}]', exp_params.items() ))
        exp_params['name'] = auto_id
    
    # Prepare the output value for al the input experiments
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ['']
    if args.output_prefix is not None:
        output_dir = [os.path.relpath(args.output_prefix + direct) for direct in output_dir]
    # If only one output dir it got repeated
    if len(output_dir) == 1:
        output_dir = [output_dir[0] for file in input_files]


    return input_files, output_dir, list_exp_params


def main(argv):
    args = parse_args(argv)
    input_files, output_dirs, all_exp_params = prepare_data(args)

    for i_file, o_dir, exp_params in zip(input_files, output_dirs, all_exp_params):
        print(f'Loading {i_file}...')
        data = tal.io.read_capture(i_file)
        print('Loaded')


        if data.is_confocal() and 'args' in data.scene_info and 'command' in data.scene_info['args']:
            tal.reconstruct.compensate_laser_cos_dsqr(data)

        print(f'Shifting {args.t_shift} time bins')
        data.H = np.roll(data.H, args.t_shift, axis = 0)

        if args.downscale is not None and args.downscale > 1:
            print(f'Downscaling data by a factor of {args.downscale}')
            data.downscale(args.downscale)

        if not os.path.isdir(o_dir):
            os.mkdir(o_dir)

        print(f'Results will be stored in {o_dir}')

        # data.H = data.H.swapaxes(1,2)

        if args.two_dimensions is not None:
            data.H = data.H[:,args.two_dimensions]
            # Preprare data for the reconstruction
            data.sensor_grid_xyz = data.sensor_grid_xyz[args.two_dimensions]

        rec_and_plot_experiment(data, exp_params, 
                                c_wavelength= args.c_wavelength, 
                                n_cycles = args.n_cycles,
                                n_zp_it = args.zero_phase_it,
                                n_threads = args.threads, 
                                output_dir = o_dir, 
                                no_dense = args.no_dense,
                                reconstruct = not args.only_plot,
                                plot = not args.only_reconstruct)



if __name__ == '__main__':
    main(sys.argv[1:])

