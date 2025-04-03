import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections.abc import Iterable
import sys
from scipy.interpolate import RegularGridInterpolator 

import tal

# Plot all meter depths python .\zppf\plot_depths_top_view.py --scale 1 --boundaries 0.0 8.5 '.\results\patches_multiple_depths\file(plane_at_1.0)_z[1.0]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' '.\results\patches_multiple_depths\file(plane_at_2.0)_z[2.0]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' '.\results\patches_multiple_depths\file(plane_at_4.0)_z[4.0]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' '.\results\patches_multiple_depths\file(plane_at_8.0)_z[8.0]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' --ground_truth ..\nlos_dataset\different_depth_planes\hdf5_ln_files\plane_at_1.0.hdf5 ..\nlos_dataset\different_depth_planes\hdf5_ln_files\plane_at_2.0.hdf5 ..\nlos_dataset\different_depth_planes\hdf5_ln_files\plane_at_4.0.hdf5 ..\nlos_dataset\different_depth_planes\hdf5_ln_files\plane_at_8.0.hdf5 --downscale 2

# Plot all microdepths at 1 m python .\zppf\plot_depths_top_view.py --scale 1 --boundaries 0.995 1.005 '.\results\patches_multiple_depths\file(plane_at_1.0)_z[1.0]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' '.\results\patches_multiple_depths\file(plane_at_1.001)_z[1.0]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' '.\results\patches_multiple_depths\file(plane_at_1.0005)_z[1.0]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' '.\results\patches_multiple_depths\file(plane_at_1.00025)_z[1.0]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' '.\results\patches_multiple_depths\file(plane_at_1.000125)_z[1.0]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' --ground_truth ..\nlos_dataset\different_depth_planes\hdf5_ln_files\plane_at_1.0.hdf5 ..\nlos_dataset\different_depth_planes\hdf5_ln_files\plane_at_1.001.hdf5 ..\nlos_dataset\different_depth_planes\hdf5_ln_files\plane_at_1.0005.hdf5 ..\nlos_dataset\different_depth_planes\hdf5_ln_files\plane_at_1.00025.hdf5 ..\nlos_dataset\different_depth_planes\hdf5_ln_files\plane_at_1.000125.hdf5 --dense_depthfiles '.\results\patches_multiple_depths\dense_depths\file(plane_at_1.0)_z[1.0]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy' '.\results\patches_multiple_depths\dense_depths\file(plane_at_1.001)_z[1.0]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy' '.\results\patches_multiple_depths\dense_depths\file(plane_at_1.0005)_z[1.0]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy' '.\results\patches_multiple_depths\dense_depths\file(plane_at_1.00025)_z[1.0]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy' '.\results\patches_multiple_depths\dense_depths\file(plane_at_1.000125)_z[1.0]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy' --downscale 2

# Plot all microdepths at 1 m python .\zppf\plot_depths_top_view.py --scale 1 --boundaries 0.995 1.005 '.\results\patches_multiple_depths\file(plane_at_1.0_confocal)_z[1.0]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' '.\results\patches_multiple_depths\file(plane_at_1.001_confocal)_z[1.0]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' '.\results\patches_multiple_depths\file(plane_at_1.0005_confocal)_z[1.0]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' '.\results\patches_multiple_depths\file(plane_at_1.00025_confocal)_z[1.0]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' '.\results\patches_multiple_depths\file(plane_at_1.000125_confocal)_z[1.0]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' --ground_truth ..\nlos_dataset\different_depth_planes\hdf5_ln_files\plane_at_1.0_confocal.hdf5 ..\nlos_dataset\different_depth_planes\hdf5_ln_files\plane_at_1.001_confocal.hdf5 ..\nlos_dataset\different_depth_planes\hdf5_ln_files\plane_at_1.0005_confocal.hdf5 ..\nlos_dataset\different_depth_planes\hdf5_ln_files\plane_at_1.00025_confocal.hdf5 ..\nlos_dataset\different_depth_planes\hdf5_ln_files\plane_at_1.000125_confocal.hdf5 --downscale 2 --dense_depthfiles '.\results\patches_multiple_depths\dense_depths\file(plane_at_1.0_confocal)_z[1.0]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy' '.\results\patches_multiple_depths\dense_depths\file(plane_at_1.001_confocal)_z[1.0]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy' '.\results\patches_multiple_depths\dense_depths\file(plane_at_1.0005_confocal)_z[1.0]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy' '.\results\patches_multiple_depths\dense_depths\file(plane_at_1.00025_confocal)_z[1.0]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy' '.\results\patches_multiple_depths\dense_depths\file(plane_at_1.000125_confocal)_z[1.0]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy'

# Plot all local distances 1 m python .\zppf\plot_depths_top_view.py --scale 1 --boundaries 0.995 1.005 '.\results\two_micro_depth_planes\file(depthplanes_z[1]_zdiff[1])_z[1.0]_zdiff[1.0]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' '.\results\two_micro_depth_planes\file(depthplanes_z[1]_zdiff[1.001])_z[1.0]_zdiff[1.001]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' '.\results\two_micro_depth_planes\file(depthplanes_z[1]_zdiff[1.0005])_z[1.0]_zdiff[1.0005]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' '.\results\two_micro_depth_planes\file(depthplanes_z[1]_zdiff[1.00025])_z[1.0]_zdiff[1.00025]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' '.\results\two_micro_depth_planes\file(depthplanes_z[1]_zdiff[1.000125])_z[1.0]_zdiff[1.000125]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' --ground_truth ..\nlos_dataset\two_local_depth_planes\hdf5_ln_files\depthplanes_z[1]_zdiff[1].hdf5 ..\nlos_dataset\two_local_depth_planes\hdf5_ln_files\depthplanes_z[1]_zdiff[1.001].hdf5 ..\nlos_dataset\two_local_depth_planes\hdf5_ln_files\depthplanes_z[1]_zdiff[1.0005].hdf5 ..\nlos_dataset\two_local_depth_planes\hdf5_ln_files\depthplanes_z[1]_zdiff[1.00025].hdf5 ..\nlos_dataset\two_local_depth_planes\hdf5_ln_files\depthplanes_z[1]_zdiff[1.000125].hdf5 --downscale 2 --dense_depthfiles '.\results\two_micro_depth_planes\dense_depths\file(depthplanes_z[1]_zdiff[1])_z[1.0]_zdiff[1.0]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy' '.\results\two_micro_depth_planes\dense_depths\file(depthplanes_z[1]_zdiff[1.001])_z[1.0]_zdiff[1.001]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy' '.\results\two_micro_depth_planes\dense_depths\file(depthplanes_z[1]_zdiff[1.0005])_z[1.0]_zdiff[1.0005]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy' '.\results\two_micro_depth_planes\dense_depths\file(depthplanes_z[1]_zdiff[1.00025])_z[1.0]_zdiff[1.00025]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy' '.\results\two_micro_depth_planes\dense_depths\file(depthplanes_z[1]_zdiff[1.000125])_z[1.0]_zdiff[1.000125]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy'

# Plot all local distances confocal 1 m python .\zppf\plot_depths_top_view.py --scale 1 --boundaries 0.995 1.005 '.\results\two_micro_depth_planes\file(depthplanes_z[1]_zdiff[1]_confocal)_z[1.0]_zdiff[1.0]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' '.\results\two_micro_depth_planes\file(depthplanes_z[1]_zdiff[1.001]_confocal)_z[1.0]_zdiff[1.001]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' '.\results\two_micro_depth_planes\file(depthplanes_z[1]_zdiff[1.0005]_confocal)_z[1.0]_zdiff[1.0005]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' '.\results\two_micro_depth_planes\file(depthplanes_z[1]_zdiff[1.00025]_confocal)_z[1.0]_zdiff[1.00025]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' '.\results\two_micro_depth_planes\file(depthplanes_z[1]_zdiff[1.000125]_confocal)_z[1.0]_zdiff[1.000125]_z_offset[0.015]_delta_z[0.0001]zero_phase_coords.npy' --ground_truth ..\nlos_dataset\two_local_depth_planes\hdf5_ln_files\depthplanes_z[1]_zdiff[1]_confocal.hdf5 ..\nlos_dataset\two_local_depth_planes\hdf5_ln_files\depthplanes_z[1]_zdiff[1.001]_confocal.hdf5 ..\nlos_dataset\two_local_depth_planes\hdf5_ln_files\depthplanes_z[1]_zdiff[1.0005]_confocal.hdf5 ..\nlos_dataset\two_local_depth_planes\hdf5_ln_files\depthplanes_z[1]_zdiff[1.00025]_confocal.hdf5 ..\nlos_dataset\two_local_depth_planes\hdf5_ln_files\depthplanes_z[1]_zdiff[1.000125]_confocal.hdf5 --downscale 2 --dense_depthfiles '.\results\two_micro_depth_planes\dense_depths\file(depthplanes_z[1]_zdiff[1]_confocal)_z[1.0]_zdiff[1.0]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy' '.\results\two_micro_depth_planes\dense_depths\file(depthplanes_z[1]_zdiff[1.001]_confocal)_z[1.0]_zdiff[1.001]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy' '.\results\two_micro_depth_planes\dense_depths\file(depthplanes_z[1]_zdiff[1.0005]_confocal)_z[1.0]_zdiff[1.0005]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy' '.\results\two_micro_depth_planes\dense_depths\file(depthplanes_z[1]_zdiff[1.00025]_confocal)_z[1.0]_zdiff[1.00025]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy' '.\results\two_micro_depth_planes\dense_depths\file(depthplanes_z[1]_zdiff[1.000125]_confocal)_z[1.0]_zdiff[1.000125]_z_offset[0.015]_delta_z[0.0001]dense_reconstruction.npy'

def prepare_gt(gt_filename, downscale):
    d = tal.io.read_capture(gt_filename)

    assert "ground_truth" in d.scene_info.keys(), "Data without ground truth"
    gt = d.scene_info["ground_truth"]['depth']
    # Grid interpolator to match points  
    xv = np.linspace(1, -1, gt.shape[0])
    yv = np.linspace(1, -1, gt.shape[1])
    rg_interp = RegularGridInterpolator((xv, yv), gt[:,:,2].swapaxes(0,1),
                                            method="nearest")
    
    query_grid = d.sensor_grid_xyz[:,:,:2]
    if downscale > 1:
        query_grid = query_grid[::downscale, ::downscale]

    adjusted_gt = rg_interp(query_grid.reshape(-1,2))
    adjusted_gt = adjusted_gt.reshape(query_grid.shape[:2])
    return adjusted_gt

if __name__ == '__main__':
    parser = ArgumentParser('plot_depths',
            description = 'Plot the reconstructed depths of a Phasor Fields '\
                         +'reconstruction and its Zero Phase reconstruction.')
    parser.add_argument('filenames', nargs='+', type = str, 
                        help = 'Dense reconstruction and ZP coords in npy.')
    parser.add_argument('--boundaries', nargs = 2, type = float,
                        metavar='LOWER_BOUND, UPPER_BOUND',
                        help = 'Boundaries of the reconstruction')
    
    parser.add_argument('--scale', nargs = 1, type = float, default = 1,
                        help = "Define the scale of the plot")

    parser.add_argument('--ground_truth', nargs = '*', type = str, 
                        metavar='GROUNDTRUTH_FILE',
                        help = 'If indicated, plot ground truth in the '\
                            + 'GROUNDTRUTH_FILE hdf5 file')
    parser.add_argument('--dense_depthfiles', nargs = '*', type = str,
                        help = 'If indicated, plot dense depth truth in the '\
                            + 'indicated npy files')
    parser.add_argument('--downscale', nargs = 1, type = int, 
                        metavar='DOWNSCALE_FACTOR', default=[1],
                        help = 'If greater than 1, assume data reconstructed '\
                                + ' is downscaled by that factor')


    

    args = parser.parse_args(sys.argv[1:])

    

    # Load the ground truth if indicated
    gt_files = None
    if args.ground_truth is not None:
        gt_files = args.ground_truth

    scale = args.scale[0]

    densedepth_files = None
    if args.dense_depthfiles is not None:
        densedepth_files = args.dense_depthfiles

    filenames = args.filenames
    for i, file in enumerate(filenames):
        depths = np.load(file)[:,:,2]
        selected_slide = depths.shape[0]//2
        depths_v = depths[:, selected_slide]

        ground_truth = None
        mask = np.array(depths_v.shape[0], dtype = bool).fill(True)
        if gt_files is not None:
            ground_truth = prepare_gt(args.ground_truth[i], args.downscale[0])\
                            [:, selected_slide]
            mask = ground_truth > 0.01

        plt.figure()
        plt.xlim((-1, 1))
        if args.boundaries is not None:
            plt.ylim((args.boundaries[0]/scale, args.boundaries[1]/scale))

        xv = np.linspace(-1,1, depths_v.shape[0])
        plt.plot(xv[mask], depths_v[mask]/scale, '-o', color = 'forestgreen', label = 'Depth estimation', alpha = 0.8)
        if ground_truth is not None:
            plt.plot(xv[mask], ground_truth[mask]/scale, 'blueviolet', label = 'Ground truth', alpha = 0.8)

        if densedepth_files is not None:
            dense_depth = np.load(densedepth_files[i])
            plt.plot(xv[mask], dense_depth[mask, selected_slide], '-^', color = 'orange', label = 'Dense depth', alpha = 0.8)

        plt.legend()
        plt.grid(visible = True)
        title = file.split('/')[-1].split('\\')[-1]
        plt.title(file.split('/')[-1].split('\\')[-1])
        plt.savefig( title[:-4] + '.svg')

        if i == 0:
            depth_base = depths_v[mask]
            gt_base = ground_truth[mask]
        print(f'Mean distance: {np.mean(depths_v[mask])}')
        print(f'Difference:{np.mean(np.abs(depth_base - depths_v[mask]))}')
        print(f'Difference gt:{np.mean(np.abs(gt_base - ground_truth[mask]))}')
        print(f'GT difference: {np.mean(np.abs(ground_truth[mask] - depths_v[mask]))}')

    plt.show()
