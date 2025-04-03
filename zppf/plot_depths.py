import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections.abc import Iterable
import sys
from scipy.interpolate import RegularGridInterpolator 
import os

import tal

output_filename = ""

def prepare_depth_plots(dense, zp_coords, m_thres, delta_z, z_offset):
    dense_max = np.max(np.abs(dense), axis = 0)
    dense_idx = np.argmax(np.abs(dense), axis = 0)
    dense_depth = dense_idx *delta_z + z_offset
    mask = dense_max <= np.max(dense_max)*m_thres

    ims = []
    # Plot dense results
    fig1, axsf1 = plt.subplots(1,2, sharex=True, sharey=True, figsize=(21, 14))
    fig1.suptitle('Dense results')
    ims.append(axsf1[0].imshow(dense_max, cmap='hot')) 
    axsf1[0].set_title('Dense PF image') 
    ims.append(axsf1[1].imshow(np.ma.masked_array(dense_max, mask), cmap='hot')) 
    axsf1[1].set_title('Masked dense PF image')
    plt.savefig(output_filename + '_dense_hm.svg')

    # Plot phase of maximum plane results
    fig2, axsf2 = plt.subplots(1,2, sharex=True, sharey=True)
    max_idx = dense_idx.flat[np.argmax(dense_max)]
    phase_max = np.angle(dense[max_idx])
    fig2.suptitle('Dense phase results')
    ims.append(axsf2[0].imshow(phase_max, cmap='bwr')) 
    axsf2[0].set_title('Dense phase PF') 
    ims.append(axsf2[1].imshow(np.ma.masked_array(phase_max, mask), cmap='bwr')) 
    axsf2[1].set_title('Masked dense phase PF')

    # Plot unmasked depths
    f_vmin = np.min([dense_depth, zp_coords[...,2]])
    f_vmax = np.max([dense_depth, zp_coords[...,2]])
    fig3, axsf3 = plt.subplots(1,2, sharex=True, sharey=True)
    fig3.suptitle('Unmasked depths')
    ims.append(axsf3[0].imshow(dense_depth, vmin=f_vmin, vmax=f_vmax, cmap='nipy_spectral'))
    axsf3[0].set_title('Dense PF depth')
    ims.append(axsf3[1].imshow(zp_coords[...,2], vmin=f_vmin, vmax=f_vmax, cmap='nipy_spectral'))
    axsf3[1].set_title('Zero-Phase PF depth')

    # Plot masked depths
    ma_dense_depth = np.ma.masked_array(dense_depth, mask)
    ma_zp_depth = np.ma.masked_array(zp_coords[...,2], mask)
    ma_vmin = np.ma.min([ma_dense_depth, ma_zp_depth])
    ma_vmax = np.ma.max([ma_dense_depth, ma_zp_depth])
    fig4, axsf4 = plt.subplots(1,2, sharex=True, sharey=True)
    fig4.suptitle('Masked depths')
    ims.append(axsf4[0].imshow(ma_dense_depth, vmin=ma_vmin, vmax=ma_vmax, cmap='nipy_spectral'))
    axsf4[0].set_title('Dense PF depth')
    ims.append(axsf4[1].imshow(ma_zp_depth, vmin=ma_vmin, vmax=ma_vmax, cmap='nipy_spectral'))
    axsf4[1].set_title('Zero-Phase PF depth')   

    [plt.colorbar(im) for im in ims]

    return dense_depth

def prepare_baseline_comparison_plots(dense_depth, zp_coords, baseline,
                                      mask_threshold, delta_z_bl, offset_z_bl):
    baseline_max = np.max(np.abs(baseline), axis = 0)
    baseline_idx = np.argmax(np.abs(baseline), axis = 0)
    baseline_depth = baseline_idx *delta_z_bl + offset_z_bl
    mask = baseline_max <= np.max(baseline_max)*mask_threshold

    ims = []
    # Plot dense results
    fig1, axsf1 = plt.subplots(1,2, sharex=True, sharey=True, figsize=(21,14))
    fig1.suptitle('Dense baseline')
    ims.append(axsf1[0].imshow(baseline_max, cmap='hot')) 
    axsf1[0].set_title('Baseline image') 
    ims.append(axsf1[1].imshow(np.ma.masked_array(baseline_max, mask), cmap='hot')) 
    axsf1[1].set_title('Masked baseline image')
    plt.savefig(output_filename + '_baseline_hm.svg')


    # Plot depth comparison with the baseline
    ma_dense_depth = np.ma.masked_array(dense_depth, mask)
    ma_zp_depth = np.ma.masked_array(zp_coords[...,2], mask)
    ma_baseline_depth = np.ma.masked_array(baseline_depth, mask)
    ma_vmin = np.ma.min([ma_dense_depth, ma_baseline_depth, ma_zp_depth])
    ma_vmax = np.ma.max([ma_dense_depth, ma_baseline_depth, ma_zp_depth])
    # ma_vmin = 1.25
    # ma_vmax = 1.46


    fig2, axsf2 = plt.subplots(2,3, sharex=True, sharey=True, figsize=(21,14))
    fig2.suptitle('Masked comparison with baseline')
    ims.append(axsf2[0][0].imshow(ma_dense_depth, vmin=ma_vmin, vmax=ma_vmax, cmap='nipy_spectral'))
    axsf2[0][0].set_title('Dense PF depth')
    ims.append(axsf2[0][1].imshow(ma_zp_depth, vmin=ma_vmin, vmax=ma_vmax, cmap='nipy_spectral'))
    axsf2[0][1].set_title('Zero-Phase PF depth')  
    ims.append(axsf2[0][2].imshow(ma_baseline_depth, vmin=ma_vmin, vmax=ma_vmax, cmap='nipy_spectral'))
    axsf2[0][2].set_title('Baseline depth') 

    diff_ma_dense_depth = (ma_dense_depth - ma_baseline_depth)**2
    diff_ma_zp_depth = (ma_zp_depth - ma_baseline_depth)**2
    # diff_vmin = np.min([np.min(np.sqrt(diff_ma_dense_depth)), np.min(np.sqrt(diff_ma_zp_depth))])
    # diff_vmax = np.max([np.max(np.sqrt(diff_ma_dense_depth)), np.max(np.sqrt(diff_ma_zp_depth))])
    diff_vmin = 0.0
    diff_vmax = 0.1

    ims.append(axsf2[1][0].imshow(np.sqrt(diff_ma_dense_depth), vmin=diff_vmin, vmax=diff_vmax, cmap='nipy_spectral'))
    axsf2[1][0].set_title(f'Comparison PF depth \n (MRSE: {np.mean(np.sqrt(diff_ma_dense_depth))}, std: {np.std(np.sqrt(diff_ma_dense_depth))}))')
    ims.append(axsf2[1][1].imshow(np.sqrt(diff_ma_zp_depth), vmin=diff_vmin, vmax=diff_vmax, cmap='nipy_spectral'))
    axsf2[1][1].set_title(f'Comparison Zero-Phase PF depth \n (MRSE: {np.mean(np.sqrt(diff_ma_zp_depth))}, std: {np.std(np.sqrt(diff_ma_zp_depth))}))') 

    [plt.colorbar(im) for im in ims]
    plt.savefig(output_filename + '_baseline_depth_comparison.svg')

def prepare_ground_truth_comparison_plots(dense_depth, zp_depth, ground_truth):
    mask = ground_truth < 0.05

    ims = []
    # Plot dense results
    fig1, axsf1 = plt.subplots(2, 3, sharex=True, sharey=True)

    ma_dense_depth = np.ma.masked_array(dense_depth, mask)
    ma_zp_depth = np.ma.masked_array(zp_depth[:,:,2], mask)
    ma_gt_depth = np.ma.masked_array(ground_truth, mask)

    fig1.suptitle('Ground truth comparison')
    depth_max = np.max([np.max(ma_gt_depth), np.max(ma_zp_depth), np.max(ma_dense_depth) ])
    depth_min = np.min([np.min(ma_gt_depth), np.min(ma_zp_depth), np.min(ma_dense_depth)])

    ims.append(axsf1[0][0].imshow(ma_gt_depth, cmap='nipy_spectral', vmax=depth_max, vmin=depth_min)) 
    axsf1[0][0].set_title('Ground truth depth') 
    ims.append(axsf1[0][1].imshow(ma_dense_depth, cmap='nipy_spectral', vmax=depth_max, vmin=depth_min)) 
    axsf1[0][1].set_title('Dense depth')
    ims.append(axsf1[0][2].imshow(ma_zp_depth, cmap='nipy_spectral', vmax=depth_max, vmin=depth_min)) 
    axsf1[0][2].set_title('ZPPF depth')

    diff_ma_dense_depth = (ma_gt_depth - ma_dense_depth)**2
    diff_ma_zp_depth = (ma_gt_depth - ma_zp_depth)**2


    diff_vmin = np.min([np.min(np.sqrt(diff_ma_dense_depth)), np.min(np.sqrt(diff_ma_zp_depth))])
    diff_vmax = np.max([np.max(np.sqrt(diff_ma_dense_depth)), np.max(np.sqrt(diff_ma_zp_depth))])

    ims.append(axsf1[1][1].imshow(np.sqrt(diff_ma_dense_depth), vmin=diff_vmin, vmax=diff_vmax, cmap='nipy_spectral'))
    axsf1[1][1].set_title(f'Comparison PF depth (RMSE: {np.mean(np.sqrt(diff_ma_dense_depth))}, std: {np.std(np.sqrt(diff_ma_dense_depth))})')
    ims.append(axsf1[1][2].imshow(np.sqrt(diff_ma_zp_depth), vmin=diff_vmin, vmax=diff_vmax, cmap='nipy_spectral'))
    axsf1[1][2].set_title(f'Comparison Zero-Phase PF depth (RMSE: {np.mean(np.sqrt(diff_ma_zp_depth))}, std: {np.std(np.sqrt(diff_ma_zp_depth))})') 

    [plt.colorbar(im) for im in ims]


if __name__ == '__main__':
    parser = ArgumentParser('plot_depths',
            description = 'Plot the reconstructed depths of a Phasor Fields '\
                         +'reconstruction and its Zero Phase reconstruction.')
    parser.add_argument('filenames', nargs=2, type = str, 
                        help = 'Dense reconstruction and ZP coords in npy.')
    parser.add_argument('-m', '--mask_threshold', nargs = 1, type = float,
                        default = 0.2,
                        help = 'Threshold for the masked arrays.')
    parser.add_argument('-d', '--delta_z', nargs = 1, type = float,
                        default = 0.001,
                        help = 'Distance between z bins in the dense reconstruction')
    parser.add_argument('-o', '--offset_z', nargs = 1, type = float,
                        default = 1,
                        help = 'Distance the reconstruction starts')
    parser.add_argument('--baseline', nargs = 3, type = str, 
                        metavar=('BASELINE_FILE', 'DELTA_Z_BASELINE', 'OFFSET_Z_BASELINE'),
                        help = 'If indicated, plot the file of the '\
                            + 'BASELINE_FILE, and its difference with'\
                            + ' the ZP coords. Otherwise uses the dense')
    parser.add_argument('--ground_truth', nargs = 1, type = str, 
                        metavar='GROUNDTRUTH_FILE',
                        help = 'If indicated, plot ground truth in the '\
                            + 'GROUNDTRUTH_FILE hdf5 file')
    parser.add_argument('--downscale', nargs = 1, type = int, 
                        metavar='DOWNSCALE_FACTOR', default=[1],
                        help = 'If greater than 1, assume data reconstructed '\
                                + ' is downscaled by that factor')
    parser.add_argument('--no_baseline', action = 'store_true',
                        help = 'If indicated does not plot the baseline comparison')
    
    args = parser.parse_args(sys.argv[1:])
    filenames = args.filenames
    dense = np.load(filenames[0])
    zp_coords = np.load(filenames[1])

    src = os.path.split(filenames[1])[0]
    file = os.path.split(filenames[1])[1]
    file = file.split("zero_phase_coords")[0]
    output_filename = os.path.join(src, file)

    plot_baseline = False
    if args.baseline is not None:
        baseline_filename, delta_z_bl_str, offset_z_bl_str = args.baseline
        baseline = np.load(baseline_filename)
        delta_z_bl = float(delta_z_bl_str)
        offset_z_bl = float(offset_z_bl_str)
        plot_baseline = True

    threshold = args.mask_threshold
    if isinstance(threshold, Iterable):
        threshold = threshold[0]

    dense_depth = prepare_depth_plots(dense, zp_coords, threshold, 
                                      args.delta_z[0], args.offset_z[0])
    
    if plot_baseline and not args.no_baseline:
        prepare_baseline_comparison_plots(dense_depth, zp_coords, baseline,
                                          threshold, delta_z_bl, 
                                          offset_z_bl)
    elif not args.no_baseline:
        prepare_baseline_comparison_plots(dense_depth, zp_coords, dense,
                                            threshold, args.delta_z[0], 
                                            args.offset_z[0])
    
    if args.ground_truth is not None:
        d = tal.io.read_capture(args.ground_truth[0])

        assert "ground_truth" in d.scene_info.keys(), "Data without ground truth"
        gt = d.scene_info["ground_truth"]['depth']
        # Grid interpolator to match points  
        xv = np.linspace(1, -1, gt.shape[0])
        yv = np.linspace(1, -1, gt.shape[1])
        rg_interp = RegularGridInterpolator((xv, yv), gt[:,:,2].swapaxes(0,1),
                                             method="nearest")
        query_grid = d.sensor_grid_xyz[:,:,:2]
        if args.downscale[0] > 1:
            query_grid = query_grid[::args.downscale[0], ::args.downscale[0]]
        
        adjusted_gt = rg_interp(query_grid.reshape(-1,2))
        adjusted_gt = adjusted_gt.reshape(query_grid.shape[:2])

        prepare_ground_truth_comparison_plots(dense_depth, zp_coords,
                                               adjusted_gt)
        # prepare_ground_truth_comparison_plots(zp_coords[:,:,2], zp_coords,
        #                                        adjusted_gt)
        
    # plt.show()

    plt.savefig(output_filename + '.svg')

