import os
import sys
import tal
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


o_dir = './results/phase_shift_analysis/'
if __name__ == "__main__":
    directory = '../nlos_dataset/phase_analysis/'
    for src_files in os.listdir(directory):
        full_src_files = os.path.join(directory, src_files)
        if os.path.isdir(full_src_files):
            for file in os.listdir(full_src_files):
                if 'hdf5' in file:
                    print(file)
                    filepath = os.path.join(full_src_files, file)
                    d = tal.io.read_capture(filepath)

                    if d.is_confocal():
                        tal.reconstruct.compensate_laser_cos_dsqr(d)
                        d.laser_grid_xyz = d.laser_grid_xyz[:, 64:65]

                    d.H = d.H[:, :, 64:65]
                    d.sensor_grid_xyz = d.sensor_grid_xyz[:, 64:65]
                    
                    zv = np.linspace(0.9, 1.412, 256, endpoint = False)
                    V = tal.reconstruct.get_volume_project_rw(d, zv)
                    rec = tal.reconstruct.pf.solve(d, 0.05, 5, 
                                            tal.enums.CameraSystem.DIRECT_LIGHT,
                                            V)
                    o_file = file[:-5] + f'_delta_z_{zv[1]- zv[0]}_starts_{zv[0]}'
                    np.save(o_dir + o_file, rec)

                    gt = d.scene_info["ground_truth"]['depth']
                    
                    # Grid interpolator to match points  
                    xv = np.linspace(1, -1, gt.shape[0])
                    yv = np.linspace(1, -1, gt.shape[1])
                    rg_interp = RegularGridInterpolator((xv, yv), gt[:,:,2].swapaxes(0,1),
                                                        method="nearest")
                    query_grid = d.sensor_grid_xyz[:,:,:2]
                    
                    adjusted_gt = rg_interp(query_grid.reshape(-1,2))
                    adjusted_gt = adjusted_gt.reshape(query_grid.shape[:2])

                    mask = adjusted_gt > 0.9

                    extent = (V[0,0,0,0], V[-1,0,-1, 0], 
                              V[0,0,0, 2], V[-1,0,-1, 2])
                    
                    # extent = (5, 10, 3, 1)

                    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
                    figManager = plt.get_current_fig_manager()
                    figManager.window.showMaximized()
                    fig.suptitle(file)
                    im1 = axs[0].imshow(np.abs(rec[:,0]).T, cmap='hot', extent = extent, aspect = 4, origin='lower')
                    axs[0].set_title('Absolute')
                    axs[0].plot(query_grid[mask, 0], adjusted_gt[mask], color='purple', label = 'Ground Truth')
                    axs[0].legend()
                    im2 = axs[1].imshow(np.angle(rec[:,0]).T, cmap='BrBG', extent = extent, aspect = 4, origin='lower')
                    axs[1].plot(query_grid[mask, 0], adjusted_gt[mask], color='purple', label = 'Ground Truth')
                    axs[1].legend()
                    axs[1].set_title('Phase')
                    fig.colorbar(im1)
                    fig.colorbar(im2)
                    plt.savefig(o_dir + o_file + '.svg')

