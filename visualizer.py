
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

import matplotlib

class StreakPlotter(object):

    class SubPlotter2D(object):
        def __init__(self, V_2d_coords, dense_V_2d, adaptive_max_V_2d,
                      adaptive_zero_phase_V_2d, ground_truth):
            assert dense_V_2d.ndim == 2 and adaptive_max_V_2d.ndim == 2 \
                    and adaptive_zero_phase_V_2d.ndim == 2 \
                    and V_2d_coords.ndim == 3, "Only works in 2D"
            self.coords = V_2d_coords
            self.dense_V = dense_V_2d
            self.adaptive_max_V = adaptive_max_V_2d
            self.adaptive_zero_V = adaptive_zero_phase_V_2d
            self.ground_truth = ground_truth
            self.title = ''
            self.outputfile = 'reconstruction'
            self.backend = 'svg'
            
        def set_backend(self, str):
            """
            Set the backend. See matplotlib.backends for more information
            """
            self.backend = str


        def plot(self):
            og_backend = matplotlib.get_backend()
            matplotlib.use(self.backend)
            # Plot the results
            min_x, min_y = self.coords[0,0]
            delta_x, delta_y = self.coords[1,1] - self.coords[0,0]

            plotify = lambda d: ((d[:, 0] - min_x) / delta_x,
                                 (d[:, 1] - min_y) / delta_y) 
            
            max_dense_V_idx = np.argmax(self.dense_V, axis = 0)
            z_dense_V = self.coords[max_dense_V_idx, 
                                    np.arange(self.dense_V.shape[1])]  
            printable_max_V = plotify(z_dense_V)
            printable_max_add_V = plotify(self.adaptive_max_V)
            printable_zpp = plotify(self.adaptive_zero_V)
            if self.ground_truth is not None:
                mid_gt = self.ground_truth.shape[0]//2
                interest_idx = np.arange(mid_gt - 10, mid_gt + 11)
                # interest_idx = np.arange(self.dense_V.shape[1])\
                #                 [self.ground_truth > 0]
                valid_ground_truth = self.ground_truth[interest_idx]
                printable_gt = [interest_idx,
                                (valid_ground_truth- min_y)\
                                    / delta_y]
                # printable_gt = [np.arange(self.dense_V.shape[1]),
                #                 (self.ground_truth- min_y)\
                #                     / delta_y]
            
                valid_adaptive_zero_V = self.adaptive_zero_V[interest_idx, 1]
                zero_RMSE = (valid_adaptive_zero_V - valid_ground_truth)**2
                zero_RMSE = np.sum(zero_RMSE)

                dense_RMSE = np.sum(z_dense_V[interest_idx, 1] - valid_ground_truth)**2


            plt.figure()
            fig, ax = plt.subplots(1,3, sharex=True, sharey=True, figsize=(8,20))
            ax[0].set_title('Amplitude')
            ax[0].imshow(np.abs(self.dense_V), cmap = 'hot')
            ytick_label = ax[0].get_yticks().astype(int) * delta_y + min_y
            ax[0].set_yticklabels(["{:0.4f}".format(tick) for tick in ytick_label] )
            if self.ground_truth is not None:
                ax[0].plot(printable_gt[0], printable_gt[1], color = 'purple', 
                           alpha = 0.9)

            ax[1].set_title('Phase')
            ax[1].imshow(np.angle(self.dense_V), cmap = 'bwr')
            if self.ground_truth is not None:
                ax[1].plot(printable_gt[0], printable_gt[1], color = 'purple',
                           alpha = 0.9)


            ax[2].set_title('Found surfaces')
            if self.ground_truth is not None:
                ax[2].plot(printable_gt[0], printable_gt[1], color = 'purple',
                           alpha = 0.9, label= 'Ground truth')
            ax[2].plot(printable_zpp[0], printable_zpp[1], color = 'g',
                       alpha = 0.6, label='Zero phase depth')
            ax[2].plot(printable_max_add_V[0], printable_max_add_V[1], 
                       color = 'b', alpha = 0.6,
                       label='Max in adaptative voxel depth')
            ax[2].plot(printable_max_V[0], printable_max_V[1], 
                       color = 'darkorange', alpha = 0.6,
                       label = 'Max in voxel depth')
            ax[2].legend()


            if self.ground_truth is not None:
                fig.suptitle(self.title + \
                          f'(RMSE max dense:{dense_RMSE}, '\
                         +f' RMSE zero phase{zero_RMSE})')
            else:
                fig.suptitle(self.title)
            plt.savefig(self.outputfile)

            matplotlib.use(og_backend)



    def __init__(self, dense_coords:np.ndarray, dense_V_path:str,
                 adaptive_max_V_path:str, adaptive_zero_phase_V_path:str,
                 ground_truth: np.ndarray|None = None):
        self.coords = dense_coords
        shape = self.coords.shape[1:-1]
        self.dense_V = np.load(dense_V_path)
        self.adaptive_max_V = np.load(adaptive_max_V_path).reshape(shape + (3,))
        self.adaptive_zero_phase_V = np.load(adaptive_zero_phase_V_path)\
                                        .reshape(shape + (3,))
        if ground_truth is not None:
            xv = np.linspace(np.min(self.coords[0,:,0,0]), 
                             np.max(self.coords[0,:,0,0]), 
                             ground_truth.shape[0], 
                             endpoint=True)
            yv = np.linspace(np.min(self.coords[0,0,:,1]), 
                             np.max(self.coords[0,0,:,1]), 
                             ground_truth.shape[1], 
                             endpoint=True)
            self.ground_truth = RegularGridInterpolator((xv, yv),
                                                        ground_truth[:,:,2],
                                                        method="nearest")
        else:
            self.ground_truth = None

    def __getitem__(self, idx: int|slice|tuple):
        # Select the cardinals for the x and y coordinates
        cardinal_coords_idx = [1,2]
        if type(idx) == tuple and idx[0] == slice(None, None, None):
            cardinal_coords_idx = [0,2]
        elif type(idx) == tuple and len(idx) < 2:
            idx = idx + (slice(None,None,None),)
        elif not type(idx) == tuple:
            idx = (idx,slice(None,None,None))


        query_coords_3d = self.coords[(slice(None,None,None),) + idx]
        if self.ground_truth:
            query_ground_truth = self.ground_truth(query_coords_3d[0,:,:2])
        else:
            query_ground_truth = None

        return StreakPlotter.SubPlotter2D(query_coords_3d[..., cardinal_coords_idx],
                                      self.dense_V[(slice(None,None,None),) + idx],
                                      self.adaptive_max_V[idx][..., cardinal_coords_idx],
                                      self.adaptive_zero_phase_V[idx][...,cardinal_coords_idx],
                                      query_ground_truth)