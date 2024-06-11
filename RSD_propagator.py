import numpy as np
import tal
import matplotlib.pyplot as plt

import time

class RSD_kernel(object):
    """
    Rayleigh Sommerfeld Diffraction kernels for parallel plane to plane 
    reconstructions
    """

    def __init__(self, origin_grid):
        self.coords_z0 = RSD_kernel.__z_0_kernel_coordinates(origin_grid)
        self.normal = RSD_kernel.__get_normal(origin_grid)

    def get(self, dist, wavelengths):
        # Get the kernel coordinates and distances
        kernel_coords = self.coords_z0 + self.normal*dist
        kernel_dists = np.linalg.norm(kernel_coords, axis = -1)
        ndim = kernel_dists.ndim
        resh_wavelengths = wavelengths.reshape((-1,)+ndim*(1,))
        phase_prop = np.exp(2j*np.pi*kernel_dists/resh_wavelengths)
        return phase_prop #/ kernel_dists

    def __z_0_kernel_coordinates(origin_grid):
        # Extract maximum and minimum distances from the origin grid
        grid_dim = origin_grid.ndim - 1
        # Eye to iterate over a unknown dimensional grid
        tensor_idx = np.eye(grid_dim, dtype=int)
        # Find minimum and maximum (or delta) distances
        max_grid_dirs = origin_grid[(0,)*grid_dim]\
                         - origin_grid[tuple(-1*tensor_idx)]
        max_grid_dist = np.linalg.norm(max_grid_dirs, axis = -1)

        # shape_divisor = np.reciprocal(shape, where=shape > 0)
        delta_grid_dist = max_grid_dist / origin_grid.shape[:-1]

        # Slices for mgrid of dimensions depending on the origin grid
        slices_idx = [slice(-max_i+ delta_i, max_i , delta_i)
                      if not delta_i == 0 else slice(0, 1, 2)
                      for max_i, delta_i in zip(max_grid_dist, delta_grid_dist)]
        # Return the coordinates for the kernel
        return np.array(np.mgrid[slices_idx + [slice(0, 1, 2)]])\
                    .swapaxes(0, -1)[0].copy()


    def __get_normal(origin_grid):
        # Extract the normal from the grid
        squeezed_origin = origin_grid.squeeze()
        grid_dim = origin_grid.squeeze().ndim - 1
        if grid_dim > 1:
            # Eye to iterate over a unknown dimensional grid
            tensor_idx = np.eye(grid_dim, dtype=int)*-1
            max_grid_dirs = origin_grid[tuple(tensor_idx)]\
                            - origin_grid[(0,)*grid_dim]
            normals = [np.cross(max_grid_dirs[i], max_grid_dirs[i+1]) 
                            for i in range(max_grid_dirs.shape[0] - 1) ]
            return np.array(normals) / np.linalg.norm(normals, axis = -1)
        else:
            # Extract one of the normals from the segment
            max_grid_dirs = squeezed_origin[-1] - squeezed_origin[0]
            perpendiculater = np.ones(max_grid_dirs.shape)
            perpendiculater[1::2] = -1
            normal = np.roll(max_grid_dirs, 1) * perpendiculater
            return normal / np.linalg.norm(normal, axis = -1)


def point_to_point_propagator(target_points, origin_point, wavelengths):
    directions = target_points - origin_point
    distances = np.linalg.norm(directions, axis = -1)
    ndim = distances.ndim
    resh_wavelengths = wavelengths.reshape((-1,)+ndim*(1,))
    phase_prop = np.exp(2j*np.pi*distances/resh_wavelengths)
    return phase_prop #/ distances