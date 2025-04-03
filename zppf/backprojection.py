import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor as Pool
from functools import partial

def back_project_point(xv, H, sensor_grid, illumination_grid, delta_t):
    sensor_dist = np.linalg.norm(sensor_grid - xv, axis = -1)
    illumination_dist = np.linalg.norm(illumination_grid - xv, axis = -1)
    ill_new_shape = (1,)*sensor_dist.ndim + illumination_dist.shape
    travelled_distance = sensor_dist + illumination_dist.reshape(ill_new_shape)
    
    H_slices = [slice(0, ni, 1) if ni>0 else slice(0,1,2) for ni in H.shape[1:]]
    H_idx = np.mgrid[H_slices]
    t_idx = np.round(travelled_distance/delta_t).astype(int)
    t_idx[t_idx >= H.shape[0]-1] = -1
    return np.sum(H[t_idx.flat, H_idx[0].flat, H_idx[1].flat])


def back_projection(tensor_volume, H, sensor_grid, illumination_grid, 
                    delta_t, n_threads = 1):
    # TODO add H[-1] = 0
    I = np.zeros(tensor_volume.shape[:-1], dtype = np.complex128)
    H_aux = np.append(H, [np.zeros(H.shape[1:])], axis = 0)
    
    if n_threads <= 1:
        for idx in tqdm(range(I.size), 
                        desc = 'reconstructed points', total = I.size):
            I.flat[idx] = back_project_point(tensor_volume.reshape(-1,3)[idx],
                                             H_aux, sensor_grid,
                                             illumination_grid, delta_t)
    else:
        with Pool(n_threads) as p:
            con_bp_point = partial(
                        back_project_point, H = H_aux, sensor_grid = sensor_grid,
                                       illumination_grid = illumination_grid,
                                       delta_t = delta_t)
            I.flat = list(tqdm(p.map(con_bp_point, tensor_volume.reshape(-1,3),
                                    chunksize = I.size // (n_threads**2)),
                                    desc = 'reconstructed points',
                                    total = I.size))

    return I