import numpy as np
from concurrent.futures import ThreadPoolExecutor as Pool
from multiprocessing import Manager

from RSD_propagator import RSD_kernel, point_to_point_propagator
from functools import partial
from tqdm import tqdm

PF_by_point = False

def set_by_point():
    global PF_by_point
    PF_by_point = True

def set_planar_RSD():
    global PF_by_point
    PF_by_point = False

def phasor_fields_filter(central_wavelength, n_pulses, tal_data, 
                         fourier_data = False, analysis={}):
    pulse_sigma = n_pulses*central_wavelength / 6
    angular_freqs = 2*np.pi*np.fft.fftfreq(tal_data.H.shape[0],
                                           tal_data.delta_t)
    # Eq. 17 from Liu, X., Bauer, S. & Velten, A. Phasor field diffraction based
    # reconstruction for fast non-line-of-sight imaging systems. Nat Commun 11, 
    # 1645 (2020). https://doi.org/10.1038/s41467-020-15157-4
    exp_pow =-0.5*(pulse_sigma*(angular_freqs - 2*np.pi/central_wavelength))**2
    pulse_in_fourier = pulse_sigma*np.sqrt((2*np.pi)**3) * np.exp(exp_pow)

    idx = np.argwhere(pulse_in_fourier > 1e-2).squeeze()
    if fourier_data:
        return (2*np.pi/angular_freqs[idx], pulse_in_fourier[idx], idx)
    else:
        fH = np.fft.fft(tal_data.H, axis = 0)[idx]
        return (2*np.pi/angular_freqs[idx], pulse_in_fourier[idx], fH)


def __reconstruct_pf_plane(RSD_prop, wavelengths, coords, xl, fH, freq_weights,
                           image, grid_z, analysis = {}, idx_z = 0):
    z = grid_z[idx_z]
    kernels = RSD_prop.get(z, wavelengths)
    center_prop = point_to_point_propagator(coords[idx_z], xl, wavelengths)

    # Apply the convolution with the RSD kernels
    fV = np.fft.fft2(fH, s = (2*fH.shape[1]-1, 2*fH.shape[2]-1), 
                        axes = (-2,-1))\
        * np.fft.fft2(kernels, axes = (-2,-1))
    # Apply the center propagation and return the reconstruction at time 0
    image[idx_z] = np.sum(np.fft.ifft2(fV, axes = (-2,-1))\
                                    [..., -fH.shape[1]:, -fH.shape[2]:]\
                          *freq_weights.reshape(-1, 1, 1)*center_prop, axis = 0)
    analysis['kernels used'] += kernels.shape[1]*kernels.shape[2]
    return 0


def __reconstruct_pf_plane_by_point(grid_coords, rec_coords, wavelengths, xl,
                                    fH, freq_weights, image, grid_z, 
                                    analysis = {}, idx_z = 0):
    l2v_prop = point_to_point_propagator(rec_coords[idx_z], xl, wavelengths)
    for it in np.ndindex(grid_coords.shape[:-1]):
        xv = rec_coords[(idx_z,) + it]
        s2v_prop = point_to_point_propagator(grid_coords, xv, wavelengths)
        half_prop = np.sum(s2v_prop * fH, axis = tuple(range(1, fH.ndim )))
        image[(idx_z,) + it] = np.sum(half_prop*l2v_prop[(slice(None),)+it]\
                                      *freq_weights)

    return 0


def phasor_fields_reconstruction(data, central_wavelength, n_pulses, z_begin,
                                 z_end, delta_z, xl, fH_all = None, 
                                 RSD_prop = None, analysis={}, n_threads = 1):
    
    z_grid = np.mgrid[z_begin:z_end:delta_z]

    # Generate filter
    if fH_all is not None:
        wavelengths, freq_weights, f_idx = phasor_fields_filter(
                                                    central_wavelength, 
                                                    n_pulses, 
                                                    data,
                                                    fourier_data = True)
        fH = fH_all[f_idx]
    else:
        wavelengths, freq_weights, fH = phasor_fields_filter(central_wavelength, 
                                                            n_pulses,
                                                            data,
                                                            fourier_data= False)
    # Reconstruction grid
    coords = np.array(np.meshgrid(data.sensor_grid_xyz[:,0,0],
                                  data.sensor_grid_xyz[0,:,1],
                                  z_grid)).swapaxes(0, -1).swapaxes(1,2)
    
    # Extract propagators
    global PF_by_point
    if RSD_prop is None and not PF_by_point:
        RSD_prop = RSD_kernel(data.sensor_grid_xyz)

    image = np.zeros(coords.shape[:-1], dtype = np.complex128)

    # Iterate over planes
    analysis_concurrent = Manager().dict()
    analysis_concurrent['kernels used'] = 0
    
    if PF_by_point:
        plane_rec_func = partial(__reconstruct_pf_plane_by_point,
                                 data.sensor_grid_xyz, coords, wavelengths, xl, 
                                 fH, freq_weights, image, z_grid, 
                                 analysis_concurrent)
    else:
        plane_rec_func = partial(__reconstruct_pf_plane, RSD_prop, wavelengths, 
                                 coords, xl, fH, freq_weights, image, z_grid, 
                                 analysis_concurrent)

    if n_threads > 1:
        # Concurrent
        with Pool(n_threads) as p:
            list(tqdm( p.map( plane_rec_func, range(len(z_grid)) ), 
                      total = len(z_grid),
                      desc = 'Reconstructed'))
    else:
        # Iterative
        for idx_z in tqdm(range(len(z_grid)), total = len(z_grid),
                          desc='reconstruction'):
            plane_rec_func(idx_z)

    analysis['kernels used'] = analysis_concurrent['kernels used']        
    analysis['reconstructed voxels'] = image.ravel().shape[0]
    return image