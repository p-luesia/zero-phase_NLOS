from .phasor_fields import phasor_fields_reconstruction
from .RSD_propagator import RSD_kernel
from tal.io import NLOSCaptureData
import numpy as np

import tal

def reconstruct(data:NLOSCaptureData, central_wavelength:float, n_cycles:float,
                z_v: np.ndarray, xl:np.ndarray = None, 
                analysis:dict = {}, n_threads:int = 1):
    """
    Given data, it reconstruct parallel planes starting at z_begin
    and finishing at z_end, spaced at delta_z with a Phasor Fields 
    reconstruction with a gaussian pulse.
    :param data:                Y-tal format data containging the transient 
                                impulse response
    :param central_wavelength:  Defines the central wavelength of the virtual
                                gaussian illumination function for Phasor Fields.
    :param n_cycles:            Number of cycles of the central wavelength of 
                                the gaussian illumination function for Phasor
                                Fields. It defines the width of the pulse.
    :param z_v:                 Distances to the planes to reconstruct
    :param xl:                  Location of a single illumination point. It can
                                be not used if data capture is confocal.
    :param fH_all:              Impulse response previously filtered by the 
                                Phasor Fields virtual illumination. Optional.
    :param RSD_prop:            Rayleigh Sommerfeld Diffraction kernels to 
                                perform the propagation into the reconstruction
                                volume. Optional.
    :param analyisis:           Dictionary used to store parameters of the
                                reconstruction.
    :params n_threads:          Number of threads to use for the reconstruction.
    """
    V = tal.reconstruct.get_volume_project_rw(data, z_v)
    tal.set_resources(cpu_processes = n_threads)
    solution =  tal.reconstruct.pf.solve(data,
                                        central_wavelength, n_cycles,
                                        camera_system=\
                                            tal.enums.CameraSystem.DIRECT_LIGHT,
                                        volume = V )
    return np.moveaxis(solution, 2, 0)
    return phasor_fields_reconstruction(data, central_wavelength, n_cycles,
                                        z_begin, z_end, delta_z, xl, fH_all, 
                                        RSD_prop, analysis, n_threads) 