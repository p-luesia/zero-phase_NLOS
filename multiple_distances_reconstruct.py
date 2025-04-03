import os
import sys
sys.path.insert(0, '')
from zppf import launch

if __name__ == "__main__":
    directory = '../nlos_dataset/different_depth_planes/hdf5_ln_files/'
    # for z in [1.0, 2.0, 4.0, 8.0]:
    for z in [1.0]:
        # for d in [0]:
        for d in [0, 0.000125, 0.00025, 0.0005, 0.001]:
            # file = f'plane_at_{z+d}.hdf5'
            file = f'plane_at_{z+d}_confocal.hdf5'
            filepath = os.path.join(directory, file)
            query = []
            # if d >= 0:
            #     query = ['--no_dense']

            print(filepath)
            launch([filepath, "-o", "./results/patches_multiple_depths", "-t", "8",
                "--c_wavelength", "0.05" , "--downscale", "2", "-z", str(z), 
                "--z_offset", "0.015", "--only_reconstruct"] + query)