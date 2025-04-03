import os
import sys
sys.path.insert(0, '')
from zppf import launch

if __name__ == "__main__":
    directory = '../nlos_dataset/two_local_depth_planes/hdf5_ln_files/'
    for file in os.listdir(directory):
        if "depthplanes_z[1]" not in file:
            continue
        
        if "1.000" not in file or "1.0000" in file:
            continue 
        # if "confocal" not in file:
        #     continue
        query=[]
        # query = ["--only_plot"]

        filepath = os.path.join(directory, file)
        print(filepath)
        launch([filepath, "-o", "./results/two_micro_depth_planes", "-t", "8",
                "--c_wavelength", "0.05" , "--downscale", "2", "-z", "1", 
                "--z_offset", "0.015", "--delta_z", "0.0001", "--only_reconstruct"] + query)