import os
import sys
sys.path.insert(0, '')
from zppf import launch

if __name__ == "__main__":
    directory = '../nlos_dataset/fk_dataset_parsed2tal'
    for file in os.listdir(directory):
            if "dragon" in file:
                filepath = os.path.join(directory, file)
                print(filepath)
                launch([filepath, "-o", "./results/fk_migration", "-t", "1",
                    "--c_wavelength", "0.05" , "-z", "1.4", "--delta_z", "0.01",
                    "--z_offset", "0.2", "--only_reconstruct", "--zero_phase_it", "1"])