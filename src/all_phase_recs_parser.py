import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

rec_by_freqs = np.zeros((4001, 544, 256, 1), np.complex128)

for i in tqdm(range(4001), desc = "Loading from disk", total = 4001):
    rec_by_freqs[i] = np.load(f"phase_shift_by_freq/{i}_by_freq_rec.npy")

rec_by_freqs = rec_by_freqs.swapaxes(0,1)[..., 0]

center_line = rec_by_freqs[:,:,128].copy()
angle_by_freq = np.angle(rec_by_freqs)

del rec_by_freqs

zero_phase_by_freqs = np.abs(angle_by_freq) < np.pi/30
del angle_by_freq

all_phases = np.sum(zero_phase_by_freqs.astype(int), axis = 0)
del zero_phase_by_freqs

plt.imshow(all_phases)
plt.show()

plt.plot(center_line.swapaxes(0,1))
plt.show()