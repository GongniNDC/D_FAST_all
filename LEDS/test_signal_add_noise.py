# Define the station and types for processing
import os

import numpy as np
from matplotlib import pyplot as plt
 # Type for px pairs
results_folder = "E:/D_FAST_data/results/pair_sim/snr2/D_FAST"  # Path to the results folder

# Load results from .npy files using os.path.join to construct complete file paths
auto_eq = np.load(os.path.join(results_folder, 'auto_signal.npy'))  # Auto-correlation for eq pairs
cross_eq_noise = np.load(os.path.join(results_folder, "noise_signal.npy"))  # Cross-correlation between eq and noise
auto_noise = np.load(os.path.join(results_folder, "auto_noise.npy"))  # Auto-correlation for noise pairs
# auto_px = np.load(os.path.join(results_folder, "auto_px.npy"))  # Auto-correlation for px pairs
# cross_px_noise = np.load(os.path.join(results_folder, "noise_px.npy"))  # Cross-correlation between px and noise

# Plot the histogram for the results
plt.figure(figsize=(10, 6))  # Set figure size
# Set font size globally for the plot
plt.rcParams.update({'font.size': 16})

# Plot histogram for auto-correlation of eq pairs
plt.hist(auto_eq, np.arange(0.1, 1.02, 0.01), color='b', alpha=0.6, label='eq_pairs', density=True)
# Plot histogram for auto-correlation of noise pairs
plt.hist(auto_noise, np.arange(0.1, 1.02, 0.01), color='green', alpha=0.6, label='noise_pairs', density=True)
# Plot histogram for cross-correlation of eq and noise pairs
plt.hist(cross_eq_noise, np.arange(0.1, 1.02, 0.01), color='y', alpha=0.6, label='eq_noise_pairs', density=True)

# Set labels and ticks for the plot
plt.xlabel('Hamming Similarity')
plt.ylabel('Number of Fingerprint Pairs')
plt.xticks(np.arange(0.1, 1.1, 0.1))  # Set x-ticks
plt.xticks(fontsize=16)  # Set font size for x-ticks
plt.yticks(fontsize=16)  # Set font size for y-ticks
plt.legend(fontsize=16)  # Set font size for legend
plt.grid(False)  # Disable grid
plt.axvline(x=0.6, color='gray', linestyle='--')  # Add a vertical dashed line at x=0.6
plt.savefig(os.path.join(results_folder, 'd_fast.png'))
plt.show()