import os
import numpy as np
import matplotlib.pyplot as plt

results_folder = "E:/D_FAST_data/results/pair_sim/snr2/fast"  # Path to the results folder

# Load results from .npy files using os.path.join to construct complete file paths
auto_eq = np.load(os.path.join(results_folder, "auto_signal.npy"))  # Auto-correlation for eq pairs
cross_eq_noise = np.load(os.path.join(results_folder, "noise_signal.npy"))  # Cross-correlation between eq and noise
auto_noise = np.load(os.path.join(results_folder, "auto_noise.npy"))  # Auto-correlation for noise pairs

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
plt.xlabel('Jaccard Similarity')
plt.ylabel('Number of Fingerprint Pairs')
plt.xticks(np.arange(0.1, 1.1, 0.1))  # Set x-ticks
plt.xticks(fontsize=16)  # Set font size for x-ticks
plt.yticks(fontsize=16)  # Set font size for y-ticks
plt.legend(fontsize=16)  # Set font size for legend
plt.grid(False)  # Disable grid
plt.axvline(x=0.35, color='gray', linestyle='--')  # Add a vertical dashed line at x=0.6

# Calculate True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
TP = sum(1 for sim in auto_eq if sim > 0.35)
TN = sum(1 for sim in auto_noise if sim <= 0.35) + sum(1 for sim in cross_eq_noise if sim <= 0.35)
FP = sum(1 for sim in auto_noise if sim > 0.35) + sum(1 for sim in cross_eq_noise if sim > 0.35)
FN = sum(1 for sim in auto_eq if sim <= 0.35)

# Calculate performance metrics
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0  # Calculate accuracy
precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # Calculate precision
recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # Calculate recall
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  # Calculate F1-Score

# Print performance metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")

# Save the histogram plot
plt.savefig(os.path.join(results_folder, 'fast.png'))
plt.show()
