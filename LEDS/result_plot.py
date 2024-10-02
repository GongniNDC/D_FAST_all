import os
import numpy as np
import matplotlib.pyplot as plt

# Define the station and types for processing
sta = 'B023'
type1 = 'earthquake'  # Type for eq pairs
type2 = 'explosion'  # Type for px pairs
results_folder = "E:/D_FAST_data/results/pair_sim/eq_ep_noise"  # Path to the results folder

# Load results from .npy files using os.path.join to construct complete file paths
auto_eq = np.load(os.path.join(results_folder, f"auto_{type1}.npy"))  # Auto-correlation for eq pairs
cross_eq_noise = np.load(os.path.join(results_folder, f"noise_{type1}.npy"))  # Cross-correlation between eq and noise
auto_noise = np.load(os.path.join(results_folder, "auto_noise.npy"))  # Auto-correlation for noise pairs
auto_px = np.load(os.path.join(results_folder, f"auto_{type2}.npy"))  # Auto-correlation for px pairs
cross_px_noise = np.load(os.path.join(results_folder,f"noise_{type2}.npy"))  # Cross-correlation between px and noise

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

# Calculate True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
TP = sum(1 for sim in auto_eq if sim > 0.6)  # Count of eq pairs with similarity > 0.6
TN = sum(1 for sim in auto_noise if sim <= 0.6) + sum(1 for sim in cross_eq_noise if sim <= 0.6)  # Count of noise pairs and eq_noise pairs with similarity <= 0.6
FP = sum(1 for sim in auto_noise if sim > 0.6) + sum(1 for sim in cross_eq_noise if sim > 0.6)  # Count of noise pairs incorrectly detected as eq pairs
FN = sum(1 for sim in auto_eq if sim <= 0.6)  # Count of eq pairs not detected (similarity <= 0.6)

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
plt.savefig(os.path.join(results_folder, 'detection_eq.png'))
# Display the plot
plt.show()

# Plot another histogram for px pairs
plt.figure(figsize=(10, 6))  # Set figure size
# Set font size globally for the plot
plt.rcParams.update({'font.size': 16})

# Plot histogram for auto-correlation of px pairs
plt.hist(auto_px, np.arange(0.1, 1.02, 0.01), color='r', alpha=0.6, label='px_pairs', density=True)
# Plot histogram for auto-correlation of noise pairs
plt.hist(auto_noise, np.arange(0.1, 1.02, 0.01), color='green', alpha=0.6, label='noise_pairs', density=True)
# Plot histogram for cross-correlation of px and noise pairs
plt.hist(cross_px_noise, np.arange(0.1, 1.02, 0.01), color='y', alpha=0.6, label='px_noise_pairs', density=True)

# Set labels and ticks for the plot
plt.xlabel('Hamming Similarity')
plt.ylabel('Number of Fingerprint Pairs')
plt.xticks(np.arange(0.1, 1.1, 0.1))  # Set x-ticks
plt.xticks(fontsize=16)  # Set font size for x-ticks
plt.yticks(fontsize=16)  # Set font size for y-ticks
plt.legend(fontsize=16)  # Set font size for legend
plt.grid(False)  # Disable grid
plt.axvline(x=0.6, color='gray', linestyle='--')  # Add a vertical dashed line at x=0.6

# Calculate True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN) for px pairs
TP = sum(1 for sim in auto_px if sim > 0.6)  # Count of px pairs with similarity > 0.6
TN = sum(1 for sim in auto_noise if sim <= 0.6) + sum(1 for sim in cross_px_noise if sim <= 0.6)  # Count of noise pairs and px_noise pairs with similarity <= 0.6
FP = sum(1 for sim in auto_noise if sim > 0.6) + sum(1 for sim in cross_px_noise if sim > 0.6)  # Count of noise pairs incorrectly detected as px pairs
FN = sum(1 for sim in auto_px if sim <= 0.6)  # Count of px pairs not detected (similarity <= 0.6)

# Calculate performance metrics for px pairs
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0  # Calculate accuracy
precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # Calculate precision
recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # Calculate recall
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  # Calculate F1-Score

# Print performance metrics for px pairs
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")

# Save the histogram plot for px pairs
plt.savefig(os.path.join(results_folder, 'detection_px.png'))
# Display the plot
plt.show()
