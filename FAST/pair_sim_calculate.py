import os
import time
import numpy as np

def jaccard_similarity(vec1, vec2):
    intersection = np.sum(np.logical_and(vec1, vec2))
    union = np.sum(np.logical_or(vec1, vec2))
    return intersection / union

def test_jaccard(fp1,fp2):
    signal1 = np.loadtxt(fp1)
    signal2 = np.loadtxt(fp2)

    binary_signal1=[int(float_num) for float_num in signal1]
    binary_signal2 = [int(float_num) for float_num in signal2]
    # sim = hamming_sim(binary_signal1, binary_signal2)
    sim = jaccard_similarity(binary_signal1, binary_signal2)
    return sim


def process_auto_correlation(input_folder):
    """
    Process the auto-correlation of fingerprint files in a given folder.

    Parameters:
    input_folder (str): The path to the folder containing fingerprint files.

    Returns:
    tuple: Average similarity, maximum similarity, minimum similarity, detailed similarity information, and similarity values.
    """
    # Get all .txt file paths in the input folder
    file_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.txt')]

    sim_values = []  # List to store similarity values
    for i in range(len(file_paths)):
        for j in range(i + 1, len(file_paths)):
            file1_name = os.path.basename(file_paths[i])  # Get the name of the first file
            file2_name = os.path.basename(file_paths[j])  # Get the name of the second file
            sim = test_jaccard(file_paths[i], file_paths[j])  # Calculate Jaccard similarity
            sim_values.append(sim)
    return sim_values  # Return results


def process_cross_correlation(input_folder1, input_folder2):
    """
    Process the cross-correlation between fingerprint files in two folders.

    Parameters:
    input_folder1 (str): The path to the first folder containing fingerprint files.
    input_folder2 (str): The path to the second folder containing fingerprint files.

    Returns:
    tuple: Average similarity, maximum similarity, minimum similarity, and similarity values.
    """
    # Get all .txt file paths in both input folders
    file_paths1 = [os.path.join(input_folder1, f) for f in os.listdir(input_folder1) if f.endswith('.txt')]
    file_paths2 = [os.path.join(input_folder2, f) for f in os.listdir(input_folder2) if f.endswith('.txt')]

    sim_values = []  # List to store similarity values
    sim_details = []  # List to store detailed similarity information
    for file1 in file_paths1:
        for file2 in file_paths2:
            file1_name = os.path.basename(file1)  # Get the name of the first file
            file2_name = os.path.basename(file2)  # Get the name of the second file
            signal1 = np.loadtxt(file1)  # Load the first signal
            signal2 = np.loadtxt(file2)  # Load the second signal

            # Convert the signals to binary (0s and 1s)
            binary_signal1 = [int(float_num) for float_num in signal1]
            binary_signal2 = [int(float_num) for float_num in signal2]
            sim = jaccard_similarity(binary_signal1, binary_signal2)  # Calculate Jaccard similarity
            sim_values.append(sim)  # Store similarity value
    return sim_values  # Return results


result_folder = "E:/D_FAST_data/results/pair_sim/fast"  # Folder to save results
start_time = time.time()  # Start timing
input_folder_signal = 'E:/D_FAST_data/snr2/signal_add_noise_fp_fast'
input_folder_noise = 'E:/D_FAST_data/snr2/noise_fp_fast'
# Process cross-correlation between signal and noise fingerprints
noise_signal = process_cross_correlation(input_folder_signal, input_folder_noise)
# Process auto-correlation for signal fingerprints
auto_signal = process_auto_correlation(input_folder_signal)
auto_noise = process_auto_correlation(input_folder_noise)
# Save the results to .npy files
np.save(os.path.join(result_folder, 'noise_signal.npy'), noise_signal)
np.save(os.path.join(result_folder, 'auto_noise.npy'), auto_noise)
np.save(os.path.join(result_folder, 'auto_signal.npy'), auto_signal)
