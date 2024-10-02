import os
import time
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw


def hamming_distance(str1, str2):
    """
    Calculate the Hamming distance between two strings.

    Parameters:
    str1 (str): The first string.
    str2 (str): The second string.

    Returns:
    int: The Hamming distance between the two strings.
    """
    if len(str1) != len(str2):
        raise ValueError("Two strings must have the same length")  # Ensure both strings have the same length

    return sum(c1 != c2 for c1, c2 in zip(str1, str2))  # Count differing characters


def hamming_sim(str1, str2):
    """
    Calculate the similarity between two strings based on Hamming distance.

    Parameters:
    str1 (str): The first string.
    str2 (str): The second string.

    Returns:
    float: The similarity score between 0 and 1.
    """
    distance = hamming_distance(str1, str2)  # Calculate Hamming distance
    similarity = 1 - distance / len(str1)  # Compute similarity
    return similarity  # Return similarity score


def test_hamming(fp1, fp2):
    """
    Test the Hamming similarity between two fingerprint files.

    Parameters:
    fp1 (str): The path to the first fingerprint file.
    fp2 (str): The path to the second fingerprint file.

    Returns:
    float: The similarity score between the two fingerprints.
    """
    signal1 = np.loadtxt(fp1)  # Load the first fingerprint
    signal2 = np.loadtxt(fp2)  # Load the second fingerprint

    # Convert the signals to binary (0s and 1s)
    binary_signal1 = [int(float_num) for float_num in signal1]
    binary_signal2 = [int(float_num) for float_num in signal2]

    sim = hamming_sim(binary_signal1, binary_signal2)  # Calculate similarity
    return sim  # Return similarity score


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
    sim_details = []  # List to store detailed similarity information
    for i in range(len(file_paths)):
        for j in range(i + 1, len(file_paths)):
            file1_name = os.path.basename(file_paths[i])  # Get the name of the first file
            file2_name = os.path.basename(file_paths[j])  # Get the name of the second file
            sim = test_hamming(file_paths[i], file_paths[j])  # Calculate similarity
            sim_values.append(sim)  # Store similarity value
            sim_details.append((file1_name, file2_name, sim))  # Store details

    # Create a results folder to store output
    results_folder = os.path.join(input_folder, "results_cross")
    os.makedirs(results_folder, exist_ok=True)

    # Save the similarity details to a text file
    txt_file_path = os.path.join(results_folder, "sim_details.txt")
    # Filter for similarities less than 0.6
    filtered_sim_details = [item for item in sim_details if item[2] < 0.6]

    # Open the text file for writing
    with open(txt_file_path, "w") as file:
        for item in filtered_sim_details:
            file.write(f"{item[0].split('.IGP')[0]}, {item[1].split('.IGP')[0]}, {item[2]}\n")  # Write each detail

    # Calculate the average similarity
    average_sim = np.mean(sim_values)
    # Calculate maximum and minimum similarity, rounded to 3 decimal places
    max_sim = round(np.max(sim_values), 3)
    min_sim = round(np.min(sim_values), 3)
    return average_sim, max_sim, min_sim, sim_details, sim_values  # Return results


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
            sim = hamming_sim(binary_signal1, binary_signal2)  # Calculate similarity
            sim_values.append(sim)  # Store similarity value
            sim_details.append((file1_name, file2_name, sim))  # Store details

    # Calculate average similarity
    average_sim = np.mean(sim_values)
    # Calculate maximum and minimum similarity
    max_sim = np.max(sim_values)
    min_sim = np.min(sim_values)
    return average_sim, max_sim, min_sim, sim_values  # Return results


# Cross-correlation processing section
type = 'earthquake'
input_folder_signal = f"E:/D_FAST_data/{type}/B023/fp"  # Folder containing signal fingerprints
input_folder_noise = "E:/D_FAST_data/noise/fp_D_FAST"  # Folder containing noise fingerprints
result_folder = "E:/D_FAST_data/results/pair_sim/eq_ep_noise"  # Folder to save results
# Process cross-correlation between signal and noise fingerprints
average_sim, max_sim, min_sim, noise_signal = process_cross_correlation(input_folder_signal, input_folder_noise)
# Process auto-correlation for signal fingerprints
average_sim, max_sim, min_sim, sim_details, auto_signal = process_auto_correlation(input_folder_signal)
#average_sim, max_sim, min_sim, sim_details, auto_noise = process_auto_correlation(input_folder_noise)
# Save the results to .npy files
np.save(os.path.join(result_folder, f'noise_{type}.npy'), noise_signal)
#np.save(os.path.join(result_folder, 'auto_noise.npy'), auto_noise)
np.save(os.path.join(result_folder, f'auto_{type}.npy'), auto_signal)








# result_folder = "E:/D_FAST_data/results/pair_sim/snr2/D_FAST"  # Folder to save results
# input_folder_signal='E:/D_FAST_data/snr2/fp_D_FAST'
# input_folder_noise ='E:/D_FAST_data/snr2/noise_fp_D_FAST'
# # Process cross-correlation between signal and noise fingerprints
# average_sim, max_sim, min_sim, noise_signal = process_cross_correlation(input_folder_signal, input_folder_noise)
# # Process auto-correlation for signal fingerprints
# average_sim, max_sim, min_sim, sim_details, auto_signal = process_auto_correlation(input_folder_signal)
# average_sim, max_sim, min_sim, sim_details, auto_noise = process_auto_correlation(input_folder_noise)
# # Save the results to .npy files
# np.save(os.path.join(result_folder, 'noise_signal.npy'), noise_signal)
# np.save(os.path.join(result_folder, 'auto_noise.npy'), auto_noise)
# np.save(os.path.join(result_folder, 'auto_signal.npy'), auto_signal)