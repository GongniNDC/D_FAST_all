import os
import numpy as np
import pickle

def convert_float_to_binary(float_vector):
    """
    Convert a vector of floating-point numbers to a binary vector in integer form.

    Parameters:
    float_vector (list): A vector of floating-point numbers (expected to be 0.0 or 1.0).

    Returns:
    list: A binary vector in integer form.
    """
    # Assuming the floating-point numbers are 0.0 or 1.0, convert them to integers
    return [int(round(num)) for num in float_vector]

def load_fingerprints_from_folder(folder_path):
    """
    Load fingerprints from a specified folder, extracting binary vectors and their corresponding filenames.

    Parameters:
    folder_path (str): The path to the folder containing fingerprint files.

    Returns:
    list: A list of tuples containing binary vectors and their corresponding filenames [(binary_vector, filename), ...].
    """
    fingerprints = []  # Initialize an empty list to store fingerprints

    # Iterate through each file in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file has a .txt extension
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)  # Construct the full file path

            # Read the contents of the file
            with open(file_path, 'r') as f:
                # Convert each line in the file to a float and create a list
                float_vector = [float(line.strip()) for line in f.readlines()]

            # Convert the float vector to a binary vector
            binary_vector = convert_float_to_binary(float_vector)
            # Append the binary vector and filename as a tuple to the fingerprints list
            fingerprints.append((binary_vector, filename))

    return fingerprints  # Return the list of fingerprints

def save_fingerprints_to_pickle(fingerprints, output_file):
    """
    Save fingerprints to a pickle file.

    Parameters:
    fingerprints (list): A list containing binary vectors and their corresponding filenames.
    output_file (str): The path for the output pickle file.
    """
    with open(output_file, 'wb') as f:
        pickle.dump(fingerprints, f)  # Dump the fingerprints list to the pickle file

# Example usage
# folder_path = 'E:/D_FAST_code/fp_database'  # Specify folder path for fingerprints
# output_file = 'E:/D_FAST_code/fp_database.pkl'  # Specify output path for the pickle file
# folder_path = 'E:/D_FAST_data/detection/database'  # Specify folder path for detection database
# output_file = 'E:/D_FAST_data/detection/database_detection.pkl'  # Specify output path for the detection pickle file
folder_path = 'E:/D_FAST_data/detection_ep_eq_noise/database'  # Specify folder path for detection database
output_file = 'E:/D_FAST_data/detection_ep_eq_noise/database.pkl'
# Load fingerprints from the specified folder
fingerprints = load_fingerprints_from_folder(folder_path)
# Save the loaded fingerprints to a pickle file
save_fingerprints_to_pickle(fingerprints, output_file)

# Check the number of loaded fingerprints
print(f"Number of loaded fingerprints: {len(fingerprints)}")  # Print the count of loaded fingerprints
