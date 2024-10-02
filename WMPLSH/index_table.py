import pickle
import random

def load_fingerprints_from_pickle(pickle_file):
    """
    Load fingerprints and their corresponding filenames from a pickle file.

    Parameters:
    pickle_file (str): The path to the pickle file containing fingerprints.

    Returns:
    tuple: A tuple containing two lists - fingerprints and filenames.
    """
    with open(pickle_file, 'rb') as f:
        fingerprints_data = pickle.load(f)  # Load the data from the pickle file
    fingerprints, filenames = zip(*fingerprints_data)  # Unzip the loaded data into two separate lists
    return fingerprints, filenames  # Return the lists of fingerprints and filenames

def save_keys_to_pickle(keys, output_file):
    """
    Save keys to a pickle file.

    Parameters:
    keys (list): A list of keys to be saved.
    output_file (str): The path for the output pickle file.
    """
    with open(output_file, 'wb') as f:
        pickle.dump(keys, f)  # Dump the keys list into the specified pickle file

def save_index_table_to_pickle(index_table, output_file):
    """
    Save the index table to a pickle file.

    Parameters:
    index_table (list): The index table to be saved.
    output_file (str): The path for the output pickle file.
    """
    with open(output_file, 'wb') as f:
        pickle.dump(index_table, f)  # Dump the index table into the specified pickle file

def load_index_table_from_pickle(input_file):
    """
    Load an index table from a pickle file.

    Parameters:
    input_file (str): The path to the input pickle file containing the index table.

    Returns:
    list: The loaded index table.
    """
    with open(input_file, 'rb') as f:
        index_table = pickle.load(f)  # Load the index table from the pickle file
    return index_table  # Return the loaded index table

def initialize_index_table(pickle_file, m, n, l, keys_output_file, index_table_output_file):
    """
    Initialize an index table using fingerprints loaded from a pickle file and generate random keys.

    Parameters:
    pickle_file (str): The path to the pickle file containing fingerprints.
    m (int): The number of bits in each key.
    n (int): The number of keys to generate.
    l (int): The length of each fingerprint.
    keys_output_file (str): The path for the output pickle file for keys.
    index_table_output_file (str): The path for the output pickle file for the index table.

    Returns:
    tuple: A tuple containing the index table and the list of keys.
    """
    fingerprints, filenames = load_fingerprints_from_pickle(pickle_file)  # Load fingerprints and filenames

    M = 2 ** m  # Calculate the size of the index table
    index_table = [[[] for _ in range(n)] for _ in range(M)]  # Initialize the index table with empty lists

    keys = []  # Initialize an empty list to store keys
    for i in range(n):
        key = [random.randint(0, l - 1) for _ in range(m)]  # Generate a random key
        keys.append(key)  # Append the key to the keys list

        # Iterate through each fingerprint to populate the index table
        for f_idx, fingerprint in enumerate(fingerprints):
            if len(fingerprint) < l:  # Skip fingerprints shorter than l
                continue

            # Create a hash vector based on the key
            hash_vector = [fingerprint[k] for k in key]
            # Calculate the hash value for the fingerprint
            hash_value = sum(bit << (len(hash_vector) - 1 - idx) for idx, bit in enumerate(hash_vector)) % M
            # Append the filename to the corresponding position in the index table
            index_table[hash_value][i].append(filenames[f_idx])

    # Save the generated keys and index table to pickle files
    save_keys_to_pickle(keys, keys_output_file)
    save_index_table_to_pickle(index_table, index_table_output_file)

    return index_table, keys  # Return the index table and keys

pickle_file = 'E:/D_FAST_data/detection_ep_eq_noise/database.pkl'  # Path to the input pickle file
keys_output_file = 'E:/D_FAST_data/detection_ep_eq_noise/keys.pkl'  # Path for the output pickle file for keys
index_table_output_file = 'E:/D_FAST_data/detection_ep_eq_noise/index_table.pkl'
# Example usage
# pickle_file = 'E:/D_FAST_data/detection/database_detection.pkl'  # Path to the input pickle file
# keys_output_file = 'E:/D_FAST_data/keys1.pkl'  # Path for the output pickle file for keys
# index_table_output_file = 'E:/D_FAST_data/index_table1.pkl'  # Path for the output pickle file for the index table
m = 10  # Number of bits in each key
n = 20  # Number of keys to generate
l = 3200  # Length of each fingerprint

# Initialize the index table and generate keys
index_table, keys = initialize_index_table(pickle_file, m, n, l, keys_output_file, index_table_output_file)

# Print the dimensions of the index table and the number of generated keys
print(f"Index table has {len(index_table)} rows and {len(index_table[0])} columns.")
print(f"Number of generated keys: {len(keys)}")
print(index_table)  # Print the index table
print(keys)  # Print the generated keys

# Continue with the implementation of pseudo-hashing and error-weighted hashing
