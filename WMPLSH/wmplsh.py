import pickle
import random
from itertools import combinations

def hamming_distance(x, y):
    """Calculate the Hamming distance between two binary strings."""
    return sum(c1 != c2 for c1, c2 in zip(x, y))  # Count the number of differing bits

def generate_pseudo_hashes(query_hash, r):
    """
    Generate all pseudo-hashes that differ from the query hash vector by r bits.

    Parameters:
    query_hash (list): The query hash vector (a list of integers consisting of 0s and 1s).
    r (int): The number of bit differences.

    Returns:
    pseudo_hashes (set): A set of all possible pseudo-hash vectors.
    """
    indices = range(len(query_hash))  # Get the indices of the hash vector
    pseudo_hashes = set()  # Initialize a set to store unique pseudo-hashes

    # Find all combinations of r indices and flip those bits
    for positions in combinations(indices, r):
        hash_list = list(query_hash)  # Create a mutable copy of the query hash
        for pos in positions:
            hash_list[pos] = 1 if hash_list[pos] == 0 else 0  # Flip the bit at the position
        # Convert the list of integers back to a string and add to the set
        pseudo_hashes.add(''.join(map(str, hash_list)))

    return pseudo_hashes  # Return the set of pseudo-hashes

def weighted_multi_probe_hashing(query_fingerprint, fingerprints, filenames, index_table, keys, weights, similarity_threshold, e):
    """
    Error Weighted Hashing (EWH) algorithm.

    Parameters:
    query_fingerprint (str): The binary fingerprint to query.
    fingerprints (list): The fingerprint database.
    filenames (list): The corresponding filenames for the fingerprints.
    index_table (list): A 2D index table containing M rows and n columns.
    keys (list): A list of randomly generated keys.
    weights (list): A list of weights corresponding to different levels of pseudo-hash vectors.
    similarity_threshold (float): The similarity score threshold.
    e (int): The maximum number of bit differences.

    Returns:
    nearest_neighbors (list): A list of filenames for the fingerprints most similar to the query fingerprint.
    """
    # Initialize similarity scores for each filename
    scores = {filename: 0 for filename in filenames}

    M = len(index_table)  # Number of rows in the index table
    n = len(keys)  # Number of keys

    # Iterate over all columns (corresponding to different keys)
    for i in range(n):
        key = keys[i]
        # Compute the hash vector for the query fingerprint using the key
        hash_vector = [query_fingerprint[k] for k in key]
        # Calculate the hash value
        hash_value = sum(bit << (len(hash_vector) - 1 - idx) for idx, bit in enumerate(hash_vector)) % M

        # Increase weight 0 for fingerprints at the row corresponding to the calculated hash value
        for filename in index_table[hash_value][i]:
            scores[filename] += weights[0]

        # Iterate over bit difference values r from 1 to e
        for r in range(1, e + 1):
            pseudo_hashes = generate_pseudo_hashes(hash_vector, r)  # Generate pseudo-hashes differing by r bits
            for pseudo_hash in pseudo_hashes:
                row_hr = int(pseudo_hash, 2) % M  # Get the row number corresponding to the pseudo-hash
                # Increase weight r for fingerprints at the corresponding row and column
                for filename in index_table[row_hr][i]:
                    scores[filename] += weights[r]

    # Select candidate filenames with scores above the similarity threshold
    candidates = [f for f, score in scores.items() if score > similarity_threshold]
    return candidates  # Return the list of candidate filenames

def load_fingerprints_from_pickle(pickle_file):
    """Load fingerprints and their corresponding filenames from a pickle file."""
    with open(pickle_file, 'rb') as f:
        fingerprints_data = pickle.load(f)  # Load the data from the pickle file
    fingerprints, filenames = zip(*fingerprints_data)  # Unzip the loaded data into two separate lists
    return fingerprints, filenames  # Return the lists of fingerprints and filenames

def load_pickle_file(pickle_file):
    """Load data from a pickle file."""
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)  # Load the data from the pickle file
        return data  # Return the loaded data

# Specify paths to data files
pickle_file = 'E:/D_FAST_code/detection/database_detection.pkl'
keys_output_file = 'E:/D_FAST_code/keys1.pkl'
index_table_output_file = 'E:/D_FAST_code/index_table1.pkl'

# Load data
fingerprints, filenames = load_fingerprints_from_pickle(pickle_file)  # Load fingerprints and filenames
keys = load_pickle_file(keys_output_file)  # Load keys
index_table = load_pickle_file(index_table_output_file)  # Load index table

m = 10  # Number of bits in each key
n = 20  # Number of keys to generate
# Configure parameters
weights = [1 / n, (m - 1) / (m * n), (m - 2) / (m * n)]  # Weights for scoring
similarity_threshold = 0.10  # Set similarity threshold
e = 2  # Maximum number of bit differences
query_fingerprint = fingerprints[1]  # Select a query fingerprint

# Run the Error Weighted Hashing algorithm
candidates = weighted_multi_probe_hashing(query_fingerprint, fingerprints, filenames, index_table, keys, weights, similarity_threshold, e)

# Count the number of candidate filenames that start with "noise"
noise_candidates = [candidate for candidate in candidates if candidate.startswith('noise')]
FP = len(noise_candidates)  # Count false positives
TP = len(candidates) - len(noise_candidates)  # Count true positives
TN = 341 - len(noise_candidates)  # Count true negatives (assuming total negatives)
FN = 145 - TP  # Count false negatives (assuming total positives)

# Calculate precision and recall
precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # Precision calculation
recall = TP / 145  # Recall calculation (assuming total positives)

# Print precision and recall values
print('precision=', precision)
print('recall=', recall)
