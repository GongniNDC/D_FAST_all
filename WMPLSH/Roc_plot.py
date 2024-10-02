import os
import pickle
import random
from itertools import combinations
import matplotlib.pyplot as plt


def hamming_distance(x, y):
    """Calculate the Hamming distance between two binary strings."""
    return sum(c1 != c2 for c1, c2 in zip(x, y))  # Count the number of differing bits


def generate_pseudo_hashes(query_hash, r):
    """
    Generate all pseudo-hashes that differ from the query hash vector by r bits.

    Parameters:
    query_hash (list): The original hash vector as a list of bits.
    r (int): The number of bits to flip in the query hash to generate pseudo-hashes.

    Returns:
    set: A set of pseudo-hashes differing by r bits.
    """
    indices = range(len(query_hash))  # Get the indices of the hash vector
    pseudo_hashes = set()  # Initialize a set to store unique pseudo-hashes
    for positions in combinations(indices, r):  # Generate combinations of indices to flip
        hash_list = list(query_hash)  # Create a mutable copy of the query hash
        for pos in positions:
            hash_list[pos] = 1 if hash_list[pos] == 0 else 0  # Flip the bit at the position
        pseudo_hashes.add(''.join(map(str, hash_list)))  # Convert list back to string and add to set
    return pseudo_hashes  # Return the set of pseudo-hashes


def weighted_multi_probe_hashing(query_fingerprint, fingerprints, filenames, index_table, keys, weights,
                                 similarity_threshold, e):
    """
    Error Weighted Hashing (EWH) algorithm for matching fingerprints.

    Parameters:
    query_fingerprint (list): The fingerprint to query.
    fingerprints (list): The list of all fingerprints.
    filenames (list): The corresponding filenames for the fingerprints.
    index_table (list): The index table for efficient retrieval.
    keys (list): The generated keys for hashing.
    weights (list): The weights associated with each level of probing.
    similarity_threshold (float): The threshold for considering a candidate as a match.
    e (int): The maximum number of bit flips allowed.

    Returns:
    list: A list of candidate filenames that exceed the similarity threshold.
    """
    scores = {filename: 0 for filename in filenames}  # Initialize scores for each filename
    M = len(index_table)  # Number of rows in the index table
    n = len(keys)  # Number of keys

    # For each key, compute the hash and update scores
    for i in range(n):
        key = keys[i]
        hash_vector = [query_fingerprint[k] for k in key]  # Create hash vector using the key
        hash_value = sum(
            bit << (len(hash_vector) - 1 - idx) for idx, bit in enumerate(hash_vector)) % M  # Calculate hash value

        # Update scores for exact matches
        for filename in index_table[hash_value][i]:
            scores[filename] += weights[0]

        # Generate pseudo-hashes and update scores for them
        for r in range(1, e + 1):
            pseudo_hashes = generate_pseudo_hashes(hash_vector, r)  # Generate pseudo-hashes differing by r bits
            for pseudo_hash in pseudo_hashes:
                row_hr = int(pseudo_hash, 2) % M  # Calculate the row index for the pseudo-hash
                for filename in index_table[row_hr][i]:
                    scores[filename] += weights[r]  # Update scores based on weights

    # Collect candidates that exceed the similarity threshold
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

pickle_file = 'E:/D_FAST_data/detection_ep_eq_noise/database.pkl'
keys_output_file = 'E:/D_FAST_data/detection_ep_eq_noise/keys.pkl'
index_table_output_file = 'E:/D_FAST_data/detection_ep_eq_noise/index_table.pkl'
results_folder = "E:/D_FAST_data/results"
# Specify paths to data files
# pickle_file = 'E:/D_FAST_code/detection/database_detection.pkl'
# keys_output_file = 'E:/D_FAST_code/keys1.pkl'
# index_table_output_file = 'E:/D_FAST_code/index_table1.pkl'

# Load data
fingerprints, filenames = load_fingerprints_from_pickle(pickle_file)  # Load fingerprints and filenames
keys = load_pickle_file(keys_output_file)  # Load keys
index_table = load_pickle_file(index_table_output_file)  # Load index table

m = 10  # Number of bits in each key
n = 20  # Number of keys to generate
weights = [1 / n, (m - 1) / (m * n), (m - 2) / (m * n)]  # Weights for scoring
e = 2  # Maximum number of bit flips allowed

# Adjust the range of similarity_threshold from 0.1 to 0.9 with a step of 0.02
threshold_values = [round(0.1 + i * 0.02, 2) for i in range(41)]  # Generate 41 threshold values

# Calculate average precision and recall, and plot the ROC curve
avg_precisions = []  # List to store average precision values
avg_recalls = []  # List to store average recall values

for similarity_threshold in threshold_values:
    precisions = []  # List to store precision values for each query
    recalls = []  # List to store recall values for each query

    # Iterate over query fingerprints
    for query_idx in range(0, 145):
        query_fingerprint = fingerprints[query_idx]  # Get the query fingerprint
        candidates = weighted_multi_probe_hashing(query_fingerprint, fingerprints, filenames, index_table, keys,
                                                  weights,
                                                  similarity_threshold, e)  # Get candidates based on EWH

        # Calculate true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN)
        noise_candidates = [candidate for candidate in candidates if candidate.startswith('noise')]
        FP = len(noise_candidates)
        TP = len(candidates) - FP

        # Calculate precision and recall
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / 145 # Assuming total positive cases

        precisions.append(precision)  # Append precision for this query
        recalls.append(recall)  # Append recall for this query

    # Calculate average precision and recall for the current threshold
    avg_precisions.append(sum(precisions) / len(precisions))
    avg_recalls.append(sum(recalls) / len(recalls))

# Plot the ROC curve
plt.figure()
plt.plot(avg_recalls, avg_precisions, marker='o', color='b')  # Plot recall vs precision
plt.xlabel('Recall')  # Label for x-axis
plt.ylabel('Precision')  # Label for y-axis
plt.title('ROC Curve under Different Thresholds for D_FAST')  # Title
plt.savefig(os.path.join(results_folder, 'd_fast.png'))
plt.show()
print('Recall',avg_recalls)
print('Precision',avg_precisions)