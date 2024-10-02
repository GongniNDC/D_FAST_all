import os
import shutil

def organize_waveform_files_by_station(type):
    """Organize waveform files into subdirectories based on their prefix."""
    folder_path = f"E:/D_FAST_data/{type}_waveforms"  # Define the directory containing waveform files

    # Get all .pkl files in the specified directory
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.pkl')]

    # Create a dictionary to store filenames with the same prefix
    file_dict = {}

    # Iterate through all files
    for file in files:
        # Extract the prefix from the filename (the last part before '.pkl')
        prefix = file.split('_')[-1].split('.pkl')[0]

        # Check if the prefix is already in the dictionary
        if prefix in file_dict:
            # If the prefix exists, move the file to the corresponding subdirectory
            shutil.move(os.path.join(folder_path, file), os.path.join(folder_path, prefix, file))
        else:
            # If the prefix does not exist, add it to the dictionary and create a new subdirectory
            file_dict[prefix] = [file]  # Store the file under the prefix
            os.makedirs(os.path.join(folder_path, prefix))  # Create a new directory for the prefix
            shutil.move(os.path.join(folder_path, file), os.path.join(folder_path, prefix, file))  # Move the file

# Call the function to organize waveform files
type1='eq'
type2='px'
organize_waveform_files_by_station(type1)
organize_waveform_files_by_station(type2)
