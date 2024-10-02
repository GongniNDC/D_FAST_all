import time
import numpy as np
import matplotlib.pyplot as plt
from obspy import Trace
from scipy.signal import spectrogram
import os


class FeatureExtractor(object):
    def __init__(self, sampling_rate, window_length, window_lag, fingerprint_length, fingerprint_lag,
                 min_freq=0, max_freq=None, new_d1=33, new_d2=100):
        """
        Initialize the FeatureExtractor with the given parameters.

        Parameters:
        sampling_rate (int): The sampling rate of the input signal.
        window_length (float): Length of the window used for the spectrogram (in seconds).
        window_lag (float): Lag between windows (in seconds).
        fingerprint_length (int): Width of the fingerprint (in samples).
        fingerprint_lag (int): Lag between fingerprints (in samples).
        min_freq (int): Minimum frequency for the bandpass filter.
        max_freq (int): Maximum frequency for the bandpass filter.
        new_d1 (int): Number of frequency bands to use in the output.
        new_d2 (int): Not used in this implementation but included for consistency.
        """
        self.sampling_rate = sampling_rate  # Sampling rate
        self.window_len = window_length  # Length of the window for spectrogram
        self.window_lag = window_lag  # Lag between windows
        self.fp_len = fingerprint_length  # Width of fingerprint (in samples)
        self.fp_lag = fingerprint_lag  # Lag between fingerprints
        self.max_freq = self._initialize_frequencies(max_freq)  # Initialize max frequency
        self.min_freq = min_freq  # Minimum frequency
        self.new_d1 = new_d1  # Number of frequency bands
        self.new_d2 = new_d2  # Not used in this implementation
        self.d1 = None  # Dimension of spectral images before resizing
        self.d2 = None

    def _initialize_frequencies(self, max_freq):
        """Initialize the maximum frequency for the bandpass filter."""
        if max_freq is None:
            max_freq = self.sampling_rate / 2.0  # Nyquist frequency
        return max_freq

    def get_window_params(self, N, L, dL):
        """
        Calculate window parameters for the spectrogram.

        Parameters:
        N (int): Total number of time samples.
        L (int): Length of the window.
        dL (int): Lag between windows.

        Returns:
        nWindows (int): Number of windows.
        idx1 (np.ndarray): Start indices of the windows.
        idx2 (np.ndarray): End indices of the windows.
        """
        idx0 = np.asarray(range(0, N + 1, dL))  # Indices for all windows
        idx2 = np.asarray(range(L, N + 1, dL))  # End indices for windows
        nWindows = len(idx2)  # Number of windows
        idx1 = idx0[0:nWindows]  # Start indices for windows
        return nWindows, idx1, idx2

    ########################################################################
    ##     FOR COMPUTING FINGERPRINTS                                     ##
    ########################################################################

    def data_to_spectrogram(self, x_data):
        """
        Compute the spectrogram from continuous time-series data.

        Parameters:
        x_data (np.ndarray): Input time-series data.

        Returns:
        f (np.ndarray): Frequency bins.
        t (np.ndarray): Time bins.
        Sxx (np.ndarray): Spectrogram of the input data.
        """
        # Calculate the spectrogram
        f, t, Sxx = spectrogram(x_data, fs=self.sampling_rate,
                                nperseg=int(self.sampling_rate * self.window_len),
                                noverlap=int(self.sampling_rate * (self.window_len - self.window_lag)))

        # Truncate the spectrogram to keep only passband frequencies
        if self.min_freq > 0:
            fidx_keep = (f >= self.min_freq)
            Sxx = Sxx[fidx_keep, :]  # Keep only the frequencies above min_freq
            f = f[fidx_keep]
        if self.max_freq < f[-1]:
            fidx_keep = (f <= self.max_freq)
            Sxx = Sxx[fidx_keep, :]  # Keep only the frequencies below max_freq
            f = f[fidx_keep]

        self.frequencies = f  # Store frequencies for later use
        self.times = t  # Store time bins for later use
        return f, t, Sxx  # Return frequency, time, and spectrogram

    def spectrogram_to_spectral_images(self, Sxx):
        """
        Convert the spectrogram into overlapping frames (spectral images).

        Parameters:
        Sxx (np.ndarray): Spectrogram of the input data.

        Returns:
        spectral_images (np.ndarray): Array of spectral images.
        nWindows (int): Number of windows.
        idx1 (np.ndarray): Start indices of the windows.
        idx2 (np.ndarray): End indices of the windows.
        """
        nFreq, nTimes = np.shape(Sxx)  # Get the shape of the spectrogram
        nWindows, idx1, idx2 = self.get_window_params(nTimes, self.fp_len, self.fp_lag)  # Get window parameters
        spectral_images = np.zeros([nWindows, nFreq, self.fp_len])  # Initialize spectral images array

        # Fill the spectral images with data from the spectrogram
        for i in range(nWindows):
            spectral_images[i, :, :] = Sxx[:, idx1[i]:idx2[i]]

        self.nwindows = nWindows  # Store number of windows
        nWindows, self.d1, self.d2 = np.shape(spectral_images)  # Get the shape of the spectral images
        return spectral_images, nWindows, idx1, idx2  # Return spectral images and parameters

    def oversample(self, oversampling_factor, input_vector):
        """
        Perform oversampling on the input vector.

        Parameters:
        oversampling_factor (int): Factor by which to oversample.
        input_vector (np.ndarray): Input data to oversample.

        Returns:
        output_vector (np.ndarray): Oversampled data.
        """
        indices = np.arange(len(input_vector) - 1)  # Original indices
        new_indices = np.linspace(0, len(input_vector) - 1, len(input_vector) * oversampling_factor)  # New indices
        output_vector = np.interp(new_indices, indices, input_vector)  # Interpolate to create oversampled data
        return output_vector  # Return oversampled data

    def philips(self, input_file, output_file, folder):
        """
        Process the input file to extract features and save to output file.

        Parameters:
        input_file (str): Path to the input .npy file.
        output_file (str): Path to the output .txt file.
        folder (str): Folder to save output files.
        """
        # Load data from the input file
        data = np.load(input_file)
        x_data = np.array(data)  # Convert data to a NumPy array

        # Calculate the spectrogram for the entire x_data
        f, t, Sxx = self.data_to_spectrogram(x_data)
        Freq, Times = np.shape(Sxx)  # Get the shape of the spectrogram
        nWindows, idx1, idx2 = self.get_window_params(Times, self.fp_len, self.fp_lag)
        self.nwindows = nWindows  # Store number of windows

        # Divide frequencies into new_d1 bands
        m1 = Freq // self.new_d1  # Number of frequency bands
        spectral_matrix = Sxx[:m1 * self.new_d1, :]  # Truncate the spectrogram to the desired number of bands

        # Sum rows to create the energy matrix
        sum_rows = np.array([spectral_matrix[i:i + m1].sum(axis=0) for i in range(0, len(spectral_matrix), m1)])
        E = np.zeros((nWindows, self.new_d1))  # Initialize energy matrix

        # Fill the energy matrix with summed spectral data
        for i in range(nWindows):
            spectral_sets = sum_rows[:, idx1[i]:idx2[i]]
            E[i, :] = np.sum(spectral_sets, axis=1)  # Sum the energy for each window

        F = np.zeros((nWindows, self.new_d1 - 1))  # Initialize local energy difference matrix

        # Calculate local energy differences
        for n in range(1, nWindows):
            for m in range(self.new_d1 - 1):
                if E[n, m] - E[n - 1, m] - (E[n, m + 1] - E[n - 1, m + 1]) > 0:
                    F[n - 1, m] = 1  # Mark as significant change
                else:
                    F[n - 1, m] = 0  # Mark as no significant change

        fp = F.flatten()  # Flatten the matrix into a vector
        fp = fp[:3200]  # Truncate to the first 3200 samples
        np.savetxt(output_file, fp)  # Save the fingerprint to a text file


# Initialize the FeatureExtractor with specified parameters
fe = FeatureExtractor(sampling_rate=100, window_length=10, window_lag=0.1, fingerprint_length=100,
                      fingerprint_lag=4, min_freq=1, max_freq=50, new_d1=33, new_d2=100)


def process_file(input_folder, output_base_folder):
    """
    Process all .npy files in the input folder and extract features.

    Parameters:
    input_folder (str): Path to the input folder containing .npy files.
    output_base_folder (str): Path to the output folder for saving results.

    Returns:
    average_cpu_time (float): Average CPU time taken for processing.
    """
    start_time = time.time()  # Start timing
    for filename in os.listdir(input_folder):  # Iterate through all files in the input folder
        if filename.endswith(".npy"):  # Process only .npy files
            input_file = os.path.join(input_folder, filename).replace("\\", "/")  # Create full input file path
            input_file_name = os.path.basename(filename)  # Get the base filename
            input_file_name_without_extension = os.path.splitext(input_file_name)[0]  # Remove the extension
            new_file_name = f"{output_base_folder}/{input_file_name_without_extension}.txt"  # Create output file path

            # Process the input file and save the output
            fe.philips(input_file, new_file_name, output_base_folder)

    end_time = time.time()  # End timing
    average_cpu_time = end_time - start_time  # Calculate average CPU time
    return average_cpu_time  # Return the average CPU time


# # Specify input and output directories for processing
# type1 = 'eq'
# type2 = 'ep'
#
# input_folder = f"E:/D_FAST_data/{type1}_waveforms/B023/P_waveform"
# output_folder = f"E:/D_FAST_data/{type1}_waveforms/B023/fp"
# if not os.path.exists(output_folder):  # Create output folder if it doesn't exist
#     os.makedirs(output_folder)
# average_cpu_time = process_file(input_folder, output_folder)  # Process the files
#
# input_folder = f"E:/D_FAST_data/{type2}_waveforms/B023/P_waveform"
# output_folder = f"E:/D_FAST_data/{type2}_waveforms/B023/fp"
# if not os.path.exists(output_folder):  # Create output folder if it doesn't exist
#     os.makedirs(output_folder)
# average_cpu_time = process_file(input_folder, output_folder)  # Process the files

# Uncomment the following block to process noise waveform files
input_folder = "E:/D_FAST_data/noise/wave_strim"
output_folder = "E:/D_FAST_data/noise/fp_D_FAST"
if not os.path.exists(output_folder):  # Create output folder if it doesn't exist
    os.makedirs(output_folder)
average_cpu_time = process_file(input_folder, output_folder)  # Process the files

input_folder = 'E:/D_FAST_data/snr2/wav'
output_folder = 'E:/D_FAST_data/snr2/fp_D_FAST'
if not os.path.exists(output_folder):  # Create output folder if it doesn't exist
    os.makedirs(output_folder)
average_cpu_time = process_file(input_folder, output_folder)  # Process the files
