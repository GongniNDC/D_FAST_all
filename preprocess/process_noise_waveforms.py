import os
import shutil
import pandas as pd
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import filtfilt, butter, spectrogram


class Trace:
    def __init__(self, data, metadata):
        self.data = data  # Initialize Trace with data and metadata
        self.metadata = metadata


def process_h5(input_file, trace_name, row):
    """Process HDF5 file to extract trace data."""
    f = h5py.File(input_file, "r")  # Open HDF5 file
    bucket, array = trace_name.split('$')  # Split trace name to get bucket and array
    x, y, z = iter([int(i) for i in array.split(',:')])  # Extract indices
    data = f[f'/data/{bucket}'][x, :y, :z]  # Read data from HDF5 file

    trace = Trace(data, row)  # Create a Trace object
    return trace  # Return the Trace object


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a Butterworth bandpass filter."""
    nyquist = 0.5 * fs  # Calculate Nyquist frequency
    low = lowcut / nyquist  # Normalize low cutoff frequency
    high = highcut / nyquist  # Normalize high cutoff frequency
    b, a = butter(order, [low, high], btype='band')  # Get filter coefficients
    return b, a  # Return filter coefficients


def data_to_spectrogram(x_data, sampling_rate, window_len, window_lag, min_freq, max_freq):
    """Convert data to a spectrogram."""
    f, t, Sxx = spectrogram(x_data, fs=sampling_rate,
                            nperseg=int(sampling_rate * window_len),
                            noverlap=int(sampling_rate * (window_len - window_lag)))

    # Truncate spectrogram, keep only passband frequencies
    if min_freq > 0:
        fidx_keep = (f >= min_freq)
        Sxx = Sxx[fidx_keep, :]
        f = f[fidx_keep]
    if max_freq < f[-1]:
        fidx_keep = (f <= max_freq)
        Sxx = Sxx[fidx_keep, :]
        f = f[fidx_keep]
    frequencies = f
    times = t
    return f, t, Sxx  # Return frequencies, times, and the spectrogram


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply Butterworth bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)  # Get filter coefficients
    y = filtfilt(b, a, data)  # Apply filter to data
    return y  # Return filtered data


def process_noise_waveforms():
    """Process noise waveforms by filtering and saving them."""
    sta = 'B023'  # Define station
    folder_path1 = "E:/D_FAST_data/noise_waveforms/original"  # Path to original waveforms
    wav_strim_path = "E:/D_FAST_data/noise_waveforms/wave_strim"  # Path to save processed waveforms

    # Create directory for processed waveforms if it doesn't exist
    if not os.path.exists(wav_strim_path):
        os.makedirs(wav_strim_path)

    # Get all .pkl files in the specified directory
    pkl_files = [f for f in os.listdir(folder_path1) if
                 os.path.isfile(os.path.join(folder_path1, f)) and f.endswith('.pkl')]

    # Loop through each .pkl file
    for file in pkl_files:
        file_path = os.path.join(folder_path1, file)  # Construct full file path
        with open(file_path, 'rb') as f:
            data = pickle.load(f)  # Load data from the pickle file
            wav = data.data[2]  # Extract waveform data
            # P_arrival_time = data.metadata.trace_P_arrival_sample  # (Commented out, not used)
            lowcut = 4  # Low cutoff frequency for bandpass filter
            highcut = 10  # High cutoff frequency for bandpass filter
            fs = 100  # Sampling frequency
            order = 5  # Filter order

            # Apply bandpass filter to the waveform
            wav = butter_bandpass_filter(wav, lowcut, highcut, fs, order)
            wav_strim = wav[2000:9000]  # Trim the waveform to the desired length
            np.save(os.path.join(wav_strim_path, f"{file}.npy"), wav_strim)  # Save trimmed waveform as .npy file


# Call the function to process noise waveforms
process_noise_waveforms()
