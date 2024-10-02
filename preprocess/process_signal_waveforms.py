import os
import shutil
from obspy.signal.filter import bandpass
import pickle
from scipy.signal import filtfilt, butter, spectrogram
import numpy as np
import matplotlib.pyplot as plt  # Ensure matplotlib is imported
from obspy import Trace


# Function to design a Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist  # Normalized low cutoff frequency
    high = highcut / nyquist  # Normalized high cutoff frequency
    b, a = butter(order, [low, high], btype='band')  # Filter coefficients
    return b, a


# Butterworth bandpass filter function
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)  # Get filter coefficients
    y = filtfilt(b, a, data)  # Apply filter to data
    return y


# Function to convert data to spectrogram
def data_to_spectrogram(x_data, sampling_rate, window_len, window_lag, min_freq, max_freq):
    f, t, Sxx = spectrogram(x_data, fs=sampling_rate,
                            nperseg=int(sampling_rate * window_len),
                            noverlap=int(sampling_rate * (window_len - window_lag)))
    # Truncate spectrogram to keep only passband frequencies
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


# Define station and type
sta = 'B023'
type = 'eq'
# Define paths for waveform data and output
waveform_path = f"E:/D_FAST_data/{type}_waveforms/{sta}/original"
wav_strim_path = f"E:/D_FAST_data/{type}_waveforms/{sta}/P_waveform"
spectrogram_plot_path = f"E:/D_FAST_data/{type}_waveforms/{sta}/spectrogram_plot"
waveform_plot_path = f"E:/D_FAST_data/{type}_waveforms/{sta}/waveform_plot"

# Create necessary directories
os.makedirs(waveform_path, exist_ok=True)
os.makedirs(wav_strim_path, exist_ok=True)
os.makedirs(spectrogram_plot_path, exist_ok=True)
os.makedirs(waveform_plot_path, exist_ok=True)

# List all .pkl files in the original waveform directory
pkl_files = [f for f in os.listdir(waveform_path) if
             os.path.isfile(os.path.join(waveform_path, f)) and f.endswith('.pkl')]

# Loop through each .pkl file
for file in pkl_files:
    file_path = os.path.join(waveform_path, file)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)  # Load the data from the pickle file
        wav = data.data[2]  # Extract waveform data
        P_arrival_time = data.metadata.trace_P_arrival_sample  # Get P-wave arrival time
        lowcut = 10  # Low cutoff frequency for bandpass filter
        highcut = 20  # High cutoff frequency for bandpass filter
        fs = 100  # Sampling frequency
        order = 5  # Filter order

        # ############ Plot the spectrogram of the original signal #############
        # f, t, Sxx = data_to_spectrogram(wav, fs, 10, 0.1, 1, 50)
        # plt.figure(figsize=(10, 6))
        # plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='jet')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.colorbar(label='Intensity [dB]')
        # plt.ylim(0, max(f))
        # plt.title('Original Signal Spectrogram')
        # plt.savefig(os.path.join(spectrogram_plot_path, f"{file}_spectrogram.png"))
        # plt.close()  # Close the figure to save memory

        ####### Plot the spectrogram of the bandpass filtered signal
        wav1 = bandpass(wav, freqmin=lowcut, freqmax=highcut, df=fs, corners=order, zerophase=True)
        plt.figure()
        plt.plot(wav1)  # Plot the filtered waveform
        plt.title(f"Waveform for {file}")
        plt.grid(True)
        # Mark the P-wave arrival time in the plot
        plt.axvline(x=P_arrival_time, color='red', linestyle='--', label='P-wave time (p_time)')
        plt.text(P_arrival_time, max(wav1), 'P-wave time (p_time)', rotation=90)
        plt.savefig(os.path.join(waveform_plot_path, f"{file}.png"))  # Save the waveform plot
        # Show the waveform plot
        # plt.show()

        # ####### Trim waveform ##########
        start_time = int(P_arrival_time - 200)  # Start trimming 200 samples before P-wave arrival
        end_time = int(P_arrival_time + 6800)  # End trimming 6800 samples after P-wave arrival
        wav_strim = wav1[start_time:end_time]  # Trimmed waveform
        # Save preprocessed waveform data to .npy file
        np.save(os.path.join(wav_strim_path, f"{file}.npy"), wav_strim)
