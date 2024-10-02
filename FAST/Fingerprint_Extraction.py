

import os
import time

import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from scipy.signal import spectrogram
import pywt as wt
from sklearn.preprocessing import normalize
from scipy.signal import spectrogram
from PIL import Image
from datetime import datetime

class FeatureExtractor(object):

    def __init__(self, sampling_rate, window_length, window_lag, fingerprint_length, fingerprint_lag,
                 min_freq=0, max_freq=None, nfreq=32, ntimes=64,K=400):
        self.sampling_rate = sampling_rate  # / sampling rate
        self.window_len = window_length  # / length of window (seconds) used in spectrogram
        self.window_lag = window_lag  # / window lag (seconds) used in spectrogram
        self.fp_len = fingerprint_length  # / width of fingerprint (samples)
        self.fp_lag = fingerprint_lag  # / lag between fingerprints (samples)
        self.max_freq = self._initialize_frequencies(max_freq)  # / minimum and maximum frequencies for bandpass filter
        self.min_freq = min_freq
        self.new_d1 = int(
            nfreq)  # / number of frequency / time bins in fingerprints (must be power of 2) - TODO: error checking
        self.new_d2 = int(ntimes)
        self.d1 = None  # / dimension of spectral images prior to resizing
        self.d2 = None
        self.haar_means = None
        self.haar_stddevs = None
        self.haar_medians = None
        self.haar_absdevs = None
        self.K=K

    def _initialize_frequencies(self, max_freq):  # / initializes data structure
        if max_freq is None:
            max_freq = self.sampling_rate / 2.0
        return max_freq



    ########################################################################
    ##     FOR COMPUTING FINGERPRINTS                                     ##
    ########################################################################

    # / computes spectrogram from continous timeseries data
    def data_to_spectrogram(self, x_data):
        f, t, Sxx = spectrogram(x_data, fs=self.sampling_rate,
                                nperseg=int(self.sampling_rate * self.window_len),
                                noverlap=int(self.sampling_rate * (self.window_len - self.window_lag)))
        # Truncate spectrogram, keep only passband frequencies截断频谱图，仅保留通带频率
        if self.min_freq > 0:
            fidx_keep = (f >= self.min_freq)
            Sxx = Sxx[fidx_keep, :]
            f = f[fidx_keep]
        if self.max_freq < f[-1]:
            fidx_keep = (f <= self.max_freq)
            Sxx = Sxx[fidx_keep, :]
            f = f[fidx_keep]
        self.frequencies = f
        self.times = t
        return f, t, Sxx


    # / resizes each spectral image to specified dimensions将每个光谱图像的大小调整为指定尺寸
    def _resize_spectral_images(self, Sxx, new_d1, new_d2):
        new_spectral_image = np.array(Image.fromarray(Sxx).resize(size=(new_d2, new_d1),resample=Image.Resampling.BILINEAR))
        return new_spectral_image

    # / reshapes output from PyWavelets 2d wavelet transform into image将 PyWavelets 2d 小波变换的输出重塑为图像
    def _unwrap_wavelet_coeffs(self, coeffs):
        L = len(coeffs)
        cA = coeffs[0]
        for i in range(1, L):
            (cH, cV, cD) = coeffs[i]
            cA = np.concatenate((np.concatenate((cA, cV), axis=1), np.concatenate((cH, cD), axis=1)), axis=0)
        return cA

    # / computes wavelet transform for each spectral image为每个频谱图像计算其小波变换
    def spectral_images_to_wavelet(self, Sxx, wavelet=wt.Wavelet('db1')):
        if (int(self.new_d1) != self.d1) or (int(self.new_d2) != self.d2):
            Sxx = self._resize_spectral_images(Sxx, self.new_d1, self.new_d2)
        coeffs = wt.wavedec2(Sxx, wavelet)
        haar_images = self._unwrap_wavelet_coeffs(coeffs)
        return haar_images

    # / computes (normalized) haar_images from continous timeseries data从连续时间数据中计算标准化Haar小波图像
    def data_to_haar_images(self, x_data):
        f, t, Sxx = self.data_to_spectrogram(x_data)
        haar_image = self.spectral_images_to_wavelet(Sxx)
        haar_image = normalize(self._images_to_vectors(haar_image), axis=1)
        return haar_image, Sxx, t

    # / converts set of images to array of vectors将图像集转换为向量数组
    def _images_to_vectors(self, images):
        d1, d2 = np.shape(images)
        vectors= np.reshape(images, (1, d1 * d2))
        return vectors

    def standardize_haar(self, haar_image, type='MAD'):
        if type is 'Zscore':
            haar_mean = np.mean(haar_image, axis=0)
            haar_stddev = np.std(haar_image, axis=0)
            haar_image = (haar_image - haar_mean) / haar_stddev
            return haar_image
        elif type is 'MAD':
            medians=np.median(haar_image)
            tmp = abs(haar_image - medians)
            mad=np.median(tmp)
            haar_image = (haar_image - medians) / mad
            return haar_image
        else:
            print('Warning: invalid type - select type MAD or Zscore')
            return None

    def binarize_vectors_topK_sign(self, coeff_vectors):
        N, M = np.shape(coeff_vectors)
        binary_vectors = np.zeros((N, 2 * M), dtype=bool)
        for i in range(N):
            idx = np.argsort(abs(coeff_vectors[i, :]))[-self.K:]
            binary_vectors[i, idx] = coeff_vectors[i, idx] > 0
            binary_vectors[i, idx + M] = coeff_vectors[i, idx] < 0
        return binary_vectors



    def binarize_vectors_topK(self, coeff_vectors):
        N, M = np.shape(coeff_vectors)
        sign_vectors = np.zeros((N, M), dtype=bool)
        for i in range(N):
            idx = np.argsort(coeff_vectors[i, :])[-self.K:]
            sign_vectors[i, idx] = 1
        return sign_vectors


    def process_spectrogram_data(self,input_file, output_file):
        data = np.load(input_file)
        # 将数据转化为矩阵形式
        x_data = np.array(data)

        # 执行特征提取操作
        #f, t, Sxx = feats.data_to_spectrogram(x_data, window_type='hanning')
        haar_image, Sxx, t = self.data_to_haar_images(x_data)
        std_haar_images = self.standardize_haar(haar_image, type='MAD')
        binaryFingerprints = self.binarize_vectors_topK_sign(std_haar_images)
        binary_matrix = binaryFingerprints.astype(int)
        fp = binary_matrix.flatten()

        np.savetxt(output_file, fp)

        ##绘制其中一个haar小波图像的灰度图检验一下
        figure_matrix=binary_matrix.reshape(64, 64)

        plt.figure()
        plt.imshow(figure_matrix, cmap='binary', interpolation='nearest')
        plt.grid(True, which='both', color='black', linewidth=0.5)
        plt.show()
        # # 获取输入文件的文件名（不包括路径和扩展名）
        # file_name = os.path.basename(input_file)
        # file_name_without_extension = os.path.splitext(file_name)[0]
        #
        # # 设置图片标题为输入文件的名称
        # plt.title(file_name_without_extension)
        #
        # # 创建 fp_figure_folder 文件夹
        # fp_figure_folder = os.path.join(fig_folder, 'fp_figure')
        # if not os.path.exists(fp_figure_folder):
        #     os.makedirs(fp_figure_folder)
        #
        # # 保存图像到 fp_figure 文件夹
        # fig_file = os.path.join(fp_figure_folder, file_name_without_extension + '.png')
        # plt.savefig(fig_file)
        # 显示图像
        plt.show()


##初始化一个类
fe = FeatureExtractor(sampling_rate=100, window_length=10, window_lag=0.1, fingerprint_length=128,
                          fingerprint_lag=10, min_freq=1, max_freq=20, nfreq=32, ntimes=64,K=400)

def process_file(input_folder, output_base_folder):
    start_time = time.time()
    for filename in os.listdir(input_folder):
        if filename.endswith(".npy"):
            input_file=os.path.join(input_folder, filename).replace("\\", "/")
            input_file_name = os.path.basename(filename)
            input_file_name_without_extension = os.path.splitext(input_file_name)[0]
            new_file_name = f"{output_base_folder}/{input_file_name_without_extension}.txt"
            with open(new_file_name, "w") as file:
                # Write some content to the new txt file
                fe.process_spectrogram_data(input_file, new_file_name)
    end_time=time.time()
    average_cpu_time=end_time-start_time
    return average_cpu_time
input_folder="E:/D_FAST_data/snr2/signal_add_noise_wav"
output_folder ="E:/D_FAST_data/snr2/signal_add_noise_fp_fast"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
average_cpu_time=process_file(input_folder, output_folder)
input_folder="E:/D_FAST_data/noise/wave_strim"
output_folder ="E:/D_FAST_data/noise/fp_fast"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
average_cpu_time=process_file(input_folder, output_folder)