import numpy as np
import os
import random
import matplotlib.pyplot as plt

# 读取波形文件
waveform = np.load('E:/D_FAST_data/ep_waveforms/B023/P_waveform/uw10778083_B023.pkl.npy')
noise_folder = 'E:/D_FAST_data/noise_waveforms/wave_strim'
signal_noise_folder = 'E:/D_FAST_data/signal_noise'
waveform_plot_folder = 'E:/D_FAST_data/waveform_plots'

# 创建保存波形图的文件夹（如果不存在）
if not os.path.exists(waveform_plot_folder):
    os.makedirs(waveform_plot_folder)

# 获取噪声文件列表
noise_files = [f for f in os.listdir(noise_folder) if f.endswith('.npy')]

# 选择前100个噪声文件
selected_noise_files = noise_files[:100]


# 计算原始信号的功率
signal_power = np.mean(waveform ** 2)

# 设置信噪比（以 dB 为单位）
SNR_dB = 2  # 设置 SNR 为 2 dB

# 将 SNR 转换为线性形式
SNR_linear = 10 ** (SNR_dB / 10)

# 根据 SNR 计算噪声的功率
noise_power = signal_power / SNR_linear

for i, noise_file in enumerate(selected_noise_files):
    # 加载噪声信号
    noise_signal = np.load(os.path.join(noise_folder, noise_file))

    # 确保噪声信号与波形长度一致
    if len(noise_signal) != len(waveform):
        print(f"Warning: Noise signal {noise_file} length does not match the waveform length.")
        continue

    # 计算噪声信号的当前功率
    current_noise_power = np.mean(noise_signal ** 2)

    # 调整噪声信号的功率到指定的噪声功率
    scaling_factor = np.sqrt(noise_power / current_noise_power)
    adjusted_noise_signal = noise_signal * scaling_factor

    # 将波形插入到调整后的噪声信号中
    combined_signal = adjusted_noise_signal + waveform

    # 保存带噪声的信号到指定文件夹
    combined_signal_filename = os.path.join(signal_noise_folder, f'combined_signal_{i+1}.npy')
    np.save(combined_signal_filename, combined_signal)

    # 绘制波形图
    plt.figure(figsize=(10, 4))
    plt.plot(combined_signal, label=f'Combined Signal {i+1} (SNR = {SNR_dB} dB)')
    plt.title(f'Combined Signal {i+1} (SNR = {SNR_dB} dB)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()

    # 保存波形图到新的文件夹
    plot_filename = os.path.join(waveform_plot_folder, f'combined_signal_{i+1}.png')
    plt.savefig(plot_filename)
    plt.close()

    print(f'Saved combined signal {i+1} with SNR {SNR_dB} dB and its waveform plot.')
