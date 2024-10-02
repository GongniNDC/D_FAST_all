import os

import matplotlib.pyplot as plt
import numpy as np

# 数据持续时间（以天为单位）
x_values = np.array([1, 3, 7, 14, 30, 90, 180, 365, 3650])  # 1天, 3天, 7天, 14天, 30天, 90天, 180天, 1年, 10年

# FAST 算法内存使用量（根据数据点）
y_fast = np.array([0.5, 1.2, 2.5, 5.5, 10, 30, 65, 113.5, 1368])  # 包括估算值


# LEDS+加权多探针LSH算法内存使用量（计算值）
def compute_memory_leds_lsh(x):
    # 计算 y1
    y1_leds_lsh = (24 * 3600 * 640 * x) / (8 * 10 ** 9)

    # 根据 FAST 算法数据计算 y2（基于 FAST 算法的 1/5）
    y2_fast = np.array([0.49277, 1.17831, 2.44939, 5.38229, 9.7553, 19.1, 45.7, 91.85, 1225.0])  # 包括估算值
    y2_leds_lsh = y2_fast / 5

    # 总内存使用量
    y_leds_lsh = y1_leds_lsh + y2_leds_lsh
    return y_leds_lsh


# 计算 LEDS+加权多探针LSH算法的 y 值
y_leds_lsh = compute_memory_leds_lsh(x_values)
result_folder='E:/D_FAST_data/results'
# 绘图
plt.figure(figsize=(12, 6))
plt.plot(x_values, y_fast, label='FAST Algorithm', marker='o', linestyle='-', color='blue')
plt.plot(x_values, y_leds_lsh, label='D-FAST Algorithm', marker='o', linestyle='-', color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Data Duration (days)',fontsize=16)
plt.ylabel('Memory Usage (GB)',fontsize=16)
#plt.title('Memory Usage Comparison: FAST Algorithm vs D-FAST Algorithm',fontsize=16)
plt.legend(fontsize=16)  # 设置图例字体大小
# 设置坐标轴的刻度字体大小
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(x_values, [str(x) + 'd' for x in x_values])
# 保存图像到 result_folder
plt.savefig(os.path.join(result_folder, 'memory_usage_comparison.png'), dpi=300, bbox_inches='tight')

# 显示图表
plt.show()
