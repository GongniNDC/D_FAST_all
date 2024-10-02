import os

import matplotlib.pyplot as plt
recall_fast= [1.0, 1.0, 0.9992, 0.9982, 0.992, 0.9843999999999999, 0.9673999999999999, 0.9386, 0.895, 0.8376, 0.7678, 0.691, 0.6002000000000001, 0.5136, 0.4224, 0.33640000000000003, 0.257, 0.192, 0.1424, 0.0984]
precision_fast= [0.30428432327166505, 0.31547731718089467, 0.3330111648058657, 0.35905183266788965, 0.39452752147629655, 0.4447657344237112, 0.5056449926824169, 0.577920078812881, 0.6537618699780862, 0.7311452513966481, 0.8002084418968213, 0.8614885924448323, 0.9037795512723988, 0.9377396384882235, 0.9613108784706417, 0.9747899159663866, 0.9865642994241842, 0.992248062015504, 0.9951083158630328, 0.9949443882709808]
Recall_D_FAST= [0.9932, 0.9894, 0.9652, 0.9571999999999999, 0.9006000000000001, 0.8908, 0.8, 0.7844, 0.6496, 0.6278, 0.4938, 0.451, 0.3398, 0.2898, 0.218, 0.165, 0.1242, 0.08779999999999999, 0.0724, 0.0446, 0.036000000000000004, 0.023799999999999998, 0.019799999999999998, 0.0158, 0.013999999999999999, 0.012, 0.0118, 0.011200000000000002, 0.0102, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
Precision_D_FAST= [0.7848795621479083, 0.8448607864089621, 0.923914454602741, 0.9436439497208937, 0.9778564580025556, 0.9830700353989026, 0.9938775658337947, 0.9940114270056861, 0.9984271331453649, 0.9987024361696315, 0.9996309840425532, 0.9995887096774193, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

result_folder='E:/D_FAST_data/results'
# 绘制ROC曲线
plt.figure(figsize=(10, 6))

# 绘制第一组数据
plt.plot(recall_fast, precision_fast, marker='o', label='Fast Method', color='blue')

# 绘制第二组数据
plt.plot(Recall_D_FAST, Precision_D_FAST, marker='o', label='D_FAST Method', color='red')

# 设置图表标题和标签
plt.title('ROC Curves for Different Methods', fontsize=16)  # 设置标题字体大小
plt.xlabel('Recall', fontsize=16)  # 设置x轴标签字体大小
plt.ylabel('Precision', fontsize=16)  # 设置y轴标签字体大小
plt.xlim(0, 1.01)
plt.ylim(0, 1.01)
plt.grid()

# 设置坐标轴刻度值的字体大小
plt.tick_params(axis='both', labelsize=16)  # 设置x和y轴刻度值的字体大小为16号

# 设置图例字体大小
plt.legend(fontsize=16)  # 设置图例字体大小

# 保存图像到 result_folder
plt.savefig(os.path.join(result_folder, 'roc_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()