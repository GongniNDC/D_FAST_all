import os
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# 文件路径
directory = 'E:/D_FAST_data/snr2/database_fast'

# Min-Hash签名的参数
p = 1000  # 哈希函数数量
fingerprint_length = 4096  # 假设二进制指纹的长度为4096
b = 300  # 哈希表数量
r = 6    # 每个哈希表由5个哈希值组成

# 预生成随机重排索引
def generate_permuted_indices(length, num_permutations):
    return np.array([np.random.permutation(length) for _ in range(num_permutations)])

# Min-Hash签名生成函数，基于指纹的随机重排
def min_hash_signature(fingerprint, permuted_indices):
    signature = []
    for i in range(len(permuted_indices)):
        for idx in permuted_indices[i]:
            if fingerprint[idx] == 1:
                signature.append(min(idx, 255))  # 若idx > 255，则取255
                break
        else:
            signature.append(0)  # 若没有1，则使用0
    return signature

# 读取每个txt文件并处理浮点数据为二进制指纹
def read_fingerprint_files(directory):
    fingerprints = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                fingerprint = [int(float(line.strip())) for line in f.readlines()]
                fingerprints.append((filename, fingerprint))  # 记录文件名和指纹
    return fingerprints

# 创建哈希表的数据库
def create_database(fingerprints, permuted_indices):
    hash_tables = [defaultdict(set) for _ in range(b)]  # 创建b个哈希表

    for filename, fingerprint in fingerprints:
        # 生成Min-Hash签名
        signature = min_hash_signature(fingerprint, permuted_indices)

        # 将签名分成b个哈希表，每个哈希表由r个哈希值连接成一个哈希键
        for i in range(b):
            start_index = i * r
            hash_key = tuple(signature[start_index:start_index + r])  # 生成哈希键
            hash_tables[i][hash_key].add(filename)  # 将文件名添加到相应的哈希桶中

    return hash_tables

# 查找查询指纹的候选指纹
def find_candidate_fingerprints(query_fingerprint, hash_tables, permuted_indices, threshold):
    # 生成查询指纹的Min-Hash签名
    query_signature = min_hash_signature(query_fingerprint, permuted_indices)

    # 用于记录候选指纹文件名
    candidate_fingerprints = defaultdict(int)  # 使用字典来记录每个候选指纹在相同哈希桶中的出现次数

    # 生成查询指纹的哈希键并在哈希表中寻找候选指纹
    for i in range(b):
        start_index = i * r
        hash_key = tuple(query_signature[start_index:start_index + r])  # 生成哈希键

        # 检查该哈希键对应的哈希桶是否存在
        if hash_key in hash_tables[i]:
            # 记录位于相同哈希桶中的指纹
            for candidate in hash_tables[i][hash_key]:
                candidate_fingerprints[candidate] += 1  # 增加该指纹的计数

    # 判断候选指纹
    valid_candidates = [candidate for candidate, count in candidate_fingerprints.items() if count > threshold]

    return valid_candidates

# 读取指纹文件
fingerprints = read_fingerprint_files(directory)

# 生成随机重排索引
permuted_indices = generate_permuted_indices(fingerprint_length, p)

# 创建数据库
hash_tables = create_database(fingerprints, permuted_indices)

# 存储不同阈值下的召回率和精确率
recall_list = []
precision_list = []
thresholds = range(160, 200,2)

for threshold in thresholds:
    TP_list = 0
    FP_list = 0

    # 遍历前100个指纹作为查询
    for i in range(100):
        TP=0
        FP=0
        query_fingerprint = fingerprints[i][1]  # 获取第i个指纹的二进制数据
        candidates = find_candidate_fingerprints(query_fingerprint, hash_tables, permuted_indices, threshold)

        # 统计TP和FP
        for candidate in candidates:
            if candidate.startswith("noise"):
                FP += 1  # 假阳性
            else:
                TP += 1  # 真阳性
        TP_list += TP
        FP_list += FP
    TP=TP_list/100
    FP=FP_list/100
    # 计算召回率和精确率
    recall = TP / 100  # 真实正例为100
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    recall_list.append(recall)
    precision_list.append(precision)

# 绘制精确率和召回率的曲线
plt.figure()
plt.plot(recall_list, precision_list, marker='o')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.show()
print('recall=',recall_list)
print('precision=',precision_list)

