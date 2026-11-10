#!/usr/bin/env python3
"""简单测试"""

import numpy as np
import torch

print("测试空间池化...")

# 模拟数据: 2个样本, 3个band, 8x8空间
data = np.random.randn(2, 3, 8, 8).astype(np.float32)
print(f"原始数据shape: {data.shape}")

# 手动池化
pool_size = 4
N, C, H, W = data.shape
H_new = H // pool_size
W_new = W // pool_size
data_reshaped = data.reshape(N, C, H_new, pool_size, W_new, pool_size)
data_pooled = data_reshaped.mean(axis=(3, 5))
print(f"池化后shape: {data_pooled.shape}")

# 展平
data_flat = data_pooled.reshape(data_pooled.shape[0], -1)
print(f"展平后shape: {data_flat.shape}")
print(f"特征维度: {data_flat.shape[1]}")

print("\n测试通过!")