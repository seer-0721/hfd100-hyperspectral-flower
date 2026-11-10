#!/usr/bin/env python3
"""测试数据加载"""

import h5py
import numpy as np
import torch

print("测试数据加载...")

# 测试加载小部分数据
with h5py.File('./dataset/train31.h5', 'r') as f:
    # 只加载前100个样本
    hsi = f['HSI'][:100]
    labels = f['label'][:100]
    
print(f"HSI shape: {hsi.shape}")
print(f"Labels shape: {labels.shape}")

# 测试band选择
band_indices = list(range(28))  # 前28个band
hsi_subset = hsi[:, band_indices, :, :]
print(f"\n选择前{len(band_indices)}个band后:")
print(f"HSI subset shape: {hsi_subset.shape}")

# 展平
hsi_flat = hsi_subset.reshape(hsi_subset.shape[0], -1)
print(f"展平后shape: {hsi_flat.shape}")
print(f"特征维度: {hsi_flat.shape[1]}")

# 转换为tensor
x = torch.FloatTensor(hsi_flat[:4])
y = torch.LongTensor(labels[:4])
print(f"\nTensor测试:")
print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")
print(f"x range: [{x.min():.4f}, {x.max():.4f}]")
print(f"y values: {y.tolist()}")

print("\n数据加载测试通过!")