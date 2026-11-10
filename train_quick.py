#!/usr/bin/env python3
"""快速训练测试"""

import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

print("=== 快速训练测试 ===")

# 加载少量数据
with h5py.File('./dataset/train31.h5', 'r') as f:
    hsi = f['HSI'][:1000]  # 前1000个样本
    labels = f['label'][:1000]

print(f"数据shape: {hsi.shape}")
print(f"标签shape: {labels.shape}")

# 使用前28个band
band_indices = list(range(28))
hsi_subset = hsi[:, band_indices, :, :]
print(f"使用前{len(band_indices)}个band后shape: {hsi_subset.shape}")

# 空间池化 (8x8 -> 1x1，最大程度降维)
pool_size = 64  # 64x64 -> 1x1
N, C, H, W = hsi_subset.shape
H_new = H // pool_size
W_new = W // pool_size
hsi_reshaped = hsi_subset.reshape(N, C, H_new, pool_size, W_new, pool_size)
hsi_pooled = hsi_reshaped.mean(axis=(3, 5))
print(f"池化后shape: {hsi_pooled.shape}")

# 展平
hsi_flat = hsi_pooled.reshape(hsi_pooled.shape[0], -1)
print(f"展平后shape: {hsi_flat.shape}")
print(f"特征维度: {hsi_flat.shape[1]}")

# 创建简单模型
class SimpleModel(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 准备数据
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

X = torch.FloatTensor(hsi_flat)
y = torch.LongTensor(labels[:1000])

# 划分训练/验证
train_size = 800
val_size = 200
X_train, X_val = X[:train_size], X[train_size:train_size+val_size]
y_train, y_val = y[:train_size], y[train_size:train_size+val_size]

# 创建模型
model = SimpleModel(input_dim=hsi_flat.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")

# 快速训练
print("\n开始快速训练...")
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    output = model(X_train.to(device))
    loss = criterion(output, y_train.to(device))
    loss.backward()
    optimizer.step()
    
    # 验证
    model.eval()
    with torch.no_grad():
        val_output = model(X_val.to(device))
        val_loss = criterion(val_output, y_val.to(device))
        val_preds = val_output.argmax(dim=1)
        val_acc = (val_preds == y_val.to(device)).float().mean()
    
    print(f"Epoch {epoch+1}: 训练损失={loss.item():.4f}, 验证损失={val_loss.item():.4f}, 验证准确率={val_acc.item():.4f}")

print("\n=== 快速测试完成 ===")