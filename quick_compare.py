#!/usr/bin/env python3
"""快速对比不同band子集"""

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

print("=== 快速对比不同band子集 ===")

# 加载数据
with h5py.File('./dataset/train31.h5', 'r') as f:
    hsi_train = f['HSI'][:]
    labels_train = f['label'][:]

with h5py.File('./dataset/test31.h5', 'r') as f:
    hsi_test = f['HSI'][:]
    labels_test = f['label'][:]

print(f"训练集: {hsi_train.shape}, 测试集: {hsi_test.shape}")

# 定义band子集
band_subsets = {
    '10bands': list(range(10)),      # 极简短波
    '20bands': list(range(20)),      # 轻量短波
    '28bands': list(range(28)),      # 目标短波
    '31bands': list(range(31)),      # 全波段对照
}

# 简单模型
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 训练函数
def train_model(X_train, y_train, X_val, y_val, input_dim, epochs=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleClassifier(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 转换为tensor
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_t)
            val_preds = val_output.argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, val_acc={val_acc:.4f}")
    
    return best_val_acc, model

# 主实验循环
results = []

for name, band_indices in band_subsets.items():
    print(f"\n{'='*50}")
    print(f"实验: {name} (使用band: {band_indices})")
    
    # 选择band子集并应用空间池化
    pool_size = 8  # 64x64 -> 8x8
    N_train, C_train, H_train, W_train = hsi_train.shape
    N_test, C_test, H_test, W_test = hsi_test.shape
    
    # 选择band
    hsi_train_subset = hsi_train[:, band_indices, :, :]
    hsi_test_subset = hsi_test[:, band_indices, :, :]
    
    # 空间池化
    H_new = H_train // pool_size
    W_new = W_train // pool_size
    
    hsi_train_pooled = hsi_train_subset.reshape(
        N_train, len(band_indices), H_new, pool_size, W_new, pool_size
    ).mean(axis=(3, 5))
    
    hsi_test_pooled = hsi_test_subset.reshape(
        N_test, len(band_indices), H_new, pool_size, W_new, pool_size
    ).mean(axis=(3, 5))
    
    # 展平
    X_train = hsi_train_pooled.reshape(N_train, -1)
    X_test = hsi_test_pooled.reshape(N_test, -1)
    
    print(f"  特征维度: {X_train.shape[1]}")
    print(f"  训练样本: {X_train.shape[0]}")
    print(f"  测试样本: {X_test.shape[0]}")
    
    # 划分验证集
    val_size = 1000
    X_val = X_train[:val_size]
    y_val = labels_train[:val_size]
    X_train_sub = X_train[val_size:val_size+5000]  # 使用5000个样本训练
    y_train_sub = labels_train[val_size:val_size+5000]
    
    # 训练
    start_time = time.time()
    best_val_acc, model = train_model(
        X_train_sub, y_train_sub, X_val, y_val, 
        input_dim=X_train.shape[1], epochs=15
    )
    
    # 测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(labels_test).to(device)
    
    model.eval()
    with torch.no_grad():
        test_output = model(X_test_t)
        test_preds = test_output.argmax(dim=1)
        test_acc = (test_preds == y_test_t).float().mean().item()
    
    elapsed = time.time() - start_time
    
    results.append({
        'name': name,
        'num_bands': len(band_indices),
        'feature_dim': X_train.shape[1],
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'time': elapsed
    })
    
    print(f"  结果: 最佳验证准确率={best_val_acc:.4f}, 测试准确率={test_acc:.4f}")
    print(f"  耗时: {elapsed:.1f}秒")

# 输出总结
print(f"\n{'='*60}")
print("实验总结:")
print(f"{'='*60}")

print(f"{'实验':<10} {'Band数':<8} {'特征维度':<10} {'验证准确率':<12} {'测试准确率':<12} {'耗时(秒)':<10}")
print("-" * 70)

for r in results:
    print(f"{r['name']:<10} {r['num_bands']:<8} {r['feature_dim']:<10} "
          f"{r['best_val_acc']:<12.4f} {r['test_acc']:<12.4f} {r['time']:<10.1f}")

# 找出最佳结果
best_result = max(results, key=lambda x: x['test_acc'])
print(f"\n最佳结果: {best_result['name']} (测试准确率: {best_result['test_acc']:.4f})")

# 保存结果
np.savez('./quick_comparison_results.npz', results=results)
print(f"\n结果已保存到: ./quick_comparison_results.npz")