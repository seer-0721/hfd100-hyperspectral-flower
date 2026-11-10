#!/usr/bin/env python3
"""
HFD100短波子区间分类训练脚本
基于prototype.py中的模型
"""

import os
import sys
import argparse
import time
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from prototype import UltimateHybridModel, BaselineESNModel


class HFD100Dataset(Dataset):
    """HFD100数据集加载器"""
    
    def __init__(self, h5_path, band_indices=None, transform=None):
        """
        初始化数据集
        
        Args:
            h5_path: H5文件路径
            band_indices: 要使用的band索引列表，None表示使用所有band
            transform: 数据增强变换
        """
        super().__init__()
        self.h5_path = h5_path
        self.band_indices = band_indices
        self.transform = transform
        
        # 加载数据
        with h5py.File(h5_path, 'r') as f:
            self.hsi = f['HSI'][:]  # (N, 31, 64, 64)
            self.labels = f['label'][:]  # (N,)
            
        # 选择band子集
        if band_indices is not None:
            self.hsi = self.hsi[:, band_indices, :, :]
            
        # 展平空间维度，保留光谱维度作为特征
        # 将 (N, bands, 64, 64) 转换为 (N, bands*64*64)
        self.hsi = self.hsi.reshape(self.hsi.shape[0], -1)
        
        print(f"数据集加载完成: {h5_path}")
        print(f"  样本数: {len(self.hsi)}")
        print(f"  特征维度: {self.hsi.shape[1]}")
        print(f"  类别数: {len(np.unique(self.labels))}")
        if band_indices is not None:
            print(f"  使用的band索引: {band_indices}")
            print(f"  使用的band数: {len(band_indices)}")
        
    def __len__(self):
        return len(self.hsi)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.hsi[idx])
        y = torch.LongTensor([self.labels[idx]])[0]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = output.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        
        if batch_idx % 50 == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='HFD100短波子区间分类训练')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='数据目录')
    parser.add_argument('--model', type=str, default='ultimate', choices=['ultimate', 'baseline'], help='模型类型')
    parser.add_argument('--band_subset', type=str, default='28', help='band子集: 10,20,28,all')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 确定band子集
    if args.band_subset == 'all':
        band_indices = list(range(31))
    else:
        num_bands = int(args.band_subset)
        band_indices = list(range(min(num_bands, 31)))
    
    print(f"\n=== 实验配置 ===")
    print(f"模型类型: {args.model}")
    print(f"使用的band数: {len(band_indices)}")
    print(f"band索引: {band_indices}")
    print(f"批大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    
    # 加载训练数据
    train_dataset = HFD100Dataset(
        os.path.join(args.data_dir, 'train31.h5'),
        band_indices=band_indices
    )
    
    # 划分训练集和验证集
    val_size = int(len(train_dataset) * args.val_split)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    
    # 加载测试数据
    test_dataset = HFD100Dataset(
        os.path.join(args.data_dir, 'test31.h5'),
        band_indices=band_indices
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    input_dim = train_dataset.hsi.shape[1]
    num_classes = len(np.unique(train_dataset.labels))
    
    if args.model == 'ultimate':
        model = UltimateHybridModel(
            input_dim=input_dim,
            hidden_dims=(1024, 512, 256, 128),
            dropout_rates=(0.6, 0.5, 0.4, 0.3),
            num_classes=num_classes
        )
    else:
        model = BaselineESNModel(
            input_dim=input_dim,
            hidden_dims=(128, 64),
            dropout_rate=0.3,
            num_classes=num_classes
        )
    
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # 训练记录
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0
    best_model_path = os.path.join(args.output_dir, f'best_model_bands{len(band_indices)}.pth')
    
    print(f"\n=== 开始训练 ===")
    print(f"训练集: {len(train_subset)} 样本")
    print(f"验证集: {len(val_subset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 学习率调整
        scheduler.step(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'band_indices': band_indices,
                'model_config': model.get_config()
            }, best_model_path)
            print(f"  保存最佳模型，验证准确率: {val_acc:.4f}")
        
        epoch_time = time.time() - start_time
        print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        print(f"  时间: {epoch_time:.2f}秒")
    
    # 加载最佳模型进行测试
    print(f"\n=== 测试最佳模型 ===")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    
    # 生成分类报告
    print("\n分类报告:")
    print(classification_report(test_labels, test_preds, digits=4))
    
    # 保存结果
    results = {
        'band_indices': band_indices,
        'num_bands': len(band_indices),
        'model_type': args.model,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'test_preds': test_preds,
        'test_labels': test_labels,
        'config': vars(args)
    }
    
    results_path = os.path.join(args.output_dir, f'results_bands{len(band_indices)}.npz')
    np.savez(results_path, **results)
    
    # 绘制训练曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='训练损失')
    ax1.plot(val_losses, label='验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.set_title('训练和验证损失')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accs, label='训练准确率')
    ax2.plot(val_accs, label='验证准确率')
    ax2.axhline(y=test_acc, color='r', linestyle='--', label=f'测试准确率: {test_acc:.4f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率')
    ax2.set_title('训练和验证准确率')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, f'training_curve_bands{len(band_indices)}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\n训练曲线已保存: {plot_path}")
    
    print(f"\n=== 实验完成 ===")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"使用的band数: {len(band_indices)}")
    print(f"结果已保存到: {results_path}")
    print(f"模型已保存到: {best_model_path}")


if __name__ == '__main__':
    main()