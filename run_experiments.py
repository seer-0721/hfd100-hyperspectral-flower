#!/usr/bin/env python3
"""运行多个实验比较不同band子集"""

import subprocess
import time
import os

experiments = [
    # (band_subset, pool_size, description)
    ("10", 4, "前10个band，池化4x4"),
    ("20", 4, "前20个band，池化4x4"),
    ("28", 4, "前28个band，池化4x4"),
    ("all", 4, "所有31个band，池化4x4"),
]

base_cmd = [
    "python", "train_hfd100_v2.py",
    "--epochs", "30",
    "--batch_size", "64",
    "--lr", "0.001",
]

print("=== 开始运行多个实验 ===")
print(f"基础命令: {' '.join(base_cmd)}")

results = []

for band_subset, pool_size, description in experiments:
    output_dir = f"./outputs_bands{band_subset}_pool{pool_size}"
    cmd = base_cmd + [
        "--band_subset", band_subset,
        "--pool_size", str(pool_size),
        "--output_dir", output_dir,
    ]
    
    print(f"\n{'='*60}")
    print(f"实验: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 运行实验
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30分钟超时
        
        # 解析结果
        if result.returncode == 0:
            # 从输出中提取测试准确率
            for line in result.stdout.split('\n'):
                if "测试准确率:" in line:
                    test_acc = float(line.split(":")[1].strip())
                    break
            else:
                test_acc = 0.0
            
            results.append({
                'band_subset': band_subset,
                'pool_size': pool_size,
                'description': description,
                'test_acc': test_acc,
                'success': True,
                'time': time.time() - start_time,
            })
            
            print(f"✓ 实验成功完成")
            print(f"  测试准确率: {test_acc:.4f}")
            print(f"  耗时: {time.time() - start_time:.1f}秒")
            
        else:
            print(f"✗ 实验失败，返回码: {result.returncode}")
            print(f"  错误输出: {result.stderr[:500]}...")
            
            results.append({
                'band_subset': band_subset,
                'pool_size': pool_size,
                'description': description,
                'test_acc': 0.0,
                'success': False,
                'error': result.stderr[:500],
                'time': time.time() - start_time,
            })
            
    except subprocess.TimeoutExpired:
        print(f"✗ 实验超时 (30分钟)")
        results.append({
            'band_subset': band_subset,
            'pool_size': pool_size,
            'description': description,
            'test_acc': 0.0,
            'success': False,
            'error': 'Timeout',
            'time': 1800,
        })
    except Exception as e:
        print(f"✗ 实验异常: {e}")
        results.append({
            'band_subset': band_subset,
            'pool_size': pool_size,
            'description': description,
            'test_acc': 0.0,
            'success': False,
            'error': str(e),
            'time': time.time() - start_time,
        })

# 输出总结
print(f"\n{'='*60}")
print("实验总结:")
print(f"{'='*60}")

for i, r in enumerate(results, 1):
    if r['success']:
        print(f"{i}. {r['description']}: 测试准确率 = {r['test_acc']:.4f}, 耗时 = {r['time']:.1f}秒")
    else:
        print(f"{i}. {r['description']}: 失败 - {r.get('error', 'Unknown error')}")

# 保存结果到文件
with open("./experiment_results.txt", "w") as f:
    f.write("实验结果总结\n")
    f.write("="*50 + "\n")
    for r in results:
        if r['success']:
            f.write(f"{r['description']}: 测试准确率 = {r['test_acc']:.4f}\n")
        else:
            f.write(f"{r['description']}: 失败\n")
    
    # 找出最佳结果
    successful = [r for r in results if r['success']]
    if successful:
        best = max(successful, key=lambda x: x['test_acc'])
        f.write(f"\n最佳结果: {best['description']} (准确率: {best['test_acc']:.4f})\n")

print(f"\n结果已保存到: ./experiment_results.txt")