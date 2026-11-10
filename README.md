# HFD100 高光谱短波子区间花卉分类

**Hyperspectral Flower Classification on Shortwave Band Subsets (HFD100)**

用 HFD100 高光谱数据集的前 28 个 band（约 375–450 nm）做 10 类花卉分类，
验证「短波窄带子集是否足以支撑高精度判别」这一假设，为拿不到的目标波段
（290–450 nm）提供可行的替代实验方案。

| 项目 | 数值 |
| --- | --- |
| 数据集 | HFD100，31 band（由 256 band 重采样），64×64 空间分辨率 |
| 类别数 | 10（分布均衡，各约 10%） |
| 训练 / 测试样本 | 7,239 / 4,827 |
| 最高测试准确率 | **99.09%**（前 28 band，8×8 池化） |

## 方法

光谱-空间联合特征 + 深度 MLP 读出（`prototype.py` 中的 `UltimateHybridModel`）：
按 band 子集取光谱切片，做空间池化压缩维度，拼接后送入多层感知机分类。
Adam（lr=1e-3，weight_decay=1e-4），30 epoch，batch 64，20% 验证集。

## Band 消融

四个配置的结果中，前 10 / 20 / 28 band（8×8 池化）三行直接来自 `outputs_*/results_*.npz`；
28 band + 4×4 池化一行来自 `experiment_log.md` 的实验 1b 记录（该次运行未保存 npz）：

| 配置 | 特征维度 | 最佳验证 | 测试 |
| --- | --- | --- | --- |
| 前 10 band，4×4 池化 | 10×16×16 = 2,560 | 96.20% | 95.38% |
| 前 20 band，4×4 池化 | 20×16×16 = 5,120 | 99.45% | 98.96% |
| 前 28 band，8×8 池化 | 28×8×8 = 1,792 | 99.38% | 99.09% |
| 前 28 band，4×4 池化 | 28×16×16 = 7,168 | 99.10% | 98.61% |

三条结论：

1. **信息量在 20 个 band 附近饱和**：10 → 20 band 测试 +3.58 个百分点，
   20 → 28 band 只 +0.13，继续加 band 的边际收益迅速衰减。
2. **光谱比空间分辨率更关键**：同样 28 band，8×8 池化（1,792 维）反而比
   4×4 池化（7,168 维）高 0.48 个点——扩大 4 倍空间维度没有换来精度。
3. 「前 28 band 足够」的假设成立，短波窄带子集在花卉分类上是可行的替代方案。

## 结果口径（务必先读）

1. **28 band 只是 375–450 nm 的近似窗，不是真实 290–450 nm 传感器数据。**
   本项目是「在可获得数据上做的替代验证」，不能外推为「290–450 nm 也能达到 99%」。
   原始 256 band 已被重采样为 31 band，短波窗口的粒度因此较粗。
2. 精度为**单次随机划分**下的结果，未做多种子平均。表内 ±0.5 个百分点以内的差异
   （例如 99.09% vs 98.96%）不足以支撑方法优劣的结论，只应看作同一水平。
3. 验证集参与模型选择，测试集只用于最终评估；两者已分开，但划分固定为一份。

## 运行

```bash
pip install -r requirements.txt

# 数据：HFD100（不随仓库发布，约 4.3 GB），放到 dataset/
#   dataset/train31.h5   (7,239 × 31 × 64 × 64)
#   dataset/test31.h5    (4,827 × 31 × 64 × 64)

# 单次训练（前 28 band，8×8 池化）
python train_hfd100.py --bands 28 --pool 8

# 一次跑完全部 4 个消融配置
python run_experiments.py
```

每组实验的摘要写入 `outputs_bands{N}[_pool{P}]/results_bands{N}.npz`，
包含 `best_val_acc`、`test_acc`、`train_losses`、`train_accs` 与所用 `band_indices`。

## 仓库结构

```text
prototype.py           UltimateHybridModel 与特征构造
train_hfd100.py        HFD100 训练主脚本
train_hfd100_v2.py     训练脚本（v2，实验批量复用）
run_experiments.py     4 组 band/池化消融的批处理入口
quick_compare.py       配置间快速对比
experiment_log.md      实验设计、过程与结论的原始记录
outputs_bands*/        各配置的 results_*.npz（结果证据，已入库）
```

## 数据来源

HFD100（Hyperspectral Flower Dataset）为公开学术数据集，版权归发布方所有；
本仓库仅包含加载、切分与训练代码，不含数据本身。
