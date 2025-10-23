# 回译数据增强实验设计

## 📊 实验目标
测试不同回译替换比例对检索性能的影响，找出最优替换率。

## 🎯 已完成实验结果

### RSITMD数据集
| 替换比例 | 实际替换率 | 最佳Epoch | R_mean | 状态 |
|---------|-----------|----------|--------|------|
| 0% (Baseline) | 0% | 3 | **52.15%** | ✅ 完成 |
| 10% | 4.6% | 3 | 52.14% | ✅ 完成 |
| 20% | 9.2% | - | - | ⏳ 待测试 |
| 30% | 13.9% | - | - | ⏳ 待测试 |
| 40% | 18.5% | - | - | ⏳ 待测试 |
| 50% | 23.1% | - | - | ⏳ 待测试 |
| 60% | 27.7% | - | - | ⏳ 待测试 |
| 70% | 32.4% | - | - | ⏳ 待测试 |
| 80% | 37.0% | - | - | ⏳ 待测试 |

### RSICD数据集
| 替换比例 | 实际替换率 | 最佳Epoch | R_mean | 状态 |
|---------|-----------|----------|--------|------|
| 0% (Baseline) | 0% | 1 | **38.43%** | ✅ 完成 |
| 10% | 5.0% | 3 | 38.53% | ✅ 完成 |
| 20% | 10.1% | - | - | ⏳ 待测试 |
| 30% | 15.1% | - | - | ⏳ 待测试 |
| 40% | 20.1% | - | - | ⏳ 待测试 |
| 50% | 25.1% | - | - | ⏳ 待测试 |
| 60% | 30.2% | - | - | ⏳ 待测试 |
| 70% | 35.2% | - | - | ⏳ 待测试 |
| 80% | 40.2% | - | - | ⏳ 待测试 |

## 🚀 快速运行命令

### 运行单个实验
```bash
cd /data2/ls/SemRL-RS-Retrieval
conda activate /data/env/semrl

# RSITMD + 30% 增强
python run.py --task 'itr_rsitmd_geo' --dist "f023" \
  --config 'configs/Retrieval_rsitmd_geo_aug30pct.yaml' \
  --output_dir './checkpoints/HARMA/rsitmd_geo_aug30pct' \
  2>&1 | tee training_rsitmd_aug30pct.log &

# RSITMD + 50% 增强
python run.py --task 'itr_rsitmd_geo' --dist "f023" \
  --config 'configs/Retrieval_rsitmd_geo_aug50pct.yaml' \
  --output_dir './checkpoints/HARMA/rsitmd_geo_aug50pct' \
  2>&1 | tee training_rsitmd_aug50pct.log &

# RSITMD + 80% 增强
python run.py --task 'itr_rsitmd_geo' --dist "f023" \
  --config 'configs/Retrieval_rsitmd_geo_aug80pct.yaml' \
  --output_dir './checkpoints/HARMA/rsitmd_geo_aug80pct' \
  2>&1 | tee training_rsitmd_aug80pct.log &

# RSICD + 30% 增强
python run.py --task 'itr_rsicd_geo' --dist "f023" \
  --config 'configs/Retrieval_rsicd_geo_aug30pct.yaml' \
  --output_dir './checkpoints/HARMA/rsicd_geo_aug30pct' \
  2>&1 | tee training_rsicd_aug30pct.log &
```

### 查看训练状态
```bash
# 查看所有训练进程
ps aux | grep "python.*Retrieval" | grep -v grep

# 实时查看日志
tail -f training_rsitmd_aug30pct.log

# 查看GPU使用情况
nvidia-smi
```

### 提取最佳结果
```bash
# 查看某个实验的最佳结果
tail -100 checkpoints/HARMA/rsitmd_geo_aug30pct/*.log.txt | grep "best epoch"
```

## 📁 已生成文件

### 数据集文件
```
data/augmented/
├── rsitmd_train_backtrans_10pct.json  (4.6% 替换)
├── rsitmd_train_backtrans_20pct.json  (9.2% 替换)
├── rsitmd_train_backtrans_30pct.json  (13.9% 替换)
├── rsitmd_train_backtrans_40pct.json  (18.5% 替换)
├── rsitmd_train_backtrans_50pct.json  (23.1% 替换)
├── rsitmd_train_backtrans_60pct.json  (27.7% 替换)
├── rsitmd_train_backtrans_70pct.json  (32.4% 替换)
├── rsitmd_train_backtrans_80pct.json  (37.0% 替换)
├── rsicd_train_backtrans_10pct.json   (5.0% 替换)
├── rsicd_train_backtrans_20pct.json   (10.1% 替换)
├── rsicd_train_backtrans_30pct.json   (15.1% 替换)
├── rsicd_train_backtrans_40pct.json   (20.1% 替换)
├── rsicd_train_backtrans_50pct.json   (25.1% 替换)
├── rsicd_train_backtrans_60pct.json   (30.2% 替换)
├── rsicd_train_backtrans_70pct.json   (35.2% 替换)
└── rsicd_train_backtrans_80pct.json   (40.2% 替换)
```

### 配置文件（关键比例）
```
configs/
├── Retrieval_rsitmd_geo_aug30pct.yaml
├── Retrieval_rsitmd_geo_aug50pct.yaml
├── Retrieval_rsitmd_geo_aug80pct.yaml
├── Retrieval_rsicd_geo_aug30pct.yaml
├── Retrieval_rsicd_geo_aug50pct.yaml
└── Retrieval_rsicd_geo_aug80pct.yaml
```

## 💡 实验建议

### 推荐测试顺序
1. **先测试RSITMD**（数据量小，训练快）
   - 30%, 50%, 80% 三个关键点
   - 每个约1-2小时

2. **根据RSITMD结果**决定RSICD测试范围
   - 如果30%效果好 → 测试20-40%区间
   - 如果50%效果好 → 测试40-60%区间
   - 如果80%效果好 → 测试70-90%区间

### Early Stopping
- 最优结果通常在**Epoch 1-5**出现
- 如果5个epoch后持续下降，可以提前终止（Ctrl+C）
- 模型会保存`checkpoint_best.pth`（最佳epoch权重）

## 🎯 初步观察

**10%替换结果分析：**
- RSITMD: 几乎无提升 (52.15% → 52.14%)
- RSICD: 轻微提升 (38.43% → 38.53%)

**可能原因：**
- ✅ 替换率太低，影响不够显著
- ❓ 回译质量可能不如原始标注
- ❓ 遥感领域专业性强，回译容易损失精确性

**下一步：**
需要测试更高替换率，寻找性能拐点。

