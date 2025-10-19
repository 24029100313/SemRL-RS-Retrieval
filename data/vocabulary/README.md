# 遥感三层语义词表 (Remote Sensing Three-Granularity Vocabulary)

**版本**: v1.3 (Final)  
**日期**: 2025-10-18  
**作者**: Sean R. Liang

## 📊 概览

本词表为遥感图像-文本检索任务设计，包含三个语义粒度：

| 粒度 | 词数 | 描述 | 示例 |
|------|------|------|------|
| **Object** | 190 | 地物/设施 | buildings, trees, cars, airport, stadium |
| **Scene** | 177 | 场景/属性 | residential, dense, green, circular, large |
| **Layout** | 54 | 空间关系 | near, beside, surrounded, between, row |
| **总计** | **421** | | |

## ✅ 质量指标

- **内容词覆盖率**: **75.45%** 🟩 (优秀)
- **All Token覆盖率**: 49.21%
- **训练语料**: RSICD (39,310条) + RSITMD (17,175条)
- **质量评级**: 研究级/可发表级

## 📁 文件说明

### 生产文件

```
data/vocabulary/
├── rs_vocabulary.json           # ⭐ 主词表（使用这个）
├── coverage_report.json         # 覆盖率分析报告
├── METADATA.json               # 版本历史与元信息
└── README.md                   # 本文档
```

### 归档文件

```
data/vocabulary/archive/
├── rs_vocabulary_v1.0_backup.json
├── rs_vocabulary_v1.1_backup.json
├── rs_vocabulary_v1.2_backup.json
├── rs_vocabulary_v1.1.json
├── rs_vocabulary_v1.2.json
└── ...（其他中间文件）
```

## 🚀 使用方法

### Python

```python
import json

# 加载词表
with open('data/vocabulary/rs_vocabulary.json', 'r') as f:
    vocab = json.load(f)

# 访问三个粒度
object_words = vocab['object']   # 190个地物词
scene_words = vocab['scene']     # 177个场景/属性词
layout_words = vocab['layout']   # 54个空间关系词

# 示例：检查一个词属于哪个粒度
word = "buildings"
if word in object_words:
    print(f"'{word}' is an Object word")
```

### 应用场景

1. **软改写（Soft Rewriting）**: 使用BERT MLM在词向量空间进行可微改写
2. **RL语义控制**: 批级决策改写粒度和温度参数
3. **MoE文本编码**: 三粒度Expert的软路由权重计算
4. **数据增强**: 基于粒度的caption变换

## 📈 版本历史

| 版本 | 日期 | 主要改动 | 词数 | 覆盖率 |
|------|------|----------|------|--------|
| v1.0 | 2025-10-18 | 初始构建 | 409 | - |
| v1.1 | 2025-10-18 | 人工检验修正 | 391 | - |
| v1.2 | 2025-10-18 | 高频词补充 | 407 | 72.73% |
| **v1.3** | **2025-10-18** | **RL优化** | **421** | **75.45%** ⭐ |

## 🔧 维护与更新

如需更新词表：

1. 运行覆盖率分析：`python scripts/check_vocabulary_coverage.py`
2. 检查未覆盖的高频词
3. 根据任务需求决定是否增补
4. 更新版本号并重新归档

## 📞 联系方式

如有问题或建议，请联系：Sean R. Liang

---

**最后更新**: 2025-10-18
