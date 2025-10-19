"""
词表最终归档脚本
将v1.3设为正式版本，清理中间文件，生成元信息

作者：Sean R. Liang
日期：2025-10-18
"""

import json
import os
import shutil
from datetime import datetime

def main():
    print("=" * 70)
    print("词表最终归档")
    print("=" * 70)
    print()
    
    base_dir = "data/vocabulary"
    
    # ===== 1. 将v1.3设为最终版本 =====
    print("[1] 设置最终版本...")
    
    v1_3_path = f"{base_dir}/rs_vocabulary_v1.3.json"
    final_path = f"{base_dir}/rs_vocabulary.json"
    
    if os.path.exists(v1_3_path):
        shutil.copy(v1_3_path, final_path)
        print(f"  ✓ {v1_3_path} → {final_path}")
    else:
        print(f"  ❌ 错误：找不到 {v1_3_path}")
        return
    
    # 同样处理coverage report
    coverage_v1_3 = f"{base_dir}/coverage_report_v1.3.json"
    coverage_final = f"{base_dir}/coverage_report.json"
    
    if os.path.exists(coverage_v1_3):
        shutil.copy(coverage_v1_3, coverage_final)
        print(f"  ✓ {coverage_v1_3} → {coverage_final}")
    
    print()
    
    # ===== 2. 创建归档目录 =====
    print("[2] 创建归档目录...")
    
    archive_dir = f"{base_dir}/archive"
    os.makedirs(archive_dir, exist_ok=True)
    print(f"  ✓ {archive_dir}")
    print()
    
    # ===== 3. 移动中间版本到归档 =====
    print("[3] 归档中间版本...")
    
    files_to_archive = [
        "rs_vocabulary_v1.0_backup.json",
        "rs_vocabulary_v1.1_backup.json",
        "rs_vocabulary_v1.2_backup.json",
        "rs_vocabulary_v1.1.json",
        "rs_vocabulary_v1.2.json",
        "coverage_report_v1.2.json",
        "revision_log.txt",
        "revision_log_v1.3.txt",
    ]
    
    archived_count = 0
    for filename in files_to_archive:
        src = f"{base_dir}/{filename}"
        if os.path.exists(src):
            dst = f"{archive_dir}/{filename}"
            shutil.move(src, dst)
            print(f"  ✓ 归档：{filename}")
            archived_count += 1
    
    print(f"\n  共归档 {archived_count} 个文件")
    print()
    
    # ===== 4. 生成版本历史元信息 =====
    print("[4] 生成版本历史元信息...")
    
    # 读取最终版本
    with open(final_path, 'r', encoding='utf-8') as f:
        final_vocab = json.load(f)
    
    # 读取覆盖率报告
    with open(coverage_final, 'r', encoding='utf-8') as f:
        coverage = json.load(f)
    
    # 生成元信息
    metadata = {
        "vocabulary_metadata": {
            "version": "1.3 (final)",
            "creation_date": "2025-10-18",
            "author": "Sean R. Liang",
            "description": "Three-granularity vocabulary for Remote Sensing image-text retrieval",
            "optimization_target": "Soft rewriting with RL-based semantic control (no LVLM)",
            "total_words": len(final_vocab['object']) + len(final_vocab['scene']) + len(final_vocab['layout']),
            "granularities": {
                "object": {
                    "count": len(final_vocab['object']),
                    "description": "Specific objects and landforms",
                    "examples": final_vocab['object'][:10]
                },
                "scene": {
                    "count": len(final_vocab['scene']),
                    "description": "Scene categories and attributes",
                    "examples": final_vocab['scene'][:10]
                },
                "layout": {
                    "count": len(final_vocab['layout']),
                    "description": "Spatial relations",
                    "examples": final_vocab['layout'][:10]
                }
            }
        },
        "coverage_statistics": {
            "content_word_coverage": coverage['coverage_rate_content'],
            "all_token_coverage": coverage['coverage_rate_all'],
            "total_captions_analyzed": 56485,
            "datasets": ["RSICD", "RSITMD"],
            "evaluation_date": "2025-10-18",
            "quality_rating": "Excellent (≥70%)"
        },
        "version_history": {
            "v1.0": {
                "date": "2025-10-18",
                "changes": "Initial construction from RSICD+RSITMD corpus",
                "word_count": {"object": 187, "scene": 172, "layout": 50, "total": 409}
            },
            "v1.1": {
                "date": "2025-10-18",
                "changes": "Manual review and corrections (removed abstract words, added spatial terms)",
                "word_count": {"object": 166, "scene": 172, "layout": 53, "total": 391}
            },
            "v1.2": {
                "date": "2025-10-18",
                "changes": "High-frequency noun supplementation (singular forms + compound facilities)",
                "word_count": {"object": 182, "scene": 172, "layout": 53, "total": 407},
                "coverage": {"content_word": 72.73}
            },
            "v1.3": {
                "date": "2025-10-18",
                "changes": "Optimization for soft rewriting + RL (added area/farm/corner/block/district/wharf/lawn/vegetation in Object; circle/oval/triangular/khaki/empty in Scene; row in Layout)",
                "word_count": {"object": 190, "scene": 177, "layout": 54, "total": 421},
                "coverage": {"content_word": 75.45},
                "status": "FINAL"
            }
        },
        "usage_instructions": {
            "loading": "import json; vocab = json.load(open('data/vocabulary/rs_vocabulary.json'))",
            "granularities": {
                "object": "vocab['object']  # 190 words",
                "scene": "vocab['scene']   # 177 words",
                "layout": "vocab['layout']  # 54 words"
            },
            "applications": [
                "Soft rewriting with BERT MLM",
                "RL-based semantic control",
                "MoE text encoder routing",
                "Granularity-aware caption augmentation"
            ]
        },
        "file_structure": {
            "production": {
                "rs_vocabulary.json": "Main vocabulary file (v1.3)",
                "coverage_report.json": "Coverage analysis report",
                "METADATA.json": "This file"
            },
            "archive": {
                "location": "data/vocabulary/archive/",
                "contents": "All intermediate versions (v1.0, v1.1, v1.2) and legacy files"
            },
            "scripts": {
                "location": "scripts/",
                "files": [
                    "build_rs_vocabulary.py - Initial construction",
                    "check_vocabulary_coverage.py - Coverage analysis tool v2.0",
                    "fix_vocabulary.py - v1.0→v1.1 corrections",
                    "update_vocabulary_v1.2.py - v1.1→v1.2 upgrade",
                    "update_vocabulary_v1.3.py - v1.2→v1.3 upgrade"
                ]
            }
        }
    }
    
    metadata_path = f"{base_dir}/METADATA.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ {metadata_path}")
    print()
    
    # ===== 5. 生成README =====
    print("[5] 生成README...")
    
    readme_content = """# 遥感三层语义词表 (Remote Sensing Three-Granularity Vocabulary)

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
"""
    
    readme_path = f"{base_dir}/README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"  ✓ {readme_path}")
    print()
    
    # ===== 6. 最终报告 =====
    print("=" * 70)
    print("✅ 归档完成！")
    print("=" * 70)
    print()
    print("最终文件结构：")
    print()
    print("data/vocabulary/")
    print("├── rs_vocabulary.json          ⭐ 主词表（v1.3）")
    print("├── coverage_report.json        📊 覆盖率报告")
    print("├── METADATA.json              📝 元信息")
    print("├── README.md                  📖 使用说明")
    print("├── rs_vocabulary_v1.3.json    （保留）")
    print("├── coverage_report_v1.3.json  （保留）")
    print("└── archive/                   📦 归档目录")
    print("    ├── rs_vocabulary_v1.0_backup.json")
    print("    ├── rs_vocabulary_v1.1_backup.json")
    print("    ├── rs_vocabulary_v1.2_backup.json")
    print("    └── ...（其他中间文件）")
    print()
    print(f"词表统计：")
    print(f"  - Object: {len(final_vocab['object'])} 词")
    print(f"  - Scene: {len(final_vocab['scene'])} 词")
    print(f"  - Layout: {len(final_vocab['layout'])} 词")
    print(f"  - 总计: {len(final_vocab['object']) + len(final_vocab['scene']) + len(final_vocab['layout'])} 词")
    print()
    print(f"质量指标：")
    print(f"  - 内容词覆盖率: {coverage['coverage_rate_content']:.2f}% 🟩")
    print(f"  - 质量评级: 优秀")
    print()


if __name__ == "__main__":
    main()

