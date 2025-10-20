"""
创建部分替换的训练数据集

功能：
从完整的回译替换数据集中，随机选择指定比例的替换样本，
其余样本保持原始文本。

用途：
先用10%替换数据测试效果，避免过度修改导致性能下降。

使用方法：
python scripts/create_partial_augmented.py --dataset rsitmd --ratio 0.1
python scripts/create_partial_augmented.py --dataset rsicd --ratio 0.1
"""

import json
import random
import argparse
from pathlib import Path


def create_partial_augmented(dataset_name, replacement_ratio=0.1, seed=42):
    """
    创建部分替换的数据集
    
    Args:
        dataset_name: 'rsitmd' 或 'rsicd'
        replacement_ratio: 替换比例 (0.1 = 10%)
        seed: 随机种子
    """
    random.seed(seed)
    
    # 文件路径
    original_file = f"data/finetune/{dataset_name}_train.json"
    full_backtrans_file = f"data/augmented/{dataset_name}_train_backtrans.json"
    output_file = f"data/augmented/{dataset_name}_train_backtrans_{int(replacement_ratio*100)}pct.json"
    
    print("="*70)
    print(f"创建 {dataset_name.upper()} 部分替换数据集")
    print("="*70)
    
    # 1. 加载数据
    print(f"\n[1/4] 加载数据...")
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(full_backtrans_file, 'r', encoding='utf-8') as f:
        backtrans_data = json.load(f)
    
    print(f"  原始数据: {len(original_data)} 条")
    print(f"  完整回译: {len(backtrans_data)} 条")
    
    # 2. 找出所有被替换的样本索引
    print(f"\n[2/4] 识别替换样本...")
    replaced_indices = []
    
    for i, (orig, back) in enumerate(zip(original_data, backtrans_data)):
        if orig['caption'] != back['caption']:
            replaced_indices.append(i)
    
    total_replaced = len(replaced_indices)
    print(f"  发现 {total_replaced} 条替换样本 ({total_replaced/len(original_data)*100:.1f}%)")
    
    # 3. 随机选择指定比例的替换样本
    print(f"\n[3/4] 随机选择 {replacement_ratio*100:.0f}% 的替换样本...")
    num_to_keep = int(total_replaced * replacement_ratio)
    selected_indices = set(random.sample(replaced_indices, num_to_keep))
    
    print(f"  选中 {num_to_keep} 条替换样本")
    print(f"  保留原文的替换样本: {total_replaced - num_to_keep} 条")
    
    # 4. 创建新数据集
    print(f"\n[4/4] 生成新数据集...")
    new_data = []
    actually_replaced = 0
    
    for i, (orig, back) in enumerate(zip(original_data, backtrans_data)):
        if i in selected_indices:
            # 使用回译版本
            new_data.append(back)
            actually_replaced += 1
        else:
            # 使用原始版本
            new_data.append(orig)
    
    # 5. 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    # 6. 统计报告
    print("\n" + "="*70)
    print("生成完成！")
    print("="*70)
    print(f"\n输出文件: {output_file}")
    print(f"总样本数: {len(new_data)}")
    print(f"实际替换: {actually_replaced} 条 ({actually_replaced/len(new_data)*100:.1f}%)")
    print(f"原始文本: {len(new_data) - actually_replaced} 条 ({(len(new_data)-actually_replaced)/len(new_data)*100:.1f}%)")
    
    # 7. 展示几个替换样例
    print("\n" + "-"*70)
    print("替换样例（前3个）:")
    print("-"*70)
    
    shown = 0
    for i in selected_indices:
        if shown >= 3:
            break
        print(f"\n样例 {shown + 1}:")
        print(f"  原文: {original_data[i]['caption']}")
        print(f"  回译: {backtrans_data[i]['caption']}")
        shown += 1
    
    print("\n" + "="*70)
    print("\n✓ 可以在配置文件中使用此数据集:")
    print(f"  train_file: ['data/augmented/{dataset_name}_train_backtrans_{int(replacement_ratio*100)}pct.json']")
    print("="*70 + "\n")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='创建部分替换的训练数据集')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['rsitmd', 'rsicd'],
                       help='数据集名称: rsitmd 或 rsicd')
    parser.add_argument('--ratio', type=float, default=0.1,
                       help='替换比例 (默认0.1=10%%)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认42)')
    
    args = parser.parse_args()
    
    if not 0 <= args.ratio <= 1:
        print("错误: ratio 必须在 0 到 1 之间")
        return
    
    create_partial_augmented(args.dataset, args.ratio, args.seed)


if __name__ == "__main__":
    main()

