"""
从RSICD和RSITMD训练集构建遥感领域三层语义词表

功能：基于真实数据分析，提取三个粒度的词表：
  1. Object（地物/实体）：具体物体，如buildings, trees, cars
  2. Scene（场景/属性）：场景类别和属性，如residential, dense, large
  3. Layout（空间关系）：方位关系词，如near, beside, surrounded

作者：Sean R. Liang
日期：2025-10-18
"""

import json
import os
from collections import Counter
import spacy

try:
    # 加载spacy的英文模型（用于词性标注）
    # 词性标注（POS tagging）能帮我们区分：
    #   - NOUN（名词）：buildings, trees → Object
    #   - ADJ（形容词）：dense, large → Scene属性
    #   - ADP（介词）：near, in → Layout空间关系
    nlp = spacy.load("en_core_web_sm")
    print("✓ spacy模型加载成功\n")
except Exception as e:
    print(f"❌ 错误：{e}")
    print("请运行：python -m spacy download en_core_web_sm")
    exit(1)


# ====================================================================================
# 第2部分：预定义种子词表
# ====================================================================================
# 
# 为什么需要种子词？
# ————————————————————————————————————————————————————————————————————————————————
# 1. 引导分类：告诉程序哪些词应该归入哪个粒度
# 2. 扩展基础：通过共现关系（co-occurrence）扩展到数据中的相似词
# 3. 质量保证：基于遥感领域的先验知识，避免误分类
# 
# 这些种子词来自哪里？
# ————————————————————————————————————————————————————————————————————————————————
# - 我之前分析了你的RSICD和RSITMD数据（运行check_data.py看到的结果）
# - 总结了高频词和典型模式
# - 结合遥感领域的标准术语（如DOTA数据集的15类物体）
# ====================================================================================

# Object种子词：场景中的具体物体和地物
OBJECT_SEEDS = {
    # 自然地物
    'trees', 'grass', 'plants', 'water', 'ocean', 'river', 'lake', 'sea',
    'mountain', 'forest', 'field', 'meadow', 'pond', 'beach', 'sand',
    
    # 建筑物
    'buildings', 'houses', 'structures', 'building', 'house', 'structure',
    
    # 交通设施
    'roads', 'road', 'street', 'highway', 'railway', 'track', 'path',
    'cars', 'car', 'vehicles', 'vehicle',
    
    # 航空相关
    'planes', 'plane', 'aircraft', 'airplane',
    'airport', 'terminal', 'runway', 'apron',
    
    # 工业设施
    'tanks', 'tank', 'ship', 'ships', 'boat', 'boats',
    'bridge', 'bridges',
    
    # 其他人工设施
    'parking', 'lot', 'square', 'ground', 'court', 'playground'
}

# Scene种子词：场景类别和属性
SCENE_SEEDS = {
    # 场景类型（基于RSITMD的31个类别）
    'residential', 'commercial', 'industrial', 'agricultural',
    'airport', 'railway', 'harbor', 'port',
    'desert', 'forest', 'meadow', 'mountain',
    'parking', 'playground', 'stadium',
    
    # 密度和规模属性
    'dense', 'sparse', 'medium', 'large', 'small', 'huge', 'tiny',
    'many', 'few', 'several', 'some',
    
    # 颜色和外观属性
    'green', 'blue', 'white', 'gray', 'brown', 'red', 'black',
    'dark', 'light', 'bright',
    
    # 形状和排列属性
    'circular', 'square', 'rectangular', 'curved', 'straight',
    'orderly', 'scattered', 'aligned', 'arranged'
}

# Layout种子词：空间位置关系
LAYOUT_SEEDS = {
    # 核心空间介词
    'in', 'on', 'at', 'near', 'beside', 'between', 'around',
    'along', 'across', 'through', 'over', 'under',
    
    # 方向词
    'left', 'right', 'top', 'bottom', 'center', 'middle',
    'north', 'south', 'east', 'west',
    
    # 复合空间关系短语（从真实数据中提取的高频模式）
    'surrounded by', 'next to', 'close to', 'far from',
    'in the middle of', 'at the end of', 'on the side of',
    'in front of', 'at the edge of'
}

print("[步骤2] 预定义种子词表加载完成")
print(f"  - Object种子词: {len(OBJECT_SEEDS)} 个")
print(f"  - Scene种子词: {len(SCENE_SEEDS)} 个")
print(f"  - Layout种子词: {len(LAYOUT_SEEDS)} 个")
print()


# ====================================================================================
# 第3部分：数据加载函数
# ====================================================================================
# 
# 功能：从RSICD和RSITMD的训练集JSON文件中提取所有caption
# 
# 为什么只用训练集？
# ————————————————————————————————————————————————————————————————————————————————
# 1. 避免数据泄露：测试集不能参与任何训练相关的处理
# 2. 分布一致性：训练时只会看到训练集的词汇，词表应该完全基于训练集
# ====================================================================================

def load_captions_from_json(json_path):
    """
    从JSON文件加载所有caption
    
    参数：
        json_path: JSON文件路径
        
    返回：
        captions: list of str，所有描述文本
        
    数据格式示例（RSITMD）：
        [
          {
            "caption": "many buildings are in two sides of a railway",
            "image": "train/railwaystation_33.tif",
            "image_id": 4231,
            "label": 16,
            "label_name": "railway station"
          },
          ...
        ]
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取所有caption（转小写以统一处理）
        captions = [item['caption'].lower().strip() for item in data]
        
        return captions
    
    except Exception as e:
        print(f"❌ 读取文件失败：{json_path}")
        print(f"   错误信息：{e}")
        return []


def load_all_training_captions():
    """
    加载RSICD和RSITMD的所有训练集caption
    
    返回：
        all_captions: list of str
        stats: dict，统计信息
    """
    print("[步骤3] 加载训练集数据...")
    
    # 定义数据路径
    data_root = "data/finetune"
    datasets = {
        'rsitmd': f"{data_root}/rsitmd_train.json",
        'rsicd': f"{data_root}/rsicd_train.json"
    }
    
    all_captions = []
    stats = {}
    
    for dataset_name, json_path in datasets.items():
        if not os.path.exists(json_path):
            print(f"  ⚠ 警告：{json_path} 不存在，跳过")
            continue
        
        captions = load_captions_from_json(json_path)
        all_captions.extend(captions)
        stats[dataset_name] = len(captions)
        
        print(f"  ✓ {dataset_name}: {len(captions)} 条caption")
    
    print(f"\n  总计: {len(all_captions)} 条caption")
    print()
    
    return all_captions, stats


# ====================================================================================
# 第4部分：词频统计和分词
# ====================================================================================
# 
# 功能：使用spacy进行分词和词性标注，统计每个词的频率
# 
# 为什么需要词性标注？
# ————————————————————————————————————————————————————————————————————————————————
# 同一个词在不同上下文中可能有不同的词性和含义：
#   - "near the airport" → near是介词（ADP），Layout词
#   - "a near miss" → near是形容词（ADJ），Scene属性词
# 
# spacy的词性标注能帮我们准确分类
# ====================================================================================

def extract_word_statistics(captions, nlp):
    """
    从caption中提取词频和词性信息
    
    参数：
        captions: list of str，所有描述文本
        nlp: spacy模型实例
        
    返回：
        word_freq: Counter，{word: frequency}
        word_pos: dict，{word: {pos_tag: count}}，记录每个词的词性分布
    """
    print("[步骤4] 分词和词性标注...")
    
    word_freq = Counter()
    word_pos = {}  # {word: {NOUN: 10, VERB: 2, ...}}
    
    # 处理所有caption
    for i, caption in enumerate(captions):
        if (i + 1) % 5000 == 0:
            print(f"  处理进度: {i+1}/{len(captions)}")
        
        # 使用spacy处理文本
        doc = nlp(caption)
        
        for token in doc:
            # 跳过标点和空白
            if token.is_punct or token.is_space:
                continue
            
            # 跳过停用词（但保留空间介词）
            if token.is_stop and token.text not in LAYOUT_SEEDS:
                continue
            
            word = token.text.lower()
            pos = token.pos_  # 词性标签
            
            # 更新词频
            word_freq[word] += 1
            
            # 更新词性统计
            if word not in word_pos:
                word_pos[word] = Counter()
            word_pos[word][pos] += 1
    
    print(f"  ✓ 共提取 {len(word_freq)} 个不同的词")
    print(f"  ✓ 总词数: {sum(word_freq.values())}")
    print()
    
    return word_freq, word_pos


# ====================================================================================
# 第5部分：基于规则和统计的三粒度分类
# ====================================================================================
# 
# 分类策略：
# ————————————————————————————————————————————————————————————————————————————————
# 1. 优先匹配种子词（高置信度）
# 2. 基于词性标注的启发式规则：
#    - NOUN（名词）+ 高频 → Object候选
#    - ADJ（形容词）+ 中频 → Scene属性候选
#    - ADP（介词）→ Layout候选
# 3. 频率阈值过滤：太低频的词可能是噪声或特殊词汇
# ====================================================================================

def classify_words_into_granularities(word_freq, word_pos, nlp):
    """
    将词汇分类到三个语义粒度
    
    参数：
        word_freq: Counter，词频统计
        word_pos: dict，词性分布
        nlp: spacy模型
        
    返回：
        vocabularies: dict，{
            'object': set,
            'scene': set,
            'layout': set
        }
    """
    print("[步骤5] 将词汇分类到三个语义粒度...")
    
    vocabularies = {
        'object': set(),
        'scene': set(),
        'layout': set()
    }
    
    # ===== 规则1：直接匹配种子词 =====
    print("  [规则1] 匹配种子词...")
    
    for word in word_freq.keys():
        if word in OBJECT_SEEDS:
            vocabularies['object'].add(word)
        elif word in SCENE_SEEDS:
            vocabularies['scene'].add(word)
        elif word in LAYOUT_SEEDS:
            vocabularies['layout'].add(word)
    
    print(f"    - Object: {len(vocabularies['object'])} 个")
    print(f"    - Scene: {len(vocabularies['scene'])} 个")
    print(f"    - Layout: {len(vocabularies['layout'])} 个")
    
    # ===== 规则2：基于词性和频率的扩展 =====
    print("\n  [规则2] 基于词性扩展...")
    
    # 频率阈值（出现次数少于该值的词不考虑）
    MIN_FREQ_OBJECT = 20   # Object词应该比较常见
    MIN_FREQ_SCENE = 15    # Scene属性词稍微宽松
    MIN_FREQ_LAYOUT = 10   # Layout词即使不太常见也很重要
    
    for word, freq in word_freq.items():
        # 跳过已分类的词
        if word in vocabularies['object'] or \
           word in vocabularies['scene'] or \
           word in vocabularies['layout']:
            continue
        
        # 跳过太短的词（单字母，通常是代词或冠词）
        if len(word) <= 1:
            continue
        
        # 获取该词的主要词性（出现次数最多的词性）
        if word not in word_pos:
            continue
        
        dominant_pos = word_pos[word].most_common(1)[0][0]
        
        # 扩展Object词表：名词 + 高频
        if dominant_pos == 'NOUN' and freq >= MIN_FREQ_OBJECT:
            # 进一步过滤：检查是否与Object种子词共现
            # （这是一个简化策略，真实场景中可以用word2vec等计算相似度）
            if any(seed in word or word.endswith('s') for seed in OBJECT_SEEDS):
                vocabularies['object'].add(word)
        
        # 扩展Scene词表：形容词 + 中频
        elif dominant_pos == 'ADJ' and freq >= MIN_FREQ_SCENE:
            vocabularies['scene'].add(word)
        
        # 扩展Layout词表：介词/副词
        elif dominant_pos in ['ADP', 'ADV'] and freq >= MIN_FREQ_LAYOUT:
            vocabularies['layout'].add(word)
    
    print(f"    - Object扩展后: {len(vocabularies['object'])} 个")
    print(f"    - Scene扩展后: {len(vocabularies['scene'])} 个")
    print(f"    - Layout扩展后: {len(vocabularies['layout'])} 个")
    print()
    
    return vocabularies


# ====================================================================================
# 第6部分：保存词表到JSON文件
# ====================================================================================
# 
# 输出格式：
# {
#   "object": ["buildings", "trees", "cars", ...],
#   "scene": ["residential", "dense", "green", ...],
#   "layout": ["near", "beside", "surrounded", ...],
#   "metadata": {
#     "total_captions": 56485,
#     "datasets": {"rsitmd": 17175, "rsicd": 39310},
#     "build_date": "2025-10-18",
#     ...
#   }
# }
# ====================================================================================

def save_vocabularies(vocabularies, output_path, stats):
    """
    将词表保存为JSON文件
    
    参数：
        vocabularies: dict，三个粒度的词表
        output_path: str，输出文件路径
        stats: dict，统计信息
    """
    print("[步骤6] 保存词表...")
    
    # 转换为列表并排序（便于查看和版本控制）
    output_data = {
        'object': sorted(list(vocabularies['object'])),
        'scene': sorted(list(vocabularies['scene'])),
        'layout': sorted(list(vocabularies['layout'])),
        'metadata': {
            'total_captions': sum(stats.values()),
            'datasets': stats,
            'build_date': '2025-10-18',
            'author': 'Sean R. Liang',
            'description': 'Three-granularity vocabulary for Remote Sensing image-text retrieval',
            'granularities': {
                'object': {
                    'count': len(vocabularies['object']),
                    'description': 'Specific objects and landforms (buildings, trees, cars, etc.)'
                },
                'scene': {
                    'count': len(vocabularies['scene']),
                    'description': 'Scene categories and attributes (residential, dense, green, etc.)'
                },
                'layout': {
                    'count': len(vocabularies['layout']),
                    'description': 'Spatial relations (near, beside, surrounded, etc.)'
                }
            }
        }
    }
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ 词表已保存到: {output_path}")
    print()
    
    return output_data


def print_vocabulary_samples(vocabularies, n=20):
    """
    打印每个粒度的样例词汇（用于人工检查）
    
    参数：
        vocabularies: dict，三个粒度的词表
        n: int，每个粒度显示的样例数量
    """
    print("=" * 70)
    print("词表样例预览（用于人工检查）")
    print("=" * 70)
    
    for granularity, words in vocabularies.items():
        print(f"\n【{granularity.upper()}】粒度（共 {len(words)} 个词）")
        print("-" * 70)
        
        # 转为列表并排序
        word_list = sorted(list(words))
        
        # 显示前n个
        samples = word_list[:n]
        print(f"  前{len(samples)}个: {', '.join(samples)}")
        
        # 如果词表很大，也显示最后几个
        if len(word_list) > n:
            print(f"  ...(省略 {len(word_list) - n} 个)...")
    
    print("\n" + "=" * 70)
    print()


# ====================================================================================
# 第7部分：主函数 - 整合所有步骤
# ====================================================================================

def main():
    """
    主函数：执行完整的词表构建流程
    
    流程：
        1. 加载spacy模型（已在文件开头完成）
        2. 加载训练集数据
        3. 词频统计和词性标注
        4. 三粒度分类
        5. 保存词表
        6. 打印样例供检查
    """
    print("\n" + "=" * 70)
    print("开始构建遥感领域三层语义词表")
    print("=" * 70)
    print()
    
    # 步骤3：加载数据
    captions, stats = load_all_training_captions()
    
    if len(captions) == 0:
        print("❌ 错误：没有加载到任何数据！")
        print("请检查数据文件是否存在：")
        print("  - data/finetune/rsitmd_train.json")
        print("  - data/finetune/rsicd_train.json")
        return
    
    # 步骤4：词频统计
    word_freq, word_pos = extract_word_statistics(captions, nlp)
    
    # 步骤5：三粒度分类
    vocabularies = classify_words_into_granularities(word_freq, word_pos, nlp)
    
    # 步骤6：保存词表
    output_path = "data/vocabulary/rs_vocabulary.json"
    output_data = save_vocabularies(vocabularies, output_path, stats)
    
    # 步骤7：打印样例
    print_vocabulary_samples(vocabularies, n=30)
    
    # 最终统计
    print("=" * 70)
    print("✓ 词表构建完成！")
    print("=" * 70)
    print(f"\n总结：")
    print(f"  - 处理了 {len(captions)} 条caption")
    print(f"  - 提取了 {len(word_freq)} 个不同的词")
    print(f"  - Object词汇: {len(vocabularies['object'])} 个")
    print(f"  - Scene词汇: {len(vocabularies['scene'])} 个")
    print(f"  - Layout词汇: {len(vocabularies['layout'])} 个")
    print(f"  - 词表文件: {output_path}")
    print()
    print("下一步：请人工检查词表质量，确认分类是否合理")
    print()


# ====================================================================================
# 执行主函数
# ====================================================================================

if __name__ == "__main__":
    main()
