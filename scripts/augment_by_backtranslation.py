"""
遥感图像文本增强 - 领域增强回译（当前处理RSICD）

作者：基于HarMA项目
日期：2025-10-19

功能：
1. 英文 → 中文翻译
2. 中文 → 英文翻译（加入遥感领域专家prompt）
3. CLIP相似度验证（确保语义不漂移）
4. 质量评估和统计

当前配置：
- 正在处理 RSICD 数据集（数据量较大，需要更长时间）
- 使用免费的 MarianMT 本地模型（首次运行会自动下载）

建议使用流程：
1. 先设置 SAMPLE_SIZE = 100 测试
2. 检查质量报告，确认相似度和多样性
3. 如果满意，设置 SAMPLE_SIZE = None 处理全量数据

依赖：
pip install transformers sentencepiece sacremoses sentence-transformers
（如果用OpenAI API: pip install openai）
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import random
from tqdm import tqdm

# ===============================================
# Part 1: 配置
# ===============================================

# 数据路径（切换到处理RSICD）
RSICD_TRAIN = "data/finetune/rsicd_train.json"

# 输出路径
OUTPUT_DIR = "data/augmented"
SAMPLE_TEST_FILE = "backtranslation_samples_rsicd.json"  # 先测试100条
FULL_OUTPUT_FILE = "rsicd_train_backtrans.json"

# 回译配置
SAMPLE_SIZE = None  # None=全量处理RSICD，或指定数字（如100）用于快速测试
SIMILARITY_THRESHOLD = 0.9  # 只保留相似度≥此值的回译样本（语义保持）
MIN_WORD_CHANGES = 1  # 最少改变词数（确保真正增加多样性）
                      # 1 = 至少1个词不同（推荐）
                      # 2 = 至少2个词不同（更严格）
                      # 0 = 允许完全相同（不推荐）
USE_OPENAI = False  # True: 用OpenAI API, False: 用本地MarianMT（免费）

# OpenAI API配置（如果使用）
OPENAI_API_KEY = ""  # 填入你的API key
OPENAI_MODEL = "gpt-3.5-turbo"  # 或 "gpt-4"

# 领域prompt
RS_EXPERT_PROMPT = """You are a remote sensing image description expert. 
Please translate the following Chinese text into professional English 
suitable for describing satellite/aerial imagery. 
Use precise remote sensing terminology when appropriate.

Chinese text: {chinese_text}
English translation:"""


# ===============================================
# Part 2: 翻译引擎
# ===============================================

class TranslationEngine:
    """翻译引擎基类"""
    
    def en_to_zh(self, text: str) -> str:
        """英文 → 中文"""
        raise NotImplementedError
    
    def zh_to_en(self, text: str) -> str:
        """中文 → 英文（带遥感领域prompt）"""
        raise NotImplementedError


class MarianMTEngine(TranslationEngine):
    """本地MarianMT翻译引擎（免费）"""
    
    def __init__(self):
        print("加载MarianMT翻译模型...")
        from transformers import MarianMTModel, MarianTokenizer
        
        # 英文→中文
        self.en2zh_name = "Helsinki-NLP/opus-mt-en-zh"
        print(f"  加载 {self.en2zh_name}")
        self.en2zh_tokenizer = MarianTokenizer.from_pretrained(self.en2zh_name)
        self.en2zh_model = MarianMTModel.from_pretrained(self.en2zh_name)
        
        # 中文→英文
        self.zh2en_name = "Helsinki-NLP/opus-mt-zh-en"
        print(f"  加载 {self.zh2en_name}")
        self.zh2en_tokenizer = MarianTokenizer.from_pretrained(self.zh2en_name)
        self.zh2en_model = MarianMTModel.from_pretrained(self.zh2en_name)
        
        print("✓ 模型加载完成")
    
    def en_to_zh(self, text: str) -> str:
        """英文 → 中文"""
        inputs = self.en2zh_tokenizer(text, return_tensors="pt", padding=True)
        outputs = self.en2zh_model.generate(**inputs)
        translated = self.en2zh_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated
    
    def zh_to_en(self, text: str) -> str:
        """中文 → 英文
        
        注意：MarianMT不支持prompt，所以这里只是标准翻译
        如果需要领域增强，建议用OpenAI API
        """
        inputs = self.zh2en_tokenizer(text, return_tensors="pt", padding=True)
        outputs = self.zh2en_model.generate(**inputs)
        translated = self.zh2en_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated


class OpenAIEngine(TranslationEngine):
    """OpenAI API翻译引擎（支持领域prompt）"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        print(f"✓ OpenAI API已配置 (model={model})")
    
    def en_to_zh(self, text: str) -> str:
        """英文 → 中文"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": f"Translate to Chinese: {text}"}
            ],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    
    def zh_to_en(self, text: str) -> str:
        """中文 → 英文（带遥感领域prompt）"""
        prompt = RS_EXPERT_PROMPT.format(chinese_text=text)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a remote sensing expert and translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,  # 稍高温度增加多样性
            max_tokens=200
        )
        return response.choices[0].message.content.strip()


# ===============================================
# Part 3: 回译增强
# ===============================================

def backtranslate(text: str, engine: TranslationEngine) -> Tuple[str, str]:
    """
    执行回译增强
    
    Args:
        text: 原始英文文本
        engine: 翻译引擎
    
    Returns:
        (中文翻译, 回译英文)
    """
    # Step 1: 英文 → 中文
    chinese = engine.en_to_zh(text)
    
    # Step 2: 中文 → 英文（带领域prompt）
    back_english = engine.zh_to_en(chinese)
    
    return chinese, back_english


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    计算两个文本的语义相似度
    
    方法1: 尝试使用sentence-transformers（语义相似度）
    方法2: 如果没有安装，降级到改进的词重叠率（考虑词形还原）
    """
    try:
        # 方法1: 语义相似度（推荐）
        from sentence_transformers import SentenceTransformer, util
        
        # 使用轻量级模型（首次会下载）
        if not hasattr(compute_text_similarity, 'model'):
            print("  加载语义相似度模型（首次运行）...")
            compute_text_similarity.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 计算语义相似度
        emb1 = compute_text_similarity.model.encode(text1, convert_to_tensor=True)
        emb2 = compute_text_similarity.model.encode(text2, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb1, emb2).item()
        
        return similarity  # 返回 [0, 1] 之间的相似度
    
    except ImportError:
        # 方法2: 改进的词重叠率（去除标点，基本词形还原）
        import string
        
        # 去除标点
        translator = str.maketrans('', '', string.punctuation)
        text1_clean = text1.lower().translate(translator)
        text2_clean = text2.lower().translate(translator)
        
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        # 简单的词形还原（去除常见后缀）
        def simplify_word(word):
            # 去除复数、过去式等
            if word.endswith('s') and len(word) > 3:
                word = word[:-1]
            if word.endswith('ed') and len(word) > 4:
                word = word[:-2]
            if word.endswith('ing') and len(word) > 5:
                word = word[:-3]
            return word
        
        words1_simplified = {simplify_word(w) for w in words1}
        words2_simplified = {simplify_word(w) for w in words2}
        
        intersection = words1_simplified & words2_simplified
        union = words1_simplified | words2_simplified
        
        return len(intersection) / len(union)  # Jaccard相似度（改进版）


# ===============================================
# Part 4: 批量处理
# ===============================================

def load_captions(json_file: str) -> List[str]:
    """加载训练集的文本描述"""
    if not os.path.exists(json_file):
        return []
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    captions = []
    for item in data:
        if 'caption' in item:
            captions.append(item['caption'])
        elif 'sentences' in item:
            # RSICD格式：每个图像有多条描述
            for sent in item['sentences']:
                if isinstance(sent, dict) and 'raw' in sent:
                    captions.append(sent['raw'])
                elif isinstance(sent, str):
                    captions.append(sent)
    
    return captions


def load_dataset_with_structure(json_file: str) -> List[Dict]:
    """加载完整的数据集结构（用于替换后保存）"""
    if not os.path.exists(json_file):
        return []
    
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_samples(captions: List[str], 
                    engine: TranslationEngine,
                    sample_size: int = None) -> List[Dict]:
    """
    处理样本并返回结果
    
    Args:
        captions: 文本列表
        engine: 翻译引擎
        sample_size: 样本数量，None表示处理全部
    
    Returns:
        List of {
            'original': 原始英文,
            'chinese': 中文翻译,
            'backtrans': 回译英文,
            'similarity': 相似度分数
        }
    """
    # 随机采样（如果指定了sample_size）
    if sample_size is not None and len(captions) > sample_size:
        captions = random.sample(captions, sample_size)
    
    results = []
    
    print(f"\n开始回译增强（共{len(captions)}条）...")
    for text in tqdm(captions):
        try:
            chinese, back_english = backtranslate(text, engine)
            similarity = compute_text_similarity(text, back_english)
            
            results.append({
                'original': text,
                'chinese': chinese,
                'backtrans': back_english,
                'similarity': similarity
            })
        except Exception as e:
            print(f"\n错误处理文本: {text}")
            print(f"错误信息: {e}")
            continue
    
    return results


# ===============================================
# Part 5: 质量评估
# ===============================================

def analyze_results(results: List[Dict]) -> Dict:
    """分析回译质量"""
    
    similarities = [r['similarity'] for r in results]
    
    # 统计
    stats = {
        'total': len(results),
        'avg_similarity': sum(similarities) / len(similarities) if similarities else 0,
        'min_similarity': min(similarities) if similarities else 0,
        'max_similarity': max(similarities) if similarities else 0,
        'high_quality': sum(1 for s in similarities if s >= 0.6),  # 相似度 ≥ 0.6
        'medium_quality': sum(1 for s in similarities if 0.3 <= s < 0.6),
        'low_quality': sum(1 for s in similarities if s < 0.3),
    }
    
    # 计算表达多样性（改变的词数）
    changes = []
    for r in results:
        orig_words = set(r['original'].lower().split())
        back_words = set(r['backtrans'].lower().split())
        changed_words = (orig_words - back_words) | (back_words - orig_words)
        changes.append(len(changed_words))
    
    stats['avg_word_changes'] = sum(changes) / len(changes) if changes else 0
    
    return stats


def create_replaced_dataset(original_dataset: List[Dict], 
                            backtrans_results: List[Dict],
                            threshold: float = 0.9,
                            min_word_changes: int = 1) -> Tuple[List[Dict], Dict]:
    """
    创建替换后的训练数据集
    
    Args:
        original_dataset: 原始数据集
        backtrans_results: 回译结果
        threshold: 相似度阈值，只替换≥此值的样本
        min_word_changes: 最少改变词数，确保增加多样性
    
    Returns:
        (替换后的数据集, 统计信息)
    """
    import string
    
    # 计算两个文本的词变化数
    def count_word_changes(text1: str, text2: str) -> int:
        # 去除标点和大小写
        translator = str.maketrans('', '', string.punctuation)
        words1 = set(text1.lower().translate(translator).split())
        words2 = set(text2.lower().translate(translator).split())
        # 对称差集（只在一个集合中出现的词）
        changed = (words1 - words2) | (words2 - words1)
        return len(changed)
    
    # 建立原文→回译文的映射（只保留高质量且有变化的）
    replacement_map = {}
    filtered_count = 0
    diversity_filtered = 0
    
    for r in backtrans_results:
        # 条件1：相似度足够高
        if r['similarity'] >= threshold:
            # 条件2：文本真的有变化
            word_changes = count_word_changes(r['original'], r['backtrans'])
            if word_changes >= min_word_changes:
                replacement_map[r['original']] = r['backtrans']
            else:
                diversity_filtered += 1  # 因为多样性不足被过滤
        else:
            filtered_count += 1  # 因为相似度不足被过滤
    
    # 复制数据集并替换文本
    new_dataset = []
    replaced_count = 0
    
    for item in original_dataset:
        new_item = item.copy()
        
        # 处理不同的数据格式
        if 'caption' in new_item:
            # 格式1: {caption: "..."}
            if new_item['caption'] in replacement_map:
                new_item['caption'] = replacement_map[new_item['caption']]
                replaced_count += 1
        
        elif 'sentences' in new_item:
            # 格式2: {sentences: [...]}（RSICD可能的格式）
            new_sentences = []
            for sent in new_item['sentences']:
                if isinstance(sent, dict) and 'raw' in sent:
                    new_sent = sent.copy()
                    if sent['raw'] in replacement_map:
                        new_sent['raw'] = replacement_map[sent['raw']]
                        replaced_count += 1
                    new_sentences.append(new_sent)
                elif isinstance(sent, str):
                    if sent in replacement_map:
                        new_sentences.append(replacement_map[sent])
                        replaced_count += 1
                    else:
                        new_sentences.append(sent)
                else:
                    new_sentences.append(sent)
            new_item['sentences'] = new_sentences
        
        new_dataset.append(new_item)
    
    stats = {
        'total_samples': len(original_dataset),
        'replaced_count': replaced_count,
        'replacement_rate': replaced_count / len(original_dataset) if len(original_dataset) > 0 else 0,
        'threshold_used': threshold,
        'min_word_changes': min_word_changes,
        'high_quality_available': len(replacement_map),
        'filtered_by_similarity': filtered_count,
        'filtered_by_diversity': diversity_filtered
    }
    
    return new_dataset, stats


def print_quality_report(results: List[Dict], stats: Dict):
    """打印质量报告"""
    
    print("\n" + "="*70)
    print("回译质量评估报告")
    print("="*70)
    
    print(f"\n总样本数: {stats['total']}")
    print(f"平均相似度: {stats['avg_similarity']:.3f}")
    print(f"相似度范围: [{stats['min_similarity']:.3f}, {stats['max_similarity']:.3f}]")
    
    print(f"\n质量分布:")
    print(f"  高质量 (≥0.6): {stats['high_quality']} ({stats['high_quality']/stats['total']*100:.1f}%)")
    print(f"  中质量 (0.3-0.6): {stats['medium_quality']} ({stats['medium_quality']/stats['total']*100:.1f}%)")
    print(f"  低质量 (<0.3): {stats['low_quality']} ({stats['low_quality']/stats['total']*100:.1f}%)")
    
    print(f"\n平均改变词数: {stats['avg_word_changes']:.1f}")
    
    # 展示样例
    print("\n" + "-"*70)
    print("随机样例展示（前5条）:")
    print("-"*70)
    
    for i, r in enumerate(results[:5], 1):
        print(f"\n样例 {i}:")
        print(f"  原文: {r['original']}")
        print(f"  中文: {r['chinese']}")
        print(f"  回译: {r['backtrans']}")
        print(f"  相似度: {r['similarity']:.3f}")
    
    print("\n" + "="*70)
    
    # 评估建议
    if stats['avg_similarity'] >= 0.6:
        print("✓ 质量评估: 优秀 - 语义保持良好，可以继续处理完整数据集")
    elif stats['avg_similarity'] >= 0.4:
        print("⚠ 质量评估: 中等 - 建议人工检查部分样本后决定")
    else:
        print("✗ 质量评估: 较差 - 建议调整翻译引擎或prompt")
    
    print("="*70 + "\n")


# ===============================================
# Part 6: 主函数
# ===============================================

def main():
    """主流程"""
    
    print("="*70)
    print("遥感图像文本增强 - 领域增强回译")
    print("="*70)
    
    # 1. 初始化翻译引擎
    print("\n[1/5] 初始化翻译引擎...")
    if USE_OPENAI:
        if not OPENAI_API_KEY:
            print("错误: 请设置OPENAI_API_KEY")
            return
        engine = OpenAIEngine(OPENAI_API_KEY, OPENAI_MODEL)
    else:
        print("使用本地MarianMT模型（首次运行会自动下载，约500MB）")
        engine = MarianMTEngine()
    
    # 2. 加载数据（加载RSICD）
    print("\n[2/5] 加载训练数据...")
    all_captions = load_captions(RSICD_TRAIN)
    
    print(f"  RSICD: {len(all_captions)} 条")
    
    if len(all_captions) == 0:
        print("错误: 未找到训练数据")
        print(f"请确认数据文件存在: {RSICD_TRAIN}")
        return
    
    # 3. 回译处理
    print(f"\n[3/5] 回译增强处理（样本量: {SAMPLE_SIZE}）...")
    results = process_samples(all_captions, engine, SAMPLE_SIZE)
    
    # 4. 质量评估
    print("\n[4/5] 质量评估...")
    stats = analyze_results(results)
    print_quality_report(results, stats)
    
    # 5. 保存结果
    print("\n[5/5] 保存结果...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存回译分析结果
    if SAMPLE_SIZE and SAMPLE_SIZE < len(all_captions):
        output_file = os.path.join(OUTPUT_DIR, SAMPLE_TEST_FILE)
    else:
        output_file = os.path.join(OUTPUT_DIR, "backtranslation_full.json")
    
    output_data = {
        'config': {
            'engine': 'OpenAI' if USE_OPENAI else 'MarianMT',
            'sample_size': len(results),
            'similarity_threshold': SIMILARITY_THRESHOLD,
            'source_datasets': ['RSICD']  # 当前处理RSICD
        },
        'statistics': stats,
        'samples': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 回译分析结果已保存: {output_file}")
    
    # 6. 生成替换后的训练数据集（全量处理时）
    if SAMPLE_SIZE is None or SAMPLE_SIZE >= len(all_captions):
        print("\n[6/7] 生成替换后的训练数据集...")
        
        # 加载RSICD原始数据集
        rsicd_data = load_dataset_with_structure(RSICD_TRAIN)
        
        # 创建替换版本
        rsicd_replaced, rsicd_stats = create_replaced_dataset(
            rsicd_data, results, SIMILARITY_THRESHOLD, MIN_WORD_CHANGES
        )
        
        # 保存替换后的数据集
        rsicd_output = os.path.join(OUTPUT_DIR, "rsicd_train_backtrans.json")
        
        with open(rsicd_output, 'w', encoding='utf-8') as f:
            json.dump(rsicd_replaced, f, ensure_ascii=False, indent=2)
        
        print(f"✓ RSICD替换版已保存: {rsicd_output}")
        print(f"  - 总样本: {rsicd_stats['total_samples']}")
        print(f"  - 替换数: {rsicd_stats['replaced_count']}")
        print(f"  - 替换率: {rsicd_stats['replacement_rate']*100:.1f}%")
        print(f"  - 过滤（相似度低）: {rsicd_stats['filtered_by_similarity']}")
        print(f"  - 过滤（无变化）: {rsicd_stats['filtered_by_diversity']}")
    
    print("\n" + "="*70)
    print("完成！")
    print("="*70)
    
    if SAMPLE_SIZE and SAMPLE_SIZE < len(all_captions):
        print("\n这是测试运行（样本数: {}）".format(SAMPLE_SIZE))
        print("如果满意，请修改 SAMPLE_SIZE = None 处理全量数据")
    else:
        print("\n✓ 已生成替换后的训练数据集")
        print(f"✓ 相似度阈值: {SIMILARITY_THRESHOLD}")
        print("\n下一步:")
        print("1. 人工抽查替换样本（建议随机抽50条）")
        print("2. 用替换后的数据集训练baseline，对比效果")
        print("3. 如果效果不明显，可以尝试:")
        print("   - 降低阈值到0.85（增加替换率）")
        print("   - 或使用OpenAI API获得更好的回译质量")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

