"""
词表覆盖率检查脚本
分析词表在真实训练数据中的覆盖率

功能：
1. 计算每个粒度的词表覆盖率
2. 找出高频但未被词表覆盖的词汇
3. 随机抽样caption进行人工验证

作者：Sean R. Liang
日期：2025-10-18
"""

import json
import random
from collections import Counter
import spacy

# 加载spacy模型
print("加载spacy模型...")
try:
    nlp = spacy.load("en_core_web_sm")
    print("✓ 模型加载完成\n")
except:
    print("❌ 错误：spacy模型未安装")
    print("请运行：python -m spacy download en_core_web_sm")
    exit(1)


# ====================================================================================
# 词形归一映射表
# ====================================================================================

# 单复数映射（词表复数 → 语料单数）
PLURAL_SINGULAR_MAP = {
    'roofs': 'roof',
    'pools': 'pool',
    'lands': 'land',
    'barelands': 'bareland',
    'viaducts': 'viaduct',
    'factories': 'factory',
    'warehouses': 'warehouse',
    'farmlands': 'farmland',
    'parks': 'park',
    'airports': 'airport',
    'bridges': 'bridge',
    'buildings': 'building',
    'houses': 'house',
    'roads': 'road',
    'cars': 'car',
    'planes': 'plane',
    'ships': 'ship',
    'trees': 'tree',
    'fields': 'field',
}

# 复合短语映射（语料空格形式 → 词表连写形式）
PHRASE_CANONICAL_MAP = {
    'storage tank': 'storagetanks',
    'storage tanks': 'storagetanks',
    'baseball field': 'baseballfield',
    'basketball court': 'basketballcourt',
    'football field': 'footballfield',
    'swimming pool': 'swimmingpool',
    'tennis court': 'tennis',  # 词表中有tennis
    'parking lot': 'parking',
}

# 英文停用词（精简版，去掉空间介词）
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but',
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had',
    'do', 'does', 'did',
    'will', 'would', 'should', 'could', 'may', 'might', 'must',
    'this', 'that', 'these', 'those',
    'it', 'its', 'they', 'them', 'their',
    'there',
}

# 需要忽略的动词分词/时态（非地物/场景/空间关系）
VERB_FORMS_TO_IGNORE = {
    'parked', 'planted', 'located', 'built', 'arranged',
    'scattered', 'lined', 'surrounded', 'connected',
    'painted', 'covered', 'shaped', 'curved'
}


def normalize_word(word, pos):
    """
    词形归一化
    
    参数：
        word: 原始词
        pos: 词性标签
        
    返回：
        normalized_word: 归一化后的词
        should_count: 是否应该计入内容词统计
    """
    word_lower = word.lower()
    
    # 1. 停用词过滤
    if word_lower in STOPWORDS:
        return word_lower, False
    
    # 2. 动词分词过滤
    if word_lower in VERB_FORMS_TO_IGNORE:
        return word_lower, False
    
    # 3. 只保留名词/形容词/副词/介词（对应Object/Scene/Layout）
    if pos not in ['NOUN', 'PROPN', 'ADJ', 'ADV', 'ADP']:
        return word_lower, False
    
    # 4. 单复数映射
    if word_lower in PLURAL_SINGULAR_MAP:
        return PLURAL_SINGULAR_MAP[word_lower], True
    
    return word_lower, True


def load_vocabulary(vocab_path):
    """加载词表"""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {
        'object': set(data['object']),
        'scene': set(data['scene']),
        'layout': set(data['layout'])
    }


def load_captions(json_paths):
    """加载所有caption"""
    all_captions = []
    for json_path in json_paths:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            captions = [item['caption'].lower().strip() for item in data]
            all_captions.extend(captions)
            print(f"✓ 加载 {len(captions)} 条caption from {json_path}")
        except Exception as e:
            print(f"⚠ 跳过 {json_path}: {e}")
    return all_captions


def analyze_coverage(captions, vocabulary, nlp):
    """
    分析词表覆盖率（双重口径：all + content）
    
    返回：
        coverage_stats: 覆盖率统计
        uncovered_words: 未覆盖的高频词
        covered_words: 已覆盖的词统计
    """
    print("\n分析词表覆盖率（内容词口径 + 词形归一）...")
    
    # 扩展词表（加入单复数和短语映射的反向查找）
    extended_vocabulary = {}
    for granularity, vocab_set in vocabulary.items():
        extended_set = set(vocab_set)
        # 加入单复数映射的两端
        for plural, singular in PLURAL_SINGULAR_MAP.items():
            if plural in vocab_set:
                extended_set.add(singular)
            if singular in vocab_set:
                extended_set.add(plural)
        extended_vocabulary[granularity] = extended_set
    
    # 统计变量
    all_words = Counter()                     # 全部词
    content_words = Counter()                  # 内容词
    covered_words_all = {g: Counter() for g in ['object', 'scene', 'layout']}
    covered_words_content = {g: Counter() for g in ['object', 'scene', 'layout']}
    
    total_tokens_all = 0
    total_tokens_content = 0
    
    for i, caption in enumerate(captions):
        if (i + 1) % 5000 == 0:
            print(f"  处理进度: {i+1}/{len(captions)}")
        
        # 1. 短语级匹配（优先）
        caption_lower = caption.lower()
        matched_phrases = set()
        for phrase, canonical in PHRASE_CANONICAL_MAP.items():
            if phrase in caption_lower:
                matched_phrases.add(phrase)
                # 检查canonical在哪个粒度
                for granularity, vocab_set in extended_vocabulary.items():
                    if canonical in vocab_set:
                        covered_words_all[granularity][canonical] += 1
                        covered_words_content[granularity][canonical] += 1
        
        # 2. 分词级匹配
        doc = nlp(caption)
        
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            
            # 跳过已经被短语匹配的词
            if any(token.text.lower() in phrase for phrase in matched_phrases):
                continue
            
            word_raw = token.text.lower()
            pos = token.pos_
            
            # 原始词统计（all口径）
            all_words[word_raw] += 1
            total_tokens_all += 1
            
            # 词形归一化
            word_normalized, should_count = normalize_word(word_raw, pos)
            
            # 内容词统计（content口径）
            if should_count:
                content_words[word_normalized] += 1
                total_tokens_content += 1
            
            # 检查是否在词表中（all口径）
            for granularity, vocab_set in extended_vocabulary.items():
                if word_raw in vocab_set or word_normalized in vocab_set:
                    covered_words_all[granularity][word_normalized] += 1
                    if should_count:
                        covered_words_content[granularity][word_normalized] += 1
                    break
    
    # 计算覆盖率（双重口径）
    total_covered_all = sum(sum(c.values()) for c in covered_words_all.values())
    total_covered_content = sum(sum(c.values()) for c in covered_words_content.values())
    
    coverage_rate_all = (total_covered_all / total_tokens_all * 100) if total_tokens_all > 0 else 0
    coverage_rate_content = (total_covered_content / total_tokens_content * 100) if total_tokens_content > 0 else 0
    
    # 找出未覆盖的高频词（基于content口径，出现>20次）
    all_vocab = set()
    for v in extended_vocabulary.values():
        all_vocab.update(v)
    
    uncovered_content = {w: c for w, c in content_words.items() 
                         if w not in all_vocab and c >= 20}
    uncovered_sorted = sorted(uncovered_content.items(), key=lambda x: x[1], reverse=True)
    
    coverage_stats = {
        'total_tokens_all': total_tokens_all,
        'total_tokens_content': total_tokens_content,
        'covered_tokens_all': total_covered_all,
        'covered_tokens_content': total_covered_content,
        'coverage_rate_all': coverage_rate_all,
        'coverage_rate_content': coverage_rate_content,
        'unique_words_all': len(all_words),
        'unique_words_content': len(content_words),
        'covered_unique_all': len([w for w in all_words if w in all_vocab]),
        'covered_unique_content': len([w for w in content_words if w in all_vocab]),
        'granularity_stats': {
            g: {
                'vocab_size': len(vocabulary[g]),
                'used_vocab_all': len(covered_words_all[g]),
                'used_vocab_content': len(covered_words_content[g]),
                'token_count_all': sum(covered_words_all[g].values()),
                'token_count_content': sum(covered_words_content[g].values())
            }
            for g in ['object', 'scene', 'layout']
        }
    }
    
    return coverage_stats, uncovered_sorted[:50], covered_words_content


def sample_captions_analysis(captions, vocabulary, nlp, n=5):
    """随机抽样n条caption进行详细分析"""
    print(f"\n随机抽样 {n} 条caption进行人工验证...")
    print("=" * 70)
    
    samples = random.sample(captions, min(n, len(captions)))
    
    for i, caption in enumerate(samples, 1):
        print(f"\n[样本 {i}] {caption}")
        print("-" * 70)
        
        doc = nlp(caption)
        words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
        
        matched = {'object': [], 'scene': [], 'layout': []}
        unmatched = []
        
        for word in words:
            found = False
            for granularity, vocab_set in vocabulary.items():
                if word in vocab_set:
                    matched[granularity].append(word)
                    found = True
                    break
            if not found:
                unmatched.append(word)
        
        total_words = len(words)
        matched_count = sum(len(m) for m in matched.values())
        coverage = (matched_count / total_words * 100) if total_words > 0 else 0
        
        print(f"  Object: {', '.join(matched['object']) if matched['object'] else '(无)'}")
        print(f"  Scene: {', '.join(matched['scene']) if matched['scene'] else '(无)'}")
        print(f"  Layout: {', '.join(matched['layout']) if matched['layout'] else '(无)'}")
        print(f"  未匹配: {', '.join(unmatched) if unmatched else '(无)'}")
        print(f"  覆盖率: {matched_count}/{total_words} = {coverage:.1f}%")
    
    print("\n" + "=" * 70)


def print_report(coverage_stats, uncovered_words):
    """打印详细报告（双重口径）"""
    print("\n" + "=" * 70)
    print("词表覆盖率分析报告")
    print("=" * 70)
    
    print(f"\n【整体覆盖率 - 双重口径对比】")
    print(f"\n  1️⃣ All Token 口径（包含停用词/助词）：")
    print(f"     总token数: {coverage_stats['total_tokens_all']:,}")
    print(f"     被词表覆盖: {coverage_stats['covered_tokens_all']:,}")
    print(f"     覆盖率: {coverage_stats['coverage_rate_all']:.2f}%")
    
    print(f"\n  2️⃣ Content Word 口径（仅内容词：名词/形容词/副词/介词）：")
    print(f"     总内容词token数: {coverage_stats['total_tokens_content']:,}")
    print(f"     被词表覆盖: {coverage_stats['covered_tokens_content']:,}")
    print(f"     ★★★ 内容词覆盖率: {coverage_stats['coverage_rate_content']:.2f}% ★★★")
    
    print(f"\n【唯一词汇统计】")
    print(f"  All口径：{coverage_stats['unique_words_all']:,} 个词 → 覆盖 {coverage_stats['covered_unique_all']:,} 个 "
          f"({coverage_stats['covered_unique_all']/coverage_stats['unique_words_all']*100:.1f}%)")
    print(f"  Content口径：{coverage_stats['unique_words_content']:,} 个词 → 覆盖 {coverage_stats['covered_unique_content']:,} 个 "
          f"({coverage_stats['covered_unique_content']/coverage_stats['unique_words_content']*100:.1f}%)")
    
    print(f"\n【各粒度统计】")
    print(f"  {'粒度':<10} {'词表大小':<10} {'使用(All)':<12} {'使用(Content)':<15} {'token(Content)':<15}")
    print("  " + "-" * 70)
    for granularity in ['object', 'scene', 'layout']:
        stats = coverage_stats['granularity_stats'][granularity]
        print(f"  {granularity.capitalize():<10} {stats['vocab_size']:<10} "
              f"{stats['used_vocab_all']:<12} {stats['used_vocab_content']:<15} {stats['token_count_content']:<15}")
    
    print(f"\n【未覆盖的高频内容词（Top 30）】")
    print("  这些词是内容词（名/形/副/介），出现频率>=20次，但未被词表收录")
    print("  " + "-" * 60)
    for i, (word, count) in enumerate(uncovered_words[:30], 1):
        print(f"  {i:2}. {word:<20} (出现 {count} 次)")
    
    # 判断等级（基于Content口径）
    rate = coverage_stats['coverage_rate_content']
    if rate >= 70:
        grade = "优秀 ✓✓✓"
        color = "🟩"
    elif rate >= 60:
        grade = "良好 ✓✓"
        color = "🟨"
    elif rate >= 50:
        grade = "及格 ✓"
        color = "🟧"
    else:
        grade = "需改进 ✗"
        color = "🟥"
    
    print(f"\n【最终评级（Content口径）】")
    print(f"  {color} 内容词覆盖率 {rate:.2f}% → {grade}")
    print()


def main():
    print("=" * 70)
    print("词表覆盖率检查工具 v2.0")
    print("（内容词口径 + 词形归一 + 短语匹配）")
    print("=" * 70)
    print()
    
    # 配置
    vocab_path = "data/vocabulary/rs_vocabulary_v1.3.json"
    caption_paths = [
        "data/finetune/rsitmd_train.json",
        "data/finetune/rsicd_train.json"
    ]
    
    # 加载词表
    print(f"[1] 加载词表: {vocab_path}")
    vocabulary = load_vocabulary(vocab_path)
    print(f"  ✓ Object: {len(vocabulary['object'])} 个")
    print(f"  ✓ Scene: {len(vocabulary['scene'])} 个")
    print(f"  ✓ Layout: {len(vocabulary['layout'])} 个")
    print()
    
    # 加载caption
    print(f"[2] 加载训练集caption...")
    captions = load_captions(caption_paths)
    print(f"  ✓ 总计: {len(captions)} 条caption")
    
    # 分析覆盖率
    print(f"\n[3] 分析覆盖率（双重口径）...")
    coverage_stats, uncovered_words, covered_words = analyze_coverage(captions, vocabulary, nlp)
    
    # 打印报告
    print_report(coverage_stats, uncovered_words)
    
    # 随机抽样
    print(f"\n[4] 随机抽样验证...")
    sample_captions_analysis(captions, vocabulary, nlp, n=5)
    
    # 保存报告
    report_path = "data/vocabulary/coverage_report_v1.3.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        # 添加uncovered_words到报告中
        coverage_stats['uncovered_top50'] = [
            {'word': w, 'count': c} for w, c in uncovered_words
        ]
        coverage_stats['methodology'] = {
            'version': '2.0',
            'features': [
                'Dual metrics: all tokens vs. content words only',
                'Lemmatization with singular-plural mapping',
                'Phrase-level matching (e.g., football field → footballfield)',
                'POS filtering (NOUN/ADJ/ADV/ADP only for content words)',
                'Stopword removal'
            ]
        }
        json.dump(coverage_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 详细报告已保存到: {report_path}")
    print()


if __name__ == "__main__":
    random.seed(42)  # 固定随机种子，便于复现
    main()

