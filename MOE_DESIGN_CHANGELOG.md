# MoE设计修订记录

## 修订日期：2025-10-20

## 修订原因
用户指出：**Scene层应该与词表构建标准一致，改为描述性形容词（ADJ），而不是场景类别（NOUN）**

---

## 主要修改

### 1. Scene Expert 重新定义

#### 修改前
- **名称**：场景专家（Scene Expert）
- **目标**：理解整体场景类别和全局分布
- **示例词**：residential, forest, airport, parking等
- **问题**：场景类别（NOUN）与Object层混淆，边界不清

#### 修改后
- **名称**：场景属性专家（Scene Attribute Expert）
- **目标**：识别和编码描述性属性
- **分类标准**：ADJ（形容词）
- **四大类属性**：
  1. **密度/规模属性**：dense, sparse, large, small, huge, tiny
  2. **颜色/外观属性**：green, blue, white, dark, bright, colorful
  3. **形状/排列属性**：circular, rectangular, scattered, aligned, curved
  4. **质量/状态属性**：new, old, empty, crowded, clean, busy

---

### 2. 三层粒度最终定义

基于词性（POS）的清晰划分：

| 粒度 | 词性 | 关注焦点 | 示例词汇 | 语义功能 |
|------|------|----------|----------|----------|
| **Object（物体层）** | NOUN（名词） | 具体物体和实体 | buildings, trees, cars, ships, airport, runway | 场景中"**有什么**" |
| **Scene（属性层）** | ADJ（形容词） | 描述性属性特征 | dense, large, green, circular, bright, scattered | 场景"**怎么样**" |
| **Layout（空间层）** | ADP/ADV（介词/副词） | 空间关系和位置 | near, beside, surrounded, between, along | 物体"**在哪里**" |

---

### 3. 设计优势

#### ✅ 与词表构建标准一致
- `build_rs_vocabulary.py` 第357行：`elif dominant_pos == 'ADJ' and freq >= MIN_FREQ_SCENE`
- Scene词的提取规则就是基于形容词（ADJ）

#### ✅ 三层粒度边界清晰
- **Object vs Scene**：名词 vs 形容词，语法功能完全不同
- **Scene vs Layout**：属性描述 vs 空间关系，语义角色明确
- 无交叉，无重叠

#### ✅ 可直接使用现有词表
- 验证显示：现有Scene词表中69%已经是形容词
- 少量名词误分类（如business, circles）可在精炼阶段修正

#### ✅ 符合NLP的词性功能
- 名词（NOUN）→ 实体识别任务
- 形容词（ADJ）→ 属性理解任务
- 介词（ADP）→ 关系抽取任务

---

### 4. 模型设计修改

#### Scene Expert网络结构适配
```python
class SceneExpert(nn.Module):
    """
    场景属性专家 - 关注全局特征和描述性属性
    
    设计特点：
    - 全局池化捕捉整体外观和分布特征
    - 多头自注意力建模全局依赖（适合判断"dense"、"large"等属性）
    - 适合提取可用形容词描述的视觉特征
    """
```

**关键设计要点**：
1. **强调CLS token**：属性是全局的，需要整体视角
2. **全局注意力**：判断"dense"需要看整个图像的分布
3. **颜色统计**：识别"green"/"blue"需要全图颜色分布

#### 监督信号调整
- **修改前**：Scene词频 + 场景标签（如RSITMD的31类）
- **修改后**：ADJ形容词占比 + 属性标签（如颜色/密度维度）

---

### 5. 路由器（Router）调整

#### 文本先验计算
```python
def compute_text_prior(self, text_tokens):
    """
    根据文本内容计算专家的先验权重
    
    示例：
    - "many dense buildings" 
      → Object: 1/3 (buildings)
      → Scene: 1/3 (dense)
      → Layout: 1/3 (无)
      
    - "a large green field near river"
      → Object: 2/5 (field, river)
      → Scene: 2/5 (large, green)
      → Layout: 1/5 (near)
    """
```

**变化**：
- Scene权重现在基于形容词计数（ADJ），而不是场景类别名词
- 更精确地反映描述性内容的丰富程度

---

## 实验预期

### 定性提升
1. **属性检索增强**：
   - Query: "a dense residential area" → Scene Expert权重↑
   - Query: "large green meadow" → Scene Expert权重↑↑

2. **物体-属性组合理解**：
   - "three large buildings" → Object (buildings) + Scene (large)
   - 两个专家协同工作，权重均衡

3. **细粒度属性区分**：
   - "sparse vs dense"
   - "large vs small"
   - "green vs blue"

### 定量提升（预期）
- RSITMD R@1 提升：+3-5%
- 特别是包含形容词修饰的query，提升更明显

---

## 后续工作

### 短期（实现阶段）
1. [ ] 实现修订后的SceneExpert代码
2. [ ] 调整Router的词表映射逻辑
3. [ ] 添加属性相关的评估指标

### 中期（优化阶段）
1. [ ] 精炼词表：移除Scene中的名词到Object
2. [ ] 增加属性标注：标注训练集中的密度/颜色/规模
3. [ ] 消融实验：验证Scene Expert的独立贡献

### 长期（扩展方向）
1. [ ] 多粒度属性：细分为颜色专家、密度专家、形状专家
2. [ ] 跨语言验证：中文形容词是否也能这样分类
3. [ ] 零样本泛化：测试未见过的形容词

---

## 验证结果

### 现有词表统计（前100个Scene词）
```
形容词 (ADJ): 69 个 ✓
名词 (NOUN): 9 个   ⚠ 需清理
其他词性: 22 个     ⚠ 部分是spaCy标注错误
```

**形容词示例**（符合新标准）：
- adjacent, agricultural, bare, beautiful, big, black, blue, bright, broad, busy
- central, circular, clean, colorful, commercial, compact, complete, complicated

**名词示例**（应移到Object）：
- business, calm, circles, colors, complex

**结论**：现有词表已经与新标准**高度一致**（69%符合），只需少量清理。

---

## 关键决策

### ❓为什么不用场景类别（如residential, airport）？

**理由**：
1. **语义重叠**：airport既是场景类别，也是具体物体（机场建筑）
2. **分类困难**：residential是场景还是属性？难以统一
3. **扩展性差**：场景类别有限（RSITMD的31类），形容词几乎无限

### ❓形容词真的能代表"场景"吗？

**答案**：**是的！** 从认知角度：
- 人类描述场景时，大量使用形容词："这是一个**繁忙的**商业区"
- 形容词提供了场景的**属性维度**，比类别标签更丰富
- "dense residential" = **密度属性** (Scene) + **场景类型** (Object层的residential building)

### ❓这种设计的理论基础？

**支持理论**：
1. **语言学**：形容词的修饰功能（attributive function）
2. **计算机视觉**：属性学习（attribute learning）
3. **认知心理学**：场景识别的全局属性理论（global property theory）

---

**最后更新**：2025-10-20  
**修订者**：Sean R. Liang  
**状态**：✅ 设计修订完成，待实现验证

