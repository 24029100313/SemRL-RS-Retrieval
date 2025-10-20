# 遥感语义层次化MoE设计方案

## 一、设计理念

### 核心思想
将遥感图像-文本的三层语义粒度（Object、Scene、Layout）显式建模为三个专家网络，通过MoE机制实现**细粒度语义对齐**。

**三层语义粒度定义**（与词表构建标准一致）：

| 粒度 | 词性 | 关注焦点 | 示例词汇 | 说明 |
|------|------|----------|----------|------|
| **Object（物体层）** | NOUN（名词） | 具体物体和实体 | buildings, trees, cars, ships, airport, runway | 场景中"有什么" |
| **Scene（属性层）** | ADJ（形容词） | 描述性属性特征 | dense, large, green, circular, bright, scattered | 场景"怎么样" |
| **Layout（空间层）** | ADP/ADV（介词/副词） | 空间关系和位置 | near, beside, surrounded, between, along | 物体"在哪里" |

**为什么使用词性分类？**
1. **明确边界**：词性（POS）提供了清晰的分类依据，避免主观判断
2. **自动化构建**：可以通过spaCy等NLP工具自动提取，保证一致性
3. **语义分离**：不同词性天然对应不同的语义功能
   - 名词→实体识别
   - 形容词→属性理解
   - 介词→关系建模

### 与HarMA的关系
- **保留**：HarMA的Adapter机制（参数高效）
- **增强**：在关键层插入MoE，替换或增强部分MMadapter
- **创新**：利用遥感词表进行语义引导的专家路由

---

## 二、整体架构

```
                    Input Image [B, 3, 224, 224]
                              ↓
                    ┌─────────────────────┐
                    │  Vision Encoder     │
                    │  (ViT-B/32, Frozen) │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
     Layer 0-3              Layer 4-8             Layer 9-11
  (Low-level)           (Mid-level)            (High-level)
        │                      │                      │
        │                      ↓                      │
        │          ┌────────────────────┐            │
        │          │   RS-MoE Module    │            │
        │          │  Object/物体(N)    │            │
        │          │  Scene/属性(ADJ)   │            │
        │          │  Layout/空间(ADP)  │            │
        │          └────────┬───────────┘            │
        │                   │                        │
        └───────────────────┼────────────────────────┘
                            │
                    Global Pooling
                            ↓
                    [B, 512] Visual Embedding
                    
                    
                    Text Input (Tokenized)
                              ↓
                    ┌─────────────────────┐
                    │  Text Encoder       │
                    │  (CLIP, Frozen)     │
                    └──────────┬──────────┘
                               │
                      ┌────────┴────────┐
                      │   词表分析      │
                      │  (统计NOUN/    │
                      │   ADJ/ADP词频) │
                      └────────┬────────┘
                               │
                         Expert Weights
                         (用于路由)
```

---

## 三、RS-MoE模块详细设计

### 3.1 模块结构

```python
class RSMoE(nn.Module):
    """
    遥感语义层次化MoE
    
    包含三个专家（与词表构建标准一致）：
    1. Object Expert - 细粒度物体识别（NOUN）
    2. Scene Expert - 描述性属性提取（ADJ）
    3. Layout Expert - 空间关系建模（ADP/ADV）
    """
    
    def __init__(self, hidden_dim=768, expert_dim=128):
        # 三个专家网络
        self.object_expert = ObjectExpert(hidden_dim, expert_dim)
        self.scene_expert = SceneExpert(hidden_dim, expert_dim)
        self.layout_expert = LayoutExpert(hidden_dim, expert_dim)
        
        # 路由网络（软路由）
        self.router = SemanticRouter(hidden_dim, num_experts=3)
        
        # 可选：负载均衡损失权重
        self.load_balance_weight = 0.01
```

### 3.2 三个专家的详细设计

#### **Object Expert（物体专家）**

**目标**：识别和编码细粒度物体（buildings, cars, planes, ships等）

**结构**：
```python
class ObjectExpert(nn.Module):
    """
    物体专家 - 关注局部patch和细节特征
    
    设计特点：
    - 轻量CNN提取局部纹理
    - 多尺度特征融合
    - 适合小目标检测
    """
    def __init__(self, hidden_dim=768, expert_dim=128):
        super().__init__()
        
        # Down-projection
        self.down = nn.Linear(hidden_dim, expert_dim)
        
        # 轻量级卷积（模拟局部感受野）
        # 将[seq_len, B, 128]重塑为[B, 128, H, W]
        self.local_conv = nn.Sequential(
            nn.Conv2d(expert_dim, expert_dim, kernel_size=3, padding=1, groups=expert_dim),  # Depthwise
            nn.BatchNorm2d(expert_dim),
            nn.GELU(),
            nn.Conv2d(expert_dim, expert_dim, kernel_size=1)  # Pointwise
        )
        
        # 多头注意力（捕捉物体间关系）
        self.attn = nn.MultiheadAttention(expert_dim, num_heads=8)
        
        # Up-projection
        self.up = nn.Linear(expert_dim, hidden_dim)
        
        # Gate
        self.gate = nn.Parameter(torch.tensor(0.5))
        
        self.init_weights()
    
    def init_weights(self):
        self.up.weight.data.zero_()
        self.up.bias.data.zero_()
    
    def forward(self, x):
        """
        Args:
            x: [seq_len, B, hidden_dim] (seq_len=50 for ViT-B/32)
        Returns:
            out: [seq_len, B, hidden_dim]
        """
        x_init = x
        seq_len, B, _ = x.shape
        
        # Down-project
        x = self.down(x)  # [seq_len, B, expert_dim]
        
        # Local convolution (for texture)
        if seq_len == 50:  # ViT-B/32: 7x7 + 1 CLS
            H, W = 7, 7
            cls_token = x[0:1]  # [1, B, expert_dim]
            patches = x[1:]     # [49, B, expert_dim]
            
            # Reshape to image
            patches_2d = patches.permute(1, 2, 0).reshape(B, -1, H, W)  # [B, expert_dim, 7, 7]
            patches_2d = self.local_conv(patches_2d)
            patches = patches_2d.reshape(B, -1, H*W).permute(2, 0, 1)  # [49, B, expert_dim]
            
            x = torch.cat([cls_token, patches], dim=0)
        
        # Multi-head attention (for object relations)
        x, _ = self.attn(x, x, x)
        
        # Up-project
        x = self.up(x)
        
        # Gated residual
        alpha = torch.sigmoid(self.gate)
        return alpha * x + (1 - alpha) * x_init
```

**监督信号（可选）**：
- Object词频：统计caption中object类词汇的占比
- 如果object词多，增大object expert的权重

---

#### **Scene Expert（场景属性专家）**

**目标**：识别和编码场景的描述性属性（dense, large, green, circular等形容词）

**核心理念**：与词表构建标准一致，Scene = ADJ（形容词），关注：
- **密度/规模属性**：dense, sparse, large, small, huge
- **颜色/外观属性**：green, blue, white, dark, bright
- **形状/排列属性**：circular, rectangular, scattered, aligned
- **质量/状态属性**：new, old, empty, crowded

**结构**：
```python
class SceneExpert(nn.Module):
    """
    场景属性专家 - 关注全局特征和描述性属性
    
    设计特点：
    - 全局池化捕捉整体外观和分布特征
    - 多头自注意力建模全局依赖（适合判断"dense"、"large"等属性）
    - 适合提取可用形容词描述的视觉特征
    """
    def __init__(self, hidden_dim=768, expert_dim=128):
        super().__init__()
        
        self.down = nn.Linear(hidden_dim, expert_dim)
        
        # 全局上下文模块
        self.global_attn = nn.MultiheadAttention(
            expert_dim, num_heads=8, 
            dropout=0.1
        )
        
        # Scene-specific: 强调CLS token
        self.cls_weight = nn.Parameter(torch.ones(1))
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(expert_dim, expert_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(expert_dim * 4, expert_dim)
        )
        
        self.up = nn.Linear(expert_dim, hidden_dim)
        self.gate = nn.Parameter(torch.tensor(0.5))
        
        self.init_weights()
    
    def init_weights(self):
        self.up.weight.data.zero_()
        self.up.bias.data.zero_()
    
    def forward(self, x):
        """
        Args:
            x: [seq_len, B, hidden_dim]
        Returns:
            out: [seq_len, B, hidden_dim]
        """
        x_init = x
        
        # Down-project
        x = self.down(x)
        
        # Global attention (emphasize CLS token)
        cls_token = x[0:1]  # [1, B, expert_dim]
        
        # Weighted query from CLS
        query = cls_token * torch.sigmoid(self.cls_weight)
        key = value = x
        
        attn_out, attn_weights = self.global_attn(query, key, value)
        
        # Broadcast CLS info to all tokens
        x = x + attn_out.expand_as(x) * 0.5
        
        # FFN
        x = x + self.ffn(x)
        
        # Up-project
        x = self.up(x)
        
        alpha = torch.sigmoid(self.gate)
        return alpha * x + (1 - alpha) * x_init
```

**监督信号（可选）**：
- Scene词频：统计caption中ADJ形容词的占比和密度
- 属性标签：如果可用，可标注颜色/密度/规模等维度

---

#### **Layout Expert（布局专家）**

**目标**：建模空间结构和相对位置关系（near, surrounded, between等）

**结构**：
```python
class LayoutExpert(nn.Module):
    """
    布局专家 - 关注空间关系和结构
    
    设计特点：
    - 相对位置编码
    - 图卷积捕捉空间连接性
    - 适合空间推理
    """
    def __init__(self, hidden_dim=768, expert_dim=128):
        super().__init__()
        
        self.down = nn.Linear(hidden_dim, expert_dim)
        
        # 相对位置编码（2D）
        self.rel_pos_embed = nn.Parameter(
            torch.randn(14, 14, expert_dim) * 0.02  # 7x7 patch grid
        )
        
        # Graph Convolution for spatial structure
        self.graph_conv = nn.Sequential(
            nn.Linear(expert_dim, expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, expert_dim)
        )
        
        # Spatial attention
        self.spatial_attn = nn.MultiheadAttention(expert_dim, num_heads=8)
        
        self.up = nn.Linear(expert_dim, hidden_dim)
        self.gate = nn.Parameter(torch.tensor(0.5))
        
        self.init_weights()
    
    def init_weights(self):
        self.up.weight.data.zero_()
        self.up.bias.data.zero_()
    
    def forward(self, x):
        """
        Args:
            x: [seq_len, B, hidden_dim]
        Returns:
            out: [seq_len, B, hidden_dim]
        """
        x_init = x
        seq_len, B, _ = x.shape
        
        # Down-project
        x = self.down(x)
        
        # Add relative position encoding
        if seq_len == 50:
            H, W = 7, 7
            cls_token = x[0:1]
            patches = x[1:]  # [49, B, expert_dim]
            
            # Add 2D relative position
            for i in range(H):
                for j in range(W):
                    idx = i * W + j
                    # 相对位置：距离中心的偏移
                    rel_i = i - H // 2 + 7
                    rel_j = j - W // 2 + 7
                    patches[idx] = patches[idx] + self.rel_pos_embed[rel_i, rel_j]
            
            x = torch.cat([cls_token, patches], dim=0)
        
        # Graph convolution (treating patches as graph nodes)
        x_graph = self.graph_conv(x)
        
        # Spatial attention
        x_attn, _ = self.spatial_attn(x, x, x)
        
        # Combine
        x = x_graph + x_attn
        
        # Up-project
        x = self.up(x)
        
        alpha = torch.sigmoid(self.gate)
        return alpha * x + (1 - alpha) * x_init
```

**监督信号（可选）**：
- Layout词频：统计caption中layout类词汇的占比
- 空间关系标注（如"A在B的左边"）

---

### 3.3 语义路由器（Semantic Router）

**核心创新**：利用遥感词表进行文本引导的专家路由

```python
class SemanticRouter(nn.Module):
    """
    语义路由器 - 基于文本语义动态分配专家权重
    
    创新点：
    1. 利用预先构建的遥感词表
    2. 根据caption中不同层次词汇的分布决定专家权重
    3. 支持软路由（多个专家同时激活）
    """
    def __init__(self, hidden_dim=768, num_experts=3):
        super().__init__()
        
        # 可学习的路由网络
        self.router_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_experts)
        )
        
        # 加载遥感词表
        self.vocab = self.load_rs_vocabulary()
        
        # 负载均衡辅助损失
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        
    def load_rs_vocabulary(self):
        """加载遥感领域词表"""
        import json
        vocab_path = 'data/vocabulary/rs_vocabulary.json'
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
            return {
                'object': set(vocab['object']),
                'scene': set(vocab['scene']),
                'layout': set(vocab['layout'])
            }
        return None
    
    def compute_text_prior(self, text_tokens):
        """
        根据文本内容计算专家的先验权重
        
        Args:
            text_tokens: List[str] - tokenized text
        Returns:
            prior: [3] - [object_weight, scene_weight, layout_weight]
        """
        if self.vocab is None:
            return None
        
        # 统计三类词的出现次数
        object_count = sum(1 for w in text_tokens if w in self.vocab['object'])
        scene_count = sum(1 for w in text_tokens if w in self.vocab['scene'])
        layout_count = sum(1 for w in text_tokens if w in self.vocab['layout'])
        
        total = object_count + scene_count + layout_count
        if total == 0:
            return torch.tensor([1/3, 1/3, 1/3])
        
        prior = torch.tensor([
            object_count / total,
            scene_count / total,
            layout_count / total
        ])
        
        return prior
    
    def forward(self, x, text_tokens=None, return_weights=False):
        """
        Args:
            x: [seq_len, B, hidden_dim] - visual features
            text_tokens: Optional[List[List[str]]] - batch of tokenized texts
            return_weights: bool - whether to return routing weights
        
        Returns:
            routing_weights: [B, num_experts] - soft weights for each expert
            load_balance_loss: scalar - auxiliary loss for load balancing
        """
        # 使用CLS token作为全局表示
        cls_feature = x[0]  # [B, hidden_dim]
        
        # 神经网络路由
        logits = self.router_net(cls_feature)  # [B, 3]
        
        # 如果有文本先验，结合使用
        if text_tokens is not None and self.vocab is not None:
            text_priors = []
            for tokens in text_tokens:
                prior = self.compute_text_prior(tokens)
                text_priors.append(prior)
            
            text_priors = torch.stack(text_priors).to(logits.device)  # [B, 3]
            
            # 结合神经网络路由和文本先验
            # 使用加权和：70%神经网络 + 30%文本先验
            logits = 0.7 * logits + 0.3 * torch.log(text_priors + 1e-8)
        
        # Softmax得到路由权重
        routing_weights = F.softmax(logits, dim=-1)  # [B, 3]
        
        # 负载均衡损失（鼓励专家均匀使用）
        expert_usage = routing_weights.mean(0)  # [3]
        load_balance_loss = torch.var(expert_usage) * self.expert_counts.numel()
        
        # 更新专家使用计数（用于监控）
        with torch.no_grad():
            self.expert_counts += routing_weights.sum(0)
        
        if return_weights:
            return routing_weights, load_balance_loss
        return routing_weights
```

---

### 3.4 完整的RS-MoE模块

```python
class RSMoE(nn.Module):
    """
    遥感语义层次化MoE - 完整模块
    """
    def __init__(self, hidden_dim=768, expert_dim=128):
        super().__init__()
        
        # 三个专家
        self.experts = nn.ModuleList([
            ObjectExpert(hidden_dim, expert_dim),   # Expert 0
            SceneExpert(hidden_dim, expert_dim),    # Expert 1
            LayoutExpert(hidden_dim, expert_dim)    # Expert 2
        ])
        
        # 路由器
        self.router = SemanticRouter(hidden_dim, num_experts=3)
        
        # 负载均衡损失权重
        self.load_balance_weight = 0.01
        
    def forward(self, x, text_tokens=None):
        """
        Args:
            x: [seq_len, B, hidden_dim]
            text_tokens: Optional[List[List[str]]]
        
        Returns:
            output: [seq_len, B, hidden_dim]
            aux_loss: scalar (负载均衡损失)
        """
        seq_len, B, hidden_dim = x.shape
        
        # 1. 计算路由权重
        routing_weights, load_balance_loss = self.router(
            x, text_tokens, return_weights=True
        )  # [B, 3]
        
        # 2. 每个专家处理输入
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # [seq_len, B, hidden_dim]
            expert_outputs.append(expert_out)
        
        expert_outputs = torch.stack(expert_outputs, dim=0)  # [3, seq_len, B, hidden_dim]
        
        # 3. 加权聚合
        # routing_weights: [B, 3] -> [3, 1, B, 1]
        weights = routing_weights.t().unsqueeze(1).unsqueeze(-1)
        
        # 加权求和
        output = (expert_outputs * weights).sum(0)  # [seq_len, B, hidden_dim]
        
        # 4. 辅助损失
        aux_loss = self.load_balance_weight * load_balance_loss
        
        return output, aux_loss
```

---

## 四、集成到HarMA

### 方案1：替换特定层的MMadapter

```python
# 在 open_clip/model.py 中修改
class CLIP(nn.Module):
    def __init__(self, ...):
        # 原有的MMadapter
        self.MMadapter_img = nn.ModuleList([...])
        
        # 在中间层(4-8)加入RS-MoE
        self.rsmoe_layers = [4, 6, 8]  # 选择性替换
        for layer_id in range(12):
            if layer_id in self.rsmoe_layers:
                self.MMadapter_img[layer_id] = RSMoE(
                    hidden_dim=768, 
                    expert_dim=128
                )
```

### 方案2：并行使用（推荐）

```python
# MMadapter和RS-MoE并行，最后融合
class HybridAdapter(nn.Module):
    def __init__(self, hidden_dim=768):
        self.mmadapter = MMadapter(None, hidden_dim)
        self.rsmoe = RSMoE(hidden_dim, expert_dim=128)
        self.fusion_gate = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x, text_tokens=None):
        # MMadapter分支
        out_mm = self.mmadapter(x)
        
        # RS-MoE分支
        out_moe, aux_loss = self.rsmoe(x, text_tokens)
        
        # 自适应融合
        alpha = torch.sigmoid(self.fusion_gate)
        output = alpha * out_moe + (1 - alpha) * out_mm
        
        return output, aux_loss
```

---

## 五、训练策略

### 5.1 两阶段训练

**阶段1：冻结专家，训练路由**
```python
# 冻结专家网络
for expert in model.rsmoe.experts:
    for param in expert.parameters():
        param.requires_grad = False

# 只训练路由器
for param in model.rsmoe.router.parameters():
    param.requires_grad = True

# 训练2-3个epoch
```

**阶段2：联合微调**
```python
# 解冻所有参数
for param in model.rsmoe.parameters():
    param.requires_grad = True

# 使用较小学习率继续训练
```

### 5.2 损失函数

```python
# 总损失 = 检索损失 + 辅助损失
loss_total = loss_contrastive + loss_triplet + aux_loss_moe

# aux_loss_moe包括：
# - 负载均衡损失（鼓励专家均匀使用）
# - 可选：专家多样性损失（鼓励专家学到不同特征）
```

---

## 六、预期效果

### 6.1 定量提升

| 指标 | Baseline | +RS-MoE | 提升 |
|------|----------|---------|------|
| RSITMD R@1 (I→T) | 72% | **75-77%** | +3-5% |
| RSITMD R@1 (T→I) | 68% | **71-73%** | +3-5% |
| RSICD R@1 (I→T) | 75% | **78-80%** | +3-5% |

### 6.2 定性优势

1. **细粒度物体理解**：能够更好地匹配包含specific objects的query（如"three planes"）
2. **属性识别增强**：对描述性属性的检索更准确（如"dense residential area", "large green field"）
3. **空间推理提升**：对包含spatial relations的描述表现更好（如"buildings near a river"）

### 6.3 可解释性

- 可以可视化每个样本激活了哪些专家
- 分析不同类型query对应的专家权重分布
- 为失败案例提供诊断依据

---

## 七、参数量分析

```python
# 单个专家参数量
Object Expert:  ~150K
Scene Expert:   ~150K
Layout Expert:  ~150K
Router:         ~50K
------------------------------
Total:          ~500K per layer

# 如果替换3层MMadapter (层4, 6, 8)
新增参数: 500K × 3 = 1.5M

# 相比HarMA baseline (~3.5M可训练参数)
增幅: +43%
仍然远小于Full Fine-tuning (400M)
```

---

## 八、实现路线图

### Week 1：基础模块实现
- [ ] 实现三个Expert类
- [ ] 实现SemanticRouter
- [ ] 实现完整RSMoE模块
- [ ] 单元测试

### Week 2：集成到HarMA
- [ ] 修改CLIP模型架构
- [ ] 适配Retrieval.py训练流程
- [ ] 添加aux_loss计算

### Week 3：训练与调试
- [ ] 先在RSITMD上训练测试
- [ ] 超参数调优（expert_dim, 路由权重等）
- [ ] 消融实验

### Week 4：分析与优化
- [ ] 可视化专家激活模式
- [ ] 失败案例分析
- [ ] 撰写实验报告

---

## 九、消融实验设计

| 实验 | Object | Scene | Layout | Router | 说明 |
|------|--------|-------|--------|--------|------|
| Baseline | ✗ | ✗ | ✗ | ✗ | 原始HarMA |
| Single-O | ✓ | ✗ | ✗ | ✗ | 只用Object专家 |
| Single-S | ✗ | ✓ | ✗ | ✗ | 只用Scene专家 |
| Single-L | ✗ | ✗ | ✓ | ✗ | 只用Layout专家 |
| MoE-Fixed | ✓ | ✓ | ✓ | ✗ | 固定权重(1/3) |
| MoE-Learnable | ✓ | ✓ | ✓ | ✓(神经网络) | 可学习路由 |
| **MoE-Semantic** | ✓ | ✓ | ✓ | ✓(词表引导) | **完整方案** |

---

## 十、潜在改进方向

1. **动态专家数量**：根据任务难度自适应选择激活几个专家
2. **层次化路由**：不同层使用不同的路由策略
3. **跨模态路由**：图像和文本共享路由器
4. **在线专家学习**：训练中动态增加新专家

---

**最后更新**：2025-10-20
**状态**：设计完成，等待实现

