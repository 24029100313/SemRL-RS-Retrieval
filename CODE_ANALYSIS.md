# HarMA 代码架构全面分析文档

## 一、项目概述

**HarMA (Harmonized Transfer Learning and Modality Alignment)** 是一个用于遥感图像-文本检索的深度学习框架，发表于 ICLRW 2024。

### 核心特点：
1. **参数高效微调（PEFT）**：通过Adapter机制只训练少量参数
2. **层次化多模态适配器（Hierarchical Multimodal Adapter）**：受人脑信息处理启发
3. **双模态对齐**：图像-文本跨模态检索
4. **基于CLIP架构**：支持ViT-B/32和GeoRSCLIP预训练模型

---

## 二、项目结构树

```
HarMA-main/
├── configs/                    # 配置文件
│   ├── config_bert.json       # BERT配置
│   ├── config_swinT_224.json  # Swin Transformer配置
│   ├── Retrieval_rsitmd_vit.yaml    # RSITMD数据集+ViT配置
│   ├── Retrieval_rsicd_vit.yaml     # RSICD数据集+ViT配置
│   ├── Retrieval_rsitmd_geo.yaml    # RSITMD+GeoRSCLIP配置
│   └── Retrieval_rsicd_geo.yaml     # RSICD+GeoRSCLIP配置
│
├── models/                     # 核心模型
│   ├── harma.py               # HarMA基础类（损失函数、对齐策略）
│   ├── model_retrieval.py     # 检索模型主类
│   ├── mga.py                 # 多模态适配器（BiShareAdapter, MMadapter）
│   ├── bert.py                # BERT文本编码器
│   ├── swin_transformer.py    # Swin-T视觉编码器
│   ├── vit.py                 # ViT视觉编码器
│   └── __init__.py            # 模型导出
│
├── dataset/                    # 数据处理
│   ├── re_dataset.py          # 检索任务数据集（训练/评估）
│   ├── utils.py               # 数据预处理工具
│   ├── randaugment.py         # 数据增强
│   └── __init__.py            # 数据集创建函数
│
├── open_clip/                  # OpenCLIP实现
│   ├── model.py               # CLIP模型定义（集成Adapter）
│   ├── transformer.py         # Transformer模块
│   ├── tokenizer.py           # 文本分词器
│   └── factory.py             # 模型工厂
│
├── utils/                      # 工具函数
│   ├── __init__.py            # 分布式训练、指标记录
│   └── checkpointer.py        # 模型保存/加载
│
├── scripts/                    # 辅助脚本
│   └── build_rs_vocabulary.py # 遥感领域词表构建
│
├── data/finetune/              # 数据标注文件
│   ├── rsitmd_train.json      # RSITMD训练集
│   ├── rsitmd_val.json        # RSITMD验证集
│   ├── rsitmd_test.json       # RSITMD测试集
│   ├── rsicd_train.json       # RSICD训练集
│   ├── rsicd_val.json         # RSICD验证集
│   └── rsicd_test.json        # RSICD测试集
│
├── run.py                      # 训练/测试启动脚本
├── Retrieval.py               # 检索任务主程序
├── optim.py                   # 优化器配置
├── scheduler.py               # 学习率调度器
└── begin.ipynb                # Jupyter快速入门

```

---

## 三、核心组件详解

### 3.1 数据流（Data Flow）

#### **数据格式** (`data/finetune/*.json`)
```json
{
    "caption": "The river banks decorated with trees...",
    "image": "train/bridge_942.tif",
    "image_id": 539,
    "label_name": "bridge",
    "label": 0
}
```

#### **数据集类** (`dataset/re_dataset.py`)

**训练集** - `re_train_dataset`:
- 输入：图像路径 + caption文本
- 输出：`(image_tensor, caption_text, img_id, label)`
- 特点：支持数据增强（RandomAugment）

**测试集** - `re_eval_dataset`:
- 输入：图像 + 多条caption
- 输出：`(image_tensor, image_index)`
- 特点：构建 `txt2img` 和 `img2txt` 映射（用于评估）

#### **数据处理流程**
```
原始JSON → re_dataset → DataLoader → Tokenizer → Model
                ↓
         Transforms (Resize, Augment, Normalize)
```

---

### 3.2 模型架构（Model Architecture）

#### **整体架构** (`models/model_retrieval.py` + `open_clip/model.py`)

```
Input: Image + Text
         ↓
    ┌────────────────┐
    │  Vision Tower  │  (ViT-B/32 or GeoRSCLIP)
    │  + MMadapter   │  ← 12层，每层插入Adapter
    └────────────────┘
         ↓ (image_emb [B, 512])
         
    ┌────────────────┐
    │   Text Tower   │  (CLIP Text Encoder)
    │  + MMadapter   │  ← 12层，每层插入Adapter
    │ + BiShareAdapter │  ← 12个共享适配器
    └────────────────┘
         ↓ (text_emb [B, 512])
         
    ┌──────────────────────────┐
    │  Contrastive Learning    │
    │  + Triplet Loss          │
    └──────────────────────────┘
         ↓
    Loss (用于优化)
```

#### **核心创新：层次化适配器** (`models/mga.py` + `open_clip/model.py`)

##### **1. BiShareAdapter（双向共享适配器）**
```python
class BiShareAdapter(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=8):
        # Down-projection: 512→256
        self.l1 = nn.Linear(hidden_dim, hidden_dim//2)
        # Multi-head Self-Attention
        self.multihead_attention1 = nn.MultiheadAttention(hidden_dim//2, num_heads)
        # Gated fusion: α*attn + (1-α)*identity
        self.gate1 = nn.Parameter(torch.tensor(0.6))
        # Up-projection: 256→512
        self.l2 = nn.Linear(hidden_dim//2, hidden_dim)
        
    def forward(self, x):
        x_init = x
        x = self.l1(x)                          # Down
        attn_out, _ = self.multihead_attention1(x, x, x)  # Self-Attn
        x = gelu(x)
        α = sigmoid(self.gate1)
        x = α * attn_out + (1-α) * x           # Gated
        x = self.l2(x)                         # Up
        return x + x_init                       # Residual
```
**作用**：在文本模态的12层中，每层有一个独立的BiShareAdapter，用于增强文本特征。

##### **2. MMadapter（多模态适配器）**
```python
class MMadapter(nn.Module):
    def __init__(self, share_adapter, hidden_size, layer_id):
        # Down: 512→128 or 768→128
        self.img_proj_down = nn.Linear(hidden_size, 128)
        # Self-Attention
        self.multihead_attention = nn.MultiheadAttention(128, 8)
        # 引用共享适配器
        self.BiShareAdapterxx = share_adapter  # 可为None（图像）或BiShareAdapter（文本）
        # Gated fusion
        self.gate1 = nn.Parameter(torch.tensor(0.6))
        # Up: 128→512/768
        self.img_proj_up = nn.Linear(128, hidden_size)
        
    def forward(self, x):
        x_init = x
        x = self.img_proj_down(x)              # Down to 128
        x = gelu(x)
        x_mid = x
        x, _ = self.multihead_attention(x, x, x)  # Self-Attn
        if self.BiShareAdapterxx is not None:
            x = self.BiShareAdapterxx(x)       # Cross-modal interaction
        x, _ = self.multihead_attention(x, x, x)  # Self-Attn again
        α = sigmoid(self.gate1)
        x = α * x_mid + (1-α) * x             # Skip connection
        x = self.img_proj_up(x)                # Up to original dim
        return x + x_init                       # Residual
```

**使用方式**：
- **图像塔**：12个MMadapter，`share_adapter=None`（独立）
- **文本塔**：12个MMadapter，`share_adapter=BiShareAdapter[i]`（与图像共享）

**关键设计**：
- 文本MMadapter引用同层的BiShareAdapter → 实现跨模态信息交互
- 参数量极小：只训练Adapter参数（~1-5% of total params）

#### **参数冻结策略** (`Retrieval.py` - `set_trainable()`)
```python
# 1. 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 2. 解冻Adapter参数
for name, module in model.named_modules():
    if ('BiShareAdapter' in name) or 
       ('MMadapter' in name) or 
       ('mmadapter' in name):
        module.train()
        for param in module.parameters():
            param.requires_grad = True

# 3. 解冻gate和temp参数
for name, param in model.named_parameters():
    if ('gate' in name) or ('temp' in name):
        param.requires_grad = True
```

---

### 3.3 损失函数（Loss Functions）

#### **1. 对比学习损失** (`models/harma.py` - `get_contr_loss()`)

```python
def get_contr_loss(self, image_feat, text_feat, idx=None):
    """
    标准InfoNCE对比学习损失
    
    公式：
    L_i2t = -log(exp(sim(i,t+)/τ) / Σ_j exp(sim(i,t_j)/τ))
    L_t2i = -log(exp(sim(t,i+)/τ) / Σ_j exp(sim(t,i_j)/τ))
    
    其中τ=0.07（可学习温度参数）
    """
    # 跨GPU聚合特征
    image_feat_all = allgather(image_feat)  # [B*world_size, 512]
    text_feat_all = allgather(text_feat)
    
    # 计算相似度矩阵
    logits = (image_feat_all @ text_feat_all.t()) / self.temp  # [B, B]
    
    # 对角线为正样本
    labels = torch.arange(B, device=device)
    loss_i2t = CrossEntropy(logits, labels)
    loss_t2i = CrossEntropy(logits.t(), labels)
    
    return (loss_i2t + loss_t2i) / 2
```

**效果**：拉近匹配的图像-文本对，推远不匹配的对。

#### **2. 加权三元组损失** (`models/harma.py` - `weighted_triplet_loss()`)

```python
def weighted_triplet_loss(self, image_feat, text_feat, margin=0.2, gamma=2.0):
    """
    带Focal Weight的三元组损失（Hard Negative Mining）
    
    公式：
    cost_s = Σ_j w_j * max(0, margin + sim(i, t_j) - sim(i, t+))
    cost_im = Σ_i w_i * max(0, margin + sim(t, i_j) - sim(t, i+))
    
    w = (1 - exp(-cost))^γ  ← Focal weight，难样本权重大
    """
    scores = image_feat_all @ text_feat_all.t()  # [B, B]
    diagonal = scores.diag().view(B, 1)
    
    # Caption Retrieval: i→t
    cost_s = (margin + scores - diagonal).clamp(min=0)
    cost_s.masked_fill_(eye_mask, 0)
    
    # Image Retrieval: t→i
    cost_im = (margin + scores - diagonal.t()).clamp(min=0)
    cost_im.masked_fill_(eye_mask, 0)
    
    # Focal weighting
    weights_s = (1 - torch.exp(-cost_s)) ** gamma
    weights_im = (1 - torch.exp(-cost_im)) ** gamma
    
    loss = (weights_s * cost_s).sum() + (weights_im * cost_im).sum()
    return loss / 2.0
```

**效果**：
- 关注难负样本（相似但不匹配的样本）
- γ=2.0: Focal weight参数，控制难样本的强调程度

#### **总损失**
```python
loss = loss_contr + loss_triplet
```

---

### 3.4 训练流程（Training Pipeline）

#### **主程序** (`Retrieval.py`)

```
1. 初始化
   ├── 加载配置文件 (YAML)
   ├── 创建模型 HarMA(config)
   ├── 加载预训练权重 (GeoRSCLIP / ViT-B/32)
   └── 冻结backbone，解冻Adapter

2. 数据加载
   ├── create_dataset('re', config) 
   │   ├── train: re_train_dataset
   │   ├── val: re_eval_dataset
   │   └── test: re_eval_dataset
   └── DistributedSampler (多GPU)

3. 优化器&调度器
   ├── AdamW (lr=4e-6 for GeoRSCLIP, 4e-4 for ViT)
   └── Linear warmup (10% steps)

4. 训练循环 (每个epoch)
   for batch in train_loader:
       ├── image, text = batch
       ├── image_emb = model.get_vis_emb(image)
       ├── text_emb = model.get_txt_emb(text)
       ├── loss_contr = get_contr_loss(image_emb, text_emb)
       ├── loss_triplet = weighted_triplet_loss(image_emb, text_emb)
       ├── loss = loss_contr + loss_triplet
       ├── loss.backward()
       └── optimizer.step()

5. 评估 (每个epoch后)
   ├── 提取所有图像特征: [N_img, 512]
   ├── 提取所有文本特征: [N_txt, 512]
   ├── 计算相似度矩阵: sims = img_feat @ txt_feat.t()
   └── 计算指标: R@1, R@5, R@10 (Image→Text & Text→Image)

6. 保存最佳模型
   if r_mean > best:
       save_checkpoint('checkpoint_best.pth')
```

---

### 3.5 评估流程（Evaluation Pipeline）

#### **评估函数** (`Retrieval.py` - `evaluation()`)

```python
@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    """
    完整的检索评估流程
    """
    model.eval()
    
    # 1. 提取图像特征
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_embed = model.get_vis_emb(image)  # [B, 512]
        image_embeds.append(image_embed)
    image_embeds = torch.cat(image_embeds, dim=0)  # [N_img, 512]
    
    # 2. 提取文本特征
    texts = data_loader.dataset.text  # 所有caption
    text_embeds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        text_input = tokenizer(batch_texts).to(device)
        text_embed = model.get_txt_emb(text_input)  # [B, 512]
        text_embeds.append(text_embed)
    text_embeds = torch.cat(text_embeds, dim=0)  # [N_txt, 512]
    
    # 3. 计算相似度矩阵
    sims_matrix = image_embeds @ text_embeds.t()  # [N_img, N_txt]
    
    return sims_matrix, sims_matrix.t()
```

#### **指标计算** (`Retrieval.py` - `itm_eval()`)

```python
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    """
    Image-Text Matching 评估
    
    指标：
    - Text Retrieval (Image→Text): R@1, R@5, R@10
    - Image Retrieval (Text→Image): R@1, R@5, R@10
    """
    # Image→Text
    ranks = []
    for img_idx, scores in enumerate(scores_i2t):
        # 获取该图像对应的所有正样本文本索引
        gt_txt_ids = img2txt[img_idx]  # e.g., [0,1,2,3,4]
        
        # 排序：相似度从高到低
        sorted_indices = np.argsort(scores)[::-1]
        
        # 找到第一个正样本的排名
        rank = min([np.where(sorted_indices == gt)[0][0] 
                    for gt in gt_txt_ids])
        ranks.append(rank)
    
    # 计算Recall
    R1 = 100.0 * (ranks < 1).sum() / len(ranks)
    R5 = 100.0 * (ranks < 5).sum() / len(ranks)
    R10 = 100.0 * (ranks < 10).sum() / len(ranks)
    
    # 类似计算Text→Image
    # ...
    
    return {'txt_r1': R1, 'txt_r5': R5, 'txt_r10': R10,
            'img_r1': IR1, 'img_r5': IR5, 'img_r10': IR10}
```

---

### 3.6 配置文件详解

#### **RSITMD + ViT-B/32** (`configs/Retrieval_rsitmd_vit.yaml`)

```yaml
# 数据集
train_file: ['data/finetune/rsitmd_train.json']
test_file: 'data/finetune/rsitmd_test.json'
image_root: '../PIR/rsitmd/'

# 模型
is_harma: True
model: 'vit'              # 使用ViT-B/32 (OpenAI预训练)
embed_dim: 512
temp1: 0.07               # 对比学习温度

# 训练
batch_size_train: 214     # 每个GPU的batch size
optimizer:
  opt: adamW
  lr: 0.0004              # 学习率
  weight_decay: 0.04

schedular:
  sched: linear
  epochs: 50              # 训练轮数
  num_warmup_steps: 0.1   # 10% warmup

# 损失
use_affil_loss: False     # 不使用affiliation loss
use_triplet_loss: False   # 同时使用对比+三元组损失
```

#### **RSITMD + GeoRSCLIP** (`configs/Retrieval_rsitmd_geo.yaml`)

```yaml
model: 'geo'              # 使用GeoRSCLIP预训练模型
optimizer:
  lr: 4e-6                # 更小的学习率（已预训练在遥感数据）
schedular:
  epochs: 80              # 更多轮次
```

**关键差异**：
- GeoRSCLIP预训练于RS5M（500万遥感图像-文本对）
- 需要更小学习率和更多epoch来适应下游任务

---

## 四、代码执行流程图

```
┌──────────────────┐
│   run.py         │  启动脚本
│  - 解析任务      │
│  - 设置分布式    │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│  Retrieval.py    │  主程序
│  - main()        │
└────────┬─────────┘
         │
         ├─→ 1. 加载配置
         │    └─ YAML → config dict
         │
         ├─→ 2. 创建模型
         │    ├─ HarMA(config)
         │    │   ├─ open_clip.create_model_and_transforms("ViT-B/32")
         │    │   │   ├─ VisionTransformer (12 layers + MMadapter)
         │    │   │   └─ TextTransformer (12 layers + MMadapter + BiShareAdapter)
         │    │   └─ load GeoRSCLIP checkpoint (if model=='geo')
         │    └─ set_trainable() → 冻结backbone，解冻Adapter
         │
         ├─→ 3. 加载数据
         │    ├─ create_dataset('re', config)
         │    │   ├─ re_train_dataset(train_file, transform)
         │    │   └─ re_eval_dataset(test_file, transform)
         │    └─ create_loader() → DataLoader with DistributedSampler
         │
         ├─→ 4. 优化器
         │    ├─ create_optimizer() → AdamW
         │    └─ create_scheduler() → Linear warmup
         │
         └─→ 5. 训练循环
              for epoch in range(max_epoch):
                  ├─ train()
                  │   for batch in train_loader:
                  │       ├─ image, text, idx, label = batch
                  │       ├─ image_emb = model.get_vis_emb(image)
                  │       ├─ text_emb = model.get_txt_emb(text)
                  │       ├─ loss_contr = model.get_contr_loss(...)
                  │       ├─ loss_triplet = model.weighted_triplet_loss(...)
                  │       ├─ loss = loss_contr + loss_triplet
                  │       └─ optimizer.step()
                  │
                  ├─ evaluation()
                  │   ├─ 提取图像特征: [N_img, 512]
                  │   ├─ 提取文本特征: [N_txt, 512]
                  │   ├─ sims = img_feat @ txt_feat.t()
                  │   └─ return scores_i2t, scores_t2i
                  │
                  ├─ itm_eval()
                  │   └─ 计算 R@1, R@5, R@10
                  │
                  └─ save_checkpoint() if best
```

---

## 五、关键技术点总结

### 5.1 参数高效微调（PEFT）

| 方法 | 可训练参数 | 性能 |
|------|-----------|------|
| Full Fine-tuning | 100% (~400M) | Baseline |
| HarMA (Adapter) | 1-5% (~5M) | **超越Full Fine-tuning** |

**实现方式**：
1. 冻结CLIP的Vision Encoder和Text Encoder
2. 仅训练插入的Adapter模块（BiShareAdapter + MMadapter）
3. 训练gate参数和温度参数

### 5.2 层次化适配器设计

```
层级1: BiShareAdapter (跨模态共享)
       ↓
层级2: MMadapter (模态特定)
       ├─ 图像：独立MMadapter
       └─ 文本：引用BiShareAdapter的MMadapter
       
关键：文本MMadapter通过BiShareAdapter间接与图像交互
```

### 5.3 多损失函数策略

1. **对比学习损失**：全局对齐（所有样本对）
2. **加权三元组损失**：局部优化（难负样本）
3. **Focal Weight**：自动调节难样本权重

### 5.4 数据增强

```python
train_transform = [
    RandomResizedCrop(224, scale=(0.5, 1.0)),
    RandomHorizontalFlip(),
    RandomAugment(2, 7),  # 从10种操作中随机选2种
    ToTensor(),
    Normalize(CLIP_mean, CLIP_std)
]
```

---

## 六、模型输入输出规范

### 输入
```python
# 训练阶段
image: torch.Tensor        # [B, 3, 224, 224]
text: List[str]            # [B] 个字符串
text_ids: torch.Tensor     # [B, 77] (CLIP tokenizer)

# 测试阶段
image: torch.Tensor        # [B, 3, 224, 224]
text_ids: torch.Tensor     # [N_captions, 77]
```

### 输出
```python
# 训练阶段
loss_contr: torch.Tensor   # 标量
loss_triplet: torch.Tensor # 标量

# 测试阶段
image_emb: torch.Tensor    # [N_img, 512]
text_emb: torch.Tensor     # [N_txt, 512]
sims_matrix: torch.Tensor  # [N_img, N_txt]
```

---

## 七、实验配置对比

| 配置 | RSITMD+ViT | RSITMD+Geo | RSICD+ViT | RSICD+Geo |
|------|------------|------------|-----------|-----------|
| 预训练模型 | OpenAI CLIP | GeoRSCLIP | OpenAI CLIP | GeoRSCLIP |
| 学习率 | 4e-4 | **4e-6** | 4e-4 | **4e-6** |
| Epoch | 50 | **80** | 10 | **80** |
| Batch Size | 214 | 214 | 214 | 214 |
| Warmup | 10% | 10% | 10% | 10% |

**关键发现**：
- GeoRSCLIP需要更小学习率（已在遥感数据上预训练）
- RSITMD数据集较小，需要更多epoch
- RSICD数据集较大，10个epoch即可

---

## 八、推荐的修改点（为创新方案准备）

### 8.1 模型架构层面
- `models/mga.py`: 修改Adapter结构（改变维度、注意力机制）
- `open_clip/model.py`: 修改Vision/Text Tower的集成方式
- `models/harma.py`: 添加新的损失函数

### 8.2 数据处理层面
- `dataset/re_dataset.py`: 添加新的数据增强策略
- `scripts/build_rs_vocabulary.py`: 扩展词表构建方法

### 8.3 训练策略层面
- `Retrieval.py`: 修改训练循环（如添加对比学习变体）
- `optim.py`: 尝试新的优化器
- `scheduler.py`: 设计新的学习率策略

### 8.4 评估层面
- `Retrieval.py - itm_eval()`: 添加新的评估指标（如mAP）

---

## 九、常见操作命令

### 训练
```bash
# RSITMD + GeoRSCLIP
python run.py --task 'itr_rsitmd_geo' --dist "f2" \
    --config 'configs/Retrieval_rsitmd_geo.yaml' \
    --output_dir './checkpoints/HARMA/full_rsitmd_geo'

# RSICD + ViT-B/32
python run.py --task 'itr_rsicd_vit' --dist "f2" \
    --config 'configs/Retrieval_rsicd_vit.yaml' \
    --output_dir './checkpoints/HARMA/full_rsicd_vit'
```

### 测试
```bash
python run.py --task 'itr_rsitmd_geo' --dist "f2" \
    --config 'configs/Retrieval_rsitmd_geo.yaml' \
    --output_dir './checkpoints/test' \
    --checkpoint './checkpoints/HARMA/full_rsitmd_geo/checkpoint_best.pth' \
    --evaluate
```

### 查看日志
```bash
# 日志文件：output_dir/YYYY-MM-DD_HH-MM-SS-log.txt
tail -f checkpoints/HARMA/full_rsitmd_geo/2025-01-18_10-30-00-log.txt
```

---

## 十、代码依赖关系图

```
run.py
  └─→ Retrieval.py
       ├─→ models.model_retrieval.HarMA
       │    ├─→ models.harma.HarMABase
       │    │    ├─→ get_contr_loss()
       │    │    ├─→ weighted_triplet_loss()
       │    │    └─→ get_affil_loss()
       │    └─→ open_clip.create_model_and_transforms()
       │         ├─→ open_clip.model.CLIP
       │         │    ├─→ VisionTransformer (+ MMadapter)
       │         │    └─→ TextTransformer (+ MMadapter + BiShareAdapter)
       │         └─→ open_clip.model.{BiShareAdapter, MMadapter}
       │
       ├─→ dataset.create_dataset()
       │    ├─→ re_train_dataset
       │    └─→ re_eval_dataset
       │
       ├─→ optim.create_optimizer()
       └─→ scheduler.create_scheduler()
```

---

## 十一、待补充信息

以下信息需要根据你的实验环境补充：
- [ ] 预训练模型路径：`./pretrain/RS5M_ViT-B-32_RET-2.pt`
- [ ] 图像数据路径：`../PIR/rsitmd/` 和 `../PIR/rsicd/`
- [ ] GPU配置：修改 `run.py - get_dist_launch()` 中的GPU编号
- [ ] Python环境路径：修改 `run.py` 中的 `/root/miniconda3/bin/python`

---

## 总结

HarMA的核心创新在于：
1. **参数高效**：通过层次化Adapter实现1-5%参数微调
2. **跨模态交互**：BiShareAdapter实现图像-文本的隐式对齐
3. **多损失协同**：对比学习+加权三元组，全局+局部优化
4. **遥感领域优化**：支持GeoRSCLIP预训练模型

代码结构清晰，模块化程度高，易于扩展和修改。你可以基于此框架实现自己的创新方案。

---

**最后更新时间**：2025-10-18
**文档版本**：v1.0

