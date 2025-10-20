# HarMA 快速参考指南

> 这是一个快速查找手册，涵盖常用命令、关键配置、代码位置和调试技巧。

---

## 📑 目录

- [常用命令](#常用命令)
- [关键文件速查](#关键文件速查)
- [配置参数](#配置参数)
- [模型架构速查](#模型架构速查)
- [数据格式](#数据格式)
- [调试技巧](#调试技巧)
- [性能指标](#性能指标)
- [常见错误](#常见错误)

---

## 常用命令

### 训练模型

```bash
# 1. RSITMD + ViT-B/32 (OpenAI CLIP)
python run.py --task 'itr_rsitmd_vit' --dist "f2" \
    --config 'configs/Retrieval_rsitmd_vit.yaml' \
    --output_dir './checkpoints/HARMA/rsitmd_vit'

# 2. RSITMD + GeoRSCLIP (遥感预训练)
python run.py --task 'itr_rsitmd_geo' --dist "f2" \
    --config 'configs/Retrieval_rsitmd_geo.yaml' \
    --output_dir './checkpoints/HARMA/rsitmd_geo'

# 3. RSICD + ViT-B/32
python run.py --task 'itr_rsicd_vit' --dist "f2" \
    --config 'configs/Retrieval_rsicd_vit.yaml' \
    --output_dir './checkpoints/HARMA/rsicd_vit'

# 4. RSICD + GeoRSCLIP
python run.py --task 'itr_rsicd_geo' --dist "f2" \
    --config 'configs/Retrieval_rsicd_geo.yaml' \
    --output_dir './checkpoints/HARMA/rsicd_geo'
```

### 测试模型

```bash
# 测试 RSITMD + GeoRSCLIP
python run.py --task 'itr_rsitmd_geo' --dist "f2" \
    --config 'configs/Retrieval_rsitmd_geo.yaml' \
    --output_dir './checkpoints/test' \
    --checkpoint './checkpoints/HARMA/rsitmd_geo/checkpoint_best.pth' \
    --evaluate

# 测试 RSICD + ViT-B/32
python run.py --task 'itr_rsicd_vit' --dist "f2" \
    --config 'configs/Retrieval_rsicd_vit.yaml' \
    --output_dir './checkpoints/test' \
    --checkpoint './checkpoints/HARMA/rsicd_vit/checkpoint_best.pth' \
    --evaluate
```

### 查看日志

```bash
# 实时查看训练日志
tail -f checkpoints/HARMA/rsitmd_geo/2025-*-log.txt

# 查看完整日志
cat checkpoints/HARMA/rsitmd_geo/2025-*-log.txt

# 提取关键指标
grep "txt_r1\|img_r1\|r_mean" checkpoints/HARMA/rsitmd_geo/*.txt
```

### GPU配置

```bash
# 查看GPU状态
nvidia-smi

# 查看GPU占用
watch -n 1 nvidia-smi

# 设置可见GPU（修改 run.py）
# CUDA_VISIBLE_DEVICES=0,1 (使用GPU 0和1)
# CUDA_VISIBLE_DEVICES=2,3 (使用GPU 2和3)
```

---

## 关键文件速查

### 📂 核心代码文件

| 文件路径 | 功能 | 关键类/函数 |
|---------|------|-----------|
| `Retrieval.py` | 训练/测试主程序 | `main()`, `train()`, `evaluation()`, `itm_eval()` |
| `models/model_retrieval.py` | 检索模型定义 | `HarMA` (forward, get_vis_emb, get_txt_emb) |
| `models/harma.py` | 基础模型+损失函数 | `HarMABase`, `get_contr_loss()`, `weighted_triplet_loss()` |
| `models/mga.py` | 适配器模块 | `BiShareAdapter`, `MMadapter` |
| `open_clip/model.py` | CLIP模型+Adapter集成 | `CLIP` (encode_image, encode_text) |
| `dataset/re_dataset.py` | 数据集加载 | `re_train_dataset`, `re_eval_dataset` |
| `run.py` | 启动脚本 | `get_dist_launch()`, `run()` |

### 📝 配置文件

| 文件 | 用途 |
|------|------|
| `configs/Retrieval_rsitmd_vit.yaml` | RSITMD + ViT-B/32 配置 |
| `configs/Retrieval_rsitmd_geo.yaml` | RSITMD + GeoRSCLIP 配置 |
| `configs/Retrieval_rsicd_vit.yaml` | RSICD + ViT-B/32 配置 |
| `configs/Retrieval_rsicd_geo.yaml` | RSICD + GeoRSCLIP 配置 |
| `configs/config_bert.json` | BERT文本编码器配置 |
| `configs/config_swinT_224.json` | Swin-T视觉编码器配置 |

### 📊 数据文件

| 文件 | 内容 |
|------|------|
| `data/finetune/rsitmd_train.json` | RSITMD训练集 (4000+ samples) |
| `data/finetune/rsitmd_val.json` | RSITMD验证集 |
| `data/finetune/rsitmd_test.json` | RSITMD测试集 |
| `data/finetune/rsicd_train.json` | RSICD训练集 (10000+ samples) |
| `data/finetune/rsicd_val.json` | RSICD验证集 |
| `data/finetune/rsicd_test.json` | RSICD测试集 |

---

## 配置参数

### 关键参数对照表

| 参数名 | ViT配置 | Geo配置 | 说明 |
|--------|---------|---------|------|
| `model` | `'vit'` | `'geo'` | 预训练模型类型 |
| `lr` | `4e-4` | `4e-6` | 学习率（Geo更小） |
| `epochs` | `50` | `80` | 训练轮数 |
| `batch_size_train` | `214` | `214` | 每GPU批次大小 |
| `embed_dim` | `512` | `512` | 特征维度（固定） |
| `temp1` | `0.07` | `0.07` | 对比学习温度 |
| `weight_decay` | `0.04` | `0.04` | 权重衰减 |
| `num_warmup_steps` | `0.1` | `0.1` | 预热步数比例 |

### 损失函数配置

```yaml
# 同时使用对比学习+三元组损失（推荐）
use_affil_loss: False
use_triplet_loss: False

# 只使用三元组损失
use_affil_loss: False
use_triplet_loss: True

# 使用中心对齐损失（实验性）
use_affil_loss: True
use_triplet_loss: False
center_factor: 1
```

### 数据路径配置

```yaml
# 修改为你的数据路径
image_root: '/path/to/your/rsitmd/'  # 图像根目录
train_file: ['data/finetune/rsitmd_train.json']
val_file: 'data/finetune/rsitmd_val.json'
test_file: 'data/finetune/rsitmd_test.json'
```

---

## 模型架构速查

### 网络结构概览

```
输入: Image [B,3,224,224] + Text (strings)
  ↓
┌─────────────────────────────────────┐
│ Vision Encoder (12 layers)          │
│  • ViT-B/32 or GeoRSCLIP            │
│  • 每层插入 MMadapter (独立)         │
│  • 输出: [B, 512]                   │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ Text Encoder (12 layers)             │
│  • CLIP Text Transformer            │
│  • 每层插入 MMadapter                │
│  • 每层有 BiShareAdapter (共享)      │
│  • 输出: [B, 512]                   │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ Loss Functions                      │
│  • Contrastive Loss (InfoNCE)      │
│  • Weighted Triplet Loss (Focal)   │
└─────────────────────────────────────┘
```

### Adapter参数量

| 模块 | 参数量 | 位置 |
|------|--------|------|
| BiShareAdapter | ~140K/个 | 文本塔12层，共享给MMadapter |
| MMadapter (Vision) | ~200K/个 | 视觉塔12层 |
| MMadapter (Text) | ~150K/个 | 文本塔12层 |
| **总计** | **~3.5M** | **占CLIP总参数1-2%** |

### 关键超参数

```python
# Adapter降维比例
hidden_size_vision = 768 → 128  # 约6倍压缩
hidden_size_text = 512 → 128    # 约4倍压缩

# 注意力头数
num_heads = 8  # 所有Adapter使用相同头数

# Gate初始值
gate_init = 0.6  # 可学习参数

# Triplet Loss
margin = 0.2      # 间隔阈值
gamma = 2.0       # Focal权重指数
```

---

## 数据格式

### 训练数据格式 (JSON)

```json
[
    {
        "caption": "The river banks decorated with trees...",
        "image": "train/bridge_942.tif",
        "image_id": 539,
        "label_name": "bridge",
        "label": 0
    },
    ...
]
```

### 测试数据格式

```json
[
    {
        "image": "test/airport_10.tif",
        "caption": [
            "An airport with two planes on the runway.",
            "Two aircraft are parked near the terminal.",
            "The airport has multiple runways and buildings.",
            "Several planes and structures in the airport.",
            "A large airport with aircraft and facilities."
        ]
    },
    ...
]
```

### DataLoader输出

```python
# 训练时
image: torch.Tensor  # [B, 3, 224, 224]
text: List[str]      # [B] 个字符串
img_id: torch.Tensor # [B] 图像ID
label: torch.Tensor  # [B] 场景标签

# 测试时
image: torch.Tensor  # [B, 3, 224, 224]
img_idx: torch.Tensor # [B] 图像索引
```

---

## 调试技巧

### 1. 打印模型结构

```python
# 在 Retrieval.py 的 main() 中添加
print("="*50)
print("Model Structure:")
print(model)
print("="*50)

# 统计参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
```

### 2. 检查梯度流

```python
# 在 train() 函数的 loss.backward() 后添加
for name, param in model.named_parameters():
    if param.grad is None and param.requires_grad:
        print(f"⚠️  No gradient for: {name}")
```

### 3. 监控损失变化

```python
# 在 train() 中添加
if i % 10 == 0:  # 每10个batch打印一次
    print(f"Batch {i}: L_contr={loss_contr.item():.4f}, "
          f"L_triplet={loss_triplet.item():.4f}")
```

### 4. 可视化特征分布

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# 在 evaluation() 后添加
def visualize_embeddings(image_embeds, text_embeds, save_path):
    """
    使用t-SNE可视化特征分布
    """
    from sklearn.manifold import TSNE
    
    # 合并特征
    all_embeds = torch.cat([image_embeds[:100], text_embeds[:100]], dim=0)
    all_embeds = all_embeds.cpu().numpy()
    
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    embeds_2d = tsne.fit_transform(all_embeds)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    plt.scatter(embeds_2d[:100, 0], embeds_2d[:100, 1], c='blue', label='Image', alpha=0.6)
    plt.scatter(embeds_2d[100:, 0], embeds_2d[100:, 1], c='red', label='Text', alpha=0.6)
    plt.legend()
    plt.title('Image-Text Embedding Space')
    plt.savefig(save_path)
    plt.close()
```

### 5. 分析失败案例

```python
# 在 itm_eval() 后添加
def analyze_failures(scores_i2t, img2txt, top_k=10):
    """
    找出检索失败的案例
    """
    failures = []
    
    for img_idx, scores in enumerate(scores_i2t):
        gt_txt_ids = img2txt[img_idx]
        sorted_indices = np.argsort(scores)[::-1]
        
        # 找到第一个正样本的排名
        rank = min([np.where(sorted_indices == gt)[0][0] 
                    for gt in gt_txt_ids])
        
        if rank >= 10:  # R@10失败
            failures.append({
                'img_idx': img_idx,
                'rank': rank,
                'top_predictions': sorted_indices[:5].tolist(),
                'ground_truth': gt_txt_ids
            })
    
    # 保存失败案例
    import json
    with open('failures.json', 'w') as f:
        json.dump(failures[:top_k], f, indent=2)
    
    return failures
```

---

## 性能指标

### 评估指标说明

| 指标 | 全称 | 含义 |
|------|------|------|
| `txt_r1` | Text Retrieval @ 1 | 图像→文本检索，Top-1准确率 |
| `txt_r5` | Text Retrieval @ 5 | 图像→文本检索，Top-5准确率 |
| `txt_r10` | Text Retrieval @ 10 | 图像→文本检索，Top-10准确率 |
| `img_r1` | Image Retrieval @ 1 | 文本→图像检索，Top-1准确率 |
| `img_r5` | Image Retrieval @ 5 | 文本→图像检索，Top-5准确率 |
| `img_r10` | Image Retrieval @ 10 | 文本→图像检索，Top-10准确率 |
| `r_mean` | Mean Recall | 所有指标的平均值（用于选择最佳模型） |

### 预期性能范围

**RSITMD 数据集**:
```
ViT-B/32 (baseline):
  txt_r1: ~50-55%  img_r1: ~45-50%  r_mean: ~65-70%

GeoRSCLIP (预训练):
  txt_r1: ~65-70%  img_r1: ~60-65%  r_mean: ~78-82%

HarMA (ViT):
  txt_r1: ~58-63%  img_r1: ~53-58%  r_mean: ~70-75%

HarMA (Geo):
  txt_r1: ~72-77%  img_r1: ~68-73%  r_mean: ~82-87% ⭐
```

**RSICD 数据集** (更大):
```
HarMA (Geo):
  txt_r1: ~75-80%  img_r1: ~72-77%  r_mean: ~85-90%
```

### 训练时间估算

| 配置 | 数据集 | GPU | Epoch时间 | 总时间 |
|------|--------|-----|----------|--------|
| HarMA(Geo) | RSITMD | 2×V100 | ~15 min | 80 epochs ≈ 20h |
| HarMA(ViT) | RSITMD | 2×V100 | ~10 min | 50 epochs ≈ 8h |
| HarMA(Geo) | RSICD | 2×V100 | ~35 min | 80 epochs ≈ 47h |

---

## 常见错误

### ❌ 错误1: CUDA Out of Memory

```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**解决方案**:
```yaml
# 减小batch size
batch_size_train: 128  # 从214降到128

# 或使用梯度累积（修改 Retrieval.py）
accumulation_steps = 2
if (i + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### ❌ 错误2: 找不到预训练模型

```
FileNotFoundError: [Errno 2] No such file or directory: './pretrain/RS5M_ViT-B-32_RET-2.pt'
```

**解决方案**:
```bash
# 下载GeoRSCLIP预训练模型
mkdir -p pretrain
cd pretrain
wget https://huggingface.co/Zilun/GeoRSCLIP/resolve/main/RS5M_ViT-B-32_RET-2.pt

# 或修改配置使用ViT-B/32
# 在 yaml 中设置 model: 'vit'
```

### ❌ 错误3: 数据路径错误

```
FileNotFoundError: [Errno 2] No such file or directory: '../PIR/rsitmd/train/bridge_942.tif'
```

**解决方案**:
```yaml
# 修改 yaml 配置中的 image_root
image_root: '/your/actual/path/to/rsitmd/'

# 确保目录结构正确
rsitmd/
  ├── train/
  │   ├── bridge_942.tif
  │   └── ...
  └── test/
      └── ...
```

### ❌ 错误4: 分布式训练失败

```
RuntimeError: Address already in use
```

**解决方案**:
```python
# 修改 run.py 中的 master_port
# 从 9999 改为其他端口
return "... --master_port 9998 ..."
```

### ❌ 错误5: 梯度为None

```
Warning: No gradient for model.visual.xxx.weight
```

**解决方案**:
```python
# 检查 set_trainable() 函数
# 确保Adapter模块被正确解冻

# 或禁用 find_unused_parameters（在 Retrieval.py）
model = torch.nn.parallel.DistributedDataParallel(
    model, device_ids=[args.gpu], 
    find_unused_parameters=False  # 改为False
)
```

### ❌ 错误6: 评估指标为0

```
{'txt_r1': 0.0, 'txt_r5': 0.0, ...}
```

**可能原因**:
1. 模型未加载或权重错误
2. 特征未归一化
3. 数据集映射关系错误

**检查**:
```python
# 打印相似度矩阵范围
print(f"Similarity range: {sims_matrix.min():.3f} ~ {sims_matrix.max():.3f}")
# 正常应该在 [-1, 1] 或 [0, 1] 范围

# 检查特征norm
print(f"Image emb norm: {torch.norm(image_embeds, dim=1).mean():.3f}")
print(f"Text emb norm: {torch.norm(text_embeds, dim=1).mean():.3f}")
# L2归一化后应该接近1.0
```

---

## 快速实验清单

### ✅ 第一次运行

- [ ] 检查Python环境 (`python --version` >= 3.8)
- [ ] 安装依赖 (`pip install -r requirements.txt`)
- [ ] 下载预训练模型到 `pretrain/`
- [ ] 准备数据集（RSITMD或RSICD）
- [ ] 修改配置文件中的 `image_root` 路径
- [ ] 运行小规模测试（如只训练2个epoch）

### ✅ 调试checklist

- [ ] 打印模型结构，确认Adapter被正确添加
- [ ] 检查可训练参数比例（应该1-5%）
- [ ] 查看第一个batch的损失值（不应为nan或inf）
- [ ] 监控GPU显存占用
- [ ] 检查学习率是否合理（观察前几个step）
- [ ] 保存训练日志到文件

### ✅ 修改模型时

- [ ] 备份原始代码
- [ ] 在小数据集上测试新改动
- [ ] 对比修改前后的性能
- [ ] 记录超参数变化
- [ ] 保存实验结果和配置

---

## 有用的代码片段

### 1. 加载checkpoint进行推理

```python
import torch
from models.model_retrieval import HarMA
from open_clip import tokenizer
from PIL import Image
from torchvision import transforms

# 加载模型
config = {...}  # 从yaml加载
model = HarMA(config)
checkpoint = torch.load('checkpoint_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()
model.cuda()

# 准备数据
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

# 推理
image = Image.open('test.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0).cuda()
text = ["An airport with planes"]
text_tokens = tokenizer.tokenize(text).cuda()

with torch.no_grad():
    img_emb = model.get_vis_emb(image_tensor)
    txt_emb = model.get_txt_emb(text_tokens)
    similarity = (img_emb @ txt_emb.T).item()
    print(f"Similarity: {similarity:.4f}")
```

### 2. 批量测试不同超参数

```bash
#!/bin/bash
# test_hyperparams.sh

learning_rates=(1e-4 5e-5 1e-5)
batch_sizes=(128 256)

for lr in "${learning_rates[@]}"; do
    for bs in "${batch_sizes[@]}"; do
        echo "Testing lr=$lr, bs=$bs"
        
        # 修改yaml配置
        sed -i "s/lr: .*/lr: $lr/" configs/test.yaml
        sed -i "s/batch_size_train: .*/batch_size_train: $bs/" configs/test.yaml
        
        # 运行训练
        python run.py --task 'itr_rsitmd_vit' --dist "f2" \
            --config 'configs/test.yaml' \
            --output_dir "./checkpoints/exp_lr${lr}_bs${bs}"
    done
done
```

### 3. 计算参数量

```python
def count_parameters(model, verbose=True):
    """统计模型参数"""
    total = 0
    trainable = 0
    frozen = 0
    
    param_dict = {}
    for name, param in model.named_parameters():
        num = param.numel()
        total += num
        
        if param.requires_grad:
            trainable += num
            module = name.split('.')[0]
            param_dict[module] = param_dict.get(module, 0) + num
        else:
            frozen += num
    
    if verbose:
        print(f"{'='*60}")
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,} ({trainable/total*100:.2f}%)")
        print(f"Frozen parameters: {frozen:,} ({frozen/total*100:.2f}%)")
        print(f"{'='*60}")
        print("Trainable modules:")
        for module, num in sorted(param_dict.items(), key=lambda x: -x[1]):
            print(f"  {module}: {num:,} ({num/trainable*100:.2f}%)")
    
    return total, trainable, frozen
```

---

## 相关资源

### 📚 论文和文档

- **HarMA论文**: [arXiv:2404.18253](https://arxiv.org/abs/2404.18253)
- **CLIP论文**: [Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- **GeoRSCLIP**: [Hugging Face Model](https://huggingface.co/Zilun/GeoRSCLIP)

### 🗂️ 数据集

- **RSITMD**: [GitHub - AMFMN](https://github.com/xiaoyuan1996/AMFMN/tree/master/RSITMD)
- **RSICD**: [GitHub - RSICD](https://github.com/201528014227051/RSICD_optimal)

### 🔧 工具

- **OpenCLIP**: [GitHub](https://github.com/mlfoundations/open_clip)
- **Timm**: [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)

---

## 联系方式

- **项目Issue**: [GitHub Issues](https://github.com/seekerhuang/HarMA/issues)
- **论文作者**: Tengjun Huang

---

**最后更新**: 2025-10-18  
**配合阅读**: `CODE_ANALYSIS.md`, `ARCHITECTURE_DIAGRAM.md`

---

> 💡 **提示**: 将此文件加入书签，需要时快速查找！



