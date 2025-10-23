# CLIP模型手动下载指南

## 需要下载的文件

**仓库**: `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`

### 文件列表：
1. **open_clip_pytorch_model.bin** (约 605 MB) - 主模型权重文件
2. **open_clip_config.json** (约 1 KB) - 配置文件

---

## 下载链接（选一个可用的）

### 方案1: HuggingFace 官网（国外）
```
https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_pytorch_model.bin
https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_config.json
```

### 方案2: HF国内镜像站
```
https://hf-mirror.com/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_pytorch_model.bin
https://hf-mirror.com/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_config.json
```

### 方案3: ModelScope（国内）
如果以上都不行，可以尝试在 https://modelscope.cn 搜索 CLIP 模型

---

## 上传到服务器的目标路径

下载完成后，需要放到以下路径：

```bash
/data2/huggingface_cache/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/snapshots/1a25a446712ba5ee05982a381eed697ef9b435cf/
```

### 目录结构应该是：
```
/data2/huggingface_cache/
└── models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/
    ├── blobs/
    │   └── <hash>.blob (模型文件的blob存储)
    ├── refs/
    │   └── main
    └── snapshots/
        └── 1a25a446712ba5ee05982a381eed697ef9b435cf/
            ├── open_clip_pytorch_model.bin  ← 放这里
            └── open_clip_config.json        ← 放这里
```

---

## 上传步骤

### 1. 在本地下载文件
使用浏览器或下载工具下载上述两个文件

### 2. 创建目标目录（在服务器上）
```bash
mkdir -p /data2/huggingface_cache/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/snapshots/1a25a446712ba5ee05982a381eed697ef9b435cf/
```

### 3. 上传文件到服务器
使用 `scp`、`rsync`、`FileZilla` 或其他文件传输工具上传到目标路径

### 4. 验证文件
```bash
ls -lh /data2/huggingface_cache/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/snapshots/1a25a446712ba5ee05982a381eed697ef9b435cf/
```

应该看到：
- open_clip_pytorch_model.bin (约 605MB)
- open_clip_config.json (约 1KB)

### 5. 设置环境变量后运行训练
```bash
export HF_HOME=/data2/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/data2/huggingface_cache

# 然后运行训练命令
cd /data2/ls/SemRL-RS-Retrieval
conda activate /data/env/semrl
python run.py --task 'itr_rsicd_vit' --dist "f023" --config 'configs/Retrieval_rsicd_vit_aug80pct.yaml' --output_dir './checkpoints/HARMA/rsicd_vit_aug80pct'
```

---

## 备注
- 文件下载大约需要 605MB 空间
- 建议使用断点续传工具（如 wget、aria2c）
- 上传到服务器建议使用 rsync 以支持断点续传

