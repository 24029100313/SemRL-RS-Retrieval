#!/bin/bash
# 批量测试不同回译替换比例的影响
# 用法: bash scripts/run_augmentation_sweep.sh [dataset] [ratios]
# 例如: bash scripts/run_augmentation_sweep.sh rsitmd "30 50 80"

DATASET=${1:-"rsitmd"}  # 默认rsitmd
RATIOS=${2:-"30 50 80"}  # 默认测试30%, 50%, 80%

cd /data2/ls/SemRL-RS-Retrieval

echo "=================================================="
echo "开始批量实验: ${DATASET} 数据集"
echo "测试替换比例: ${RATIOS}"
echo "=================================================="

for ratio in ${RATIOS}; do
    echo ""
    echo "==================== ${ratio}% 替换 ===================="
    
    # 创建配置文件
    CONFIG_FILE="configs/Retrieval_${DATASET}_geo_aug${ratio}pct.yaml"
    cp "configs/Retrieval_${DATASET}_geo.yaml" "${CONFIG_FILE}"
    
    # 修改训练文件路径
    sed -i "s|train_file:.*|train_file: ['data/augmented/${DATASET}_train_backtrans_${ratio}pct.json']  # ${ratio}% backtranslation|" "${CONFIG_FILE}"
    
    echo "✓ 配置文件已创建: ${CONFIG_FILE}"
    
    # 启动训练
    OUTPUT_DIR="./checkpoints/HARMA/${DATASET}_geo_aug${ratio}pct"
    LOG_FILE="training_${DATASET}_aug${ratio}pct.log"
    
    echo "✓ 开始训练: ${DATASET} + ${ratio}% 增强"
    echo "  输出目录: ${OUTPUT_DIR}"
    echo "  日志文件: ${LOG_FILE}"
    
    conda activate /data/env/semrl
    python run.py \
        --task "itr_${DATASET}_geo" \
        --dist "f023" \
        --config "${CONFIG_FILE}" \
        --output_dir "${OUTPUT_DIR}" \
        2>&1 | tee "${LOG_FILE}" &
    
    PID=$!
    echo "  进程ID: ${PID}"
    echo ""
    
    # 等待一会儿再启动下一个（避免同时启动太多）
    sleep 5
done

echo "=================================================="
echo "所有实验已启动！"
echo "使用 'ps aux | grep python.*Retrieval' 查看进程"
echo "使用 'tail -f training_*.log' 查看日志"
echo "=================================================="

