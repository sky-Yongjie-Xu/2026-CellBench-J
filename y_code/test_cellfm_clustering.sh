#!/bin/bash
# Nicheformer 提取所有数据集 Embedding 脚本

# 设置环境变量以解决MindSpore的protobuf版本冲突
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# 设置参数
gpu_id=0
seed=${1:-41}  # 默认随机种子

# 定义所有数据集列表（来自 dataset.py）
# 单细胞转录组数据

# 单细胞扰动数据
# 单细胞多批次数据
BATCH_DATASETS=(
   uc_imm1
   
)


# 合并所有数据集
ALL_DATASETS=(
    "${BATCH_DATASETS[@]}"
)

echo "=========================================="
echo "开始提取 Nicheformer 模型的所有数据集 Embedding"
echo "总共 ${#ALL_DATASETS[@]} 个数据集"
echo "随机种子: $seed"
echo "GPU ID: $gpu_id"
echo "=========================================="
echo ""

# 记录开始时间
start_time=$(date +%s)

# 循环处理所有数据集
for i in "${!ALL_DATASETS[@]}"; do
    dataset_name="${ALL_DATASETS[$i]}"
    dataset_num=$((i + 1))
    
    echo "=========================================="
    echo "[$dataset_num/${#ALL_DATASETS[@]}] 正在处理数据集: $dataset_name"
    echo "=========================================="
    
    # 运行 embedding 提取任务
    # 注意：任务名称可能是 "embedding" 或 "clustering"，根据实际情况调整
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
        --cfgs ./configs/nicheformer.yaml \
        --task correction\
        --dataset_name $dataset_name \
        --seed $seed \
        --gpu $gpu_id
    
    # 检查是否成功
    if [ $? -eq 0 ]; then
        echo "✓ 数据集 $dataset_name 处理成功"
    else
        echo "✗ 数据集 $dataset_name 处理失败，继续下一个..."
    fi
    
    echo ""
done

# 记录结束时间并计算耗时
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo "=========================================="
echo "所有数据集 Embedding 提取完成！"
echo "总耗时: ${hours}小时 ${minutes}分钟 ${seconds}秒"
echo "=========================================="

