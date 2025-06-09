# Multi-GPU Training Guide for VLA Model

## Overview

This guide explains how to run multi-GPU training for the Vision-Language-Action (VLA) model using PyTorch DistributedDataParallel (DDP).

## Files Created

1. **`main_train_ddp.py`** - Main multi-GPU training script
2. **`training/trainer_ddp.py`** - Distributed trainer class
3. **`scripts/run_ddp_training.sh`** - Convenience script for launching training

## Key Features

### Multi-GPU Support
- **DistributedDataParallel (DDP)**: Efficient data parallelism across multiple GPUs
- **Automatic batch size splitting**: Total batch size is divided among available GPUs
- **Gradient synchronization**: Gradients are automatically synchronized across all processes
- **Memory efficiency**: Each GPU only loads its portion of the data

### Training Features
- **Mixed precision training**: Support for FP16 and BFP16
- **Gradient clipping**: Prevents gradient explosion
- **Learning rate scheduling**: Compatible with various LR schedulers
- **Checkpoint resumption**: Can resume training from saved checkpoints
- **WandB integration**: Logging only on the main process to avoid conflicts

### Data Loading
- **DistributedSampler**: Ensures each GPU processes different data samples
- **Epoch shuffling**: Proper data shuffling across epochs in distributed setting
- **Load balancing**: Data is evenly distributed across all GPUs

## Requirements

- **Multiple GPUs**: At least 2 CUDA-compatible GPUs
- **PyTorch with CUDA support**
- **NCCL backend**: For GPU communication (usually included with PyTorch)

## 使用方法

### 方法1: 使用便捷脚本 (推荐)

```bash
# 给脚本执行权限
chmod +x scripts/train_multi_gpu.sh

# 使用所有可用GPU训练
./scripts/train_multi_gpu.sh

# 使用指定数量的GPU (例如2个GPU)
./scripts/train_multi_gpu.sh 2

# 指定配置文件和实验名称
./scripts/train_multi_gpu.sh 4 configs/vla_config.yaml my_experiment

# 添加额外参数
./scripts/train_multi_gpu.sh 2 configs/vla_config.yaml my_exp --batch_size 4 --epochs 20
```

### 方法2: 直接运行Python脚本

```bash
# 使用4个GPU训练
python main_train_ddp.py \
    --config_path configs/vla_config.yaml \
    --world_size 4 \
    --experiment_name paligemma_vla_multi_gpu \
    --keep_effective_batch_size \
    --use_wandb

# 从checkpoint恢复训练
python main_train_ddp.py \
    --config_path configs/vla_config.yaml \
    --world_size 4 \
    --resume_checkpoint experiments/paligemma_vla_finetune_20241201_120000/checkpoints/best_model.pth \
    --keep_effective_batch_size
```

## 重要参数说明

### 批大小调整

- **`--keep_effective_batch_size`**: 保持有效批大小不变
  - 开启时: 每个GPU的批大小 = 配置中的批大小 / GPU数量
  - 关闭时: 每个GPU使用配置中的完整批大小，有效批大小 = 批大小 × GPU数量

例如：
- 配置文件中 `batch_size: 8`，使用4个GPU
- 开启 `--keep_effective_batch_size`: 每GPU批大小=2，有效批大小=8
- 关闭 `--keep_effective_batch_size`: 每GPU批大小=8，有效批大小=32

### 其他重要参数

- **`--world_size`**: 使用的GPU数量（不指定则使用所有可用GPU）
- **`--port`**: 分布式训练通信端口（默认12355）
- **`--resume_checkpoint`**: 恢复训练的checkpoint路径

## 性能优化建议

### 1. 批大小调整

```bash
# 如果显存充足，可以增加每GPU的批大小
python main_train_ddp.py --batch_size 16 --world_size 4

# 如果显存不足，可以减少批大小但保持有效批大小
python main_train_ddp.py --batch_size 4 --world_size 4 --keep_effective_batch_size
```

### 2. 数据加载优化

在配置文件中调整：
```yaml
data:
  num_workers: 8  # 增加数据加载worker数量
  batch_size: 8   # 根据GPU显存调整
```

### 3. 混合精度训练

确保配置文件中启用了混合精度：
```yaml
model:
  vlm_config:
    dtype: "torch.bfloat16"  # 或 "torch.float16"
```

## 监控和调试

### 1. 检查GPU使用情况

```bash
# 在另一个终端中监控GPU使用
watch -n 1 nvidia-smi
```

### 2. 日志查看

多GPU训练的日志只在rank 0进程中输出，可以在实验目录中找到：
```
experiments/your_experiment_name_timestamp/train.log
```

### 3. Weights & Biases监控

使用 `--use_wandb` 开启W&B监控，可以实时查看：
- 训练损失
- 验证损失  
- 学习率变化
- GPU利用率

## 常见问题解决

### 1. CUDA_VISIBLE_DEVICES设置

如果系统有多个GPU但只想使用部分GPU：
```bash
# 只使用GPU 0和1
export CUDA_VISIBLE_DEVICES=0,1
python main_train_ddp.py --world_size 2
```

### 2. 端口冲突

如果遇到端口占用错误：
```bash
python main_train_ddp.py --port 12356  # 使用其他端口
```

### 3. 显存不足

```bash
# 减少批大小
python main_train_ddp.py --batch_size 2

# 或减少worker数量
# 在配置文件中设置 num_workers: 2
```

### 4. 数据不均衡

确保训练数据可以被均匀分配到所有GPU：
- 使用 `drop_last=True` 确保批大小一致
- 检查数据集大小是否能被GPU数量整除

## 性能对比

预期的多GPU训练性能提升：

| GPU数量 | 相对单GPU训练时间 | 实际加速比 |
|---------|------------------|------------|
| 2       | ~50%             | ~1.8x      |
| 4       | ~25%             | ~3.5x      |
| 8       | ~12.5%           | ~6.5x      |

*注：实际性能取决于模型大小、批大小、网络带宽等因素*

## 从单GPU迁移

如果你已有单GPU训练的checkpoint，可以直接用于多GPU训练：

```bash
# 从单GPU checkpoint开始多GPU训练
python main_train_ddp.py \
    --resume_checkpoint path/to/single_gpu_checkpoint.pth \
    --world_size 4
```

模型权重会自动在所有GPU间同步，无需额外转换。
