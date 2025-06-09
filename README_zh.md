# PaliGemma-VLA：视觉-语言-动作模型

该项目使用PaliGemma架构作为其骨干，实现了一个视觉-语言-动作（VLA）模型。它专为需要理解视觉输入和自然语言指令以预测一系列动作的任务而设计。项目结构和工程原理受到`miravla`和`RoboVLMs`（特别是`robopaligemma.py`）等项目的启发，并致力于实现兼容性。

## 特性

*   **PaliGemma骨干**：利用预训练的PaliGemma模型进行多模态特征提取。
*   **多步动作预测**：支持基于时间窗口的动作序列，具有可配置的动作维度（例如，`horizon=8`，`per_action_dim=7`，`action_dim=56`）。
*   **流匹配架构**：实现先进的流匹配技术，用于连续动作空间预测，并在推理过程中使用欧拉积分。
*   **多GPU训练**：使用PyTorch DistributedDataParallel (DDP) 支持分布式训练，具有自动端口分配和NCCL优化。
*   **灵活的数据加载**：支持基于Parquet的数据集，包含可变长度的动作序列，包括多摄像头输入和机器人状态信息。
*   **高级归一化**：自动计算和应用归一化统计数据，以实现稳定的训练。
*   **模块化设计**：将关注点分离到数据加载、模型组件（VLM、动作头、集成的VLA模型）、训练和推理。
*   **可配置的训练和评估**：使用YAML配置文件轻松管理超参数和实验设置。
*   **检查点**：保存和加载模型检查点，以便恢复训练和进行推理。
*   **推理支持**：提供用于数据集批量评估和单项推理的脚本。
*   **实验跟踪**：集成Weights & Biases (WandB) 支持实验监控。

## 目录结构

```
paligemma-VLA/
├── configs/                     # YAML配置文件
│   ├── vla_config.yaml         # 单GPU训练配置
│   └── vla_config_ddp.yaml     # 多GPU分布式训练配置
├── data/
│   └── loader.py               # VLADataset, VLAImageProcessor, collate_fn
├── model/
│   ├── __init__.py
│   ├── paligemma_vlm.py        # PaliGemmaVLM骨干类
│   ├── vla_model.py            # 集成VLM和ActionHead的VLAModel
│   ├── action_head/            # 动作预测模块
│   │   └── flow_matching.py    # 流匹配动作头实现
│   └── vision_encoder_module.py # 视觉编码器组件
├── training/
│   ├── trainer.py              # 单GPU VLATrainer类
│   └── trainer_ddp.py          # 多GPU VLATrainerDDP类
├── inference/
│   └── predictor.py            # 用于推理的VLAPredictor类
├── utils/
│   ├── misc.py                 # 实用函数（日志记录、检查点、动作离散化）
│   ├── config_utils.py         # 配置实用程序
│   └── calculate_normalization_stats.py  # 归一化统计数据计算
├── scripts/
│   └── run_ddp_training.sh     # 自动化多GPU训练脚本
├── main_train.py               # 单GPU训练主脚本
├── main_train_ddp.py           # 多GPU分布式训练主脚本
├── main_eval.py                # 评估和推理主脚本
├── test_vla_offline.py         # 离线推理测试脚本
├── requirements.txt            # Python依赖项
└── README.md                   # 本文件
```

## 设置与安装

### 1. 克隆并设置环境

```bash
# 克隆仓库（如果适用）
cd paligemma-VLA

# 创建Python虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Windows上: venv\Scripts\activate
```

### 2. 安装依赖项

```bash
# 安装所有必需的依赖项
pip install -r requirements.txt

# 可选：安装特定CUDA版本的PyTorch
# 访问 https://pytorch.org/ 获取特定CUDA的安装命令
# CUDA 11.8示例:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Hugging Face身份验证（可选）

对于私有模型或避免速率限制：
```bash
# 如果尚未安装，请安装Hugging Face CLI
pip install huggingface_hub[cli]

# 登录Hugging Face
huggingface-cli login
```

### 4. 下载预训练模型

该项目使用PaliGemma模型。您可以：
- 让系统从Hugging Face自动下载（需要互联网）
- 手动下载并放置在 `./weight/paligemma-3b-pt-224/`

### 5. GPU设置验证

对于多GPU训练，请验证您的设置：
```bash
# 检查CUDA可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# 测试多GPU设置（如果您有多个GPU）
python diagnose_ddp.py
```

## 数据准备与归一化

### 数据集格式

模型期望数据以Parquet文件格式存储。每行代表动作序列/片段中的一个时间步。

**必需的数据字段**：
*   `image_1`: 当前步骤主RGB图像的字节
*   `image_2`: (可选) 次要（例如，手腕）RGB图像的字节
*   `state`: 机器人状态向量（例如，关节位置、夹爪状态），作为浮点数列表/数组
*   `action`: 当前步骤的动作向量，作为浮点数列表/数组
*   `is_first`: 布尔值，如果这是片段的第一个步骤，则为true
*   `is_last`: 布尔值，如果这是片段的最后一个步骤，则为true
*   `is_terminal`: 布尔值，如果此步骤是终止状态，则为true
*   `prompt`: 自然语言指令字符串

### 多步动作配置

模型支持多步动作预测，参数如下：
- **horizon**: 要预测的动作步数（例如，8）
- **per_action_dim**: 每个动作步骤的维度（例如，7自由度手臂为7）
- **action_dim**: 总动作维度 (horizon × per_action_dim = 56)

### 归一化统计数据计算

**重要**：在训练前计算归一化统计数据以实现稳定收敛：

```bash
# 为您的数据集计算归一化统计数据
python utils/calculate_normalization_stats.py \
    --config_path configs/vla_config_ddp.yaml \
    --output_path normalization_stats.json \
    --train_data_override "/path/to/your/training/data/"

# 生成的 normalization_stats.json 将包含：
# - 状态归一化（每个状态维度的最小值/最大值）
# - 动作归一化（每个动作维度的最小值/最大值）
```

**添加到您的配置文件**：
```yaml
data:
  normalization_stats_path: "normalization_stats.json"
  # ... 其他数据配置
```

### 序列处理

`VLADataset` 类：
1. 根据 `is_first` 标志将行分组为片段
2. 从片段中采样固定长度的窗口 (`max_seq_len`)
3. 对较短的片段应用填充
4. 使用计算出的统计数据对状态和动作数据进行归一化

## 配置

### 单GPU训练 (`configs/vla_config.yaml`)
用于单GPU训练和开发。

### 多GPU训练 (`configs/vla_config_ddp.yaml`)
针对跨多个GPU的分布式训练优化的配置。

**关键配置部分：**

#### 数据配置
```yaml
data:
  train_parquet_files: "/path/to/training/data/"
  val_parquet_files: "/path/to/validation/data/"
  tokenizer_name_or_path: "./weight/paligemma-3b-pt-224"
  image_processor_name_or_path: "./weight/paligemma-3b-pt-224"
  siglip_model_name: "google/siglip-base-patch16-224"
  max_seq_len: 4                    # 模型输入的序列长度
  prompt_max_len: 77                # 最大提示符标记长度
  batch_size: 20                    # 总批量大小（分布在GPU上）
  num_workers: 8                    # DataLoader工作线程数
  normalization_stats_path: "normalization_stats.json"  # 归一化统计数据
  state_dim: 7                      # 机器人状态维度
```

#### 模型架构
```yaml
model:
  vlm_config:
    model_name_or_path: "./weight/paligemma-3b-pt-224"
    use_aux_camera: false           # 使用辅助摄像头
    freeze_vision_tower: false      # 冻结视觉编码器
    freeze_language_model: false    # 冻结语言模型
    dtype: "torch.bfloat16"         # 模型精度
    num_image_tokens: 256           # 图像标记数量

  vision_resampler_config:
    type: "mlp"                     # 视觉重采样器类型
    output_dim: 2048                # 输出嵌入维度

  action_head_config:
    use_state_input: true           # 包括机器人状态
    horizon: 8                      # 动作步数
    per_action_dim: 7               # 每个动作步骤的维度
    action_dim: 56                  # 总动作维度 (8×7)
    num_action_bins: 1024           # 离散化区间数
    mlp_hidden_dims: [512, 256]     # 隐藏层大小
    dropout_prob: 0.1               # Dropout概率
```

#### 训练配置
```yaml
training:
  epochs: 50                        # 训练轮数
  log_interval: 50                  # 日志记录频率（批次）
  checkpoint_dir: "./experiments"   # 检查点目录
  experiment_name: "vla_ddp_run"    # 实验标识符
  grad_clip_norm: 1.0              # 梯度裁剪
  seed: 123                        # 随机种子
  save_every_n_epochs: 5           # 检查点频率

optimizer:
  type: "AdamW"                    # 优化器类型
  lr: 1e-4                         # 学习率
  weight_decay: 0.01               # 权重衰减
  betas: [0.9, 0.999]             # Adam betas

lr_scheduler:
  type: "CosineAnnealingLR"        # 学习率调度器
  T_max: 50                        # 余弦退火周期
```

## 快速入门示例

### 完整的训练流程

以下是从头开始训练VLA模型的完整示例：

```bash
# 1. 设置环境
python3 -m venv vla_env
source vla_env/bin/activate
pip install -r requirements.txt

# 2. 准备数据（确保Parquet文件格式正确）
# 您的数据应位于：/path/to/your/training/data/

# 3. 计算归一化统计数据
python utils/calculate_normalization_stats.py \
    --config_path configs/vla_config_ddp.yaml \
    --output_path normalization_stats.json

# 4. 更新配置
# 编辑 configs/vla_config_ddp.yaml 指向您的数据：
# data:
#   train_parquet_files: "/path/to/your/training/data/"
#   val_parquet_files: "/path/to/your/validation/data/"
#   normalization_stats_path: "normalization_stats.json"

# 5. 首先使用单个GPU进行测试
python main_train.py \
    --config_path configs/vla_config.yaml \
    --epochs 2 \
    --experiment_name test_run

# 6. 运行完整的多GPU训练
python main_train_ddp.py \
    --config_path configs/vla_config_ddp.yaml \
    --experiment_name production_training \
    --use_wandb

# 7. 评估训练好的模型
python main_eval.py \
    --checkpoint_path experiments/production_training/checkpoints/model_best.pth.tar \
    --eval_data_path /path/to/test/data/ \
    --output_dir ./results/evaluation \
    --save_predictions

# 8. 测试推理
python test_vla_offline.py \
    --checkpoint_path experiments/production_training/checkpoints/model_best.pth.tar \
    --test_data_dir output_data/ \
    --output_file inference_results.json
```

### 配置概述

您需要修改的关键文件：

1.  **`configs/vla_config_ddp.yaml` 中的数据路径**：
```yaml
data:
  train_parquet_files: "/your/training/data/path/"
  val_parquet_files: "/your/validation/data/path/"
```

2.  **模型架构**（如果需要）：
```yaml
model:
  action_head_config:
    horizon: 8                    # 动作预测步数
    per_action_dim: 7             # 每步的动作维度
    action_dim: 56                # 总计：horizon × per_action_dim
```

3.  **训练设置**：
```yaml
training:
  epochs: 50                      # 训练时长
  experiment_name: "my_vla_model" # 实验标识符
```

## 训练

### 先决条件

1.  **计算归一化统计数据**（稳定训练所必需）：
```bash
python utils/calculate_normalization_stats.py \
    --config_path configs/vla_config_ddp.yaml \
    --output_path normalization_stats.json
```

2.  **更新配置**：将归一化文件路径添加到您的配置中：
```yaml
data:
  normalization_stats_path: "normalization_stats.json"
```

### 单GPU训练

用于开发和较小的数据集：

```bash
python main_train.py --config_path configs/vla_config.yaml
```

**命令行覆盖**：
```bash
python main_train.py --config_path configs/vla_config.yaml \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --experiment_name my_experiment \
    --use_wandb  # 启用实验跟踪
```

**恢复训练**：
```bash
python main_train.py \
    --config_path configs/vla_config.yaml \
    --resume_checkpoint experiments/my_experiment/checkpoints/checkpoint_epoch_10.pth.tar
```

### 多GPU分布式训练

使用多个GPU加快训练速度：

#### 使用脚本快速启动
```bash
# 自动化多GPU训练
./scripts/run_ddp_training.sh \
    --config configs/vla_config_ddp.yaml \
    --experiment_name multi_gpu_xarm_training \
    --wandb_project VLA_XArm_Project
```

#### 手动多GPU训练
```bash
python main_train_ddp.py \
    --config_path configs/vla_config_ddp.yaml \
    --experiment_name paligemma_vla_ddp_finetune \
    --use_wandb \
    --wandb_project_name VLA_Project_DDP
```

#### 高级多GPU选项
```bash
python main_train_ddp.py \
    --config_path configs/vla_config_ddp.yaml \
    --experiment_name custom_training \
    --batch_size 32 \
    --lr 2e-4 \
    --epochs 100 \
    --train_data_override "/path/to/custom/training/data/" \
    --val_data_override "/path/to/custom/validation/data/" \
    --model_path_override "./weight/paligemma-3b-pt-224" \
    --use_wandb \
    --wandb_project_name My_VLA_Experiments \
    --wandb_entity your_wandb_team
```

#### 恢复多GPU训练
```bash
python main_train_ddp.py \
    --config_path configs/vla_config_ddp.yaml \
    --resume_checkpoint experiments/paligemma_vla_ddp_finetune/checkpoints/checkpoint_epoch_25.pth.tar \
    --use_wandb
```

### 训练监控

#### Weights & Biases集成
```bash
# 如果尚未安装wandb，请安装
pip install wandb

# 登录您的帐户
wandb login

# 使用wandb日志记录进行训练
python main_train_ddp.py \
    --config_path configs/vla_config_ddp.yaml \
    --use_wandb \
    --wandb_project_name VLA_Experiments \
    --wandb_entity your_team_name
```

#### 日志文件
- 训练日志：`experiments/{experiment_name}/logs/train.log`
- 检查点：`experiments/{experiment_name}/checkpoints/`
- WandB日志：`wandb/` 目录

### 多GPU训练故障排除

#### 测试GPU设置
```bash
# 检查GPU可用性并测试基本的DDP功能
python diagnose_ddp.py
```

#### 常见问题与解决方案

**NCCL超时错误**：
```bash
# 训练前设置环境变量
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

python main_train_ddp.py --config_path configs/vla_config_ddp.yaml
```

**端口绑定问题**：
系统会自动查找空闲端口，但您可以指定：
```bash
export MASTER_PORT=12355
python main_train_ddp.py --config_path configs/vla_config_ddp.yaml
```

**内存问题**：
```bash
# 在配置中或通过命令行减少批量大小
python main_train_ddp.py \
    --config_path configs/vla_config_ddp.yaml \
    --batch_size 16  # 从默认值20减少
```

## 评估与推理

### 数据集批量评估

在验证或测试数据集上评估您训练好的模型：

```bash
python main_eval.py \
    --checkpoint_path experiments/paligemma_vla_ddp_finetune/checkpoints/model_best.pth.tar \
    --eval_data_path /path/to/validation/data/ \
    --output_dir ./results/evaluation_results \
    --save_predictions
```

**多数据集评估**：
```bash
python main_eval.py \
    --checkpoint_path experiments/my_experiment/checkpoints/model_best.pth.tar \
    --eval_data_path /path/to/test_data_1.parquet /path/to/test_data_2.parquet \
    --output_dir ./results/multi_dataset_eval \
    --save_predictions
```

### 单项推理

使用您训练好的模型测试单个示例：

```bash
python main_eval.py \
    --checkpoint_path experiments/my_experiment/checkpoints/model_best.pth.tar \
    --image1_paths /path/to/camera1/image.jpg \
    --prompt "拿起红色的杯子并将其放在桌子上" \
    --output_dir ./results/single_inference \
    --save_predictions
```

**带序列的多步推理**：
```bash
python main_eval.py \
    --checkpoint_path experiments/my_experiment/checkpoints/model_best.pth.tar \
    --image1_paths /path/to/step1.jpg /path/to/step2.jpg /path/to/step3.jpg \
    --image2_paths /path/to/wrist1.jpg /path/to/wrist2.jpg /path/to/wrist3.jpg \
    --prompt "导航到厨房并拿起蓝色物体" \
    --state_vector 0.1 0.2 0.3 0.4 0.5 0.6 0.7 \
    --output_dir ./results/multi_step_inference \
    --save_predictions
```

### 离线推理测试

独立测试推理流程：

```bash
# 使用预保存的数据测试离线推理
python test_vla_offline.py \
    --checkpoint_path experiments/my_experiment/checkpoints/model_best.pth.tar \
    --test_data_dir output_data/ \
    --output_file test_inference_results.json
```

### 理解模型输出

#### 动作预测
模型输出多步动作，形状为 `(horizon, per_action_dim)`：
- **horizon=8**: 预测未来8个动作步骤
- **per_action_dim=7**: 每个步骤有7个动作维度（例如，6自由度手臂+夹爪）
- 动作使用归一化统计数据自动反归一化

#### 评估指标
- **准确率**: 离散化动作预测准确率
- **MSE/MAE**: 连续动作预测误差
- **每步分析**: 按预测时间窗口的性能细分

#### 输出文件
- `predictions.csv`: 包含真实值的详细预测
- `metrics.json`: 汇总的评估指标
- `inference_log.txt`: 详细的推理日志

### 性能优化

#### GPU内存管理
```bash
# 对于大型模型或有限的GPU内存
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python main_eval.py \
    --checkpoint_path path/to/checkpoint.pth.tar \
    --eval_data_path path/to/data/ \
    --batch_size 1  # 减少推理的批量大小
```

#### 推理速度
```bash
# 启用混合精度以加快推理速度
python main_eval.py \
    --checkpoint_path path/to/checkpoint.pth.tar \
    --eval_data_path path/to/data/ \
    --use_amp  # 如果评估脚本支持
```

## 核心组件与架构

### 模型架构概述

VLA模型由三个主要组件组成：

1.  **视觉-语言模型 (VLM)**: PaliGemma骨干，用于多模态理解
2.  **动作头**: 流匹配网络，用于连续动作预测
3.  **集成层**: 将VLM嵌入与机器人状态相结合以进行动作预测

### 关键组件

#### `PaliGemmaVLM` (`model/paligemma_vlm.py`)
- 包装Hugging Face `PaliGemmaForConditionalGeneration`
- 处理图像序列和文本提示
- 为每个步骤输出多模态嵌入
- 支持辅助摄像头输入和视觉重采样

#### `FlowmatchingActionHead` (`model/action_head/flow_matching.py`)
- 实现用于连续动作空间的流匹配
- 支持多步动作预测（基于时间窗口）
- 推理过程中使用欧拉积分
- 处理动作归一化和离散化

#### `VLAModel` (`model/vla_model.py`)
- 集成PaliGemmaVLM和ActionHead
- 管理从原始数据到动作预测的前向传播
- 支持训练（流匹配）和推理模式
- 处理设备放置和混合精度

#### `VLATrainerDDP` (`training/trainer_ddp.py`)
- 使用PyTorch DDP进行分布式训练
- 自动梯度缩放和裁剪
- WandB集成用于实验跟踪
- 针对多GPU环境的稳健错误处理

#### `VLADataset` (`data/loader.py`)
- 使用片段分组加载Parquet文件
- 用于视觉输入的SigLIP图像处理
- 自动应用归一化
- 多步动作序列处理

### 训练与推理模式

#### 训练模式（流匹配）
- 使用 `actions_gt_seq` 参数进行真实值指导
- 应用流匹配损失进行连续动作学习
- 支持教师强制以实现稳定训练

#### 推理模式（欧拉积分）
- 不提供真实值动作
- 使用欧拉积分进行动作预测
- 自主生成多步动作序列

## 高级用法

### 自定义数据集集成

#### 1. 准备您的数据格式
确保您的Parquet文件包含必需的字段：
```python
# Parquet文件中的必需列
required_columns = [
    'image_1_bytes',      # 主摄像头图像（字节）
    'image_2_bytes',      # 辅助摄像头（可选，字节）
    'state',              # 机器人状态（浮点数列表）
    'action',             # 动作向量（浮点数列表）
    'prompt',             # 文本指令（字符串）
    'is_first',           # 片段开始标志（布尔值）
    'is_last',            # 片段结束标志（布尔值）
    'is_terminal'         # 终止状态标志（布尔值）
]
```

#### 2. 计算数据集统计数据
```bash
python utils/calculate_normalization_stats.py \
    --config_path configs/your_custom_config.yaml \
    --output_path your_dataset_normalization.json
```

#### 3. 创建自定义配置
```yaml
# configs/your_custom_config.yaml
data:
  train_parquet_files: "/path/to/your/train/data/"
  val_parquet_files: "/path/to/your/val/data/"
  normalization_stats_path: "your_dataset_normalization.json"
  state_dim: 7                    # 匹配您的机器人状态维度
  max_seq_len: 4                  # 根据您的片段进行调整

model:
  action_head_config:
    horizon: 8                    # 预测时间窗口
    per_action_dim: 7             # 每步的动作维度
    action_dim: 56                # horizon × per_action_dim
```

### 微调预训练模型

#### 1. 从预训练检查点开始
```bash
python main_train_ddp.py \
    --config_path configs/vla_config_ddp.yaml \
    --resume_checkpoint path/to/pretrained/model.pth.tar \
    --experiment_name fine_tuned_model \
    --lr 1e-5  # 微调时使用较低的学习率
```

#### 2. 冻结特定组件
```yaml
# 在您的配置文件中
model:
  vlm_config:
    freeze_vision_tower: true     # 冻结视觉编码器
    freeze_language_model: false  # 微调语言模型
```

### 多摄像头设置

#### 配置
```yaml
model:
  vlm_config:
    use_aux_camera: true          # 启用辅助摄像头
```

#### 数据准备
确保您的Parquet文件中同时存在 `image_1_bytes` 和 `image_2_bytes`。

### 动作空间自定义

#### 不同的动作维度
```yaml
model:
  action_head_config:
    horizon: 4                    # 较短的预测时间窗口
    per_action_dim: 6             # 6自由度机器人（无夹爪）
    action_dim: 24                # 4 × 6
    num_action_bins: 512          # 调整离散化分辨率
```

#### 自定义动作边界
归一化统计数据将自动处理您的动作范围。确保正确计算归一化：
```bash
python utils/calculate_normalization_stats.py \
    --config_path configs/custom_action_config.yaml \
    --output_path custom_action_normalization.json
```

### 性能调优

#### 内存优化
```yaml
data:
  batch_size: 8                   # 如果内存有限则减少
  num_workers: 4                  # 根据CPU核心数调整

model:
  vlm_config:
    dtype: "torch.bfloat16"       # 使用混合精度
```

#### 训练速度
```yaml
training:
  grad_clip_norm: 1.0             # 防止梯度爆炸
  log_interval: 100               # 减少日志记录频率

data:
  num_workers: 8                  # 增加以加快数据加载速度
```

### 调试与监控

#### 检查模型权重
```bash
python diagnose_weights.py --checkpoint_path path/to/checkpoint.pth.tar
```

#### 分析训练数据
```bash
python inspect_parquet.py --data_path /path/to/your/data/
```

#### 简单分析
```bash
python simple_analysis.py --config_path configs/vla_config_ddp.yaml
```

## 最佳实践

### 训练建议

#### 1. 始终首先计算归一化统计数据
```bash
# 任何训练前计算
python utils/calculate_normalization_stats.py \
    --config_path configs/vla_config_ddp.yaml \
    --output_path normalization_stats.json
```

#### 2. 从单GPU开始进行调试
```bash
# 首先测试您的配置
python main_train.py --config_path configs/vla_config.yaml --epochs 2
```

#### 3. 使用多GPU进行生产训练
```bash
# 验证后扩展
python main_train_ddp.py --config_path configs/vla_config_ddp.yaml
```

#### 4. 监控训练进度
- 启用WandB日志记录进行实验跟踪
- 频繁保存检查点 (`save_every_n_epochs: 5`)
- 监控训练和验证损失

#### 5. 学习率策略
- 新模型从 `1e-4` 开始
- 微调预训练模型时使用 `1e-5`
- 启用余弦退火以实现稳定收敛

### 数据准备最佳实践

#### 1. 片段结构
- 确保正确的 `is_first`, `is_last`, `is_terminal` 标志
- 尽可能保持一致的片段长度
- 在训练数据中包含多样化的场景

#### 2. 图像质量
- 使用一致的图像分辨率（推荐224x224）
- 确保良好的光照和对比度
- 如果可用，包括多个摄像机角度

#### 3. 动作空间设计
- 将动作归一化到合理的范围
- 使用一致的动作表示
- 包含足够的动作多样性

### 性能优化

#### 内存管理
```yaml
# 针对可用GPU内存进行优化
data:
  batch_size: 16                  # 根据GPU内存调整
  num_workers: 4                  # 与CPU核心数平衡

model:
  vlm_config:
    dtype: "torch.bfloat16"       # 减少内存使用
```

#### 多GPU效率
```bash
# 设置最佳NCCL设置
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
```

## 故障排除

### 常见训练问题

#### 1. 验证损失高而训练损失低
**症状**：训练损失正常下降（例如，0.005），但验证损失仍然很高（例如，0.38）

**解决方案**：
- 检查训练和验证是否使用一致的参数
- 确保正确应用归一化统计数据
- 验证数据划分质量和多样性
- 监控过拟合（降低模型复杂度或添加正则化）

#### 2. NCCL超时错误（多GPU）
**症状**：
```
RuntimeError: NCCL operation timed out
```

**解决方案**：
```bash
# 设置环境变量
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# 使用诊断脚本
python diagnose_ddp.py
```

#### 3. 端口绑定问题
**症状**：
```
RuntimeError: Address already in use
```

**解决方案**：
- 系统会自动查找空闲端口
- 手动设置端口：`export MASTER_PORT=12355`
- 检查挂起的进程：`ps aux | grep python`

#### 4. 内存不足 (OOM)
**症状**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：
```bash
# 减少批量大小
python main_train_ddp.py --config_path configs/vla_config_ddp.yaml --batch_size 8

# 使用混合精度
# 在配置中设置：dtype: "torch.bfloat16"

# 减少序列长度
# 在配置中设置：max_seq_len: 2
```

#### 5. 数据加载错误
**症状**：图像损坏、字段缺失、加载缓慢

**解决方案**：
```bash
# 检查您的数据
python inspect_parquet.py --data_path /path/to/your/data/

# 检查数据完整性
python data/loader.py  # 运行测试脚本

# 如果出现内存问题，减少num_workers
# 在配置中设置：num_workers: 2
```

### 模型架构问题

#### 1. 动作维度不匹配
**症状**：
```
RuntimeError: size mismatch for action_head layers
```

**解决方案**：
- 确保 `action_dim = horizon × per_action_dim`
- 验证您的数据具有正确的动作维度
- 检查归一化统计数据是否与您的动作空间匹配

#### 2. 视觉处理错误
**症状**：黑色图像、图像形状不正确

**解决方案**：
- 验证VLADataset中的图像预处理
- 检查SigLIP处理器兼容性
- 确保图像为RGB格式

### 推理问题

#### 1. 预测质量差
**解决方案**：
- 验证模型是否使用正确的归一化进行训练
- 检查推理是否使用相同的归一化统计数据
- 确保输入数据与训练分布匹配

#### 2. 推理速度慢
**解决方案**：
```bash
# 使用混合精度
# 减少推理的批量大小
# 启用编译（如果支持）
```

### 获取帮助

#### 1. 检查日志
- 训练日志：`experiments/{experiment_name}/logs/`
- WandB仪表板用于指标可视化
- 系统日志用于硬件问题

#### 2. 调试工具
```bash
# GPU诊断
python diagnose_ddp.py

# 权重分析
python diagnose_weights.py --checkpoint_path path/to/checkpoint.pth.tar

# 数据检查
python inspect_parquet.py --data_path /path/to/data/
```

#### 3. 验证步骤
```bash
# 测试单个批次
python simple_analysis.py --config_path configs/vla_config_ddp.yaml

# 测试推理流程
python test_vla_offline.py --checkpoint_path path/to/checkpoint.pth.tar
```

## 版本历史与更新

### 近期改进
- **多GPU训练**：稳定的分布式训练，具有自动端口分配功能
- **流匹配架构**：先进的连续动作预测
- **归一化流程**：自动统计数据计算和应用
- **训练/验证一致性**：修复了训练和验证之间的参数一致性问题
- **推理流程**：针对多步动作预测进行了完全重写

### 架构演变
- **v1.0**: 基本的离散动作预测
- **v2.0**: 使用连续动作的流匹配
- **v2.1**: 多步时间窗口预测
- **v2.2**: 分布式训练支持
- **v2.3**: 训练/验证一致性修复

## 待办事项/未来工作

*   **增强的评估指标**：更全面的机器人特定指标
*   **实时推理**：针对实时机器人控制的优化
*   **多模态扩展**：支持其他传感器模态
*   **模型压缩**：用于部署的量化和剪枝
*   **模拟器集成**：与机器人模拟器的直接集成
*   **迁移学习**：针对不同机器人平台的预训练模型

