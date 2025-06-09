# PaliGemma-VLA: Vision-Language-Action Model

This project implements a Vision-Language-Action (VLA) model using the PaliGemma architecture
as its backbone. It is designed for tasks that require understanding visual input and 
natural language instructions to predict a sequence of actions. The project structure and
engineering principles are inspired by and aim for compatibility with projects like `miravla`
and `RoboVLMs` (specifically `robopaligemma.py`).

## Features

*   **PaliGemma Backbone**: Utilizes a pre-trained PaliGemma model for multimodal feature extraction.
*   **Multi-Step Action Prediction**: Supports horizon-based action sequences with configurable action dimensions (e.g., `horizon=8`, `per_action_dim=7`, `action_dim=56`).
*   **Flow Matching Architecture**: Implements advanced flow matching for continuous action space prediction with Euler integration during inference.
*   **Multi-GPU Training**: Distributed training support using PyTorch DistributedDataParallel (DDP) with automatic port allocation and NCCL optimization.
*   **Flexible Data Loading**: Supports Parquet-based datasets with variable-length action sequences, including multiple camera inputs and robot state information.
*   **Advanced Normalization**: Automatic calculation and application of normalization statistics for stable training.
*   **Modular Design**: Separates concerns into data loading, model components (VLM, action head, integrated VLA model), training, and inference.
*   **Configurable Training and Evaluation**: Uses YAML configuration files for easy management of hyperparameters and experiment settings.
*   **Checkpointing**: Saves and loads model checkpoints for resuming training and for inference.
*   **Inference Support**: Provides scripts for both batch evaluation on datasets and single-item inference.
*   **Experiment Tracking**: Integrated Weights & Biases (WandB) support for experiment monitoring.

## Directory Structure

```
paligemma-VLA/
├── configs/                     # YAML configuration files
│   ├── vla_config.yaml         # Single-GPU training configuration
│   └── vla_config_ddp.yaml     # Multi-GPU distributed training configuration
├── data/
│   └── loader.py               # VLADataset, VLAImageProcessor, collate_fn
├── model/
│   ├── __init__.py
│   ├── paligemma_vlm.py        # PaliGemmaVLM backbone class
│   ├── vla_model.py            # VLAModel integrating VLM and ActionHead
│   ├── action_head/            # Action prediction modules
│   │   └── flow_matching.py    # Flow matching action head implementation
│   └── vision_encoder_module.py # Vision encoder components
├── training/
│   ├── trainer.py              # Single-GPU VLATrainer class
│   └── trainer_ddp.py          # Multi-GPU VLATrainerDDP class
├── inference/
│   └── predictor.py            # VLAPredictor class for inference
├── utils/
│   ├── misc.py                 # Utility functions (logging, checkpoints, action discretization)
│   ├── config_utils.py         # Configuration utilities
│   └── calculate_normalization_stats.py  # Normalization statistics calculation
├── scripts/
│   └── run_ddp_training.sh     # Automated multi-GPU training script
├── main_train.py               # Main script for single-GPU training
├── main_train_ddp.py           # Main script for multi-GPU distributed training
├── main_eval.py                # Main script for evaluation and inference
├── test_vla_offline.py         # Offline inference testing script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Setup and Installation

### 1. Clone and Setup Environment

```bash
# Clone the repository (if applicable)
cd paligemma-VLA

# Create a Python virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install all required dependencies
pip install -r requirements.txt

# Optional: Install PyTorch with specific CUDA version
# Visit https://pytorch.org/ for CUDA-specific installation commands
# Example for CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Hugging Face Authentication (Optional)

For private models or to avoid rate limits:
```bash
# Install Hugging Face CLI if not already installed
pip install huggingface_hub[cli]

# Login to Hugging Face
huggingface-cli login
```

### 4. Download Pre-trained Models

The project uses PaliGemma models. You can either:
- Let the system auto-download from Hugging Face (requires internet)
- Manually download and place in `./weight/paligemma-3b-pt-224/`

### 5. GPU Setup Verification

For multi-GPU training, verify your setup:
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Test multi-GPU setup (if you have multiple GPUs)
python diagnose_ddp.py
```

## Data Preparation and Normalization

### Dataset Format

The model expects data in Parquet files. Each row represents one time step in an action sequence/episode.

**Required Data Fields**:
*   `image_1`: Bytes of the main RGB image for the current step
*   `image_2`: (Optional) Bytes of a secondary (e.g., wrist) RGB image
*   `state`: Robot state vector (e.g., joint positions, gripper status) as a list/array of floats
*   `action`: Action vector for the current step as a list/array of floats
*   `is_first`: Boolean, true if this is the first step of an episode
*   `is_last`: Boolean, true if this is the last step of an episode
*   `is_terminal`: Boolean, true if this step is a terminal state
*   `prompt`: Natural language instruction string

### Multi-Step Action Configuration

The model supports multi-step action prediction with the following parameters:
- **horizon**: Number of action steps to predict (e.g., 8)
- **per_action_dim**: Dimensions per action step (e.g., 7 for 7-DOF arm)
- **action_dim**: Total action dimensions (horizon × per_action_dim = 56)

### Normalization Statistics Calculation

**Important**: Calculate normalization statistics before training for stable convergence:

```bash
# Calculate normalization stats for your dataset
python utils/calculate_normalization_stats.py \
    --config_path configs/vla_config_ddp.yaml \
    --output_path normalization_stats.json \
    --train_data_override "/path/to/your/training/data/"

# The generated normalization_stats.json will contain:
# - State normalization (min/max values for each state dimension)
# - Action normalization (min/max values for each action dimension)
```

**Add to your config file**:
```yaml
data:
  normalization_stats_path: "normalization_stats.json"
  # ... other data config
```

### Sequence Handling

The `VLADataset` class:
1. Groups rows into episodes based on the `is_first` flag
2. Samples fixed-length windows (`max_seq_len`) from episodes
3. Applies padding for shorter episodes
4. Normalizes state and action data using calculated statistics

## Configuration

### Single-GPU Training (`configs/vla_config.yaml`)
For single GPU training and development.

### Multi-GPU Training (`configs/vla_config_ddp.yaml`)
Optimized configuration for distributed training across multiple GPUs.

**Key Configuration Sections:**

#### Data Configuration
```yaml
data:
  train_parquet_files: "/path/to/training/data/"
  val_parquet_files: "/path/to/validation/data/"
  tokenizer_name_or_path: "./weight/paligemma-3b-pt-224"
  image_processor_name_or_path: "./weight/paligemma-3b-pt-224"
  siglip_model_name: "google/siglip-base-patch16-224"
  max_seq_len: 4                    # Sequence length for model input
  prompt_max_len: 77                # Maximum prompt token length
  batch_size: 20                    # Total batch size (distributed across GPUs)
  num_workers: 8                    # DataLoader workers
  normalization_stats_path: "normalization_stats.json"  # Normalization statistics
  state_dim: 7                      # Robot state dimensions
```

#### Model Architecture
```yaml
model:
  vlm_config:
    model_name_or_path: "./weight/paligemma-3b-pt-224"
    use_aux_camera: false           # Use secondary camera
    freeze_vision_tower: false      # Freeze vision encoder
    freeze_language_model: false    # Freeze language model
    dtype: "torch.bfloat16"         # Model precision
    num_image_tokens: 256           # Number of image tokens
  
  vision_resampler_config:
    type: "mlp"                     # Vision resampler type
    output_dim: 2048                # Output embedding dimension
  
  action_head_config:
    use_state_input: true           # Include robot state
    horizon: 8                      # Number of action steps
    per_action_dim: 7               # Dimensions per action step
    action_dim: 56                  # Total action dimensions (8×7)
    num_action_bins: 1024           # Discretization bins
    mlp_hidden_dims: [512, 256]     # Hidden layer sizes
    dropout_prob: 0.1               # Dropout probability
```

#### Training Configuration
```yaml
training:
  epochs: 50                        # Number of training epochs
  log_interval: 50                  # Logging frequency (batches)
  checkpoint_dir: "./experiments"   # Checkpoint directory
  experiment_name: "vla_ddp_run"    # Experiment identifier
  grad_clip_norm: 1.0              # Gradient clipping
  seed: 123                        # Random seed
  save_every_n_epochs: 5           # Checkpoint frequency

optimizer:
  type: "AdamW"                    # Optimizer type
  lr: 1e-4                         # Learning rate
  weight_decay: 0.01               # Weight decay
  betas: [0.9, 0.999]             # Adam betas

lr_scheduler:
  type: "CosineAnnealingLR"        # Learning rate scheduler
  T_max: 50                        # Cosine annealing period
```

## Quick Start Example

### Complete Training Pipeline

Here's a complete example of training a VLA model from scratch:

```bash
# 1. Setup environment
python3 -m venv vla_env
source vla_env/bin/activate
pip install -r requirements.txt

# 2. Prepare your data (ensure Parquet files are in correct format)
# Your data should be in: /path/to/your/training/data/

# 3. Calculate normalization statistics
python utils/calculate_normalization_stats.py \
    --config_path configs/vla_config_ddp.yaml \
    --output_path normalization_stats.json

# 4. Update configuration
# Edit configs/vla_config_ddp.yaml to point to your data:
# data:
#   train_parquet_files: "/path/to/your/training/data/"
#   val_parquet_files: "/path/to/your/validation/data/"
#   normalization_stats_path: "normalization_stats.json"

# 5. Test with single GPU first
python main_train.py \
    --config_path configs/vla_config.yaml \
    --epochs 2 \
    --experiment_name test_run

# 6. Run full multi-GPU training
python main_train_ddp.py \
    --config_path configs/vla_config_ddp.yaml \
    --experiment_name production_training \
    --use_wandb

# 7. Evaluate the trained model
python main_eval.py \
    --checkpoint_path experiments/production_training/checkpoints/model_best.pth.tar \
    --eval_data_path /path/to/test/data/ \
    --output_dir ./results/evaluation \
    --save_predictions

# 8. Test inference
python test_vla_offline.py \
    --checkpoint_path experiments/production_training/checkpoints/model_best.pth.tar \
    --test_data_dir output_data/ \
    --output_file inference_results.json
```

### Configuration Overview

The key files you'll need to modify:

1. **Data paths** in `configs/vla_config_ddp.yaml`:
```yaml
data:
  train_parquet_files: "/your/training/data/path/"
  val_parquet_files: "/your/validation/data/path/"
```

2. **Model architecture** (if needed):
```yaml
model:
  action_head_config:
    horizon: 8                    # Action prediction steps
    per_action_dim: 7             # Action dimensions per step
    action_dim: 56                # Total: horizon × per_action_dim
```

3. **Training settings**:
```yaml
training:
  epochs: 50                      # Training duration
  experiment_name: "my_vla_model" # Experiment identifier
```

## Training

### Prerequisites

1. **Calculate Normalization Statistics** (Required for stable training):
```bash
python utils/calculate_normalization_stats.py \
    --config_path configs/vla_config_ddp.yaml \
    --output_path normalization_stats.json
```

2. **Update Configuration**: Add the normalization file path to your config:
```yaml
data:
  normalization_stats_path: "normalization_stats.json"
```

### Single-GPU Training

For development and smaller datasets:

```bash
python main_train.py --config_path configs/vla_config.yaml
```

**Command-line overrides**:
```bash
python main_train.py --config_path configs/vla_config.yaml \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --experiment_name my_experiment \
    --use_wandb  # Enable experiment tracking
```

**Resume training**:
```bash
python main_train.py \
    --config_path configs/vla_config.yaml \
    --resume_checkpoint experiments/my_experiment/checkpoints/checkpoint_epoch_10.pth.tar
```

### Multi-GPU Distributed Training

For faster training with multiple GPUs:

#### Quick Start with Script
```bash
# Automated multi-GPU training
./scripts/run_ddp_training.sh \
    --config configs/vla_config_ddp.yaml \
    --experiment_name multi_gpu_xarm_training \
    --wandb_project VLA_XArm_Project
```

#### Manual Multi-GPU Training
```bash
python main_train_ddp.py \
    --config_path configs/vla_config_ddp.yaml \
    --experiment_name paligemma_vla_ddp_finetune \
    --use_wandb \
    --wandb_project_name VLA_Project_DDP
```

#### Advanced Multi-GPU Options
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

#### Resume Multi-GPU Training
```bash
python main_train_ddp.py \
    --config_path configs/vla_config_ddp.yaml \
    --resume_checkpoint experiments/paligemma_vla_ddp_finetune/checkpoints/checkpoint_epoch_25.pth.tar \
    --use_wandb
```

### Training Monitoring

#### Weights & Biases Integration
```bash
# Install wandb if not already installed
pip install wandb

# Login to your account
wandb login

# Training with wandb logging
python main_train_ddp.py \
    --config_path configs/vla_config_ddp.yaml \
    --use_wandb \
    --wandb_project_name VLA_Experiments \
    --wandb_entity your_team_name
```

#### Log Files
- Training logs: `experiments/{experiment_name}/logs/train.log`
- Checkpoints: `experiments/{experiment_name}/checkpoints/`
- WandB logs: `wandb/` directory

### Troubleshooting Multi-GPU Training

#### Test GPU Setup
```bash
# Check GPU availability and test basic DDP functionality
python diagnose_ddp.py
```

#### Common Issues and Solutions

**NCCL Timeout Errors**:
```bash
# Set environment variables before training
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

python main_train_ddp.py --config_path configs/vla_config_ddp.yaml
```

**Port Binding Issues**:
The system automatically finds free ports, but you can specify:
```bash
export MASTER_PORT=12355
python main_train_ddp.py --config_path configs/vla_config_ddp.yaml
```

**Memory Issues**:
```bash
# Reduce batch size in config or via command line
python main_train_ddp.py \
    --config_path configs/vla_config_ddp.yaml \
    --batch_size 16  # Reduce from default 20
```

## Evaluation and Inference

### Batch Evaluation on Datasets

Evaluate your trained model on validation or test datasets:

```bash
python main_eval.py \
    --checkpoint_path experiments/paligemma_vla_ddp_finetune/checkpoints/model_best.pth.tar \
    --eval_data_path /path/to/validation/data/ \
    --output_dir ./results/evaluation_results \
    --save_predictions
```

**Multiple dataset evaluation**:
```bash
python main_eval.py \
    --checkpoint_path experiments/my_experiment/checkpoints/model_best.pth.tar \
    --eval_data_path /path/to/test_data_1.parquet /path/to/test_data_2.parquet \
    --output_dir ./results/multi_dataset_eval \
    --save_predictions
```

### Single-Item Inference

Test individual examples with your trained model:

```bash
python main_eval.py \
    --checkpoint_path experiments/my_experiment/checkpoints/model_best.pth.tar \
    --image1_paths /path/to/camera1/image.jpg \
    --prompt "Pick up the red cup and place it on the table" \
    --output_dir ./results/single_inference \
    --save_predictions
```

**Multi-step inference with sequence**:
```bash
python main_eval.py \
    --checkpoint_path experiments/my_experiment/checkpoints/model_best.pth.tar \
    --image1_paths /path/to/step1.jpg /path/to/step2.jpg /path/to/step3.jpg \
    --image2_paths /path/to/wrist1.jpg /path/to/wrist2.jpg /path/to/wrist3.jpg \
    --prompt "Navigate to the kitchen and pick up the blue object" \
    --state_vector 0.1 0.2 0.3 0.4 0.5 0.6 0.7 \
    --output_dir ./results/multi_step_inference \
    --save_predictions
```

### Offline Inference Testing

Test the inference pipeline independently:

```bash
# Test offline inference with pre-saved data
python test_vla_offline.py \
    --checkpoint_path experiments/my_experiment/checkpoints/model_best.pth.tar \
    --test_data_dir output_data/ \
    --output_file test_inference_results.json
```

### Understanding Model Outputs

#### Action Predictions
The model outputs multi-step actions with shape `(horizon, per_action_dim)`:
- **horizon=8**: Predicts 8 future action steps
- **per_action_dim=7**: Each step has 7 action dimensions (e.g., 6-DOF arm + gripper)
- Actions are automatically denormalized using the normalization statistics

#### Evaluation Metrics
- **Accuracy**: Discretized action prediction accuracy
- **MSE/MAE**: Continuous action prediction errors
- **Per-step Analysis**: Performance breakdown by prediction horizon

#### Output Files
- `predictions.csv`: Detailed predictions with ground truth
- `metrics.json`: Aggregated evaluation metrics
- `inference_log.txt`: Detailed inference logs

### Performance Optimization

#### GPU Memory Management
```bash
# For large models or limited GPU memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python main_eval.py \
    --checkpoint_path path/to/checkpoint.pth.tar \
    --eval_data_path path/to/data/ \
    --batch_size 1  # Reduce batch size for inference
```

#### Inference Speed
```bash
# Enable mixed precision for faster inference
python main_eval.py \
    --checkpoint_path path/to/checkpoint.pth.tar \
    --eval_data_path path/to/data/ \
    --use_amp  # If supported by the evaluation script
```

## Core Components and Architecture

### Model Architecture Overview

The VLA model consists of three main components:

1. **Vision-Language Model (VLM)**: PaliGemma backbone for multimodal understanding
2. **Action Head**: Flow matching network for continuous action prediction
3. **Integration Layer**: Combines VLM embeddings with robot state for action prediction

### Key Components

#### `PaliGemmaVLM` (`model/paligemma_vlm.py`)
- Wraps Hugging Face `PaliGemmaForConditionalGeneration`
- Processes image sequences and text prompts
- Outputs multimodal embeddings for each step
- Supports auxiliary camera inputs and vision resampling

#### `FlowmatchingActionHead` (`model/action_head/flow_matching.py`)
- Implements flow matching for continuous action spaces
- Supports multi-step action prediction (horizon-based)
- Uses Euler integration during inference
- Handles action normalization and discretization

#### `VLAModel` (`model/vla_model.py`)
- Integrates PaliGemmaVLM and ActionHead
- Manages forward pass from raw data to action predictions
- Supports both training (flow matching) and inference modes
- Handles device placement and mixed precision

#### `VLATrainerDDP` (`training/trainer_ddp.py`)
- Distributed training with PyTorch DDP
- Automatic gradient scaling and clipping
- WandB integration for experiment tracking
- Robust error handling for multi-GPU environments

#### `VLADataset` (`data/loader.py`)
- Parquet file loading with episode grouping
- SigLIP image processing for vision inputs
- Automatic normalization application
- Multi-step action sequence handling

### Training vs Inference Modes

#### Training Mode (Flow Matching)
- Uses `actions_gt_seq` parameter for ground truth guidance
- Applies flow matching loss for continuous action learning
- Supports teacher forcing for stable training

#### Inference Mode (Euler Integration)
- No ground truth actions provided
- Uses Euler integration for action prediction
- Generates multi-step action sequences autonomously

## Advanced Usage

### Custom Dataset Integration

#### 1. Prepare Your Data Format
Ensure your Parquet files contain the required fields:
```python
# Required columns in your Parquet files
required_columns = [
    'image_1_bytes',      # Main camera image (bytes)
    'image_2_bytes',      # Secondary camera (optional, bytes)
    'state',              # Robot state (list of floats)
    'action',             # Action vector (list of floats)
    'prompt',             # Text instruction (string)
    'is_first',           # Episode start flag (boolean)
    'is_last',            # Episode end flag (boolean)
    'is_terminal'         # Terminal state flag (boolean)
]
```

#### 2. Calculate Dataset Statistics
```bash
python utils/calculate_normalization_stats.py \
    --config_path configs/your_custom_config.yaml \
    --output_path your_dataset_normalization.json
```

#### 3. Create Custom Configuration
```yaml
# configs/your_custom_config.yaml
data:
  train_parquet_files: "/path/to/your/train/data/"
  val_parquet_files: "/path/to/your/val/data/"
  normalization_stats_path: "your_dataset_normalization.json"
  state_dim: 7                    # Match your robot's state dimensions
  max_seq_len: 4                  # Adjust based on your episodes

model:
  action_head_config:
    horizon: 8                    # Prediction horizon
    per_action_dim: 7             # Action dimensions per step
    action_dim: 56                # horizon × per_action_dim
```

### Fine-tuning Pre-trained Models

#### 1. Start from Pre-trained Checkpoint
```bash
python main_train_ddp.py \
    --config_path configs/vla_config_ddp.yaml \
    --resume_checkpoint path/to/pretrained/model.pth.tar \
    --experiment_name fine_tuned_model \
    --lr 1e-5  # Lower learning rate for fine-tuning
```

#### 2. Freeze Specific Components
```yaml
# In your config file
model:
  vlm_config:
    freeze_vision_tower: true     # Freeze vision encoder
    freeze_language_model: false  # Fine-tune language model
```

### Multi-Camera Setup

#### Configuration
```yaml
model:
  vlm_config:
    use_aux_camera: true          # Enable secondary camera
```

#### Data Preparation
Ensure both `image_1_bytes` and `image_2_bytes` are present in your Parquet files.

### Action Space Customization

#### Different Action Dimensions
```yaml
model:
  action_head_config:
    horizon: 4                    # Shorter prediction horizon
    per_action_dim: 6             # 6-DOF robot (no gripper)
    action_dim: 24                # 4 × 6
    num_action_bins: 512          # Adjust discretization resolution
```

#### Custom Action Bounds
The normalization statistics will automatically handle your action ranges. Ensure proper normalization calculation:
```bash
python utils/calculate_normalization_stats.py \
    --config_path configs/custom_action_config.yaml \
    --output_path custom_action_normalization.json
```

### Performance Tuning

#### Memory Optimization
```yaml
data:
  batch_size: 8                   # Reduce if memory limited
  num_workers: 4                  # Adjust based on CPU cores

model:
  vlm_config:
    dtype: "torch.bfloat16"       # Use mixed precision
```

#### Training Speed
```yaml
training:
  grad_clip_norm: 1.0             # Prevent gradient explosion
  log_interval: 100               # Reduce logging frequency

data:
  num_workers: 8                  # Increase for faster data loading
```

### Debugging and Monitoring

#### Check Model Weights
```bash
python diagnose_weights.py --checkpoint_path path/to/checkpoint.pth.tar
```

#### Analyze Training Data
```bash
python inspect_parquet.py --data_path /path/to/your/data/
```

#### Simple Analysis
```bash
python simple_analysis.py --config_path configs/vla_config_ddp.yaml
```

## Best Practices

### Training Recommendations

#### 1. Always Calculate Normalization Statistics First
```bash
# Calculate before any training
python utils/calculate_normalization_stats.py \
    --config_path configs/vla_config_ddp.yaml \
    --output_path normalization_stats.json
```

#### 2. Start with Single-GPU for Debugging
```bash
# Test your configuration first
python main_train.py --config_path configs/vla_config.yaml --epochs 2
```

#### 3. Use Multi-GPU for Production Training
```bash
# Scale up after validation
python main_train_ddp.py --config_path configs/vla_config_ddp.yaml
```

#### 4. Monitor Training Progress
- Enable WandB logging for experiment tracking
- Save checkpoints frequently (`save_every_n_epochs: 5`)
- Monitor both training and validation losses

#### 5. Learning Rate Strategy
- Start with `1e-4` for new models
- Use `1e-5` for fine-tuning pre-trained models
- Enable cosine annealing for stable convergence

### Data Preparation Best Practices

#### 1. Episode Structure
- Ensure proper `is_first`, `is_last`, `is_terminal` flags
- Maintain consistent episode lengths when possible
- Include diverse scenarios in your training data

#### 2. Image Quality
- Use consistent image resolution (224x224 recommended)
- Ensure good lighting and contrast
- Include multiple camera angles if available

#### 3. Action Space Design
- Normalize actions to reasonable ranges
- Use consistent action representations
- Include sufficient action diversity

### Performance Optimization

#### Memory Management
```yaml
# Optimize for available GPU memory
data:
  batch_size: 16                  # Adjust based on GPU memory
  num_workers: 4                  # Balance with CPU cores

model:
  vlm_config:
    dtype: "torch.bfloat16"       # Reduce memory usage
```

#### Multi-GPU Efficiency
```bash
# Set optimal NCCL settings
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
```

## Troubleshooting

### Common Training Issues

#### 1. High Validation Loss with Low Training Loss
**Symptoms**: Training loss decreases normally (e.g., 0.005) but validation loss remains high (e.g., 0.38)

**Solutions**:
- Check that training and validation use consistent parameters
- Ensure normalization statistics are applied correctly
- Verify data split quality and diversity
- Monitor for overfitting (reduce model complexity or add regularization)

#### 2. NCCL Timeout Errors (Multi-GPU)
**Symptoms**: 
```
RuntimeError: NCCL operation timed out
```

**Solutions**:
```bash
# Set environment variables
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Use diagnosis script
python diagnose_ddp.py
```

#### 3. Port Binding Issues
**Symptoms**:
```
RuntimeError: Address already in use
```

**Solutions**:
- The system automatically finds free ports
- Manually set port: `export MASTER_PORT=12355`
- Check for hanging processes: `ps aux | grep python`

#### 4. Out of Memory (OOM)
**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
```bash
# Reduce batch size
python main_train_ddp.py --config_path configs/vla_config_ddp.yaml --batch_size 8

# Use mixed precision
# Set in config: dtype: "torch.bfloat16"

# Reduce sequence length
# Set in config: max_seq_len: 2
```

#### 5. Data Loading Errors
**Symptoms**: Corrupt images, missing fields, slow loading

**Solutions**:
```bash
# Inspect your data
python inspect_parquet.py --data_path /path/to/your/data/

# Check data integrity
python data/loader.py  # Run the test script

# Reduce num_workers if seeing memory issues
# Set in config: num_workers: 2
```

### Model Architecture Issues

#### 1. Action Dimension Mismatch
**Symptoms**:
```
RuntimeError: size mismatch for action_head layers
```

**Solutions**:
- Ensure `action_dim = horizon × per_action_dim`
- Verify your data has the correct action dimensions
- Check normalization statistics match your action space

#### 2. Vision Processing Errors
**Symptoms**: Black images, incorrect image shapes

**Solutions**:
- Verify image preprocessing in VLADataset
- Check SigLIP processor compatibility
- Ensure images are RGB format

### Inference Issues

#### 1. Poor Prediction Quality
**Solutions**:
- Verify model was trained with proper normalization
- Check that inference uses the same normalization statistics
- Ensure input data matches training distribution

#### 2. Slow Inference Speed
**Solutions**:
```bash
# Use mixed precision
# Reduce batch size for inference
# Enable compilation (if supported)
```

### Getting Help

#### 1. Check Logs
- Training logs: `experiments/{experiment_name}/logs/`
- WandB dashboard for metrics visualization
- System logs for hardware issues

#### 2. Debug Tools
```bash
# GPU diagnosis
python diagnose_ddp.py

# Weight analysis
python diagnose_weights.py --checkpoint_path path/to/checkpoint.pth.tar

# Data inspection
python inspect_parquet.py --data_path /path/to/data/
```

#### 3. Validation Steps
```bash
# Test single batch
python simple_analysis.py --config_path configs/vla_config_ddp.yaml

# Test inference pipeline
python test_vla_offline.py --checkpoint_path path/to/checkpoint.pth.tar
```

## Version History and Updates

### Recent Improvements
- **Multi-GPU Training**: Stable distributed training with automatic port allocation
- **Flow Matching Architecture**: Advanced continuous action prediction
- **Normalization Pipeline**: Automatic statistics calculation and application
- **Training/Validation Consistency**: Fixed parameter consistency between training and validation
- **Inference Pipeline**: Complete rewrite for multi-step action prediction

### Architecture Evolution
- **v1.0**: Basic discrete action prediction
- **v2.0**: Flow matching with continuous actions
- **v2.1**: Multi-step horizon prediction
- **v2.2**: Distributed training support
- **v2.3**: Training/validation consistency fixes

## TODO / Future Work

*   **Enhanced Evaluation Metrics**: More comprehensive robotics-specific metrics
*   **Real-time Inference**: Optimizations for live robot control
*   **Multi-modal Extensions**: Support for additional sensor modalities
*   **Model Compression**: Quantization and pruning for deployment
*   **Simulation Integration**: Direct integration with robot simulators
*   **Transfer Learning**: Pre-trained models for different robot platforms 