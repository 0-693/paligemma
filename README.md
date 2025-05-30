# PaliGemma-VLA: Vision-Language-Action Model

This project implements a Vision-Language-Action (VLA) model using the PaliGemma architecture
as its backbone. It is designed for tasks that require understanding visual input and 
natural language instructions to predict a sequence of actions. The project structure and
engineering principles are inspired by and aim for compatibility with projects like `miravla`
and `RoboVLMs` (specifically `robopaligemma.py`).

## Features

*   **PaliGemma Backbone**: Utilizes a pre-trained PaliGemma model for multimodal feature extraction.
*   **Flexible Data Loading**: Supports Parquet-based datasets with variable-length action sequences, including multiple camera inputs and robot state information.
*   **Discrete Action Prediction**: Implements an action head to predict discretized action tokens for multiple action dimensions.
*   **Modular Design**: Separates concerns into data loading, model components (VLM, action head, integrated VLA model), training, and inference.
*   **Configurable Training and Evaluation**: Uses YAML configuration files for easy management of hyperparameters and experiment settings.
*   **Checkpointing**: Saves and loads model checkpoints for resuming training and for inference.
*   **Inference Support**: Provides scripts for both batch evaluation on datasets and single-item inference.
*   **Metrics Calculation**: Computes accuracy and MSE/MAE for action predictions during evaluation.

## Directory Structure

```
paligemma-VLA/
├── configs/                  # YAML configuration files
│   └── vla_config.yaml       # Example main configuration file
├── data/
│   └── loader.py             # VLADataset, VLAImageProcessor, collate_fn
├── model/
│   ├── __init__.py
│   ├── paligemma_vlm.py      # PaliGemmaVLM backbone class
│   ├── action_head.py        # ActionHead class
│   └── vla_model.py          # VLAModel integrating VLM and ActionHead
├── training/
│   └── trainer.py            # VLATrainer class
├── inference/
│   └── predictor.py          # VLAPredictor class
├── utils/
│   └── misc.py               # Utility functions (logging, checkpoints, action discretization)
├── main_train.py             # Main script for training
├── main_eval.py              # Main script for evaluation and inference
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone ...
    cd paligemma-VLA
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate 
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    You might also need to install PyTorch separately according to your CUDA version. Visit [https://pytorch.org/](https://pytorch.org/) for instructions.

4.  **Hugging Face Hub Login (Optional, for private models or to avoid rate limits):**
    If you are using gated models or want to ensure smooth downloads from Hugging Face Hub:
    ```bash
    huggingface-cli login
    ```

## Data Preparation

The model expects data in Parquet files. Each row in a Parquet file typically represents one time step in an action sequence/episode.

*   **Key Data Fields**: 
    *   `image_1`: Bytes of the main RGB image for the current step.
    *   `image_2`: (Optional) Bytes of a secondary (e.g., wrist) RGB image.
    *   `state`: Robot state vector (e.g., joint positions, gripper status) as a list/array of floats.
    *   `action`: Action vector for the current step as a list/array of floats.
    *   `is_first`: Boolean, true if this is the first step of an episode.
    *   `is_last`: Boolean, true if this is the last step of an episode (for sampling fixed-length windows).
    *   `is_terminal`: Boolean, true if this step is a terminal state in the episode.
    *   `prompt`: Natural language instruction string, typically associated with the start of an episode.

*   **Sequence Handling**: The `VLADataset` class groups rows into episodes based on the `is_first` flag and then samples fixed-length windows (`max_seq_len`) from these episodes. Shorter episodes are padded.

## Configuration (`configs/vla_config.yaml`)

Training and model parameters are controlled via a YAML configuration file (e.g., `configs/vla_config.yaml`). 

Key sections in the configuration file:

*   **`data`**: Parameters for data loading.
    *   `train_parquet_files`: List of paths to training Parquet files.
    *   `val_parquet_files`: (Optional) List of paths to validation Parquet files.
    *   `tokenizer_name_or_path`: Hugging Face tokenizer name or path (e.g., "bert-base-uncased").
    *   `image_processor_name_or_path`: Hugging Face image processor name or path (e.g., "google/paligemma-3b-pt-224") or path to a compatible processor config.
    *   `max_seq_len`: Maximum sequence length for model input (padding/truncation).
    *   `prompt_max_len`: Maximum length for tokenized prompts.
    *   `batch_size`: Training and evaluation batch size.
    *   `num_workers`: Number of workers for DataLoader.
    *   `action_bounds`: Tuple `(min_val, max_val)` for continuous action values, used for discretization.

*   **`model`**: Configuration for the VLA model architecture.
    *   **`vlm_config`**: Parameters for the `PaliGemmaVLM` backbone.
        *   `model_name_or_path`: Hugging Face model name (e.g., "google/paligemma-3b-pt-224").
        *   `use_aux_camera`: Boolean, whether to use `image_2`.
        *   `freeze_vision_tower`: Boolean, whether to freeze vision encoder weights.
        *   `freeze_language_model`: Boolean, whether to freeze language model weights.
        *   `dtype`: Model data type (e.g., "torch.float16", "torch.float32").
    *   **`action_head_config`**: Parameters for the `ActionHead`.
        *   `use_state_input`: Boolean, whether to concatenate robot state with VLM embeddings.
        *   `num_action_dims`: Number of dimensions in the action vector (e.g., 7 for a 7-DOF arm).
        *   `num_action_bins`: Number of discrete bins per action dimension.
        *   `mlp_hidden_dims`: List of hidden layer sizes for the MLP in the action head (e.g., `[512, 256]`). Can be empty.
        *   `dropout_prob`: Dropout probability for the action head MLP.

*   **`training`**: Parameters for the training process.
    *   `epochs`: Number of training epochs.
    *   `log_interval`: Log training progress every N batches.
    *   `checkpoint_dir`: Root directory to save model checkpoints and logs.
    *   `experiment_name`: Subdirectory under `checkpoint_dir` for this specific experiment.
    *   `grad_clip_norm`: Maximum norm for gradient clipping.
    *   `seed`: Random seed for reproducibility.

*   **`optimizer`**: Optimizer configuration.
    *   `type`: Optimizer type (e.g., "AdamW", "SGD").
    *   `lr`: Learning rate.
    *   `weight_decay`: Weight decay.

*   **`lr_scheduler`**: (Optional) Learning rate scheduler configuration.
    *   `type`: Scheduler type (e.g., "StepLR", "CosineAnnealingLR").
    *   Other scheduler-specific parameters.

## Running the Code

### 1. Training

Modify `configs/vla_config.yaml` with your dataset paths and desired hyperparameters.

```bash
python main_train.py --config_path configs/vla_config.yaml
```

**Command-line overrides:**
Many parameters from the config file can be overridden via command-line arguments. For example:

```bash
python main_train.py --config_path configs/vla_config.yaml \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --experiment_name my_experiment_run \
    --model_name_or_path google/paligemma-3b-pt-448 # if using a different model size
```

To resume training from a checkpoint:
```bash
python main_train.py --config_path configs/vla_config.yaml --resume_checkpoint <path_to_checkpoint.pth.tar>
```

### 2. Evaluation / Inference

**Batch Evaluation on a Dataset:**

```bash
python main_eval.py --checkpoint_path <path_to_your_trained_checkpoint.pth.tar> \
    --eval_data_path /path/to/your/eval_data_1.parquet /path/to/your/eval_data_2.parquet \
    --output_dir ./results/my_eval \
    --save_predictions
```

*   `--checkpoint_path`: Path to the trained model checkpoint.
*   `--eval_data_path`: One or more paths to Parquet files for evaluation.
*   `--output_dir`: Directory to save metrics and predictions.
*   `--save_predictions`: If specified, saves detailed predictions to a CSV file.

**Single Item Inference:**

```bash
python main_eval.py --checkpoint_path <path_to_your_trained_checkpoint.pth.tar> \
    --image1_paths /path/to/img1_step1.jpg /path/to/img1_step2.jpg \
    --prompt "your robot instruction here" \
    --image2_paths /path/to/img2_step1.jpg /path/to/img2_step2.jpg  # Optional
    --state_vector 0.1 0.2 0.3 0.4 0.5 0.6 0.7                     # Optional, ensure 7 values for 7-dim state
    --output_dir ./results/single_inference \
    --save_predictions
```

*   `--image1_paths`: List of paths to main camera images (sequence).
*   `--prompt`: The text prompt.
*   `--image2_paths`: (Optional) List of paths to auxiliary camera images.
*   `--state_vector`: (Optional) List of floats representing the robot state.

## Core Components

*   **`PaliGemmaVLM` (`model/paligemma_vlm.py`)**: Wraps a Hugging Face `PaliGemmaForConditionalGeneration` model. It processes sequences of images and a text prompt, outputting multimodal embeddings for each step in the sequence.
*   **`ActionHead` (`model/action_head.py`)**: An MLP-based module that takes VLM embeddings (and optionally robot state) as input and predicts logits for discretized action bins.
*   **`VLAModel` (`model/vla_model.py`)**: Integrates `PaliGemmaVLM` and `ActionHead`. Handles the forward pass from raw data to action logits.
*   **`VLATrainer` (`training/trainer.py`)**: Manages the training and validation loops, including loss computation, optimization, checkpointing, and logging.
*   **`VLAPredictor` (`inference/predictor.py`)**: Loads a trained `VLAModel` from a checkpoint and provides methods for preprocessing input and performing inference to get action predictions.
*   **`VLADataset` (`data/loader.py`)**: Handles loading and preprocessing of data from Parquet files, including image processing, tokenization, and padding.

## TODO / Future Work

*   Implement more comprehensive unit and integration tests.
*   Add support for more advanced logging/monitoring tools (e.g., TensorBoard, W&B).
*   Explore options for distributed training.
*   Extend evaluation with more sophisticated robotics metrics. 