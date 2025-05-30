# Main script for training the VLA model 

import argparse
import yaml
import os
import torch
import torch.optim as optim
import random
import numpy as np
from datetime import datetime
import logging

from data.loader import VLADataset, vla_collate_fn
from torch.utils.data import DataLoader
from model.vla_model import VLAModel
from training.trainer import VLATrainer
from utils.misc import setup_logging, load_checkpoint, save_checkpoint, get_lr_scheduler
from utils.config_utils import OmegaConfAttrDict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Potentially slower, but good for reproducibility if needed
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def main(args):
    # --- Initial basic logger setup ---
    # This logger will be used until the full config is loaded and a more specific logger is set up.
    logger = logging.getLogger(__name__) # Use a named logger
    # Basic configuration for this initial logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Setup ---    
    if args.config_path:
        try:
            with open(args.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            # Convert to OmegaConf AttrDict for easier access (config.data.batch_size)
            config = OmegaConfAttrDict(config_dict) 
            logger.info(f"Loaded configuration from: {args.config_path}")
        except Exception as e:
            logger.error(f"Error loading YAML config from {args.config_path}: {e}")
            return
    else:
        logger.error("No configuration file provided. Exiting.")
        return

    # Override config with command-line arguments if any
    if args.batch_size: config.data.batch_size = args.batch_size
    if args.epochs: config.training.epochs = args.epochs
    if args.lr: config.optimizer.lr = args.lr
    if args.checkpoint_dir: config.training.checkpoint_dir = args.checkpoint_dir
    if args.experiment_name: config.training.experiment_name = args.experiment_name
    if args.model_path_override: config.model.vlm_config.model_name_or_path = args.model_path_override
    if args.train_data_override: config.data.train_parquet_files = args.train_data_override.split(',')
    if args.val_data_override: config.data.val_parquet_files = args.val_data_override.split(',')

    # Setup logging
    log_dir = os.path.join(config.training.checkpoint_dir, config.training.experiment_name, "logs")
    # Get a unique log file name using timestamp, or use a fixed name if preferred
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_log_{current_time}.log")
    # Reconfigure logger or get a new one based on config
    logger = setup_logging(name=config.training.experiment_name, log_file=log_file, log_level=logging.INFO) 
    logger.info(f"Main logger configured. Log file: {log_file}")
    # Pass this main logger to sub-modules for consistent logging

    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set random seeds for reproducibility
    seed = config.training.get('seed', 123)
    set_seed(seed)
    logger.info(f"Random seed set to: {seed}")

    # --- Data Loading --- #
    logger.info("Initializing datasets and dataloaders...")
    # For VLADataset, ensure all necessary config args are passed, including processor_name_or_path
    # and dimensions if needed by __init__.
    # The new VLADataset needs: processor_name_or_path, action_dim, state_dim from config.
    # These should be under config.data or config.model.action_head_config

    common_dataset_params = {
        "processor_name_or_path": config.model.vlm_config.model_name_or_path, # Crucial for AutoProcessor
        "max_seq_len": config.data.max_seq_len,
        "prompt_max_len": config.data.prompt_max_len,
        "action_dim": config.model.action_head_config.num_action_dims, # From action head config
        "state_dim": config.data.get('state_dim', 7), # Get state_dim from data config, default 7
        "logger": logger,
        "use_siglip": True,
        "siglip_model_name": getattr(config.data, 'siglip_model_name', 'google/siglip-base-patch16-224'),
    }

    train_dataset = VLADataset(
        parquet_files=config.data.train_parquet_files,
        **common_dataset_params
    )
    sample0 = train_dataset[0]
    state0 = sample0['state']            # 这是一个 Tensor，形状 (max_seq_len, D_state)
    print("state0 shape:", state0.shape)
    print("state0 data:\n", state0)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.data.batch_size, 
        shuffle=True, 
        num_workers=config.data.num_workers, 
        collate_fn=vla_collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    logger.info(f"Training dataset initialized. Number of samples: {len(train_dataset)}")

    val_dataloader = None
    if config.data.val_parquet_files and os.path.exists(config.data.val_parquet_files[0] if isinstance(config.data.val_parquet_files, list) else config.data.val_parquet_files ):
        val_dataset = VLADataset(
            parquet_files=config.data.val_parquet_files,
            **common_dataset_params
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=config.data.batch_size, 
            shuffle=False, 
            num_workers=config.data.num_workers, 
            collate_fn=vla_collate_fn,
            pin_memory=True if device.type == 'cuda' else False
        )
        logger.info(f"Validation dataset initialized. Number of samples: {len(val_dataset)}")
    else:
        logger.info("No validation data provided or path invalid. Skipping validation dataloader.")

    # --- Model Initialization --- #
    logger.info("Initializing VLAModel...")
    # VLAModel's __init__ now expects the full config object.
    vla_model = VLAModel(config=config, model_logger=logger).to(device)
    logger.info(f"VLAModel initialized and moved to device: {device}")

    # --- Optimizer and Scheduler --- #
    logger.info("Initializing optimizer and learning rate scheduler...")
    if config.optimizer.type.lower() == "adamw":
        optimizer = optim.AdamW(vla_model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.get('weight_decay', 0.01))
    elif config.optimizer.type.lower() == "adam":
        optimizer = optim.Adam(vla_model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.get('weight_decay', 0.01))
    else:
        logger.warning(f"Unsupported optimizer type: {config.optimizer.type}. Defaulting to AdamW.")
        optimizer = optim.AdamW(vla_model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.get('weight_decay', 0.01))
    logger.info(f"Optimizer: {config.optimizer.type} with LR: {config.optimizer.lr}")

    # LR Scheduler (example: CosineAnnealingLR, StepLR. Add more as needed from config)
    lr_scheduler = get_lr_scheduler(optimizer, config.lr_scheduler, logger, steps_per_epoch=len(train_dataloader))
    if lr_scheduler:
         logger.info(f"LR Scheduler: {config.lr_scheduler.type} initialized.")
    else:
        logger.info("No LR scheduler configured or type not supported.")

    # --- Checkpoint Loading (Resumption) --- #
    start_epoch = 0
    if args.resume_checkpoint:
        if os.path.isfile(args.resume_checkpoint):
            logger.info(f"Resuming training from checkpoint: {args.resume_checkpoint}")
            start_epoch, _ = load_checkpoint(args.resume_checkpoint, vla_model, optimizer, lr_scheduler, logger, device)
            logger.info(f"Resumed from epoch {start_epoch}")
        else:
            logger.warning(f"Resume checkpoint not found: {args.resume_checkpoint}. Starting from scratch.")
    else:
        logger.info("No checkpoint provided for resumption. Starting training from scratch.")

    # --- Trainer Initialization and Training --- #
    logger.info("Initializing VLATrainer...")
    trainer = VLATrainer(
        model=vla_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config, # Pass the full config
        device=device,
        logger=logger, # Pass the main logger
        model_dtype=vla_model.paligemma_vlm.dtype # Pass model's dtype for autocast
    )
    logger.info("VLATrainer initialized.")

    try:
        logger.info(f"Starting training from epoch {start_epoch + 1} for {config.training.epochs} epochs...")
        trainer.train(start_epoch=start_epoch) # VLATrainer's train method will handle epochs
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        # Optionally save a crash checkpoint
        crash_checkpoint_dir = os.path.join(config.training.checkpoint_dir, config.training.experiment_name, "crash_checkpoints")
        os.makedirs(crash_checkpoint_dir, exist_ok=True)
        crash_checkpoint_path = os.path.join(crash_checkpoint_dir, f"crash_checkpoint_epoch_unknown_{current_time}.pth")
        # 构造 state 字典
        state = {
            'epoch': start_epoch if 'start_epoch' in locals() else 0,
            'state_dict': vla_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_metric': 0.0
        }
        save_checkpoint(state, is_best=False, filename=crash_checkpoint_path)
        logger.info(f"Saved crash checkpoint to {crash_checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VLA Model")
    parser.add_argument("--config_path", type=str, default="configs/vla_config.yaml", help="Path to the YAML configuration file.")
    # Add command-line overrides for common parameters
    parser.add_argument("--batch_size", type=int, help="Override batch size.")
    parser.add_argument("--epochs", type=int, help="Override number of epochs.")
    parser.add_argument("--lr", type=float, help="Override learning rate.")
    parser.add_argument("--checkpoint_dir", type=str, help="Override checkpoint directory root.")
    parser.add_argument("--experiment_name", type=str, help="Override experiment name.")
    parser.add_argument("--resume_checkpoint", type=str, help="Path to checkpoint to resume training from.")
    parser.add_argument("--model_path_override", type=str, help="Override VLM model name or path (e.g., for PaliGemma).")
    parser.add_argument("--train_data_override", type=str, help="Comma-separated list of train Parquet files/dirs to override config.")
    parser.add_argument("--val_data_override", type=str, help="Comma-separated list of val Parquet files/dirs to override config.")

    args = parser.parse_args()
    
    # Initial base logger for setup until config is loaded - THIS BLOCK CAN BE REMOVED NOW
    # base_logger = logging.getLogger(__name__)
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main(args) 