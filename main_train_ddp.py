# Multi-GPU training script for VLA model using DistributedDataParallel
import os
import argparse
import logging
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.optim as optim
import yaml
import glob
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from data.loader import VLADataset, vla_collate_fn
from model.vla_model import VLAModel
from training.trainer_ddp import VLATrainerDDP
from utils.misc import setup_logging, load_checkpoint, save_checkpoint, get_lr_scheduler
from utils.config_utils import OmegaConfAttrDict


def get_parquet_files_from_path(path):
    """
    Get all parquet files from a path (file or directory).
    
    Args:
        path (str or list): Path to parquet file(s) or directory containing parquet files
        
    Returns:
        list: List of parquet file paths
    """
    # Convert OmegaConf ListConfig to regular Python list
    if hasattr(path, '__iter__') and not isinstance(path, str):
        # If it's a list-like object (including ListConfig), process each item
        all_files = []
        for p in path:
            all_files.extend(get_parquet_files_from_path(str(p)))  # Convert to string
        return all_files
    
    # Convert to string to handle any path-like objects
    path = str(path)
    
    if os.path.isfile(path) and path.endswith('.parquet'):
        # Single parquet file
        return [path]
    elif os.path.isdir(path):
        # Directory containing parquet files
        parquet_files = glob.glob(os.path.join(path, "*.parquet"))
        parquet_files.sort()  # Sort for consistent ordering
        return parquet_files
    else:
        # Try to find parquet files with pattern matching
        parquet_files = glob.glob(path)
        parquet_files = [f for f in parquet_files if f.endswith('.parquet')]
        parquet_files.sort()
        return parquet_files


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Optional: for complete reproducibility (may slow down training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def setup_ddp(rank, world_size, backend='nccl', master_port=None):
    """Initialize the distributed environment"""
    import socket
    
    os.environ['MASTER_ADDR'] = 'localhost'
    
    # Find an available port if not specified
    if master_port is None:
        # Try to find an available port starting from 12355
        for port in range(12355, 12400):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    master_port = str(port)
                    break
            except OSError:
                continue
        
        if master_port is None:
            raise RuntimeError("无法找到可用的端口进行分布式训练")
    
    os.environ['MASTER_PORT'] = str(master_port)
    
    # Set NCCL environment variables for better stability
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_TIMEOUT'] = '1800'  # 30 minutes timeout
    os.environ['NCCL_SOCKET_TIMEOUT'] = '600'  # 10 minutes socket timeout
    os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand if not available
    os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable P2P to avoid potential issues
    
    # Initialize the process group with timeout
    from datetime import timedelta
    dist.init_process_group(
        backend, 
        rank=rank, 
        world_size=world_size,
        timeout=timedelta(minutes=30)  # 30 minutes timeout
    )
    
    # Set the current CUDA device
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up the distributed environment"""
    dist.destroy_process_group()


def main_worker(rank, world_size, args, master_port=None):
    """Main worker function for each GPU process"""
    
    # Setup distributed training
    setup_ddp(rank, world_size, master_port=master_port)
    
    # Only setup logging on the main process (rank 0)
    if rank == 0:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)  # Only warnings and errors for other ranks
    
    try:
        # --- Load Configuration ---
        if args.config_path:
            try:
                with open(args.config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                config = OmegaConfAttrDict(config_dict)
                if rank == 0:
                    logger.info(f"Loaded configuration from: {args.config_path}")
            except Exception as e:
                if rank == 0:
                    logger.error(f"Error loading YAML config from {args.config_path}: {e}")
                return
        else:
            if rank == 0:
                logger.error("No configuration file provided. Exiting.")
            return

        # Override config with command-line arguments
        if args.batch_size: config.data.batch_size = args.batch_size
        if args.epochs: config.training.epochs = args.epochs
        if args.lr: config.optimizer.lr = args.lr
        if args.checkpoint_dir: config.training.checkpoint_dir = args.checkpoint_dir
        if args.experiment_name: config.training.experiment_name = args.experiment_name
        if args.model_path_override: config.model.vlm_config.model_name_or_path = args.model_path_override
        if args.train_data_override: config.data.train_parquet_files = args.train_data_override.split(',')
        if args.val_data_override: config.data.val_parquet_files = args.val_data_override.split(',')

        # Setup logging (only on main process)
        if rank == 0:
            log_dir = os.path.join(config.training.checkpoint_dir, config.training.experiment_name, "logs")
            os.makedirs(log_dir, exist_ok=True)  # Create log directory if it doesn't exist
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"train_log_ddp_{current_time}.log")
            logger = setup_logging(name=f"{config.training.experiment_name}_ddp", log_file=log_file, log_level=logging.INFO)
            logger.info(f"Multi-GPU training logger configured. Log file: {log_file}")
            logger.info(f"Training on {world_size} GPUs")

        # --- WandB Initialization (only on main process) ---
        config.wandb_enabled = False
        if args.use_wandb and rank == 0 and WANDB_AVAILABLE:
            try:
                run_id = wandb.util.generate_id()
                if args.resume_checkpoint and wandb.run is not None and wandb.run.resumed:
                    run_id = wandb.run.id
                
                wandb.init(
                    project=args.wandb_project_name or f"VLA_Project_{config.training.experiment_name}_DDP",
                    entity=args.wandb_entity,
                    name=f"{config.training.experiment_name}_ddp_{current_time}",
                    config=config_dict,
                    resume="allow",
                    id=run_id,
                    mode="offline"
                )
                if rank == 0:
                    logger.info(f"Weights & Biases initialized. Run name: {wandb.run.name}, Run ID: {wandb.run.id}")
                config.wandb_enabled = True
            except Exception as e:
                if rank == 0:
                    logger.error(f"Failed to initialize Weights & Biases: {e}. Proceeding without wandb.")
                config.wandb_enabled = False
        elif args.use_wandb and not WANDB_AVAILABLE and rank == 0:
            logger.warning("WandB requested but not available. Install with: pip install wandb")

        # Set device for this process
        device = torch.device(f'cuda:{rank}')
        if rank == 0:
            logger.info(f"Process {rank} using device: {device}")

        # Set random seeds for reproducibility
        seed = config.training.get('seed', 123) + rank  # Different seed for each process
        set_seed(seed)
        if rank == 0:
            logger.info(f"Random seed set to: {seed} (base seed + rank)")

        # --- Data Loading ---
        if rank == 0:
            logger.info("Initializing datasets and dataloaders...")

        common_dataset_params = {
            "processor_name_or_path": config.model.vlm_config.model_name_or_path,
            "max_seq_len": config.data.max_seq_len,
            "prompt_max_len": config.data.prompt_max_len,
            "action_dim": config.model.action_head_config.action_dim,  # Total action dim (horizon * per_action_dim)
            "state_dim": config.data.get('state_dim', 7),
            "logger": logger if rank == 0 else None,
            "use_siglip": True,  # VLADataset requires SigLIP for image processing
            "siglip_model_name": getattr(config.data, 'siglip_model_name', 'google/siglip-base-patch16-224'),
            "normalization_stats_path": config.data.get('normalization_stats_path', None),
            # Multi-step action prediction parameters
            "horizon": config.model.action_head_config.horizon,
            "per_action_dim": config.model.action_head_config.per_action_dim
        }

        # Training dataset
        # Get parquet files from path (supports both files and directories)
        train_parquet_files = get_parquet_files_from_path(config.data.train_parquet_files)
        if rank == 0:
            logger.info(f"Found {len(train_parquet_files)} training parquet files")
            logger.info(f"Training files: {train_parquet_files[:3]}{'...' if len(train_parquet_files) > 3 else ''}")
        
        train_dataset = VLADataset(
            parquet_files=train_parquet_files,
            **common_dataset_params
        )

        # Create distributed sampler
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True,
            drop_last=True
        )

        # Adjust batch size for distributed training
        per_gpu_batch_size = config.data.batch_size // world_size
        if config.data.batch_size % world_size != 0:
            if rank == 0:
                logger.warning(f"Batch size {config.data.batch_size} is not divisible by world size {world_size}. "
                             f"Using per-GPU batch size: {per_gpu_batch_size}")
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=per_gpu_batch_size,
            sampler=train_sampler,
            num_workers=config.data.num_workers,
            collate_fn=vla_collate_fn,
            pin_memory=True,
            drop_last=True
        )

        if rank == 0:
            logger.info(f"Training dataset initialized. Number of samples: {len(train_dataset)}")
            logger.info(f"Per-GPU batch size: {per_gpu_batch_size}, Total effective batch size: {per_gpu_batch_size * world_size}")

        # Validation dataset (if available)
        val_dataloader = None
        val_sampler = None
        if config.data.val_parquet_files:
            try:
                # Get parquet files from validation path
                val_parquet_files = get_parquet_files_from_path(config.data.val_parquet_files)
                if val_parquet_files:  # Check if we found any files
                    if rank == 0:
                        logger.info(f"Found {len(val_parquet_files)} validation parquet files")
                        logger.info(f"Validation files: {val_parquet_files[:3]}{'...' if len(val_parquet_files) > 3 else ''}")
                    
                    val_dataset = VLADataset(
                        parquet_files=val_parquet_files,
                        **common_dataset_params
                    )
                    
                    val_sampler = DistributedSampler(
                        val_dataset,
                        num_replicas=world_size,
                        rank=rank,
                        shuffle=False,
                        drop_last=False
                    )
                    
                    val_dataloader = DataLoader(
                        val_dataset,
                        batch_size=per_gpu_batch_size,
                        sampler=val_sampler,
                        num_workers=config.data.num_workers,
                        collate_fn=vla_collate_fn,
                        pin_memory=True,
                        drop_last=False
                    )
                    if rank == 0:
                        logger.info(f"Validation dataset initialized. Number of samples: {len(val_dataset)}")
                else:
                    if rank == 0:
                        logger.warning(f"No parquet files found in validation path: {config.data.val_parquet_files}")
            except Exception as e:
                if rank == 0:
                    logger.warning(f"Failed to load validation data from {config.data.val_parquet_files}: {e}")
        else:
            if rank == 0:
                logger.info("No validation data provided or path invalid. Skipping validation dataloader.")

        # --- Model Initialization ---
        if rank == 0:
            logger.info("Initializing VLAModel...")
        
        vla_model = VLAModel(config=config, model_logger=logger if rank == 0 else None).to(device)
        
        # Wrap model with DDP
        # Set find_unused_parameters=True to handle unused parameters in multi-step action prediction
        vla_model = DDP(vla_model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        
        if rank == 0:
            logger.info(f"VLAModel initialized and wrapped with DDP on device: {device}")

        # --- Optimizer and Scheduler ---
        if rank == 0:
            logger.info("Initializing optimizer and learning rate scheduler...")
        
        if config.optimizer.type.lower() == "adamw":
            optimizer = optim.AdamW(
                vla_model.parameters(), 
                lr=config.optimizer.lr, 
                weight_decay=config.optimizer.get('weight_decay', 0.01)
            )
        elif config.optimizer.type.lower() == "adam":
            optimizer = optim.Adam(
                vla_model.parameters(), 
                lr=config.optimizer.lr, 
                weight_decay=config.optimizer.get('weight_decay', 0.01)
            )
        else:
            if rank == 0:
                logger.warning(f"Unsupported optimizer type: {config.optimizer.type}. Defaulting to AdamW.")
            optimizer = optim.AdamW(
                vla_model.parameters(), 
                lr=config.optimizer.lr, 
                weight_decay=config.optimizer.get('weight_decay', 0.01)
            )
        
        if rank == 0:
            logger.info(f"Optimizer: {config.optimizer.type} with LR: {config.optimizer.lr}")

        # LR Scheduler
        lr_scheduler = get_lr_scheduler(
            optimizer, 
            config.lr_scheduler, 
            logger if rank == 0 else None, 
            steps_per_epoch=len(train_dataloader)
        )
        if lr_scheduler and rank == 0:
            logger.info(f"LR Scheduler: {config.lr_scheduler.type} initialized.")
        elif rank == 0:
            logger.info("No LR scheduler configured or type not supported.")

        # --- Checkpoint Loading (Resumption) ---
        start_epoch = 0
        resumed_best_metric = None

        if args.resume_checkpoint:
            if os.path.isfile(args.resume_checkpoint):
                if rank == 0:
                    logger.info(f"Resuming training from checkpoint: {args.resume_checkpoint}")
                
                # Load checkpoint (only on main process, then broadcast)
                map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                epoch_completed_from_ckpt, best_metric_from_ckpt = load_checkpoint(
                    model=vla_model.module,  # Use .module to access the underlying model
                    optimizer=optimizer,
                    filename=args.resume_checkpoint,
                    device=device,
                    strict=True
                )
                
                start_epoch = epoch_completed_from_ckpt + 1
                resumed_best_metric = best_metric_from_ckpt

                # Load LR scheduler state
                checkpoint_data = torch.load(args.resume_checkpoint, map_location=map_location)
                if lr_scheduler and 'lr_scheduler' in checkpoint_data:
                    try:
                        lr_scheduler.load_state_dict(checkpoint_data['lr_scheduler'])
                        if rank == 0:
                            logger.info("Successfully loaded LR scheduler state from checkpoint.")
                    except Exception as e:
                        if rank == 0:
                            logger.warning(f"Could not load LR scheduler state from checkpoint: {e}")

                if rank == 0:
                    logger.info(f"Resumed from checkpoint. Training will start at epoch {start_epoch}. "
                              f"Previous best metric: {resumed_best_metric if resumed_best_metric is not None else 'N/A'}")
            else:
                if rank == 0:
                    logger.warning(f"Resume checkpoint not found: {args.resume_checkpoint}. Starting from scratch.")
        else:
            if rank == 0:
                logger.info("No checkpoint provided for resumption. Starting training from scratch at epoch 0.")

        # Synchronize all processes
        dist.barrier()

        # --- Trainer Initialization and Training ---
        if rank == 0:
            logger.info("Initializing VLATrainerDDP...")
        
        trainer = VLATrainerDDP(
            model=vla_model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            train_sampler=train_sampler,
            val_sampler=val_sampler,
            config=config,
            device=device,
            rank=rank,
            world_size=world_size,
            logger=logger if rank == 0 else None,
            model_dtype=vla_model.module.paligemma_vlm.dtype
        )

        # Set best metric if resumed
        if resumed_best_metric is not None:
            trainer.best_metric = resumed_best_metric
            if rank == 0:
                logger.info(f"Trainer's best_metric initialized to {resumed_best_metric} from checkpoint.")

        if rank == 0:
            logger.info("VLATrainerDDP initialized.")

        try:
            if rank == 0:
                logger.info(f"Starting multi-GPU training from epoch {start_epoch + 1} for {config.training.epochs} total epochs...")
            trainer.train(start_epoch=start_epoch)
            if rank == 0:
                logger.info("Multi-GPU training completed successfully.")
        except Exception as e:
            if rank == 0:
                logger.error(f"An error occurred during training: {e}", exc_info=True)
                # Save crash checkpoint
                crash_checkpoint_dir = os.path.join(
                    config.training.checkpoint_dir, 
                    config.training.experiment_name, 
                    "crash_checkpoints"
                )
                os.makedirs(crash_checkpoint_dir, exist_ok=True)
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                crash_checkpoint_path = os.path.join(
                    crash_checkpoint_dir, 
                    f"crash_checkpoint_ddp_epoch_unknown_{current_time}.pth"
                )
                state = {
                    'epoch': start_epoch if 'start_epoch' in locals() else 0,
                    'state_dict': vla_model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_metric': 0.0
                }
                save_checkpoint(state, is_best=False, filename=crash_checkpoint_path)
                logger.info(f"Saved crash checkpoint to {crash_checkpoint_path}")
        finally:
            if config.wandb_enabled and wandb and wandb.run and rank == 0:
                wandb.finish()

    except Exception as e:
        if rank == 0:
            print(f"Error in main_worker: {e}")
        raise e
    finally:
        cleanup_ddp()


def main(args):
    """Main entry point for multi-GPU training"""
    import socket
    
    # Get the number of available GPUs
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("Multi-GPU training requires at least 2 GPUs. Falling back to single GPU training.")
        print("Please use main_train.py for single GPU training.")
        return
    
    print(f"Starting multi-GPU training on {world_size} GPUs")
    
    # Find an available port before spawning processes
    master_port = None
    for port in range(12355, 12400):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                master_port = port
                break
        except OSError:
            continue
    
    if master_port is None:
        print("错误: 无法找到可用的端口进行分布式训练")
        return
    
    print(f"使用端口 {master_port} 进行分布式训练")
    
    # Check for existing processes on the port (safer approach)
    print("检查端口使用情况...")
    try:
        import subprocess
        result = subprocess.run(["fuser", str(master_port) + "/tcp"], 
                              capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            print(f"端口 {master_port} 被进程占用: {result.stdout.strip()}")
            print("尝试使用其他端口...")
            # Find another port
            for port in range(master_port + 1, 12400):
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('localhost', port))
                        master_port = port
                        print(f"改用端口: {master_port}")
                        break
                except OSError:
                    continue
    except Exception as e:
        print(f"端口检查时出现警告: {e}，继续使用原端口")
    
    # Spawn processes for each GPU
    mp.spawn(main_worker, args=(world_size, args, master_port), nprocs=world_size, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VLA Model with Multi-GPU Support")
    parser.add_argument("--config_path", type=str, default="configs/vla_config.yaml", 
                       help="Path to the YAML configuration file.")
    
    # Command-line overrides
    parser.add_argument("--batch_size", type=int, help="Override total batch size (will be divided among GPUs).")
    parser.add_argument("--epochs", type=int, help="Override number of epochs.")
    parser.add_argument("--lr", type=float, help="Override learning rate.")
    parser.add_argument("--checkpoint_dir", type=str, help="Override checkpoint directory root.")
    parser.add_argument("--experiment_name", type=str, help="Override experiment name.")
    parser.add_argument("--resume_checkpoint", type=str, help="Path to checkpoint to resume training from.")
    parser.add_argument("--model_path_override", type=str, 
                       help="Override VLM model name or path (e.g., for PaliGemma).")
    parser.add_argument("--train_data_override", type=str, 
                       help="Comma-separated list of train Parquet files/dirs to override config.")
    parser.add_argument("--val_data_override", type=str, 
                       help="Comma-separated list of val Parquet files/dirs to override config.")
    
    # WandB arguments
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project_name", type=str, default=None, 
                       help="Weights & Biases project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, 
                       help="Weights & Biases entity (team name).")

    args = parser.parse_args()
    main(args)
