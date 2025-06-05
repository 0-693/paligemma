# Training and fine-tuning logic will be implemented here 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm
import numpy as np
import logging
import torch.nn.functional as F
import wandb

# Assuming scripts like main_train.py are run from a context where 
# 'model', 'utils', and 'data' are top-level directories/packages.
from model.vla_model import VLAModel 
from utils.misc import save_checkpoint, load_checkpoint, setup_logging, discretize_actions
from data.loader import VLADataset, vla_collate_fn # For potential testing or direct use

class VLATrainer:
    def __init__(self, model, optimizer, lr_scheduler, train_dataloader, val_dataloader, 
                 config, device, logger=None, model_dtype=torch.float32):
        """
        Trainer for the Vision-Language-Action Model.
        Args:
            model (VLAModel): The VLA model instance.
            optimizer (torch.optim.Optimizer): The optimizer.
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
            train_dataloader (DataLoader): DataLoader for the training set.
            val_dataloader (DataLoader, optional): DataLoader for the validation set.
            config (dict): Configuration dictionary containing training parameters like:
                           'epochs', 'log_interval', 'checkpoint_dir', 'experiment_name', 
                           'grad_clip_norm', 'action_loss_weight', etc.
                           It should also contain 'num_action_dims' and 'num_action_bins' for loss calculation.
            device (torch.device): Device to train on ('cpu' or 'cuda').
            logger (logging.Logger, optional): Logger instance.
            model_dtype (torch.dtype): Model data type for automatic mixed precision (AMP) scaler.
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.logger = logger if logger else logging.getLogger(__name__)
        self.model_dtype = model_dtype

        # Extract necessary parameters from config for convenience
        self.epochs = self.config.training.epochs
        self.log_interval = self.config.training.log_interval
        self.checkpoint_dir = os.path.join(self.config.training.checkpoint_dir, self.config.training.experiment_name, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.grad_clip_norm = self.config.training.get('grad_clip_norm', 1.0)
        
        # Action discretization parameters (if needed for loss calculation, from VLAModel perspective actions are already discretized by dataset)
        # However, if labels are continuous and need discretization here:
        self.action_bounds = self.config.data.get('action_bounds', [-1.0, 1.0])
        self.num_action_bins = self.config.model.action_head_config.num_action_bins
        self.num_action_dims = self.config.model.action_head_config.num_action_dims # For assertion

        # AMP/bfloat16兼容性
        self.use_amp = (self.model_dtype == torch.float16 and self.device.type == 'cuda')
        self.use_bfloat16 = (self.model_dtype == torch.bfloat16 and self.device.type == 'cuda')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.logger.info(f"VLATrainer initialized. AMP scaler enabled: {self.scaler.is_enabled()}, use_bfloat16: {self.use_bfloat16}")
        self.best_val_loss = float('inf')
        self.wandb_enabled = config.get('wandb_enabled', False)

    def _compute_loss(self, action_pred, action_labels, vlm_attention_mask):
        """
        计算连续动作的MSE损失。
        Args:
            action_pred (torch.Tensor): (B, S, num_action_dims) 或 (B, S, D)
            action_labels (torch.Tensor): (B, S, num_action_dims) 或 (B, S, D)
            vlm_attention_mask (torch.Tensor): (B, S) boolean mask for valid sequence steps.
        """
        # 只对有效帧做MSE
        mask = vlm_attention_mask.unsqueeze(-1).float() # (B, S, 1)
        mse = (action_pred - action_labels) ** 2
        masked_mse = mse * mask
        loss = masked_mse.sum() / mask.sum().clamp(min=1)
        return loss

    def train_one_epoch(self, epoch_num):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch_num + 1}/{self.epochs} [Training]")

        for batch_idx, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()

            image_1_batch = batch['image_1'].to(self.device)
            raw_prompt_texts_batch = batch['raw_prompt_text']
            vlm_attention_mask = batch['vlm_attention_mask'].to(self.device)
            action_labels = batch['action'].to(self.device) # 直接用连续动作
            state_batch = batch.get('state', None)
            if state_batch is not None:
                state_batch = state_batch.to(self.device)
            image_2_batch = batch.get('image_2', None)
            if image_2_batch is not None:
                image_2_batch = image_2_batch.to(self.device)

            with torch.cuda.amp.autocast(enabled=(self.use_amp or self.use_bfloat16), dtype=self.model_dtype if self.device.type == 'cuda' else None):
                action_pred = self.model(
                    image_1_batch=image_1_batch,
                    raw_prompt_texts_batch=raw_prompt_texts_batch,
                    vlm_attention_mask_batch=vlm_attention_mask,
                    state_batch=state_batch,
                    image_2_batch=image_2_batch 
                )
                loss = self._compute_loss(action_pred, action_labels, vlm_attention_mask)

            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(f"NaN or Inf loss detected at epoch {epoch_num+1}, batch {batch_idx}. Skipping batch.")
                del loss, action_pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            total_loss += loss.item()
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(loss=loss.item(), avg_loss=total_loss / (batch_idx + 1), lr=current_lr)

            # Log to wandb (per batch/step)
            if self.wandb_enabled and (batch_idx + 1) % self.log_interval == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/avg_loss_so_far": total_loss / (batch_idx + 1),
                    "train/learning_rate": current_lr,
                    "train/epoch": epoch_num + 1,
                    "train/step": epoch_num * len(self.train_dataloader) + batch_idx 
                })

            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss_so_far = total_loss / (batch_idx + 1)
                self.logger.info(f"Epoch {epoch_num + 1}/{self.epochs}, Batch {batch_idx + 1}/{len(self.train_dataloader)}, Train Loss: {loss.item():.4f}, Avg Train Loss: {avg_loss_so_far:.4f}, LR: {current_lr:.2e}")
        avg_epoch_loss = total_loss / len(self.train_dataloader)
        self.logger.info(f"Epoch {epoch_num + 1} Training Summary: Average Loss: {avg_epoch_loss:.4f}")
        # Log to wandb (per epoch)
        if self.wandb_enabled:
            wandb.log({
                "train/epoch_loss": avg_epoch_loss,
                "train/epoch": epoch_num + 1
            })
        return avg_epoch_loss

    def validate_one_epoch(self, epoch_num):
        if self.val_dataloader is None:
            self.logger.info("No validation dataloader provided. Skipping validation.")
            return None

        self.model.eval()
        total_val_loss = 0
        progress_bar = tqdm(self.val_dataloader, desc=f"Epoch {epoch_num + 1}/{self.epochs} [Validation]")

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                image_1_batch = batch['image_1'].to(self.device)
                raw_prompt_texts_batch = batch['raw_prompt_text']
                vlm_attention_mask = batch['vlm_attention_mask'].to(self.device)
                action_labels = batch['action'].to(self.device)
                state_batch = batch.get('state', None)
                if state_batch is not None:
                    state_batch = state_batch.to(self.device)
                image_2_batch = batch.get('image_2', None)
                if image_2_batch is not None:
                    image_2_batch = image_2_batch.to(self.device)

                with torch.cuda.amp.autocast(enabled=(self.use_amp or self.use_bfloat16), dtype=self.model_dtype if self.device.type == 'cuda' else None):
                    action_pred = self.model(
                        image_1_batch=image_1_batch,
                        raw_prompt_texts_batch=raw_prompt_texts_batch,
                        vlm_attention_mask_batch=vlm_attention_mask,
                        state_batch=state_batch,
                        image_2_batch=image_2_batch
                    )
                    loss = self._compute_loss(action_pred, action_labels, vlm_attention_mask)
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"NaN or Inf validation loss detected at epoch {epoch_num+1}, batch {batch_idx}. Skipping batch loss accumulation for this one.")
                    continue
                total_val_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item(), avg_loss=total_val_loss / (batch_idx + 1))
        avg_val_loss = total_val_loss / len(self.val_dataloader) if len(self.val_dataloader) > 0 else float('inf')
        self.logger.info(f"Epoch {epoch_num + 1} Validation Summary: Average Loss: {avg_val_loss:.4f}")
        # Log to wandb (per epoch)
        if self.wandb_enabled and self.val_dataloader is not None: # 确保有验证数据
            wandb.log({
                "val/epoch_loss": avg_val_loss,
                "val/epoch": epoch_num + 1
            })
        return avg_val_loss

    def train(self, start_epoch=0):
        self.logger.info(f"Starting training loop from epoch {start_epoch + 1} to {self.epochs}.")
        if self.wandb_enabled and hasattr(wandb, 'watch'): # 监视模型梯度和参数（可选）
            wandb.watch(self.model, log="all", log_freq=self.log_interval * 10, log_graph=False) # 每 N 个 batch 记录一次

        for epoch in range(start_epoch, self.epochs):
            self.logger.info(f"--- Epoch {epoch + 1}/{self.epochs} ---")
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate_one_epoch(epoch)

            if self.lr_scheduler:
                # Step scheduler based on its type (epoch-wise or validation loss based)
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loss is not None:
                        self.lr_scheduler.step(val_loss)
                        self.logger.info(f"Stepped LR scheduler with validation loss: {val_loss:.4f}")
                    else:
                        self.logger.warning("ReduceLROnPlateau scheduler needs validation loss, but it's None. Skipping scheduler step.")
                else:
                    self.lr_scheduler.step()
                    self.logger.info("Stepped LR scheduler (epoch-wise).")

            # Save checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            state = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            }
            save_checkpoint(state, is_best=False, filename=checkpoint_path)

            # Save best model based on validation loss
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint_best_val.pth")
                save_checkpoint(state, is_best=True, filename=best_checkpoint_path)
                self.logger.info(f"Saved new best validation model (Epoch {epoch + 1}, Val Loss: {val_loss:.4f}) to {best_checkpoint_path}")
            elif val_loss is None and self.val_dataloader is None: # If no validation, save based on train loss (e.g. save every N epochs or last one)
                 # For now, just saving epoch checkpoint is enough. Could add logic to save based on train_loss if desired.
                 pass 

        self.logger.info("Training finished.")

    def load_model_checkpoint(self, checkpoint_path):
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        # Create a temporary optimizer of the same type as the one used during training if available
        # This is needed because load_checkpoint might try to load optimizer state
        temp_optimizer = None
        if self.optimizer: # If an optimizer is already configured for the trainer
            temp_optimizer = self.optimizer
        elif 'optimizer_config' in self.config: # Try to create one from config
            opt_conf = self.config['optimizer_config']
            optimizer_class = getattr(optim, opt_conf.get('type', 'Adam'))
            temp_optimizer = optimizer_class(self.model.parameters(), lr=opt_conf.get('lr', 1e-4))
        
        loaded_epoch, best_metric = load_checkpoint(self.model, temp_optimizer, checkpoint_path, self.device)
        self.start_epoch = loaded_epoch
        self.best_val_metric = best_metric
        
        # If the trainer had an optimizer, its state should now be loaded by load_checkpoint
        # If a temp_optimizer was created and the trainer didn't have one, 
        # we might want to assign this loaded optimizer to self.optimizer if it makes sense for the workflow.
        # For now, we assume self.optimizer is either pre-set or re-initialized after loading a model for fine-tuning.
        if self.optimizer is None and temp_optimizer is not None and 'optimizer' in self.config: 
             self.logger.info("Trainer optimizer was None. Loaded optimizer state into a temporary one.")
        
        if self.lr_scheduler and 'lr_scheduler' in torch.load(checkpoint_path, map_location='cpu'):
            try:
                self.lr_scheduler.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['lr_scheduler'])
                self.logger.info("Loaded LR scheduler state from checkpoint.")
            except Exception as e:
                self.logger.warning(f"Could not load LR scheduler state: {e}. Scheduler may be re-initialized.")


# Example Usage (Conceptual - requires actual data and full config)
if __name__ == '__main__':
    print("Conceptual VLATrainer Test")
    # 1. Setup Configurations (normally from YAML or args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        'epochs': 3,
        'log_interval': 2,
        'checkpoint_dir': './test_vla_checkpoints',
        'experiment_name': 'test_vla_exp',
        'grad_clip_norm': 1.0,
        'action_bounds': (-1.0, 1.0),
        'save_checkpoint_interval': 1,
        'vlm_config': {
            'model_name_or_path': "google/paligemma-3b-pt-224", # Use a tiny model for actual test
            'use_secondary_camera': False,
        },
        'action_head_config': {
            'state_dim': 7,
            'num_action_dims': 7,
            'num_action_bins': 20, # Smaller for faster test
            'use_state_input': True,
            'hidden_layers_config': [64, 32],
        },
        'optimizer_config': {
            'type': 'AdamW',
            'lr': 1e-4,
            'weight_decay': 1e-2
        },
        'lr_scheduler_config': {
            'type': 'StepLR',
            'step_size': 1,
            'gamma': 0.8
        },
        'data_config': {
            'train_parquet_files': ['dummy_train_data.parquet'], # Need to create these
            'val_parquet_files': ['dummy_val_data.parquet'],
            'tokenizer_name_or_path': 'bert-base-uncased', # Placeholder
            'image_processor_name_or_path': 'google/vit-base-patch16-224-in21k', # Placeholder
            'max_seq_len': 5,
            'prompt_max_len': 10,
            'batch_size': 2,
            'num_workers': 0
        }
    }
    logger = setup_logging(log_file=os.path.join(config['checkpoint_dir'], f"{config['experiment_name']}.log"))

    # 2. Create Dummy Data and DataLoaders (mimicking data/loader.py)
    def create_dummy_pq_for_trainer(file_path, num_seq=5, max_frames=8, num_action_dims=7):
        # Simplified from data/loader.py for this test
        data = []
        from PIL import Image; import io; import pandas as pd; import numpy as np # Local import for this func
        dummy_img_byte_stream = io.BytesIO()
        Image.new('RGB', (10, 10), color = 'red').save(dummy_img_byte_stream, format='PNG')
        dummy_img_bytes = dummy_img_byte_stream.getvalue()
        for _ in range(num_seq):
            n_frames = np.random.randint(2, max_frames + 1)
            for i in range(n_frames):
                data.append({
                    'image_1': dummy_img_bytes, 'image_2': dummy_img_bytes,
                    'state': np.random.rand(num_action_dims).astype(np.float32) * 2 - 1,
                    'action': np.random.rand(num_action_dims).astype(np.float32) * 2 - 1,
                    'is_first': i == 0, 'is_last': i == n_frames - 1, 'is_terminal': False,
                    'prompt': 'test prompt'
                })
        pd.DataFrame(data).to_parquet(file_path)
        logger.info(f"Created dummy parquet: {file_path}")

    os.makedirs(os.path.dirname(config['data_config']['train_parquet_files'][0]), exist_ok=True)
    create_dummy_pq_for_trainer(config['data_config']['train_parquet_files'][0], num_action_dims=config['action_head_config']['num_action_dims'])
    create_dummy_pq_for_trainer(config['data_config']['val_parquet_files'][0], num_action_dims=config['action_head_config']['num_action_dims'])
    
    train_dataset = VLADataset(**config['data_config'], parquet_files=config['data_config']['train_parquet_files'])
    val_dataset = VLADataset(**config['data_config'], parquet_files=config['data_config']['val_parquet_files'])
    train_dl = DataLoader(train_dataset, batch_size=config['data_config']['batch_size'], collate_fn=vla_collate_fn, num_workers=config['data_config']['num_workers'], shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=config['data_config']['batch_size'], collate_fn=vla_collate_fn, num_workers=config['data_config']['num_workers'])

    # 3. Initialize Model, Optimizer, Scheduler
    # IMPORTANT: The PaliGemmaVLM part will try to download a large model by default.
    # For a real test, you'd mock this or use a tiny pre-trained model.
    try:
        vla_model_instance = VLAModel(config['vlm_config'], config['action_head_config'], device, dtype=torch.float32 if device==torch.device('cpu') else torch.float16)
        
        opt_conf = config['optimizer_config']
        optimizer_class = getattr(optim, opt_conf.get('type', 'AdamW'))
        optimizer_instance = optimizer_class(vla_model_instance.parameters(), lr=opt_conf['lr'], weight_decay=opt_conf.get('weight_decay', 0))
        
        scheduler_instance = None
        if 'lr_scheduler_config' in config and config['lr_scheduler_config']:
            sched_conf = config['lr_scheduler_config']
            scheduler_class = getattr(optim.lr_scheduler, sched_conf.get('type', 'StepLR'))
            scheduler_instance = scheduler_class(optimizer_instance, step_size=sched_conf['step_size'], gamma=sched_conf['gamma'])

        # 4. Initialize Trainer
        trainer = VLATrainer(vla_model_instance, optimizer_instance, scheduler_instance, train_dl, val_dl, config, device, logger)

        # 5. (Optional) Load checkpoint if resuming
        # trainer.load_model_checkpoint('path_to_checkpoint.pth.tar')

        # 6. Start Training
        logger.info("Starting conceptual training run...")
        trainer.train()
        logger.info("Conceptual training run finished.")

    except Exception as e:
        logger.error(f"Error during VLATrainer conceptual test: {e}", exc_info=True)
        logger.error("This test (especially model loading) might require specific setup or a mock model for PaliGemmaVLM.")
    finally:
        # Clean up dummy files
        import shutil
        if os.path.exists(config['checkpoint_dir']): shutil.rmtree(config['checkpoint_dir'])
        if os.path.exists(config['data_config']['train_parquet_files'][0]): os.remove(config['data_config']['train_parquet_files'][0])
        if os.path.exists(config['data_config']['val_parquet_files'][0]): os.remove(config['data_config']['val_parquet_files'][0])
        logger.info("Cleaned up dummy files and directories.") 