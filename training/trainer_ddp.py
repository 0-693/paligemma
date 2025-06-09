# Distributed training logic for VLA model using PyTorch DDP
import os
import torch
import torch.distributed as dist
import logging
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from utils.misc import save_checkpoint, load_checkpoint, discretize_actions


class VLATrainerDDP:
    """
    Distributed trainer for the Vision-Language-Action Model using PyTorch DDP.
    """
    
    def __init__(self, model, optimizer, lr_scheduler, train_dataloader, val_dataloader,
                 train_sampler, val_sampler, config, device, rank, world_size, 
                 logger=None, model_dtype=torch.float32):
        """
        Initialize the distributed trainer.
        
        Args:
            model (torch.nn.parallel.DistributedDataParallel): DDP-wrapped VLA model
            optimizer (torch.optim.Optimizer): The optimizer
            lr_scheduler: Learning rate scheduler
            train_dataloader (DataLoader): Training dataloader with DistributedSampler
            val_dataloader (DataLoader): Validation dataloader with DistributedSampler
            train_sampler (DistributedSampler): Training sampler for epoch shuffling
            val_sampler (DistributedSampler): Validation sampler
            config: Configuration object
            device (torch.device): Device for this process
            rank (int): Process rank (GPU ID)
            world_size (int): Total number of processes
            logger: Logger instance (only used on rank 0)
            model_dtype: Model data type for autocast
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        self.logger = logger if self.is_main_process else logging.getLogger(__name__)
        self.model_dtype = model_dtype

        # Extract parameters from config
        self.epochs = self.config.training.epochs
        self.log_interval = self.config.training.log_interval
        self.grad_clip_norm = self.config.training.get('grad_clip_norm', 1.0)
        self.save_checkpoint_interval = self.config.training.get('save_checkpoint_interval', 1)
        
        # Checkpoint directory (only create on main process)
        if self.is_main_process:
            self.checkpoint_dir = os.path.join(
                self.config.training.checkpoint_dir, 
                self.config.training.experiment_name, 
                "checkpoints"
            )
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        else:
            self.checkpoint_dir = None

        # Action parameters
        self.action_bounds = self.config.data.get('action_bounds', [-1.0, 1.0])
        self.num_action_bins = self.config.model.action_head_config.num_action_bins
        self.num_action_dims = self.config.model.action_head_config.num_action_dims

        # AMP/bfloat16 compatibility
        self.use_amp = (self.model_dtype == torch.float16 and self.device.type == 'cuda')
        self.use_bfloat16 = (self.model_dtype == torch.bfloat16 and self.device.type == 'cuda')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        if self.is_main_process:
            self.logger.info(f"VLATrainerDDP initialized on {world_size} GPUs. "
                           f"AMP scaler enabled: {self.scaler.is_enabled()}, "
                           f"use_bfloat16: {self.use_bfloat16}")
        
        self.best_val_loss = float('inf')
        self.best_metric = 0.0
        self.wandb_enabled = config.get('wandb_enabled', False) and self.is_main_process and WANDB_AVAILABLE

    def _compute_loss(self, action_pred, action_labels, vlm_attention_mask):
        """
        Compute MSE loss for continuous multi-step actions.
        
        Args:
            action_pred (torch.Tensor): (B, action_dim) predicted multi-step actions
            action_labels (torch.Tensor): (B, action_dim) ground truth multi-step actions
            vlm_attention_mask (torch.Tensor): (B,) or (B, S) - not used for single-step prediction
        """
        # For multi-step action prediction, we compute simple MSE loss
        mse = (action_pred - action_labels) ** 2
        loss = mse.mean()
        return loss

    def _all_reduce_tensor(self, tensor):
        """All-reduce a tensor across all processes with error handling"""
        if self.world_size > 1:
            try:
                # Add timeout for all-reduce operation
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor /= self.world_size
            except Exception as e:
                if self.is_main_process:
                    self.logger.error(f"All-reduce operation failed: {e}")
                # Return original tensor if all-reduce fails
                pass
        return tensor

    def train_one_epoch(self, epoch_num):
        """Train for one epoch"""
        self.model.train()
        
        # Set epoch for sampler to ensure proper shuffling
        if self.train_sampler:
            self.train_sampler.set_epoch(epoch_num)
        
        total_loss = 0
        num_batches = len(self.train_dataloader)
        
        # Only show progress bar on main process
        if self.is_main_process:
            progress_bar = tqdm(self.train_dataloader, 
                              desc=f"Epoch {epoch_num + 1}/{self.epochs} [Training]")
        else:
            progress_bar = self.train_dataloader

        for batch_idx, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()

            # Move batch to device
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

            # Forward pass with autocast
            with torch.cuda.amp.autocast(enabled=(self.use_amp or self.use_bfloat16), 
                                        dtype=self.model_dtype if self.device.type == 'cuda' else None):
                # Training mode: use flow matching with ground truth actions
                action_pred = self.model(
                    image_1_batch=image_1_batch,
                    raw_prompt_texts_batch=raw_prompt_texts_batch,
                    vlm_attention_mask_batch=vlm_attention_mask,
                    state_batch=state_batch,
                    image_2_batch=image_2_batch,
                    actions_gt_seq=action_labels
                )
                
                loss = self._compute_loss(action_pred, action_labels, vlm_attention_mask)

            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                if self.is_main_process:
                    self.logger.warning(f"NaN or Inf loss detected at batch {batch_idx}. Skipping batch.")
                continue

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            # Update learning rate scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()

            total_loss += loss.item()

            # Update progress bar on main process
            if self.is_main_process:
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix(
                    loss=loss.item(), 
                    avg_loss=total_loss / (batch_idx + 1), 
                    lr=current_lr
                )

                # Log to wandb (per batch/step)
                if self.wandb_enabled and (batch_idx + 1) % self.log_interval == 0:
                    step = epoch_num * num_batches + batch_idx + 1
                    if WANDB_AVAILABLE:
                        wandb.log({
                            "train/batch_loss": loss.item(),
                            "train/lr": current_lr,
                            "train/step": step
                        })

                if (batch_idx + 1) % self.log_interval == 0:
                    self.logger.info(
                        f"Epoch {epoch_num + 1}/{self.epochs}, "
                        f"Batch {batch_idx + 1}/{num_batches}, "
                        f"Loss: {loss.item():.4f}, "
                        f"Avg Loss: {total_loss / (batch_idx + 1):.4f}, "
                        f"LR: {current_lr:.2e}"
                    )

        # Calculate average epoch loss across all processes
        avg_epoch_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_epoch_loss_tensor = torch.tensor(avg_epoch_loss, device=self.device)
        avg_epoch_loss_tensor = self._all_reduce_tensor(avg_epoch_loss_tensor)
        avg_epoch_loss = avg_epoch_loss_tensor.item()

        # Synchronize all processes before logging
        if self.world_size > 1:
            try:
                dist.barrier()
            except Exception as e:
                if self.is_main_process:
                    self.logger.warning(f"Barrier synchronization failed: {e}")

        if self.is_main_process:
            self.logger.info(f"Epoch {epoch_num + 1} Training Summary: Average Loss: {avg_epoch_loss:.4f}")
            
            # Log to wandb (per epoch)
            if self.wandb_enabled and WANDB_AVAILABLE:
                wandb.log({
                    "train/epoch_loss": avg_epoch_loss,
                    "train/epoch": epoch_num + 1
                })

        return avg_epoch_loss

    def validate_one_epoch(self, epoch_num):
        """Validate for one epoch"""
        if self.val_dataloader is None:
            if self.is_main_process:
                self.logger.info("No validation dataloader provided. Skipping validation.")
            return None

        self.model.eval()
        total_val_loss = 0
        num_batches = len(self.val_dataloader)

        # Set epoch for validation sampler
        if self.val_sampler:
            self.val_sampler.set_epoch(epoch_num)

        # Only show progress bar on main process
        if self.is_main_process:
            progress_bar = tqdm(self.val_dataloader, 
                              desc=f"Epoch {epoch_num + 1}/{self.epochs} [Validation]")
        else:
            progress_bar = self.val_dataloader

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
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

                # Forward pass
                with torch.cuda.amp.autocast(enabled=(self.use_amp or self.use_bfloat16), 
                                            dtype=self.model_dtype if self.device.type == 'cuda' else None):
                    # Validation mode: use inference mode (no ground truth actions)
                    action_pred = self.model(
                        image_1_batch=image_1_batch,
                        raw_prompt_texts_batch=raw_prompt_texts_batch,
                        vlm_attention_mask_batch=vlm_attention_mask,
                        state_batch=state_batch,
                        image_2_batch=image_2_batch
                        # No actions_gt_seq - this enables inference mode
                    )
                    
                    val_loss = self._compute_loss(action_pred, action_labels, vlm_attention_mask)

                total_val_loss += val_loss.item()

                # Update progress bar on main process
                if self.is_main_process:
                    progress_bar.set_postfix(
                        val_loss=val_loss.item(), 
                        avg_val_loss=total_val_loss / (batch_idx + 1)
                    )

        # Calculate average validation loss across all processes
        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else float('inf')
        avg_val_loss_tensor = torch.tensor(avg_val_loss, device=self.device)
        avg_val_loss_tensor = self._all_reduce_tensor(avg_val_loss_tensor)
        avg_val_loss = avg_val_loss_tensor.item()

        if self.is_main_process:
            self.logger.info(f"Epoch {epoch_num + 1} Validation Summary: Average Loss: {avg_val_loss:.4f}")
            
            # Log to wandb (per epoch)
            if self.wandb_enabled and WANDB_AVAILABLE:
                wandb.log({
                    "val/epoch_loss": avg_val_loss,
                    "val/epoch": epoch_num + 1
                })

        return avg_val_loss

    def save_checkpoint_if_needed(self, epoch, val_loss=None):
        """Save checkpoint if needed (only on main process)"""
        if not self.is_main_process:
            return

        is_best = False
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_metric = val_loss
            is_best = True

        # Save checkpoint at specified intervals or if it's the best
        if (epoch + 1) % self.save_checkpoint_interval == 0 or is_best:
            state = {
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),  # Use .module for DDP
                'optimizer': self.optimizer.state_dict(),
                'best_metric': self.best_metric,
                'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                'config': self.config
            }
            
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(state, is_best=is_best, filename=checkpoint_path)
            
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            if is_best:
                best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                save_checkpoint(state, is_best=True, filename=best_path)
                self.logger.info(f"New best model saved: {best_path}")

    def train(self, start_epoch=0):
        """Main training loop"""
        if self.is_main_process:
            self.logger.info(f"Starting distributed training loop from epoch {start_epoch + 1} to {self.epochs}.")
            
            if self.wandb_enabled and hasattr(wandb, 'watch') and WANDB_AVAILABLE:
                wandb.watch(self.model, log="all", log_freq=self.log_interval * 10, log_graph=False)

        for epoch in range(start_epoch, self.epochs):
            if self.is_main_process:
                self.logger.info(f"--- Epoch {epoch + 1}/{self.epochs} ---")

            # Training phase
            train_loss = self.train_one_epoch(epoch)

            # Validation phase
            val_loss = None
            if self.val_dataloader is not None:
                val_loss = self.validate_one_epoch(epoch)

            # Save checkpoint (only on main process)
            self.save_checkpoint_if_needed(epoch, val_loss)

            # Synchronize all processes before next epoch
            if self.world_size > 1:
                dist.barrier()

        if self.is_main_process:
            self.logger.info("Distributed training finished.")

    def load_checkpoint_ddp(self, checkpoint_path):
        """Load checkpoint for DDP training"""
        if self.is_main_process:
            self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        # Map checkpoint to current device
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        
        # Load checkpoint
        epoch, best_metric = load_checkpoint(
            self.model.module,  # Use .module for DDP
            self.optimizer, 
            checkpoint_path, 
            self.device, 
            strict=True
        )
        
        self.best_metric = best_metric
        
        # Load scheduler state if available
        checkpoint_data = torch.load(checkpoint_path, map_location=map_location)
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_data:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_data['lr_scheduler'])
                if self.is_main_process:
                    self.logger.info("Successfully loaded LR scheduler state from checkpoint.")
            except Exception as e:
                if self.is_main_process:
                    self.logger.warning(f"Could not load LR scheduler state: {e}")
        
        # Synchronize all processes after loading
        if self.world_size > 1:
            dist.barrier()
        
        return epoch, best_metric


# Example usage
if __name__ == '__main__':
    print("VLATrainerDDP - Distributed trainer for VLA models")
    print("This module should be imported and used with main_train_ddp.py")
