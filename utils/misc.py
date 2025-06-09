# Utility functions will be implemented here 

import torch
import numpy as np
import logging
import os
import shutil # For saving checkpoints

def discretize_actions(actions_continuous, num_bins, action_bounds=(-1.0, 1.0)):
    """
    Discretizes continuous actions into bin indices.
    Args:
        actions_continuous (np.ndarray or torch.Tensor): Continuous actions, shape (..., num_action_dims).
                                                        Expected to be in [action_bounds[0], action_bounds[1]].
        num_bins (int): Number of discrete bins for each action dimension.
        action_bounds (tuple): Min and max values for continuous actions.
    Returns:
        torch.Tensor: Discretized action bin indices, shape (..., num_action_dims), dtype torch.long.
    """
    is_torch = isinstance(actions_continuous, torch.Tensor)
    if not is_torch:
        actions_np = np.array(actions_continuous, dtype=np.float32)
    else:
        actions_np = actions_continuous.detach().cpu().numpy().astype(np.float32)

    # Normalize to [0, 1]
    actions_normalized = (actions_np - action_bounds[0]) / (action_bounds[1] - action_bounds[0])
    # Scale to [0, num_bins - 1] and round to nearest int for bin index
    # Ensure values are clipped to be valid indices, e.g. for values exactly at bounds
    action_bin_indices = np.clip(np.floor(actions_normalized * num_bins), 0, num_bins - 1).astype(np.int64)
    
    return torch.from_numpy(action_bin_indices).long()

def undiscretize_actions(action_bin_indices, num_bins, action_bounds=(-1.0, 1.0)):
    """
    Converts discretized action bin indices back to continuous action values (bin centers).
    Args:
        action_bin_indices (np.ndarray or torch.Tensor): Discretized action bin indices, shape (..., num_action_dims).
        num_bins (int): Number of discrete bins for each action dimension.
        action_bounds (tuple): Min and max values for continuous actions.
    Returns:
        torch.Tensor: Continuous action values, shape (..., num_action_dims), dtype torch.float32.
    """
    is_torch = isinstance(action_bin_indices, torch.Tensor)
    if not is_torch:
        indices_np = np.array(action_bin_indices, dtype=np.float32)
    else:
        indices_np = action_bin_indices.detach().cpu().numpy().astype(np.float32)

    # Calculate bin width and center offset
    bin_width = (action_bounds[1] - action_bounds[0]) / num_bins
    # Get the start of each bin, then add half bin_width for center
    actions_normalized_centers = (indices_np + 0.5) / num_bins 
    # Denormalize from [0, 1] back to [action_bounds[0], action_bounds[1]]
    actions_continuous = actions_normalized_centers * (action_bounds[1] - action_bounds[0]) + action_bounds[0]
    
    return torch.from_numpy(actions_continuous).float()

def setup_logging(log_level=logging.INFO, log_file=None, name='VLATrainer'):
    """Set up logging for the training process."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename_prefix='model_best'):
    """
    Saves model and training parameters at checkpoint.
    Args:
        state (dict): Contains model's state_dict, optimizer's state_dict, epoch, etc.
        is_best (bool): True if it is the best model seen so far (based on some metric).
        filename (str): Base filename for the checkpoint.
        best_filename_prefix (str): Prefix for the best model filename.
    """
    saved_dir = os.path.dirname(filename)
    if saved_dir and not os.path.exists(saved_dir):
        os.makedirs(saved_dir, exist_ok=True)
        
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(saved_dir, f"{best_filename_prefix}.pth.tar")
        shutil.copyfile(filename, best_filename)
        print(f"Saved new best model to {best_filename}")

def load_checkpoint(model, optimizer=None, filename='checkpoint.pth.tar', device='cpu', strict=True):
    """
    Loads model and training parameters from a checkpoint file.
    Args:
        model (torch.nn.Module): Model instance.
        optimizer (torch.optim.Optimizer, optional): Optimizer instance.
        filename (str): Path to the checkpoint file.
        device (str): Device to load the model to ('cpu' or 'cuda').
        strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this module’s state_dict(). Default True.
    Returns:
        int: Start epoch for training (or epoch of the loaded model).
        float: Best metric score recorded (if available in checkpoint, else 0.0).
    """
    if not os.path.isfile(filename):
        print(f"=> No checkpoint found at '{filename}'")
        return 0, 0.0 # Default start epoch and best_metric

    print(f"=> Loading checkpoint '{filename}'")
    checkpoint = torch.load(filename, map_location=device) # Load to specified device
    
    start_epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', 0.0)
    
    # Handle potential DataParallel or DistributedDataParallel model saving
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] # remove `module.` prefix
        else:
            name = k
        new_state_dict[name] = v
    
    # Pass the strict argument to load_state_dict
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict)
    
    if missing_keys:
        print(f"Warning: Missing keys in state_dict: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
        
    model.to(device) # Ensure model is on the correct device after loading

    if optimizer and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            # Move optimizer states to the correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        except Exception as e:
            print(f"Could not load optimizer state: {e}. Optimizer will be re-initialized.")

    print(f"=> Loaded checkpoint '{filename}' (epoch {start_epoch}, best_metric {best_metric:.4f})")
    return start_epoch, best_metric

def get_lr_scheduler(optimizer, scheduler_config, logger=None, steps_per_epoch=None):
    """
    Creates a learning rate scheduler based on the provided configuration.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to create the scheduler.
        scheduler_config (dict or OmegaConfAttrDict): Configuration for the scheduler, including 'type' and other params.
                                                     Example: {'type': 'StepLR', 'step_size': 30, 'gamma': 0.1}
        logger (logging.Logger, optional): Logger instance.
        steps_per_epoch (int, optional): Number of steps per epoch, required by some schedulers like OneCycleLR.

    Returns:
        torch.optim.lr_scheduler._LRScheduler or None: The learning rate scheduler, or None if config is empty/invalid.
    """
    if not scheduler_config or not scheduler_config.get('type'):
        if logger:
            logger.info("No LR scheduler type specified or scheduler_config is empty. No scheduler will be used.")
        return None

    scheduler_type = scheduler_config.type
    params = {k: v for k, v in scheduler_config.items() if k != 'type'}

    if logger:
        logger.info(f"Attempting to create LR scheduler of type: {scheduler_type} with params: {params}")

    try:
        if scheduler_type.lower() == 'steplr':
            return torch.optim.lr_scheduler.StepLR(optimizer, **params)
        elif scheduler_type.lower() == 'multisteplr':
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, **params)
        elif scheduler_type.lower() == 'exponentiallr':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, **params)
        elif scheduler_type.lower() == 'cosineannealinglr':
            # T_max is often set to total epochs or total steps depending on update frequency
            if 'T_max' not in params:
                if logger: logger.warning("T_max not specified for CosineAnnealingLR. Ensure it's set appropriately.")
                # Defaulting T_max to a large number if not specified, but should be configured.
                # For epoch-wise stepping, T_max is often config.training.epochs.
                # For step-wise stepping, T_max is often total_training_steps.
                # This function is generic, so caller should ensure T_max is meaningful.
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
        elif scheduler_type.lower() == 'reducelronplateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
        elif scheduler_type.lower() == 'onecyclelr':
            if steps_per_epoch is None and 'total_steps' not in params:
                msg = "OneCycleLR requires either 'steps_per_epoch' (passed to get_lr_scheduler) or 'total_steps' in its config."
                if logger: logger.error(msg)
                raise ValueError(msg)
            if 'total_steps' not in params and steps_per_epoch is not None:
                 # Estimate total_steps if epochs is available in params (e.g. from global training config)
                 # This is a bit of a hack; ideally total_steps is explicitly configured for OneCycleLR.
                 epochs_for_onecycle = params.pop('epochs', None) # Pop to avoid passing to OneCycleLR constructor if it was just for total_steps calc
                 if epochs_for_onecycle is not None:
                     params['total_steps'] = epochs_for_onecycle * steps_per_epoch
                     if logger: logger.info(f"Calculated total_steps for OneCycleLR: {params['total_steps']} ({epochs_for_onecycle} epochs * {steps_per_epoch} steps/epoch)")
                 else:
                     # If epochs not in scheduler_config, one might use a default or assume it's handled by total_steps.
                     # For simplicity, if total_steps isn't there and can't be derived, it will error out if required by OneCycleLR.
                     if logger: logger.warning("OneCycleLR: 'total_steps' not set and 'epochs' not in scheduler_config for derivation. Ensure 'max_lr' is set.")
            
            # Ensure max_lr is correctly passed; it's often specified as 'lr' in optimizer and then OneCycleLR needs it as max_lr.
            if 'max_lr' not in params and 'lr' in params: # common to have lr in params from main optimizer config part
                params['max_lr'] = params.pop('lr') 
            elif 'max_lr' not in params: # if optimizer's lr is the max_lr to be used
                params['max_lr'] = optimizer.defaults.get('lr', 0.01) # Fallback, should be configured
                if logger: logger.warning(f"OneCycleLR: 'max_lr' not specified in scheduler params. Using optimizer default lr: {params['max_lr']}")

            return torch.optim.lr_scheduler.OneCycleLR(optimizer, **params)
        else:
            if logger: logger.error(f"Unsupported LR scheduler type: {scheduler_type}")
            return None
    except Exception as e:
        if logger: logger.error(f"Error creating LR scheduler {scheduler_type}: {e}", exc_info=True)
        return None

def normalize(x, x_min, x_max, eps=1e-8):
    """Normalize x to [-1, 1] using min and max values."""
    # Ensure x_min and x_max are tensors
    if not isinstance(x_min, torch.Tensor):
        x_min = torch.tensor(x_min, dtype=x.dtype, device=x.device)
    if not isinstance(x_max, torch.Tensor):
        x_max = torch.tensor(x_max, dtype=x.dtype, device=x.device)
    
    # Reshape min/max to be broadcastable with x if x is (T, D) and min/max are (D,)
    if x.ndim == 2 and x_min.ndim == 1 and x.shape[1] == x_min.shape[0]:
        x_min = x_min.unsqueeze(0)
        x_max = x_max.unsqueeze(0)
        
    return 2 * (x - x_min) / (x_max - x_min + eps) - 1

def denormalize(x_norm, x_min, x_max):
    """Denormalize x_norm from [-1, 1] to original scale."""
    if not isinstance(x_min, torch.Tensor):
        x_min = torch.tensor(x_min, dtype=x_norm.dtype, device=x_norm.device)
    if not isinstance(x_max, torch.Tensor):
        x_max = torch.tensor(x_max, dtype=x_norm.dtype, device=x_norm.device)

    if x_norm.ndim == 2 and x_min.ndim == 1 and x_norm.shape[1] == x_min.shape[0]:
        x_min = x_min.unsqueeze(0)
        x_max = x_max.unsqueeze(0)
        
    return (x_norm + 1) / 2 * (x_max - x_min) + x_min

if __name__ == '__main__':
    print("Testing utility functions:")

    # Test action discretization/undiscretization
    print("\n--- Action Discretization/Undiscretization Test ---")
    num_bins_test = 10
    action_dims_test = 3
    test_actions_cont = np.array([[-1.0, 0.0, 1.0], [-0.5, 0.5, 0.99]]) 
    print(f"Continuous actions:\n{test_actions_cont}")
    
    test_actions_disc = discretize_actions(test_actions_cont, num_bins_test)
    print(f"Discretized actions (bin indices):\n{test_actions_disc}")
    # Expected for num_bins=10, bounds=[-1,1]:
    # [-1.0, 0.0, 1.0] -> [0, 5, 9] (actually 0, 4 or 5 depending on floor/round, 9)
    # [-0.5, 0.5, 0.99] -> [2, 7, 9]
    # Let's verify: bin_width = 2/10 = 0.2
    # -1.0 -> norm 0.0 -> 0.0*10=0 -> floor(0)=0. Correct.
    #  0.0 -> norm 0.5 -> 0.5*10=5 -> floor(5)=5. Correct.
    #  1.0 -> norm 1.0 -> 1.0*10=10 -> floor(10)=10, clipped to 9. Correct.
    # -0.5 -> norm 0.25 -> 0.25*10=2.5 -> floor(2.5)=2. Correct.
    #  0.5 -> norm 0.75 -> 0.75*10=7.5 -> floor(7.5)=7. Correct.

    test_actions_undisc = undiscretize_actions(test_actions_disc, num_bins_test)
    print(f"Undiscretized actions (bin centers):\n{test_actions_undisc}")
    # Expected centers: Bin 0 (-1 to -0.8) center -0.9. Bin 5 (0 to 0.2) center 0.1. Bin 9 (0.8 to 1.0) center 0.9.
    # Bin 2 (-0.6 to -0.4) center -0.5. Bin 7 (0.4 to 0.6) center 0.5.
    # Check a value: index 0 -> (0+0.5)/10 = 0.05. 0.05 * 2 - 1 = -0.9. Correct.
    # index 5 -> (5+0.5)/10 = 0.55. 0.55 * 2 - 1 =  0.1. Correct.

    # Test with torch tensors
    test_actions_cont_torch = torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.5, 0.99]], dtype=torch.float32)
    test_actions_disc_torch = discretize_actions(test_actions_cont_torch, num_bins_test)
    assert torch.all(test_actions_disc_torch == test_actions_disc) # Should be same
    test_actions_undisc_torch = undiscretize_actions(test_actions_disc_torch, num_bins_test)
    assert torch.allclose(test_actions_undisc_torch, test_actions_undisc) # Should be same
    print("Torch tensor discretization/undiscretization consistent.")

    # Test logging
    print("\n--- Logging Test ---")
    logger_test_file = "test_log.txt"
    test_logger = setup_logging(log_level=logging.DEBUG, log_file=logger_test_file, name="TestLogger")
    test_logger.debug("This is a debug message.")
    test_logger.info("This is an info message.")
    if os.path.exists(logger_test_file):
        print(f"Log file {logger_test_file} created. Check its content.")
        # os.remove(logger_test_file) # Clean up
    else:
        print(f"Log file {logger_test_file} NOT created.")

    # Test checkpointing (conceptual, needs a model and optimizer)
    print("\n--- Checkpoint Test (Conceptual) ---")
    class DummyModel(torch.nn.Module):
        def __init__(self): super().__init__(); self.linear = torch.nn.Linear(10,1)
        def forward(self,x): return self.linear(x)

    dummy_model = DummyModel()
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.001)
    checkpoint_dir = "./dummy_checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, "test_checkpoint.pth.tar")
    best_model_path_prefix = os.path.join(checkpoint_dir, "test_model_best")

    state_to_save = {
        'epoch': 5,
        'state_dict': dummy_model.state_dict(),
        'optimizer': dummy_optimizer.state_dict(),
        'best_metric': 0.75
    }
    save_checkpoint(state_to_save, is_best=True, filename=checkpoint_path, best_filename_prefix=best_model_path_prefix.split('.')[0]) # remove .pth.tar for prefix
    save_checkpoint(state_to_save, is_best=False, filename=checkpoint_path, best_filename_prefix=best_model_path_prefix.split('.')[0]) # Overwrite, not best

    print(f"Saved checkpoint to {checkpoint_path}")
    if os.path.exists(best_model_path_prefix + ".pth.tar"):
         print(f"Saved best model to {best_model_path_prefix}.pth.tar")

    # Test loading
    loaded_model = DummyModel()
    loaded_optimizer = torch.optim.Adam(loaded_model.parameters(), lr=0.001)
    start_epoch, best_metric_loaded = load_checkpoint(loaded_model, loaded_optimizer, checkpoint_path)
    print(f"Loaded model from epoch {start_epoch}, best metric was {best_metric_loaded}")
    assert start_epoch == 5
    assert best_metric_loaded == 0.75
    # Check if weights are loaded (simple check)
    assert torch.allclose(list(loaded_model.parameters())[0], list(dummy_model.parameters())[0])
    print("Checkpoint loading seems to work.")

    # Clean up dummy checkpoint files and dir
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print(f"Cleaned up dummy checkpoint directory: {checkpoint_dir}")
    if os.path.exists(logger_test_file):
        os.remove(logger_test_file)
        print(f"Cleaned up dummy log file: {logger_test_file}")
        
    print("\nAll utility tests completed.")