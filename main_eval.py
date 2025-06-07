# Main script for evaluating/running inference with the VLA model 

import argparse
import yaml
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json # For loading stats 
from PIL import Image
import random
from omegaconf import OmegaConf

from data.loader import VLADataset, vla_collate_fn # For loading evaluation datasets
from torch.utils.data import DataLoader
from inference.predictor import VLAPredictor
from utils.misc import setup_logging, discretize_actions, undiscretize_actions, denormalize # Added denormalize
from utils.config_utils import OmegaConfAttrDict # For config loading

def get_default_config():
    """获取与训练时兼容的默认配置"""
    # 这个配置确保模型可以被正确构建，并且会从本地加载权重而不是下载
    return {
        'model': {
            'vlm_config': {
                'model_name_or_path': "./weight/paligemma-3b-pt-224",
                'use_aux_camera': False,
                'dtype': 'torch.bfloat16', # 与vla_config.yaml保持一致
                'num_image_tokens': 256, # 显式设置以避免警告
            },
            'vision_resampler_config': { # 显式配置以避免警告
                'type': 'mlp',
                'output_dim': 2048, # 应与VLM的隐藏层大小匹配
                'mlp_projector': {
                    'hidden_dim': None # 模型内部可处理None
                }
            },
            'action_head_config': {
                'use_state_input': True,
                'state_dim': 7, 
                'num_action_dims': 7, 
                'num_action_bins': 256, 
                'hidden_layers_config': [1024, 512], # 使用正确的key
                'dropout_prob': 0.1
            }
        },
        'data': {
            'tokenizer_name_or_path': "./weight/paligemma-3b-pt-224",
            'image_processor_name_or_path': "./weight/paligemma-3b-pt-224",
            'action_bounds': [-1.0, 1.0],
            'normalization_stats_path': "normalization_stats.json" # 指向统计文件
        }
    }

def calculate_metrics(all_pred_bins, all_true_bins, all_pred_continuous, all_true_continuous, vlm_masks, num_action_dims):
    """
    Calculates evaluation metrics.
    Args:
        all_pred_bins (list of torch.Tensor): List of predicted action bins per batch.
        all_true_bins (list of torch.Tensor): List of true action bins per batch.
        all_pred_continuous (list of torch.Tensor): List of predicted continuous actions per batch.
        all_true_continuous (list of torch.Tensor): List of true continuous actions per batch.
        vlm_masks (list of torch.Tensor): List of VLM attention masks per batch.
        num_action_dims (int): Number of action dimensions.
    Returns:
        dict: Dictionary of calculated metrics.
    """
    metrics = {}

    # Concatenate all batch results, considering only valid steps based on vlm_mask
    valid_pred_bins = []
    valid_true_bins = []
    valid_pred_continuous = []
    valid_true_continuous = []

    for i in range(len(all_pred_bins)):
        mask = vlm_masks[i].bool() # (B, S)
        # Predicted bins: (B, S, D) -> select valid -> flatten to (N_valid_steps, D)
        pred_b = all_pred_bins[i][mask]
        true_b = all_true_bins[i][mask]
        pred_c = all_pred_continuous[i][mask]
        true_c = all_true_continuous[i][mask]
        
        if pred_b.numel() > 0: # Only append if there are valid steps in the batch
            valid_pred_bins.append(pred_b)
            valid_true_bins.append(true_b)
            valid_pred_continuous.append(pred_c)
            valid_true_continuous.append(true_c)

    if not valid_pred_bins: # No valid steps in the entire dataset
        logger.warning("No valid steps found in evaluation data after masking. Metrics will be zero or NaN.")
        metrics['accuracy_per_dim'] = [0.0] * num_action_dims
        metrics['overall_accuracy'] = 0.0
        metrics['mse_per_dim'] = [float('nan')] * num_action_dims
        metrics['overall_mse'] = float('nan')
        metrics['mae_per_dim'] = [float('nan')] * num_action_dims
        metrics['overall_mae'] = float('nan')
        return metrics

    valid_pred_bins = torch.cat(valid_pred_bins, dim=0) # (Total_Valid_Steps, D)
    valid_true_bins = torch.cat(valid_true_bins, dim=0)
    valid_pred_continuous = torch.cat(valid_pred_continuous, dim=0)
    valid_true_continuous = torch.cat(valid_true_continuous, dim=0)

    if valid_pred_bins.numel() == 0: # Should be caught by the above but as a safeguard
        logger.warning("Concatenated valid bins tensor is empty. Metrics will be zero or NaN.")
        # (Same return as above)
        metrics['accuracy_per_dim'] = [0.0] * num_action_dims
        metrics['overall_accuracy'] = 0.0
        metrics['mse_per_dim'] = [float('nan')] * num_action_dims
        metrics['overall_mse'] = float('nan')
        metrics['mae_per_dim'] = [float('nan')] * num_action_dims
        metrics['overall_mae'] = float('nan')
        return metrics
        
    # Accuracy (for binned actions)
    correct_preds_per_dim = (valid_pred_bins == valid_true_bins).float().sum(dim=0)
    total_valid_steps = valid_pred_bins.size(0)
    
    accuracy_per_dim = (correct_preds_per_dim / total_valid_steps).tolist() if total_valid_steps > 0 else [0.0] * num_action_dims
    metrics['accuracy_per_dim'] = [round(acc * 100, 2) for acc in accuracy_per_dim]
    
    # Overall accuracy: all action dimensions must be correct for a step to be correct
    correct_overall = (valid_pred_bins == valid_true_bins).all(dim=1).float().sum()
    metrics['overall_accuracy'] = round((correct_overall.item() / total_valid_steps) * 100, 2) if total_valid_steps > 0 else 0.0

    # MSE and MAE (for continuous actions)
    mse_per_dim = torch.mean((valid_pred_continuous - valid_true_continuous)**2, dim=0).tolist()
    mae_per_dim = torch.mean(torch.abs(valid_pred_continuous - valid_true_continuous), dim=0).tolist()
    metrics['mse_per_dim'] = mse_per_dim
    metrics['mae_per_dim'] = mae_per_dim
    metrics['overall_mse'] = torch.mean((valid_pred_continuous - valid_true_continuous)**2).item()
    metrics['overall_mae'] = torch.mean(torch.abs(valid_pred_continuous - valid_true_continuous)).item()

    return metrics

def main(args):
    # --- Setup Logger ---
    output_dir = args.output_dir if args.output_dir else \
                 os.path.join(os.path.dirname(args.checkpoint_path), 'eval_results')
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'evaluation.log')
    global logger 
    logger = setup_logging(log_file=log_file, name="VLAEvaluator")
    logger.info(f"Evaluation results will be saved in: {output_dir}")

    # --- Load Config ---
    # 首先，以默认配置为基础，这确保了所有必需的键都存在
    base_config = OmegaConf.create(get_default_config())
    config = base_config # 如果没有提供用户配置，则使用此默认配置

    # 如果用户提供了配置文件路径，则加载并合并它
    if args.config_path:
        if os.path.exists(args.config_path):
            try:
                user_config = OmegaConf.load(args.config_path)
                logger.info(f"Loaded user configuration from: {args.config_path}")
                # 将用户配置合并到基础配置之上（用户的值会覆盖默认值）
                config = OmegaConf.merge(base_config, user_config)
                logger.info("User config successfully merged with default config.")
            except Exception as e:
                logger.error(f"Error loading or merging YAML config from '{args.config_path}': {e}. Exiting.", exc_info=True)
                return
        else:
            # 如果提供了路径但文件不存在，这是一个致命错误
            logger.error(f"Provided config file not found at: {args.config_path}. Exiting.")
            return
    else:
        logger.warning("No config file path provided. Using hardcoded default config for model structure.")

    # --- Initialize Predictor ---
    logger.info(f"Initializing VLAPredictor with checkpoint: {args.checkpoint_path}")
    try:
        predictor = VLAPredictor(
            checkpoint_path=args.checkpoint_path, 
            config=config,
            device=args.device, 
            logger=logger
        )
    except Exception as e:
        logger.error(f"Error initializing VLAPredictor: {e}", exc_info=True)
        return

    # --- Load Normalization Stats for Denormalization ---
    action_norm_stats = None
    if predictor.norm_stats and 'action' in predictor.norm_stats:
        action_norm_stats = predictor.norm_stats['action']
        logger.info("Using action normalization stats from predictor for denormalization.")
        # Ensure min/max are torch tensors on CPU for upcoming operations if they aren't already
        if isinstance(action_norm_stats['min'], list):
            action_norm_stats['min'] = torch.tensor(action_norm_stats['min'], dtype=torch.float32)
        if isinstance(action_norm_stats['max'], list):
            action_norm_stats['max'] = torch.tensor(action_norm_stats['max'], dtype=torch.float32)
        action_norm_stats['min'] = action_norm_stats['min'].cpu()
        action_norm_stats['max'] = action_norm_stats['max'].cpu()
    else:
        logger.warning("Action normalization stats not found in predictor. Predictions might not be denormalized correctly.")

    all_pred_action_bins = []
    all_true_action_bins = [] 
    all_pred_actions_continuous_denorm = [] # Store denormalized predictions
    all_true_actions_continuous_gt_norm = [] # Store normalized ground truth from dataset
    all_vlm_masks = []
    all_prompts = [] 
    # --- Process Data and Predict ---
    if args.eval_data_path: # Batch evaluation from dataset files
        logger.info(f"正在从以下路径加载评估数据集: {args.eval_data_path}")
        try:
            # --- 核心修复：修正传递给VLADataset的参数 ---
            common_dataset_params = {
                # VLADataset 需要一个统一的 processor_name_or_path
                "processor_name_or_path": config.model.vlm_config.model_name_or_path,
                "max_seq_len": config.data.get('max_seq_len', 1),
                "prompt_max_len": config.data.get('prompt_max_len', 77),
                # 这些参数现在直接从action_head_config获取
                "action_dim": config.model.action_head_config.num_action_dims,
                "state_dim": config.model.action_head_config.state_dim,
                "logger": logger,
                "normalization_stats_path": config.data.get('normalization_stats_path', None),
                # 可能还需要 use_siglip, siglip_model_name 等，如果您的 VLADataset 需要的话
                # 暂时我们只包含最核心的参数
            }
            eval_dataset = VLADataset(
                parquet_files=args.eval_data_path,
                **common_dataset_params
            )
            # 注意：这里的 batch_size 设为 1，以便逐个样本进行对比
            eval_loader = DataLoader(
                eval_dataset, 
                batch_size=1, 
                shuffle=False, 
                collate_fn=vla_collate_fn, 
                num_workers=args.num_workers or config.data.get('num_workers', 0)
            )
        except Exception as e:
            logger.error(f"Error initializing evaluation data loader: {e}", exc_info=True)
            return

        logger.info(f"Evaluation dataset size: {len(eval_dataset)}. Batch size: {1}")
        progress_bar = tqdm(eval_loader, desc="Evaluating Batches")
        for batch_data in progress_bar:
            processed_batch = {}
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    processed_batch[key] = value.to(predictor.device, dtype=predictor.model_dtype if value.is_floating_point() else value.dtype)
                else:
                    processed_batch[key] = value 
            
            predictions_dict = predictor.predict(processed_batch) # Returns dict with normalized continuous actions
            pred_actions_continuous_normalized = predictions_dict['predicted_actions_continuous'].cpu()
            
            # Denormalize continuous predictions
            pred_actions_denormalized = pred_actions_continuous_normalized # Default to normalized if no stats
            if action_norm_stats and action_norm_stats['min'] is not None and action_norm_stats['max'] is not None:
                # Ensure stats are tensors on the same device as the data for denormalize function
                action_min = action_norm_stats['min'].to(pred_actions_continuous_normalized.device)
                action_max = action_norm_stats['max'] .to(pred_actions_continuous_normalized.device)
                pred_actions_denormalized = denormalize(pred_actions_continuous_normalized, action_min, action_max)
            else:
                logger.warning("Predictor action norm_stats not available for denormalization. Using normalized predictions as denormalized.")
            
            all_pred_actions_continuous_denorm.append(pred_actions_denormalized)
            all_vlm_masks.append(processed_batch['vlm_attention_mask'].cpu())
            
            if 'raw_prompt_text' in processed_batch:
                all_prompts.extend(processed_batch['raw_prompt_text'])
            elif 'prompt_input_ids' in processed_batch: 
                all_prompts.extend([predictor.tokenizer.decode(ids.cpu(), skip_special_tokens=True) for ids in processed_batch['prompt_input_ids']])

            if 'action' in processed_batch: 
                true_actions_continuous_normalized_gt = processed_batch['action'].cpu() # These are from VLADataset, already normalized
                all_true_actions_continuous_gt_norm.append(true_actions_continuous_normalized_gt)

                # --- Binning for metrics: Use denormalized predictions and denormalized GT ---
                # Denormalize GT actions for consistent binning if stats are available
                true_actions_denormalized_gt = true_actions_continuous_normalized_gt # Default
                if action_norm_stats and action_norm_stats['min'] is not None and action_norm_stats['max'] is not None:
                    gt_action_min = action_norm_stats['min'].to(true_actions_continuous_normalized_gt.device)
                    gt_action_max = action_norm_stats['max'].to(true_actions_continuous_normalized_gt.device)
                    true_actions_denormalized_gt = denormalize(true_actions_continuous_normalized_gt, gt_action_min, gt_action_max)
                else:
                    logger.warning("Cannot denormalize GT actions for binning. Binning GT on normalized scale.")

                pred_bins = discretize_actions(
                    pred_actions_denormalized, # Use denormalized predictions
                    predictor.config.model.action_head_config.num_action_bins, 
                    predictor.config.model.action_head_config.get('action_bounds', (-1.0, 1.0)) # Assumed to be for original scale
                ).cpu()
                all_pred_action_bins.append(pred_bins)
                
                true_bins = discretize_actions(
                    true_actions_denormalized_gt, # Use denormalized GT for binning
                    predictor.config.model.action_head_config.num_action_bins, 
                    predictor.config.model.action_head_config.get('action_bounds', (-1.0, 1.0))
                ).cpu()
                all_true_action_bins.append(true_bins)

    else: # Single item inference mode (from args or random)
        image_1 = None
        image_2 = None
        prompt = ""
        state_vector = None

        if args.image1_paths and args.prompt:
            logger.info("Received single item inference request from command line.")
            try:
                if len(args.image1_paths) > 1:
                    logger.warning(f"Multiple images provided, but current single-item mode only processes the first one. Using: {args.image1_paths[0]}")
                image_1 = Image.open(args.image1_paths[0]).convert("RGB")
                
                if args.image2_paths:
                    if len(args.image2_paths) > 1:
                        logger.warning(f"Multiple wrist images provided, using the first one: {args.image2_paths[0]}")
                    image_2 = Image.open(args.image2_paths[0]).convert("RGB")
            except FileNotFoundError as e:
                logger.error(f"Image file not found: {e}", exc_info=True)
                return
            
            prompt = args.prompt
            state_vector = np.array(args.state_vector, dtype=np.float32) if args.state_vector else None

        else:
            logger.info("No input data provided. Running with randomly generated data.")
            # 1. Generate random prompt
            sample_prompts = ["pick up the red block", "push the green square to the left", "open the drawer", "place the can in the bin"]
            prompt = random.choice(sample_prompts)
            logger.info(f"Using random prompt: '{prompt}'")

            # 2. Generate random images
            image_1 = Image.new('RGB', (224, 224), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            logger.info("Generated random base image.")
            
            # Check if model uses a second camera
            if predictor.config.model.vlm_config.get('use_aux_camera', False):
                image_2 = Image.new('RGB', (224, 224), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
                logger.info("Generated random wrist image.")

            # 3. Generate random state vector
            if predictor.config.model.action_head_config.get('use_state_input', False):
                state_dim = predictor.config.model.action_head_config.get('state_dim', 7)
                state_vector = np.random.rand(state_dim).astype(np.float32) * 2 - 1 # range [-1, 1]
                logger.info(f"Generated random state vector (dim={state_dim})")

        # --- Common logic for single item inference ---
        logger.info("Performing single item inference.")
        
        # ... (normalization of state_vector is fine) ...

        preprocessed_input = predictor.preprocess_single_item_direct(
            image_1=image_1,
            prompt_text=prompt,
            image_2=image_2,
            state_vector=state_vector # 注意：这里传递原始的state_vector，predictor内部会处理
        )
        
        predictions_dict = predictor.predict(preprocessed_input)
        action_denormalized = predictions_dict['action']
        logger.info(f"Predicted Action (denormalized): {action_denormalized}")

        all_pred_actions_continuous_denorm.append(torch.tensor(action_denormalized).unsqueeze(0).unsqueeze(0))
        all_vlm_masks.append(preprocessed_input['vlm_attention_mask'].cpu())
        all_prompts.append(prompt)
    
    # --- Calculate and Log Metrics (if true labels were available) ---
    if all_true_actions_continuous_gt_norm: # Check if we have GT actions (these are normalized from VLADataset)
        logger.info("Calculating metrics...")
        num_action_dims = predictor.config.model.action_head_config.num_action_dims
        
        # Denormalize ground truth actions for consistent metric calculation
        all_true_actions_continuous_denorm_gt = []
        if action_norm_stats and action_norm_stats['min'] is not None and action_norm_stats['max'] is not None:
            gt_action_min = action_norm_stats['min'] # Already .cpu() from earlier
            gt_action_max = action_norm_stats['max'] # Already .cpu() from earlier
            for true_actions_norm_batch in all_true_actions_continuous_gt_norm:
                # Ensure data is on the same device as stats if denormalize expects it (or handle inside denormalize)
                # Our denormalize converts stats to tensor on data's device.
                all_true_actions_continuous_denorm_gt.append(denormalize(true_actions_norm_batch.cpu(), gt_action_min, gt_action_max))
            logger.info("Ground truth actions denormalized for metrics.")
        else:
            logger.warning("Cannot denormalize ground truth actions for metrics due to missing stats. MSE/MAE might be on normalized scale if calculated using normalized GT.")
            all_true_actions_continuous_denorm_gt = all_true_actions_continuous_gt_norm # Fallback to normalized GT

        # At this point, all_pred_actions_continuous_denorm and all_true_actions_continuous_denorm_gt are in original scale.
        # Binned actions (all_pred_action_bins, all_true_action_bins) were also created based on denormalized values and original scale bounds.

        metrics = calculate_metrics(
            all_pred_action_bins,               # Bins from denormalized preds, original scale bounds
            all_true_action_bins,               # Bins from denormalized GT, original scale bounds
            all_pred_actions_continuous_denorm, # Continuous denormalized preds
            all_true_actions_continuous_denorm_gt, # Continuous denormalized GT
            all_vlm_masks, 
            num_action_dims
        )
        logger.info("Evaluation Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.yaml')
        with open(metrics_path, 'w') as f:
            yaml.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
    else:
        logger.info("No ground truth actions provided with evaluation data, skipping metrics calculation.")

    # --- Save Predictions (Optional) ---
    if args.save_predictions:
        logger.info("Saving predictions...")
        flat_predictions = []
        num_items_processed = 0 # For aligning prompts with batch items

        # Ensure all lists for metrics have the same number of batches
        num_batches = len(all_pred_actions_continuous_denorm)

        for i in range(num_batches):
            batch_preds_cont_denorm = all_pred_actions_continuous_denorm[i] 
            batch_preds_bins = all_pred_action_bins[i] if i < len(all_pred_action_bins) else None                   
            batch_mask = all_vlm_masks[i]                                 
            
            # Ground truth for saving (optional)
            batch_true_cont_denorm = all_true_actions_continuous_denorm_gt[i] if all_true_actions_continuous_denorm_gt and i < len(all_true_actions_continuous_denorm_gt) else None
            batch_true_bins = all_true_action_bins[i] if all_true_action_bins and i < len(all_true_action_bins) else None
            
            current_batch_size = batch_preds_cont_denorm.shape[0]

            for b_idx in range(current_batch_size): 
                prompt_for_item = "N/A"
                item_abs_idx = num_items_processed + b_idx
                if all_prompts and item_abs_idx < len(all_prompts):
                    prompt_for_item = all_prompts[item_abs_idx]
                
                for s_idx in range(batch_preds_cont_denorm.shape[1]): 
                    if batch_mask[b_idx, s_idx].item(): 
                        record = {
                            'item_index_in_dataset': item_abs_idx, 
                            'sequence_step': s_idx,
                            'prompt': prompt_for_item
                        }
                        for d_idx in range(batch_preds_cont_denorm.shape[2]): 
                            record[f'pred_cont_dim_{d_idx}'] = batch_preds_cont_denorm[b_idx, s_idx, d_idx].item()
                            if batch_preds_bins is not None:
                                record[f'pred_bin_dim_{d_idx}'] = batch_preds_bins[b_idx, s_idx, d_idx].item()
                            
                            if batch_true_cont_denorm is not None and s_idx < batch_true_cont_denorm.shape[1] and d_idx < batch_true_cont_denorm.shape[2]:
                                record[f'true_cont_dim_{d_idx}'] = batch_true_cont_denorm[b_idx, s_idx, d_idx].item()
                            if batch_true_bins is not None and s_idx < batch_true_bins.shape[1] and d_idx < batch_true_bins.shape[2]:
                                record[f'true_bin_dim_{d_idx}'] = batch_true_bins[b_idx, s_idx, d_idx].item()
                        flat_predictions.append(record)
            num_items_processed += current_batch_size
        
        pred_df = pd.DataFrame(flat_predictions)
        predictions_path = os.path.join(output_dir, 'predictions.csv')
        pred_df.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to {predictions_path}")

    logger.info("Evaluation script finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate or Run Inference with VLA Model")
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help='Path to the VLA model checkpoint (.pth.tar or .pth).')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to the model YAML configuration file. If not provided, a default is used.')
    
    # Group for batch evaluation from dataset files
    eval_group = parser.add_argument_group('Dataset Evaluation Arguments')
    eval_group.add_argument('--eval_data_path', type=str, nargs='+', 
                            help='Path(s) to evaluation parquet file(s).')
    eval_group.add_argument('--batch_size', type=int, default=16, help='Batch size for dataset evaluation.')
    eval_group.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers.')

    # Group for single item inference
    single_item_group = parser.add_argument_group('Single Item Inference Arguments')
    single_item_group.add_argument('--image1_paths', type=str, nargs='+', help='Path(s) to main camera image(s) for a single sequence.')
    single_item_group.add_argument('--prompt', type=str, help='Natural language prompt for single item inference.')
    single_item_group.add_argument('--image2_paths', type=str, nargs='+', help='(Optional) Path(s) to wrist camera image(s).')
    single_item_group.add_argument('--state_vector', type=float, nargs='+', help='(Optional) Robot state vector as a list of floats.')

    # General arguments
    parser.add_argument('--device', type=str, default=None, help='Device to run on (e.g., "cpu", "cuda:0"). Autodetects if None.')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save evaluation results (metrics, predictions). Defaults to checkpoint_dir/eval_results.')
    parser.add_argument('--save_predictions', action='store_true', help='Save detailed predictions to a CSV file.')
    parser.add_argument('--max_seq_len_override', type=int, help='Override max sequence length for dataloader/preprocessing if not in config.')
    parser.add_argument('--prompt_max_len_override', type=int, help='Override prompt max length for dataloader/preprocessing if not in config.')

    args = parser.parse_args()
    main(args) 