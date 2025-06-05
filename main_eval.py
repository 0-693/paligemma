# Main script for evaluating/running inference with the VLA model 

import argparse
import yaml
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json # For loading stats 

from data.loader import VLADataset, vla_collate_fn # For loading evaluation datasets
from torch.utils.data import DataLoader
from inference.predictor import VLAPredictor
from utils.misc import setup_logging, discretize_actions, undiscretize_actions, denormalize # Added denormalize
from utils.config_utils import OmegaConfAttrDict # For config loading

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

    # --- Initialize Predictor ---
    logger.info(f"Initializing VLAPredictor with checkpoint: {args.checkpoint_path}")
    try:
        predictor = VLAPredictor(checkpoint_path=args.checkpoint_path, device=args.device, logger=logger)
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
        logger.info(f"Evaluating dataset from: {args.eval_data_path}")
        eval_data_paths = args.eval_data_path # Expecting a list of parquet files
        
        # Use data configuration from the loaded model's checkpoint for consistency
        data_conf = predictor.config.get('data_config', {})
        if not data_conf:
            logger.warning("Data config not found in checkpoint. Using default/arg values for dataloader.")
        
        try:
            eval_dataset = VLADataset(
                parquet_files=eval_data_paths,
                tokenizer_name_or_path=data_conf.get('tokenizer_name_or_path', 'bert-base-uncased'),
                image_processor_name_or_path=data_conf.get('image_processor_name_or_path', 'google/vit-base-patch16-224-in21k'),
                max_seq_len=data_conf.get('max_seq_len', args.max_seq_len_override or 32),
                prompt_max_len=data_conf.get('prompt_max_len', args.prompt_max_len_override or 128)
            )
            eval_loader = DataLoader(
                eval_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                collate_fn=vla_collate_fn, 
                num_workers=args.num_workers or data_conf.get('num_workers', 0)
            )
        except Exception as e:
            logger.error(f"Error initializing evaluation data loader: {e}", exc_info=True)
            return

        logger.info(f"Evaluation dataset size: {len(eval_dataset)}. Batch size: {args.batch_size}")
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

    elif args.image1_paths and args.prompt: # Single item inference
        logger.info("Performing single item inference.")
        img1_paths_list = args.image1_paths
        if isinstance(args.image1_paths, str):
            img1_paths_list = args.image1_paths.split(',')

        img2_paths_list = None
        if args.image2_paths:
            img2_paths_list = args.image2_paths
            if isinstance(args.image2_paths, str):
                img2_paths_list = args.image2_paths.split(',')

        state_vec_orig = np.array(args.state_vector, dtype=np.float32) if args.state_vector and args.state_vector != 'None' else None
        if state_vec_orig is not None and state_vec_orig.ndim == 0 : state_vec_orig = None
        
        state_vec_normalized = state_vec_orig
        if state_vec_orig is not None and predictor.norm_stats and 'state' in predictor.norm_stats and \
           predictor.norm_stats['state']['min'] is not None and predictor.norm_stats['state']['max'] is not None:
            state_min_tensor = predictor.norm_stats['state']['min']
            state_max_tensor = predictor.norm_stats['state']['max']
            # Ensure tensors are on CPU for numpy conversion if they are not already
            state_min_np = state_min_tensor.cpu().numpy() if isinstance(state_min_tensor, torch.Tensor) else np.array(state_min_tensor)
            state_max_np = state_max_tensor.cpu().numpy() if isinstance(state_max_tensor, torch.Tensor) else np.array(state_max_tensor)
            
            # Apply normalization: x_norm = 2 * (x - min) / (max - min) - 1
            state_vec_normalized = 2 * (state_vec_orig - state_min_np) / (state_max_np - state_min_np + 1e-8) - 1
            logger.info("Single item state vector normalized.")
        elif state_vec_orig is not None:
            logger.warning("Normalization stats for state not available. Using original state vector for single item inference.")

        preprocessed_input = predictor._preprocess_single_item(
            image_1_paths=img1_paths_list,
            prompt_text=args.prompt,
            image_2_paths=img2_paths_list,
            state_vector=state_vec_normalized, # Pass the (potentially) normalized state
            max_seq_len=args.max_seq_len_override or len(img1_paths_list),
            prompt_max_len=args.prompt_max_len_override or 128
        )
        predictions_dict = predictor.predict(preprocessed_input)
        pred_actions_continuous_normalized = predictions_dict['predicted_actions_continuous'].cpu()

        pred_actions_denormalized = pred_actions_continuous_normalized # Default
        if action_norm_stats and action_norm_stats['min'] is not None and action_norm_stats['max'] is not None:
            action_min = action_norm_stats['min'].to(pred_actions_continuous_normalized.device)
            action_max = action_norm_stats['max'].to(pred_actions_continuous_normalized.device)
            pred_actions_denormalized = denormalize(pred_actions_continuous_normalized, action_min, action_max)
            logger.info("Single item predicted action denormalized.")
        else:
            logger.warning("Action norm stats not found. Using normalized action as denormalized for single item.")
        
        all_pred_actions_continuous_denorm.append(pred_actions_denormalized)
        all_vlm_masks.append(preprocessed_input['vlm_attention_mask'].cpu()) 
        all_prompts.append(args.prompt)

    else:
        logger.error("No evaluation data provided. Please specify --eval_data_path or individual input args (--image1_paths, --prompt).")
        return

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
                        help='Path to the VLA model checkpoint (.pth.tar). Contains model, config.')
    
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