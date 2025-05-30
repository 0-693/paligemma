# Main script for evaluating/running inference with the VLA model 

import argparse
import yaml
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.loader import VLADataset, vla_collate_fn # For loading evaluation datasets
from torch.utils.data import DataLoader
from inference.predictor import VLAPredictor
from utils.misc import setup_logging, discretize_actions, undiscretize_actions

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
    output_dir = args.output_dir if args.output_dir else 
                 os.path.join(os.path.dirname(args.checkpoint_path), 'eval_results')
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'evaluation.log')
    global logger # Make logger global for access in calculate_metrics if called from elsewhere
    logger = setup_logging(log_file=log_file, name="VLAEvaluator")
    logger.info(f"Evaluation results will be saved in: {output_dir}")

    # --- Initialize Predictor ---
    logger.info(f"Initializing VLAPredictor with checkpoint: {args.checkpoint_path}")
    try:
        predictor = VLAPredictor(checkpoint_path=args.checkpoint_path, device=args.device, logger=logger)
    except Exception as e:
        logger.error(f"Error initializing VLAPredictor: {e}", exc_info=True)
        return

    all_pred_action_bins = []
    all_true_action_bins = [] # Only if evaluating a dataset with labels
    all_pred_actions_continuous = []
    all_true_actions_continuous = [] # Only if evaluating a dataset with labels
    all_vlm_masks = []
    all_prompts = [] # To store prompts for context if saving results
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
            # Move data to predictor's device (predictor.predict expects data on its device)
            # VLAPredictor._preprocess_single_item does this, but for batch, we do it here
            processed_batch = {}
            for key, tensor in batch_data.items():
                if isinstance(tensor, torch.Tensor):
                    processed_batch[key] = tensor.to(predictor.device, 
                                                     dtype=predictor.model_dtype if tensor.is_floating_point() else tensor.dtype)
                else:
                    processed_batch[key] = tensor # For non-tensor data like sequence_length list
            
            predictions = predictor.predict(processed_batch)
            all_pred_action_bins.append(predictions['predicted_action_bins'].cpu())
            all_pred_actions_continuous.append(predictions['predicted_actions_continuous'].cpu())
            all_vlm_masks.append(processed_batch['vlm_attention_mask'].cpu())
            all_prompts.extend([eval_dataset.tokenizer.decode(ids, skip_special_tokens=True) for ids in processed_batch['prompt_input_ids']])

            if 'action' in processed_batch: # True actions available for metrics
                true_actions_continuous = processed_batch['action'] # Already on device
                true_action_bins = discretize_actions(
                    true_actions_continuous, 
                    predictor.num_action_bins, 
                    predictor.action_bounds
                ).cpu()
                all_true_action_bins.append(true_action_bins)
                all_true_actions_continuous.append(true_actions_continuous.cpu())

    elif args.image1_paths and args.prompt: # Single item inference
        logger.info("Performing single item inference.")
        if not isinstance(args.image1_paths, list):
            args.image1_paths = [args.image1_paths]
        if args.image2_paths and not isinstance(args.image2_paths, list):
            args.image2_paths = [args.image2_paths]
        
        state_vec = np.array(args.state_vector, dtype=np.float32) if args.state_vector else None
        if state_vec is not None and state_vec.ndim == 0 : state_vec = None # Handle case where it might be passed as empty string then None

        preprocessed_input = predictor._preprocess_single_item(
            image_1_paths=args.image1_paths,
            prompt_text=args.prompt,
            image_2_paths=args.image2_paths,
            state_vector=state_vec,
            max_seq_len=args.max_seq_len_override or len(args.image1_paths), # Use actual length or override
            prompt_max_len=args.prompt_max_len_override or 128
        )
        predictions = predictor.predict(preprocessed_input)
        all_pred_action_bins.append(predictions['predicted_action_bins'].cpu())
        all_pred_actions_continuous.append(predictions['predicted_actions_continuous'].cpu())
        all_vlm_masks.append(preprocessed_input['vlm_attention_mask'].cpu()) # Mask for this single item
        all_prompts.append(args.prompt)
        # No true actions for single inference unless passed via args (not implemented here for simplicity)

    else:
        logger.error("No evaluation data provided. Please specify --eval_data_path or individual input args (--image1_paths, --prompt).")
        return

    # --- Calculate and Log Metrics (if true labels were available) ---
    if all_true_action_bins:
        logger.info("Calculating metrics...")
        num_action_dims = predictor.config['action_head_config']['num_action_dims']
        metrics = calculate_metrics(
            all_pred_action_bins, all_true_action_bins,
            all_pred_actions_continuous, all_true_actions_continuous,
            all_vlm_masks, num_action_dims
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
        # Flatten predictions and prepare for DataFrame
        # This part can be elaborated to save more context (e.g., prompt, image_ids if available)
        flat_predictions = []
        step_counter = 0
        for i in range(len(all_pred_actions_continuous)):
            batch_preds_cont = all_pred_actions_continuous[i] # (B, S, D)
            batch_preds_bins = all_pred_action_bins[i]       # (B, S, D)
            batch_mask = all_vlm_masks[i]                    # (B, S)
            
            for b_idx in range(batch_preds_cont.shape[0]): # Iterate through batch
                prompt_for_batch_item = all_prompts[step_counter // batch_preds_cont.shape[1]] if all_prompts else "N/A"
                step_counter += batch_preds_cont.shape[1] # Assuming all_prompts length matches number of items

                for s_idx in range(batch_preds_cont.shape[1]): # Iterate through sequence
                    if batch_mask[b_idx, s_idx].item(): # Only save valid steps
                        record = {'prompt': prompt_for_batch_item, 'sequence_step': s_idx}
                        for d_idx in range(batch_preds_cont.shape[2]): # Iterate through action dimensions
                            record[f'pred_cont_dim_{d_idx}'] = batch_preds_cont[b_idx, s_idx, d_idx].item()
                            record[f'pred_bin_dim_{d_idx}'] = batch_preds_bins[b_idx, s_idx, d_idx].item()
                            if all_true_actions_continuous and i < len(all_true_actions_continuous) and b_idx < all_true_actions_continuous[i].shape[0] and s_idx < all_true_actions_continuous[i].shape[1] and d_idx < all_true_actions_continuous[i].shape[2]:
                                record[f'true_cont_dim_{d_idx}'] = all_true_actions_continuous[i][b_idx, s_idx, d_idx].item()
                                record[f'true_bin_dim_{d_idx}'] = all_true_action_bins[i][b_idx, s_idx, d_idx].item()
                        flat_predictions.append(record)
        
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