import sys
import os
# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import os
import torch
import numpy as np
import json
from tqdm import tqdm
import logging

from data.loader import VLADataset, vla_collate_fn
from torch.utils.data import DataLoader
from utils.config_utils import OmegaConfAttrDict
from utils.misc import setup_logging

def main(args):
    logger = setup_logging(name="NormalizationStats", log_level=logging.INFO)
    logger.info("Starting normalization statistics calculation.")

    if not args.config_path:
        logger.error("No configuration file provided. Exiting.")
        return

    try:
        with open(args.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = OmegaConfAttrDict(config_dict)
        logger.info(f"Loaded configuration from: {args.config_path}")
    except Exception as e:
        logger.error(f"Error loading YAML config from {args.config_path}: {e}")
        return

    # Override data paths if provided
    if args.train_data_override:
        config.data.train_parquet_files = args.train_data_override.split(',')
    if args.val_data_override:
        config.data.val_parquet_files = args.val_data_override.split(',')

    common_dataset_params = {
        "processor_name_or_path": config.model.vlm_config.model_name_or_path,
        "max_seq_len": config.data.max_seq_len,
        "prompt_max_len": config.data.prompt_max_len,
        "action_dim": config.model.action_head_config.num_action_dims,
        "state_dim": config.data.get('state_dim', 7),
        "logger": logger,
        "use_siglip": True,
        "siglip_model_name": getattr(config.data, 'siglip_model_name', 'google/siglip-base-patch16-224'),
    }

    all_states = []
    all_actions = []

    datasets_to_process = []
    if config.data.train_parquet_files:
        try:
            train_dataset = VLADataset(
                parquet_files=config.data.train_parquet_files,
                **common_dataset_params
            )
            datasets_to_process.append(train_dataset)
            logger.info(f"Loaded training dataset. Samples: {len(train_dataset)}")
        except Exception as e:
            logger.error(f"Error loading training dataset: {e}")

    if config.data.val_parquet_files and os.path.exists(
        config.data.val_parquet_files[0] if isinstance(config.data.val_parquet_files, list) else config.data.val_parquet_files
    ):
        try:
            val_dataset = VLADataset(
                parquet_files=config.data.val_parquet_files,
                **common_dataset_params
            )
            datasets_to_process.append(val_dataset)
            logger.info(f"Loaded validation dataset. Samples: {len(val_dataset)}")
        except Exception as e:
            logger.error(f"Error loading validation dataset: {e}")

    if not datasets_to_process:
        logger.error("No datasets loaded. Exiting.")
        return

    for dataset in datasets_to_process:
        logger.info(f"Processing dataset with {len(dataset)} samples...")
        for i in tqdm(range(len(dataset)), desc="Processing samples"):
            try:
                sample = dataset[i]

                valid_mask = sample.get('vlm_attention_mask', None)
                if valid_mask is None:
                    logger.warning(f"Sample {i} missing 'vlm_attention_mask'. Assuming all steps valid.")
                    valid_mask = torch.ones(sample['state'].shape[0], dtype=torch.bool)

                state_tensor = sample['state']
                action_tensor = sample['action']

                valid_states = state_tensor[valid_mask]
                valid_actions = action_tensor[valid_mask]

                if valid_states.numel() > 0:
                    all_states.append(valid_states.cpu().numpy())
                if valid_actions.numel() > 0:
                    all_actions.append(valid_actions.cpu().numpy())

            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}", exc_info=True)
                continue

    if not all_states or not all_actions:
        logger.error("No valid states or actions collected. Cannot compute statistics.")
        return

    all_states_np = np.concatenate(all_states, axis=0)
    all_actions_np = np.concatenate(all_actions, axis=0)

    logger.info(f"Total valid states collected: {all_states_np.shape}")
    logger.info(f"Total valid actions collected: {all_actions_np.shape}")

    stats = {
        'state': {
            'min': np.min(all_states_np, axis=0).tolist(),
            'max': np.max(all_states_np, axis=0).tolist(),
            'mean': np.mean(all_states_np, axis=0).tolist(),
            'std': np.std(all_states_np, axis=0).tolist(),
        },
        'action': {
            'min': np.min(all_actions_np, axis=0).tolist(),
            'max': np.max(all_actions_np, axis=0).tolist(),
            'mean': np.mean(all_actions_np, axis=0).tolist(),
            'std': np.std(all_actions_np, axis=0).tolist(),
        }
    }

    output_path = args.output_path or "normalization_stats.json"
    try:
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=4)
        logger.info(f"Normalization statistics saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving statistics to {output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate normalization statistics for VLA dataset.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--output_path", type=str, default="normalization_stats.json", help="Where to save the output JSON.")
    parser.add_argument("--train_data_override", type=str, help="Optional: override train parquet paths (comma-separated).")
    parser.add_argument("--val_data_override", type=str, help="Optional: override val parquet paths (comma-separated).")
    args = parser.parse_args()
    main(args)
