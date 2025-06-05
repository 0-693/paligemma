# Data loading and preprocessing logic will be implemented here 

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from transformers import AutoProcessor, SiglipImageProcessor
import io
import logging
import json # Added for loading normalization stats
from utils.misc import normalize # Added for normalization

# Default image transformations - adapt as needed for PaliGemma
# Typically, Paligemma might use a SigLIP image processor.
# For now, we'll use a generic one.
DEFAULT_IMAGE_SIZE = 224
DEFAULT_IMAGE_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGE_STD = [0.229, 0.224, 0.225]

class VLAImageProcessor:
    def __init__(self, image_size=DEFAULT_IMAGE_SIZE, mean=None, std=None):
        self.image_size = image_size
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
        if mean is not None and std is not None:
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        self.transform = transforms.Compose(transform_list)
        self.logger = logging.getLogger(__name__)


    def __call__(self, image_bytes_list):
        processed_images = []
        for img_bytes in image_bytes_list:
            try:
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                processed_images.append(self.transform(image))
            except Exception as e:
                self.logger.error(f"Error processing image with VLAImageProcessor: {e}, returning zeros.")
                processed_images.append(torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32))
        return torch.stack(processed_images) if processed_images else torch.empty(0)


class VLADataset(Dataset):
    def __init__(self, parquet_files, processor_name_or_path, 
                 max_seq_len=16, prompt_max_len=128, 
                 logger=None,
                 use_siglip=True, # 新增参数，默认用siglip
                 siglip_model_name="google/siglip-base-patch16-224",
                 action_dim=7, 
                 state_dim=7,
                 normalization_stats_path=None): # Added normalization_stats_path
        self.parquet_files = parquet_files if isinstance(parquet_files, list) else [parquet_files]
        self.max_seq_len = max_seq_len
        self.prompt_max_len = prompt_max_len
        self.logger = logger if logger else logging.getLogger(__name__)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.use_siglip = use_siglip
        self.siglip_model_name = siglip_model_name
        self.normalization_stats_path = normalization_stats_path
        self.norm_stats = None

        if self.normalization_stats_path:
            try:
                with open(self.normalization_stats_path, 'r') as f:
                    self.norm_stats = json.load(f)
                self.logger.info(f"Successfully loaded normalization stats from {self.normalization_stats_path}")
                # Convert relevant parts to tensors for faster processing in __getitem__
                if self.norm_stats:
                    for key in ['state', 'action']:
                        if key in self.norm_stats:
                            for stat_type in ['min', 'max']:
                                if stat_type in self.norm_stats[key]:
                                    self.norm_stats[key][stat_type] = torch.tensor(self.norm_stats[key][stat_type], dtype=torch.float32)
            except Exception as e:
                self.logger.error(f"Error loading or parsing normalization stats from {self.normalization_stats_path}: {e}. Proceeding without normalization.")
                self.norm_stats = None
        else:
            self.logger.info("No normalization_stats_path provided. Proceeding without normalization.")


        try:
            self.processor = AutoProcessor.from_pretrained(processor_name_or_path, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer
            self.logger.info(f"Successfully loaded AutoProcessor from {processor_name_or_path}")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.logger.info(f"Tokenizer pad_token was None, set to eos_token: {self.tokenizer.eos_token}")
        except Exception as e:
            self.logger.error(f"CRITICAL: Error loading AutoProcessor from {processor_name_or_path}: {e}.")
            raise

        if self.use_siglip:
            self.logger.info(f"Using SiglipImageProcessor: {self.siglip_model_name}")
            self.siglip_image_processor = SiglipImageProcessor.from_pretrained(self.siglip_model_name)
            size_info = self.siglip_image_processor.size
            h = size_info.get('height', 224) if isinstance(size_info, dict) else size_info[0] if isinstance(size_info, (list,tuple)) and len(size_info) > 0 else 224
            w = size_info.get('width', 224) if isinstance(size_info, dict) else size_info[1] if isinstance(size_info, (list,tuple)) and len(size_info) > 1 else 224
            self.default_img_H, self.default_img_W = h, w
            self.default_img_C = 3
        else:
            raise NotImplementedError("当前仅支持SigLIP图片处理。")

        self.data = []
        self._load_data()

    def _load_data(self):
        all_episode_data = []
        for file_path in self.parquet_files:
            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                self.logger.error(f"Error reading or processing parquet file {file_path}: {e}")
                continue
            
            current_sequence = []
            for _, row in df.iterrows():
                if row.get('is_first', False) and current_sequence:
                    all_episode_data.append(current_sequence)
                    current_sequence = []
                
                sample = {
                    'image_1_bytes': row.get('image_1'), 
                    'image_2_bytes': row.get('image_2'), 
                    'state': np.array(row.get('state', np.zeros(self.state_dim, dtype=np.float32)), dtype=np.float32),
                    'action': np.array(row.get('action', np.zeros(self.action_dim, dtype=np.float32)), dtype=np.float32),
                    'is_first': bool(row.get('is_first', False)),
                    'is_last': bool(row.get('is_last', False)),
                    'is_terminal': bool(row.get('is_terminal', False)),
                    'prompt': str(row.get('prompt', ""))
                }
                current_sequence.append(sample)
            
            if current_sequence:
                all_episode_data.append(current_sequence)

        for episode in all_episode_data:
            if len(episode) >= self.max_seq_len:
                 for i in range(len(episode) - self.max_seq_len + 1):
                    self.data.append(episode[i : i + self.max_seq_len])
            elif len(episode) > 0: 
                 self.data.append(episode) 

    def __len__(self):
        return len(self.data)

    def _pad_modalities(self, data_list, max_len, feature_dim_or_shape, dtype):
        current_len = len(data_list)
        
        if current_len == 0: # Empty list
            if isinstance(feature_dim_or_shape, tuple): # Image shape (C,H,W)
                pad_shape_with_len = (max_len,) + feature_dim_or_shape
            else: # State/action dim (D)
                pad_shape_with_len = (max_len, feature_dim_or_shape)
            return torch.zeros(pad_shape_with_len, dtype=dtype)

        example_item = data_list[0]
        item_dtype = example_item.dtype # Get dtype from actual items
        
        padded_list = list(data_list) # Make a mutable copy

        if current_len < max_len:
            padding_needed = max_len - current_len
            pad_shape = example_item.shape 
            for _ in range(padding_needed):
                padded_list.append(torch.zeros(pad_shape, dtype=item_dtype))
        elif current_len > max_len:
            padded_list = padded_list[:max_len]
        
        try:
            return torch.stack(padded_list, dim=0)
        except RuntimeError as e:
            self.logger.error(f"RuntimeError during torch.stack for _pad_modalities. Data length: {len(padded_list)}")
            for i, t in enumerate(padded_list):
                self.logger.error(f"Item {i} shape: {t.shape}, dtype: {t.dtype}")
            raise

    def __getitem__(self, idx):
        sequence_data_raw = self.data[idx]
        current_actual_seq_len = len(sequence_data_raw)

        # 1. 主摄像头图片处理
        processed_pixel_values_1_list = []
        processed_pixel_values_2_list = []
        for step in sequence_data_raw:
            img1_bytes = step.get('image_1_bytes')
            img2_bytes = step.get('image_2_bytes')
            # image_1
            if img1_bytes:
                try:
                    pil_img1 = Image.open(io.BytesIO(img1_bytes)).convert("RGB")
                    tensor_img1 = self.siglip_image_processor(images=pil_img1, return_tensors="pt").pixel_values.squeeze(0)
                except Exception as e:
                    self.logger.warning(f"Corrupted image_1 bytes at idx {idx}. Error: {e}")
                    tensor_img1 = torch.zeros((self.default_img_C, self.default_img_H, self.default_img_W), dtype=torch.float32)
            else:
                tensor_img1 = torch.zeros((self.default_img_C, self.default_img_H, self.default_img_W), dtype=torch.float32)
            processed_pixel_values_1_list.append(tensor_img1)
            # image_2
            if img2_bytes:
                try:
                    pil_img2 = Image.open(io.BytesIO(img2_bytes)).convert("RGB")
                    tensor_img2 = self.siglip_image_processor(images=pil_img2, return_tensors="pt").pixel_values.squeeze(0)
                except Exception as e:
                    self.logger.warning(f"Corrupted image_2 bytes at idx {idx}. Error: {e}")
                    tensor_img2 = torch.zeros((self.default_img_C, self.default_img_H, self.default_img_W), dtype=torch.float32)
            else:
                tensor_img2 = torch.zeros((self.default_img_C, self.default_img_H, self.default_img_W), dtype=torch.float32)
            processed_pixel_values_2_list.append(tensor_img2)

        images_1_padded = self._pad_modalities(processed_pixel_values_1_list, self.max_seq_len, (self.default_img_C, self.default_img_H, self.default_img_W), torch.float32)
        images_2_padded = self._pad_modalities(processed_pixel_values_2_list, self.max_seq_len, (self.default_img_C, self.default_img_H, self.default_img_W), torch.float32)

        # 2. Raw Prompt Text (consistent for the sequence)
        raw_prompt_text = sequence_data_raw[0]['prompt'] if sequence_data_raw else ""

        # 3. States and Actions
        states_list = [torch.tensor(step['state'], dtype=torch.float32) for step in sequence_data_raw]
        actions_list = [torch.tensor(step['action'], dtype=torch.float32) for step in sequence_data_raw]
        
        states_padded_orig = self._pad_modalities(states_list, self.max_seq_len, self.state_dim, torch.float32)
        actions_padded_orig = self._pad_modalities(actions_list, self.max_seq_len, self.action_dim, torch.float32)

        states_padded_normalized = states_padded_orig
        actions_padded_normalized = actions_padded_orig

        # Apply normalization if stats are available
        if self.norm_stats:
            if 'state' in self.norm_stats and self.norm_stats['state']['min'] is not None and self.norm_stats['state']['max'] is not None:
                # Create a mask for valid (non-padded) steps before normalization
                valid_steps_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
                valid_steps_mask[:current_actual_seq_len] = True
                
                # Only normalize valid steps; padded steps remain zero (or their original padding value)
                # which should be fine as (0 - min) / (max - min) - 1 will not be in [-1, 1] unless 0 is within original range.
                # It's often better to normalize, then pad. But here data is padded first.
                # So, we select valid steps, normalize, and then place them back or rely on broadcasting.
                
                # Normalize valid part of states_padded_orig
                valid_states_to_norm = states_padded_orig[valid_steps_mask]
                if valid_states_to_norm.numel() > 0: # Check if there are any valid states
                    normalized_valid_states = normalize(valid_states_to_norm, 
                                                        self.norm_stats['state']['min'], 
                                                        self.norm_stats['state']['max'])
                    # Create a new tensor for normalized states and fill it
                    states_padded_normalized = torch.zeros_like(states_padded_orig)
                    states_padded_normalized[valid_steps_mask] = normalized_valid_states
                else: # If no valid states, keep as original (should be all zeros if padding was zero)
                    states_padded_normalized = states_padded_orig


            if 'action' in self.norm_stats and self.norm_stats['action']['min'] is not None and self.norm_stats['action']['max'] is not None:
                valid_steps_mask = torch.zeros(self.max_seq_len, dtype=torch.bool) # Re-create for actions
                valid_steps_mask[:current_actual_seq_len] = True

                valid_actions_to_norm = actions_padded_orig[valid_steps_mask]
                if valid_actions_to_norm.numel() > 0:
                    normalized_valid_actions = normalize(actions_padded_orig[valid_steps_mask], 
                                                         self.norm_stats['action']['min'], 
                                                         self.norm_stats['action']['max'])
                    actions_padded_normalized = torch.zeros_like(actions_padded_orig)
                    actions_padded_normalized[valid_steps_mask] = normalized_valid_actions
                else:
                    actions_padded_normalized = actions_padded_orig


        # 4. VLM attention mask for valid (non-padded) sequence steps
        vlm_attention_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        vlm_attention_mask[:current_actual_seq_len] = True
        # 5. Meta flags (padded)
        is_first_flags = [step['is_first'] for step in sequence_data_raw] + [False] * (self.max_seq_len - current_actual_seq_len)
        is_last_flags = [step['is_last'] for step in sequence_data_raw] + [False] * (self.max_seq_len - current_actual_seq_len)
        is_terminal_flags = [step['is_terminal'] for step in sequence_data_raw] + [False] * (self.max_seq_len - current_actual_seq_len)
        is_first_tensor = torch.tensor(is_first_flags[:self.max_seq_len], dtype=torch.bool)
        is_last_tensor = torch.tensor(is_last_flags[:self.max_seq_len], dtype=torch.bool)
        is_terminal_tensor = torch.tensor(is_terminal_flags[:self.max_seq_len], dtype=torch.bool)
        # 6. Tokenized "pure" prompt (for other potential uses, not for PaliGemmaVLM directly if it uses processor)
        tokenized_pure_text_prompt = self.tokenizer(
            text=raw_prompt_text,
            return_tensors="pt",
            padding="max_length", 
            truncation=True, 
            max_length=self.prompt_max_len 
        )
        # print(f"======state:{states_padded}\n")
        # print(f"======action:{actions_padded}\n")

        # Log for debugging normalization
        if self.norm_stats and current_actual_seq_len > 0: # Log only if stats loaded and sequence is not empty
            # Log first valid original action (before padding and normalization)
            original_action_first_step = actions_list[0]
            # Log first valid normalized action (after normalization and padding)
            normalized_action_first_step = actions_padded_normalized[0]
            
            self.logger.debug(f"Dataset ID: {idx} | Actual Seq Len: {current_actual_seq_len}")
            self.logger.debug(f"  Action Original (first valid): {original_action_first_step.tolist()}")
            self.logger.debug(f"  Action Normalized (first valid): {normalized_action_first_step.tolist()}")
            if torch.allclose(original_action_first_step, normalized_action_first_step) and torch.norm(original_action_first_step) > 10: # Heuristic: if they are same and norm is large, norm likely failed
                self.logger.warning(f"Dataset ID: {idx} - Normalized action seems same as original and has large norm. Normalization might not have been applied as expected.")
                self.logger.warning(f"    Action Min Stats: {self.norm_stats['action']['min'].tolist()}")
                self.logger.warning(f"    Action Max Stats: {self.norm_stats['action']['max'].tolist()}")

        return {
            "raw_prompt_text": raw_prompt_text,         # Raw text string
            "image_1": images_1_padded,                 # (max_seq_len, C, H, W)
            "image_2": images_2_padded,                 # (max_seq_len, C, H, W)
            "state": states_padded_normalized,           # Use normalized state
            "action": actions_padded_normalized,         # Use normalized action as GT for training
            "input_ids_pure_text": tokenized_pure_text_prompt["input_ids"].squeeze(0), # (prompt_max_len)
            "attention_mask_pure_text": tokenized_pure_text_prompt["attention_mask"].squeeze(0).bool(), # (prompt_max_len)
            "vlm_attention_mask": vlm_attention_mask,   # (max_seq_len)
            "is_first_frame": is_first_tensor,          # (max_seq_len)
            "is_last_frame": is_last_tensor,            # (max_seq_len)
            "is_terminal_frame": is_terminal_tensor,    # (max_seq_len)
            "sequence_length": current_actual_seq_len   # Scalar
        }

def vla_collate_fn(batch):
    collated = {}
    elem = batch[0]

    for key in elem:
        if key == "raw_prompt_text":
            collated[key] = [d[key] for d in batch] # List of B strings
        elif isinstance(elem[key], torch.Tensor):
            collated[key] = torch.stack([d[key] for d in batch])
        elif isinstance(elem[key], (int, float, bool)): # Handle scalar metadata like sequence_length
             collated[key] = torch.tensor([d[key] for d in batch])
        # else: # Potentially other types of data
        #    collated[key] = [d[key] for d in batch]

    # Ensure sequence_length is a tensor if not already handled
    if "sequence_length" in elem and not isinstance(collated.get("sequence_length"), torch.Tensor):
        collated["sequence_length"] = torch.tensor([d["sequence_length"] for d in batch])
        
    return collated


if __name__ == '__main__':
    print("Starting VLADataset test script...")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create dummy parquet files for testing
    data1 = {
        'image_1': [b"dummy_image_bytes_1"] * 5, # Placeholder for actual image bytes
        'state': [np.random.rand(7).astype(np.float32) for _ in range(5)],
        'action': [np.random.rand(7).astype(np.float32) for _ in range(5)],
        'prompt': ["Describe this scene with a robot"] * 5,
        'is_first': [True, False, False, False, False],
        'is_last': [False, False, False, False, True],
        'is_terminal': [False, False, False, False, True],
    }
    df1 = pd.DataFrame(data1)
    df1.to_parquet("dummy_train_data1.parquet")

    data2 = {
        'image_1': [b"dummy_image_bytes_2"] * 3,
        'state': [np.random.rand(7).astype(np.float32) for _ in range(3)],
        'action': [np.random.rand(7).astype(np.float32) for _ in range(3)],
        'prompt': ["What is the robot doing?"] * 3,
        'is_first': [True, False, False],
        'is_last': [False, False, True],
        'is_terminal': [False, False, True],
    }
    df2 = pd.DataFrame(data2)
    df2.to_parquet("dummy_train_data2.parquet")
    
    # Mock processor path (ensure it exists or use a real one for actual testing)
    # For this test, we assume PaliGemma processor is available.
    # If not, this will fail at AutoProcessor.from_pretrained.
    # Use a common small model for testing if a full PaliGemma model isn't needed just for processor.
    # However, PaliGemma has specific processor needs.
    # processor_name = "google/paligemma-3b-pt-224" 
    # To make this test runnable without downloading, we need to handle processor loading more carefully
    # For now, assume the path in config (`./weight/paligemma-3b-pt-224`) is valid if running in context.
    # The test will likely fail if that path doesn't have a valid processor config.
    # This path is from the project's config.
    processor_path = "./weight/paligemma-3b-pt-224" 
    
    try:
        # Create a dummy processor config if the path doesn't exist, for basic testing structure
        import os
        from transformers import AutoTokenizer, SiglipImageProcessor
        if not os.path.exists(processor_path):
            logger.warning(f"Processor path {processor_path} not found. Attempting to create dummy processor for test.")
            try:
                # Create dummy tokenizer and image processor files
                dummy_tokenizer_path = os.path.join(processor_path, "tokenizer")
                dummy_image_processor_path = os.path.join(processor_path, "image_processor")
                os.makedirs(dummy_tokenizer_path, exist_ok=True)
                os.makedirs(dummy_image_processor_path, exist_ok=True)

                # Save dummy configs - this is very simplified and might not be enough
                # For a real test, a valid processor is needed.
                # This is more to check the VLADataset structure than processor functionality.
                try:
                    tokenizer = AutoTokenizer.from_pretrained("gpt2") # A common, small tokenizer
                    tokenizer.save_pretrained(processor_path) # Saves tokenizer.json, etc.
                    #PaliGemma uses a custom image processor often based on SigLIP
                    # image_processor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")
                    # image_processor.save_pretrained(processor_path) # Saves preprocessor_config.json
                    # For simplicity, let's assume if the dir exists, it's okay for this structural test.
                    # The actual PaliGemma processor might need more specific files.
                    logger.info(f"Saved dummy tokenizer to {processor_path}. Image processor part is complex for dummy.")
                except Exception as e_dummy_save:
                    logger.error(f"Could not create dummy processor files: {e_dummy_save}. Test may fail.")

            except Exception as e_dummy:
                 logger.error(f"Failed to create dummy processor setup: {e_dummy}. This test run might fail at processor loading.")


        logger.info(f"Attempting to load VLADataset with processor: {processor_path}")
        vla_dataset = VLADataset(
            parquet_files=["dummy_train_data1.parquet", "dummy_train_data2.parquet"],
            processor_name_or_path=processor_path, 
            max_seq_len=4,
            prompt_max_len=32,
            logger=logger,
            action_dim=7,
            state_dim=7,
            use_siglip=True, # Test with SigLIP processor first
            siglip_model_name="google/siglip-base-patch16-224"
        )
        logger.info(f"VLADataset loaded. Number of samples: {len(vla_dataset)}")

        if len(vla_dataset) > 0:
            sample = vla_dataset[0]
            logger.info(f"Sample 0 keys: {sample.keys()}")
            logger.info(f"Raw prompt: {sample['raw_prompt_text']}")
            logger.info(f"Image_1 shape: {sample['image_1'].shape}, dtype: {sample['image_1'].dtype}")
            logger.info(f"Image_2 shape: {sample['image_2'].shape}, dtype: {sample['image_2'].dtype}")
            logger.info(f"State shape: {sample['state'].shape}")
            logger.info(f"Action shape: {sample['action'].shape}")
            logger.info(f"Input IDs (pure text) shape: {sample['input_ids_pure_text'].shape}")
            logger.info(f"Attention Mask (pure text) shape: {sample['attention_mask_pure_text'].shape}")
            logger.info(f"VLM Attention Mask shape: {sample['vlm_attention_mask'].shape}")
            logger.info(f"Sequence length: {sample['sequence_length']}")

            # Test DataLoader
            vla_dataloader = DataLoader(vla_dataset, batch_size=2, collate_fn=vla_collate_fn)
            batch = next(iter(vla_dataloader))
            logger.info(f"Batch keys: {batch.keys()}")
            logger.info(f"Batch raw_prompt_text: {batch['raw_prompt_text']}")
            logger.info(f"Batch Image_1 shape: {batch['image_1'].shape}")
            logger.info(f"Batch Image_2 shape: {batch['image_2'].shape}")
            logger.info(f"Batch State shape: {batch['state'].shape}")
            logger.info(f"Batch sequence_length: {batch['sequence_length']}")

        else:
            logger.warning("VLADataset is empty after loading. Check _load_data and parquet files.")

    except ImportError as e_import:
        logger.error(f"ImportError during test: {e_import}. Make sure transformers library is installed and accessible.")
    except Exception as e:
        logger.error(f"An error occurred during VLADataset test: {e}", exc_info=True)
    finally:
        # Clean up dummy files
        if os.path.exists("dummy_train_data1.parquet"): os.remove("dummy_train_data1.parquet")
        if os.path.exists("dummy_train_data2.parquet"): os.remove("dummy_train_data2.parquet")
        # if os.path.exists(processor_path): # Be careful with removing this if it's a real path
        #     import shutil
        #     # Check if it was a dummy creation path
        #     if "dummy_processor_for_test" in processor_path : # A more robust check might be needed
        #          shutil.rmtree(processor_path) 
        #          logger.info(f"Cleaned up dummy processor at {processor_path}")
    print("VLADataset test script finished.") 