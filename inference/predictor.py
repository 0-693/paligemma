# Inference logic will be implemented here 

import torch
import yaml
import os
import json

# Assuming scripts are run from a context where 'model', 'utils', 'data' are top-level.
from model.vla_model import VLAModel
from utils.misc import load_checkpoint, undiscretize_actions, setup_logging, denormalize
from data.loader import VLAImageProcessor, DEFAULT_IMAGE_SIZE # For standalone image processing if needed
from transformers import AutoTokenizer # For standalone text processing if needed
from PIL import Image
import io
import numpy as np

class VLAPredictor:
    def __init__(self, checkpoint_path, device=None, logger=None):
        """
        Predictor for the Vision-Language-Action Model.
        Args:
            checkpoint_path (str): Path to the model checkpoint file (.pth.tar).
            device (str, optional): Device to run inference on ('cpu', 'cuda'). Autodetects if None.
            logger (logging.Logger, optional): Logger instance.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.logger = logger if logger else setup_logging(name='VLAPredictor')
        self.logger.info(f"Using device: {self.device}")

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        self.logger.info(f"Loading checkpoint from '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location='cpu') 
        
        self.config = checkpoint.get('config')
        if not self.config:
            # Try to load from old checkpoint format if config is not at top level
            if checkpoint.get('model_config') and checkpoint.get('train_config'):
                self.logger.warning("Loading config from older checkpoint format.")
                self.config = {
                    'model': checkpoint['model_config'],
                    'training': checkpoint['train_config'],
                    'data': checkpoint.get('data_config', {}), # Add data_config if present
                    'optimizer': checkpoint.get('optimizer_config', {}), # Add optimizer_config if present
                    'lr_scheduler': checkpoint.get('lr_scheduler_config', {})
                }
                # Convert to OmegaConfAttrDict if you were using it, otherwise basic dict is fine for access.
                from utils.config_utils import OmegaConfAttrDict
                self.config = OmegaConfAttrDict(self.config)
            else:
                raise ValueError("Config not found in checkpoint. Cannot initialize model.")

        # Load normalization stats if path is in config
        self.norm_stats = None
        self.normalization_stats_path = self.config.data.get('normalization_stats_path', None)
        if self.normalization_stats_path and os.path.isabs(self.normalization_stats_path):
            # If absolute path is stored, use it directly
            pass
        elif self.normalization_stats_path:
            # If relative, assume it's relative to the checkpoint's directory or a known root
            # For simplicity, let's try relative to checkpoint dir first
            potential_path = os.path.join(os.path.dirname(checkpoint_path), self.normalization_stats_path)
            if os.path.exists(potential_path):
                self.normalization_stats_path = potential_path
            else:
                # Fallback: assume it might be relative to CWD or a predefined project root if needed
                # For now, if not found relative to checkpoint, it might fail later or use None.
                self.logger.warning(f"Normalization stats path \"{self.normalization_stats_path}\" from config is relative. Tried relative to checkpoint, but {potential_path} not found. Ensure path is correct or absolute.")

        if self.normalization_stats_path and os.path.exists(self.normalization_stats_path):
            try:
                with open(self.normalization_stats_path, 'r') as f:
                    self.norm_stats = json.load(f)
                self.logger.info(f"Successfully loaded normalization stats from {self.normalization_stats_path}")
                if self.norm_stats:
                    for key in ['state', 'action']:
                        if key in self.norm_stats:
                            for stat_type in ['min', 'max']:
                                if stat_type in self.norm_stats[key]:
                                    self.norm_stats[key][stat_type] = torch.tensor(self.norm_stats[key][stat_type], dtype=self.model_dtype) # Use model_dtype
            except Exception as e:
                self.logger.error(f"Error loading or parsing normalization stats from {self.normalization_stats_path}: {e}. Proceeding without denormalization capabilities for direct calls.")
                self.norm_stats = None
        elif self.normalization_stats_path:
            self.logger.warning(f"Normalization stats file not found at resolved path: {self.normalization_stats_path}. Proceeding without denormalization capabilities for direct calls.")
        else:
            self.logger.info("No normalization_stats_path found in config. Denormalization will not be available internally.")

        # Determine model dtype from config or default to float32 for CPU / float16 for CUDA
        model_dtype_str = self.config.model.vlm_config.get('dtype', None) # Assuming dtype was stored in vlm_config
        if model_dtype_str == 'torch.float16':
            self.model_dtype = torch.float16
        elif model_dtype_str == 'torch.float32':
            self.model_dtype = torch.float32
        else:
            self.model_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        self.logger.info(f"Using model dtype: {self.model_dtype}")

        # Initialize model with loaded configuration
        self.model = VLAModel(
            vlm_config=self.config['vlm_config'], 
            action_head_config=self.config['action_head_config'],
            device=self.device, # Will be moved to device within VLAModel
            dtype=self.model_dtype
        )
        
        # Load model state dict (handling potential 'module.' prefix)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device) # Ensure model is on the final device
        self.model.eval()
        self.logger.info("Model loaded successfully and set to eval mode.")

        # For convenience if direct data prep is needed (though batch input is preferred)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.get('data_config', {}).get('tokenizer_name_or_path', 'bert-base-uncased'))
        # Assuming image processor config is compatible with VLAImageProcessor or a similar HF one was used.
        img_proc_name = self.config.get('data_config', {}).get('image_processor_name_or_path', None)
        if img_proc_name and 'vit' in img_proc_name.lower(): # Basic check for HF ViT processor
            from transformers import AutoImageProcessor
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(img_proc_name)
            except Exception as e:
                self.logger.warning(f"Could not load HF Image Processor {img_proc_name}: {e}. Using custom VLAImageProcessor.")
                self.image_processor = VLAImageProcessor(image_size=self.config.get('data_config',{}).get('image_height', DEFAULT_IMAGE_SIZE))
        else:
            self.image_processor = VLAImageProcessor(image_size=self.config.get('data_config',{}).get('image_height', DEFAULT_IMAGE_SIZE))

        self.num_action_bins = self.config['action_head_config']['num_action_bins']
        self.action_bounds = self.config.get('action_bounds', (-1.0, 1.0))


    def _preprocess_single_item(self, image_1_paths, prompt_text, image_2_paths=None, state_vector=None, max_seq_len=1, prompt_max_len=128):
        """
        Helper to preprocess a single data item (a sequence of one frame for typical inference).
        This is a simplified version. For multi-frame sequences, adapt from VLADataset.
        Args:
            image_1_paths (list of str): List of paths to main camera images (for the sequence, usually 1 for single step).
            prompt_text (str): Natural language prompt.
            image_2_paths (list of str, optional): List of paths to wrist camera images.
            state_vector (np.ndarray, optional): Robot state vector for the current step, shape (state_dim,).
            max_seq_len (int): Sequence length to pad to (usually 1 for single step inference).
            prompt_max_len (int): Max length for tokenized prompt.
        Returns:
            dict: A dictionary of tensors ready for model input.
        """
        # Image processing
        def process_imgs(img_paths_list):
            processed = []
            if not img_paths_list: return torch.empty(0)
            for img_path in img_paths_list:
                img = Image.open(img_path).convert("RGB")
                if hasattr(self.image_processor, 'preprocess'): # HF processor
                    processed.append(self.image_processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0))
                else: # Custom VLAImageProcessor
                    # VLAImageProcessor expects bytes, so we simulate it for consistency
                    byte_arr = io.BytesIO()
                    img.save(byte_arr, format='PNG')
                    processed.append(self.image_processor([byte_arr.getvalue()]).squeeze(0))
            return torch.stack(processed) if processed else torch.empty(0)

        batch_image_1 = process_imgs(image_1_paths).unsqueeze(0) # (1, S, C, H, W)
        batch_image_2 = process_imgs(image_2_paths).unsqueeze(0) if image_2_paths and self.model.paligemma_vlm.use_secondary_camera else None

        # Text tokenization
        tokenized_prompt = self.tokenizer(text=prompt_text, return_tensors="pt", padding="max_length", truncation=True, max_length=prompt_max_len)
        batch_prompt_ids = tokenized_prompt["input_ids"]
        batch_prompt_mask = tokenized_prompt["attention_mask"].bool()

        # State vector
        batch_state = None
        if state_vector is not None and self.model.action_head.use_state_input:
            batch_state = torch.tensor(state_vector, dtype=self.model_dtype).unsqueeze(0).unsqueeze(0) # (1, 1, state_dim)
            if max_seq_len > 1:
                 batch_state = batch_state.repeat(1, max_seq_len, 1) # (1, S, state_dim)
        
        # VLM attention mask (assuming all frames in this short sequence are valid)
        batch_vlm_mask = torch.ones((1, len(image_1_paths)), dtype=torch.bool)
        
        # Padding to max_seq_len (conceptual, actual padding might be more complex for sequences)
        # This simple version assumes len(image_1_paths) <= max_seq_len
        # For robust sequence padding, refer to VLADataset
        # For single step inference, max_seq_len is often 1.
        current_s = len(image_1_paths)
        pad_s = max_seq_len - current_s
        if pad_s < 0: pad_s = 0 # Should not happen if inputs are for single step or less than max_seq_len

        def pad_tensor(tensor, target_s_dim, val=0):
            if tensor is None or tensor.shape[1] == target_s_dim : return tensor
            padding = torch.full((tensor.shape[0], pad_s, *tensor.shape[2:]), val, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, padding], dim=1)

        batch_image_1 = pad_tensor(batch_image_1, max_seq_len)
        if batch_image_2 is not None: batch_image_2 = pad_tensor(batch_image_2, max_seq_len)
        if batch_state is not None: batch_state = pad_tensor(batch_state, max_seq_len)
        padding_mask = torch.zeros((1, pad_s), dtype=torch.bool, device=self.device)
        batch_vlm_mask = torch.cat([batch_vlm_mask, padding_mask], dim=1) if pad_s > 0 else batch_vlm_mask
        
        return {
            'image_1_batch': batch_image_1.to(self.device, dtype=self.model_dtype),
            'image_2_batch': batch_image_2.to(self.device, dtype=self.model_dtype) if batch_image_2 is not None else None,
            'prompt_input_ids': batch_prompt_ids.to(self.device),
            'prompt_attention_mask': batch_prompt_mask.to(self.device),
            'vlm_attention_mask': batch_vlm_mask.to(self.device),
            'state_batch': batch_state.to(self.device, dtype=self.model_dtype) if batch_state is not None else None
        }

    def predict(self, batch):
        self.model.eval()
        with torch.no_grad():
            image_1_batch = batch['image_1'].to(self.device, dtype=self.model_dtype)
            raw_prompt_texts_batch = batch['raw_prompt_text']
            vlm_attention_mask = batch['vlm_attention_mask'].to(self.device)
            state_batch = batch.get('state', None)
            if state_batch is not None:
                state_batch = state_batch.to(self.device, dtype=self.model_dtype) # state is now normalized, should match model_dtype
            image_2_batch = batch.get('image_2', None)
            if image_2_batch is not None:
                image_2_batch = image_2_batch.to(self.device, dtype=self.model_dtype)

            # Model outputs normalized actions if training was done with normalized actions
            action_pred_normalized = self.model(
                image_1_batch=image_1_batch,
                raw_prompt_texts_batch=raw_prompt_texts_batch,
                vlm_attention_mask_batch=vlm_attention_mask,
                state_batch=state_batch, # This is the normalized state from DataLoader
                image_2_batch=image_2_batch
            )
            
            # The VLAPredictor itself will return the normalized prediction.
            # Denormalization should happen in the script that calls the predictor and uses the action.
            # This keeps VLAPredictor focused on prediction and VLAModel on its core logic.
            
            # For compatibility with existing main_eval.py that might expect discretized actions:
            # If action_pred_normalized is (B, S, D_action_continuous)
            # And original code expects bins, we might need to handle that.            
            # Current VLAModel directly outputs continuous actions (or velocities if flow matching).
            # The `discretize_actions` utility is separate.

            # For now, assume `action_pred_normalized` is the continuous output.
            # The `main_eval.py` script will be responsible for denormalizing it if needed.
            # It might also discretize the original or denormalized version for some metrics.

            # Prepare a dictionary similar to what main_eval.py might expect
            # This part needs to align with how VLAModel output is structured and how main_eval consumes it.
            # Assuming VLAModel.forward returns a tensor of shape (B, S, D_action_continuous)
            
            # Let's rename for clarity, this is still normalized.
            predicted_actions_continuous_normalized = action_pred_normalized

            # If main_eval.py needs binned actions, it should create them from the continuous ones.
            # We can provide a placeholder or calculate if really needed here.
            # For now, let main_eval handle discretization from continuous.
            predicted_action_bins_placeholder = torch.zeros_like(predicted_actions_continuous_normalized, dtype=torch.long)

            return {
                "predicted_actions_continuous": predicted_actions_continuous_normalized, # This is normalized
                "predicted_action_bins": predicted_action_bins_placeholder # Placeholder, main_eval to generate if needed
            }

# Example Usage (Conceptual)
if __name__ == '__main__':
    logger = setup_logging(name="VLAPredictorTest")
    logger.info("Conceptual VLAPredictor Test")

    # This test assumes a checkpoint and its config are available.
    # In a real scenario, you'd train a model first or use a pre-trained one.
    # Create a dummy checkpoint for testing structure:
    dummy_checkpoint_dir = './dummy_predictor_checkpoint'
    os.makedirs(dummy_checkpoint_dir, exist_ok=True)
    dummy_checkpoint_path = os.path.join(dummy_checkpoint_dir, 'dummy_model.pth.tar')

    # Create a dummy config that matches what VLAModel and VLATrainer expect
    dummy_config = {
        'vlm_config': {
            'model_name_or_path': "hf-internal-testing/tiny-random-PaliGemmaForConditionalGeneration", # Needs a tiny model for init
            'use_secondary_camera': False,
            'dtype': 'torch.float32' # Specify dtype as string
        },
        'action_head_config': {
            'state_dim': 7, 'num_action_dims': 7, 'num_action_bins': 20,
            'use_state_input': False, 'hidden_layers_config': [32],
        },
        'data_config': { # For tokenizer/processor paths
             'tokenizer_name_or_path': 'bert-base-uncased',
             'image_processor_name_or_path': 'google/vit-base-patch16-224-in21k' # Or a VLAImageProcessor compatible config
        },
        'action_bounds': (-1.0, 1.0)
        # Other trainer configs are not strictly needed for predictor if model is directly loaded
    }
    
    # Create a dummy VLAModel state_dict based on the config
    try:
        # Attempt to initialize a model to get its state_dict structure
        # This requires the tiny model to be actually available or transformers to handle it gracefully.
        temp_model = VLAModel(dummy_config['vlm_config'], dummy_config['action_head_config'], device='cpu', dtype=torch.float32)
        dummy_state_dict = temp_model.state_dict()
        torch.save({'state_dict': dummy_state_dict, 'config': dummy_config}, dummy_checkpoint_path)
        logger.info(f"Created dummy checkpoint at {dummy_checkpoint_path}")

        # Initialize Predictor
        predictor = VLAPredictor(checkpoint_path=dummy_checkpoint_path, logger=logger)

        # Create dummy single item input for prediction (mimicking _preprocess_single_item)
        # For this, we need dummy image files.
        dummy_img_dir = os.path.join(dummy_checkpoint_dir, "dummy_images")
        os.makedirs(dummy_img_dir, exist_ok=True)
        dummy_img_path1 = os.path.join(dummy_img_dir, "img1.png")
        Image.new('RGB', (224,224), color='red').save(dummy_img_path1)

        # Use the predictor's preprocessing helper for a single step
        # Typically, for single-step inference, max_seq_len=1
        single_item_preprocessed = predictor._preprocess_single_item(
            image_1_paths=[dummy_img_path1],
            prompt_text="Pick up the red block.",
            max_seq_len=1, # For single step action prediction
            prompt_max_len=32
        )
        logger.info("Preprocessed single item for prediction.")

        # Perform prediction
        prediction_output = predictor.predict(single_item_preprocessed)
        logger.info(f"Prediction successful.")
        logger.info(f"Predicted action logits shape: {prediction_output['predicted_actions_continuous'].shape}")
        logger.info(f"Sample continuous action: {prediction_output['predicted_actions_continuous'][0,0,:]}") # B=0, S=0

    except Exception as e:
        logger.error(f"Error in VLAPredictor conceptual test: {e}", exc_info=True)
        logger.error("This test might fail if the tiny dummy model for PaliGemma cannot be loaded or due to other setup issues.")
    finally:
        if os.path.exists(dummy_checkpoint_dir):
            import shutil
            shutil.rmtree(dummy_checkpoint_dir)
            logger.info(f"Cleaned up dummy checkpoint directory: {dummy_checkpoint_dir}") 