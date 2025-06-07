# Paligemma VLM backbone implementation will be here 

import torch
import torch.nn as nn
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, AutoProcessor
import logging
# Import the new vision encoder modules
from .vision_encoder_module import VisionResamplerWrapper # VisionTowerAdapter and VisionTowerWrapper removed
from torchvision import transforms

class PaliGemmaVLM(nn.Module):
    def __init__(self, config, model_logger=None):
        super().__init__()
        self.config = config # This is config.model from VLAModel
        self.logger = model_logger if model_logger else logging.getLogger(__name__)

        # Determine torch.dtype
        if hasattr(torch, config.vlm_config.dtype.split('.')[-1]):
            self.dtype = getattr(torch, config.vlm_config.dtype.split('.')[-1])
        elif config.vlm_config.dtype == "auto":
             self.dtype = "auto" # For HF model loading
        else:
            self.logger.warning(f"Unsupported dtype '{config.vlm_config.dtype}' specified. Defaulting to torch.float32.")
            self.dtype = torch.float32

        self.logger.info(f"Initializing PaliGemmaVLM with model: {config.vlm_config.model_name_or_path}, dtype: {self.dtype}")

        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(config.vlm_config.model_name_or_path, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token 
                self.logger.info(f"Set tokenizer.pad_token to tokenizer.eos_token: {self.tokenizer.eos_token}")
        except Exception as e:
            self.logger.error(f"Failed to load AutoProcessor for {config.vlm_config.model_name_or_path}: {e}")
            raise

        # Load PaliGemma model
        try:
            self.paligemma_model = PaliGemmaForConditionalGeneration.from_pretrained(
                config.vlm_config.model_name_or_path,
                torch_dtype=self.dtype,
                trust_remote_code=True
            )
        except Exception as e:
            self.logger.error(f"Failed to load PaliGemmaForConditionalGeneration model: {e}")
            raise

        # Freeze components
        if config.vlm_config.freeze_vision_tower:
            self.logger.info("Freezing vision tower weights.")
            for param in self.paligemma_model.vision_tower.parameters():
                param.requires_grad = False
        
        if config.vlm_config.freeze_language_model:
            self.logger.info("Freezing language model (multi_modal_projector and language_model) weights.")
            if hasattr(self.paligemma_model, 'multi_modal_projector'):
                 for param in self.paligemma_model.multi_modal_projector.parameters():
                    param.requires_grad = False
            for param in self.paligemma_model.language_model.parameters():
                param.requires_grad = False

        model_config = self.paligemma_model.config
        try:
            self.num_image_tokens = model_config.num_image_tokens 
        except AttributeError:
            self.logger.warning("'PaliGemmaConfig' object has no attribute 'num_image_tokens'. Defaulting to 256.")
            self.num_image_tokens = 256

        if hasattr(model_config, 'text_config') and hasattr(model_config.text_config, 'hidden_size'):
            self.lm_hidden_size = model_config.text_config.hidden_size
        elif hasattr(model_config, 'hidden_size'):
            self.lm_hidden_size = model_config.hidden_size
        else:
            try:
                self.lm_hidden_size = self.paligemma_model.get_input_embeddings().embedding_dim
            except Exception as e:
                self.logger.error(f"Could not determine lm_hidden_size: {e}. Defaulting to 2048.")
                self.lm_hidden_size = 2048 
        
        self.output_embedding_dim = self.lm_hidden_size 
        self.logger.info(f"PaliGemma VLM initialized. Num image tokens: {self.num_image_tokens}, LM hidden size: {self.lm_hidden_size}")

        # Vision Tower's output dimension (input to resampler)
        # This is derived directly from the loaded PaliGemma model's vision_tower
        try:
            vision_tower_output_dim = self.paligemma_model.vision_tower.config.hidden_size
            self.logger.info(f"PaliGemma vision_tower output_dim (resampler input_dim): {vision_tower_output_dim}")
        except AttributeError as e:
            self.logger.error(f"Could not get hidden_size from paligemma_model.vision_tower.config: {e}. Defaulting vision_tower_output_dim.")
            # Provide a fallback based on common PaliGemma architectures, e.g. SigLIP base used in some PaliGemma
            # This is a guess and should ideally not be needed.
            vision_tower_output_dim = 1024 # Example, adjust if a better default is known for your model.
                                           # Or raise an error if this is critical and cannot be inferred.

        # Resampler
        # The config for resampler (e.g., config.vision_encoder_config.vision_resampler)
        # comes from the main config.model object passed to PaliGemmaVLM.
        if not hasattr(config, 'vision_encoder_config') or not hasattr(config.vision_encoder_config, 'vision_resampler'):
            self.logger.error("`vision_encoder_config.vision_resampler` not found in the provided config for PaliGemmaVLM.")
            raise ValueError("Missing vision_resampler configuration under vision_encoder_config.")
            
        resampler_config = config.vision_encoder_config.vision_resampler
        
        # Get resampler's output_dim, with a fallback and warning if missing
        resampler_output_dim = resampler_config.get('output_dim', None)
        if resampler_output_dim is None:
            self.logger.warning(
                f"'output_dim' not found in resampler_config ({resampler_config}). "
                f"Defaulting to lm_hidden_size: {self.lm_hidden_size}. "
                f"Ensure this is intended, as resampler's output_dim should ideally match LM's expected input dimension."
            )
            resampler_output_dim = self.lm_hidden_size # Fallback to lm_hidden_size

        self.vision_resampler = VisionResamplerWrapper(
            input_dim=vision_tower_output_dim, # Derived from actual vision tower
            output_dim=resampler_output_dim, # Use the retrieved or default value
            num_output_tokens=resampler_config.get('num_output_tokens', self.num_image_tokens), # Default to num_image_tokens if not specified
            resampler_type=resampler_config.get('type', 'mlp'),
            logger=self.logger
        )
        
        # This is the dimension of features AFTER resampling, which are fed to the LM.
        # It should be the resampler's output_dim.
        self.vision_output_dim = resampler_output_dim 
        self.logger.info(f"VisionResampler configured. Output dimension (features to LM): {self.vision_output_dim}")

    def _process_single_frame_batch_input(
        self, 
        images_for_frame, # Batch of images for current frame: (B, C, H, W)
        batch_raw_prompt_texts, # List of B raw prompt strings
        device
    ):
        """
        Processes a batch of single frames and corresponding text prompts using self.processor.
        The processor handles tokenizing text, adding special image tokens, and processing images.
        """
        # --- 核心修复：接收(B, C, H, W)张量，并安全地转换为PIL Image列表 ---
        processed_images = []
        if images_for_frame is not None:
            # images_for_frame is a batch tensor (B, C, H, W)
            # We need to convert each image in the batch to a PIL Image
            for i in range(images_for_frame.shape[0]):
                img_tensor = images_for_frame[i]
                # 转换前确保数据类型和值范围正确
                img_tensor_float = img_tensor.cpu().to(torch.float32)
                # ToPILImage期望(C,H,W)且值在[0,1]范围，我们的ToTensor()已保证这一点
                pil_img = transforms.ToPILImage()(img_tensor_float)
                processed_images.append(pil_img)
        
        try:
            inputs = self.processor(
                text=batch_raw_prompt_texts,
                images=processed_images if processed_images else None, # 传递PIL Image列表
                return_tensors="pt",
                padding="longest",
                truncation=True,
            ).to(device)
        except Exception as e:
            self.logger.error(f"Error during self.processor call: {e}")
            self.logger.error(f"Image type: {type(images_for_frame)}, Text type: {type(batch_raw_prompt_texts)}")
            if isinstance(images_for_frame, torch.Tensor):
                self.logger.error(f"Image tensor shape: {images_for_frame.shape}, dtype: {images_for_frame.dtype}")
            elif isinstance(images_for_frame, list) and len(images_for_frame) > 0:
                self.logger.error(f"Images list length: {len(images_for_frame)}, type of first element: {type(images_for_frame[0])}")
            if isinstance(batch_raw_prompt_texts, list) and len(batch_raw_prompt_texts) > 0:
                 self.logger.error(f"Texts list length: {len(batch_raw_prompt_texts)}, first text: \"{batch_raw_prompt_texts[0]}\"")
            raise

        # Pass processed inputs to the PaliGemma model
        # The model expects pixel_values for the image and input_ids (with image tokens) for text.
        outputs = self.paligemma_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            output_hidden_states=True # We need hidden states for embeddings
        )
        
        # Extract last hidden state from the language model part
        # PaliGemma's output.hidden_states is from the language_model component
        last_hidden_states = outputs.hidden_states[-1] # (B, total_sequence_length, lm_hidden_size)
        
        # We need to extract embeddings corresponding to the TEXT part of the prompt.
        # The processor arranges input_ids such that image tokens (e.g., `self.num_image_tokens` of them)
        # are typically at the beginning, followed by text tokens.
        # However, the exact structure can vary. A robust way is to use the attention mask or
        # identify where text tokens are.
        # For PaliGemma, `outputs.text_embeds` (if available and suitable) or by slicing `last_hidden_states`.
        # The `PaliGemmaForConditionalGeneration` forward pass documentation:
        # "The input_ids sequence length is typically `num_image_tokens + text_sequence_length`."
        # So, text embeddings should start after `num_image_tokens`.
        
        # Let B = batch_size, L_img = num_image_tokens, L_text = text_sequence_length (padded)
        # last_hidden_states shape is (B, L_img + L_text, D_lm)
        # We are interested in the text embeddings part: (B, L_text, D_lm)
        
        # Assuming image tokens are first (common pattern for PaliGemma style models)
        # The actual length of the text part in input_ids can be inferred from inputs["input_ids"].shape[1] - self.num_image_tokens
        # Or, more simply, if we only care about a fixed-length output or pooling over text tokens:
        
        # Let's pool the text token embeddings. Mean pooling is common.
        # We need to identify which part of `last_hidden_states` corresponds to text.
        # `inputs["input_ids"]` has the combined sequence. `inputs["attention_mask"]` masks padding.
        
        # The prompt to PaliGemma is typically "<prompt>\n<image>", where <image> is replaced by image embeddings.
        # The processor might prepend image tokens. If so, text tokens start after self.num_image_tokens positions.
        text_token_embeddings = last_hidden_states[:, self.num_image_tokens:, :] # (B, L_text, D_lm)
        
        # To perform masked mean pooling over actual text tokens (excluding padding):
        # We need an attention mask for the text part only.
        # `inputs['attention_mask']` is for the whole sequence (image + text).
        # The text part of the attention mask would be `inputs['attention_mask'][:, self.num_image_tokens:]`
        text_attention_mask = inputs["attention_mask"][:, self.num_image_tokens:].unsqueeze(-1) # (B, L_text, 1)
        
        sum_embeddings = (text_token_embeddings * text_attention_mask).sum(dim=1) # (B, D_lm)
        sum_mask = text_attention_mask.sum(dim=1) # (B, 1)
        sum_mask = torch.clamp(sum_mask, min=1e-9) # Avoid division by zero
        pooled_text_embeddings = sum_embeddings / sum_mask # (B, D_lm)

        return pooled_text_embeddings # Return (B, lm_hidden_size) pooled embeddings for the text part

    def forward(self, image_1_batch, raw_prompt_texts_batch, vlm_attention_mask_batch, image_2_batch=None):
        """
        Args:
            image_1_batch (torch.Tensor): Batch of primary camera image sequences (B, S, C, H, W).
            raw_prompt_texts_batch (list[str]): Batch of raw prompt strings, one per sequence (B,).
                                                 The same prompt is used for all frames in a sequence.
            vlm_attention_mask_batch (torch.Tensor): Mask for valid steps in sequence (B, S).
            image_2_batch (torch.Tensor, optional): Batch of auxiliary camera image sequences (B, S, C, H, W).

        Returns:
            torch.Tensor: VLM embeddings for each valid step in the sequence (B, S, D_vlm).
                          Returns zeros for padded/masked steps.
        """
        batch_size, seq_len, C, H, W = image_1_batch.shape
        device = image_1_batch.device
        vlm_output_embeddings_list = []

        for s_idx in range(seq_len):
            # 主摄像头
            current_image_1_s_batch = image_1_batch[:, s_idx, :, :, :] # (B, C, H, W)
            # 腕部摄像头
            current_image_2_s_batch = image_2_batch[:, s_idx, :, :, :] if image_2_batch is not None else None

            # 这里可以根据需要将image_1和image_2融合或分别处理
            # 目前仅处理image_1，image_2可用于后续多摄像头融合
            frame_vlm_embeddings = self._process_single_frame_batch_input(
                images_for_frame=current_image_1_s_batch,
                batch_raw_prompt_texts=raw_prompt_texts_batch, # List of B strings
                device=device
            )
            # TODO: 可扩展为支持image_2的多模态融合
            vlm_output_embeddings_list.append(frame_vlm_embeddings)

        # Stack embeddings from all sequence steps: (S, B, D_vlm) -> (B, S, D_vlm)
        vlm_embeddings_stacked = torch.stack(vlm_output_embeddings_list, dim=1) # (B, S, D_vlm)

        # Apply sequence mask (vlm_attention_mask_batch)
        masked_vlm_embeddings = vlm_embeddings_stacked * vlm_attention_mask_batch.unsqueeze(-1).float()

        return masked_vlm_embeddings


if __name__ == '__main__':
    # This test block needs to be updated significantly due to changes in:
    # 1. Config structure (direct dict vs. OmegaConf/AttrDict)
    # 2. VLADataset output (raw_prompt_text, image_1 as (B,S,C,H,W))
    # 3. PaliGemmaVLM.forward signature (raw_prompt_texts_batch)
    # 4. Processor loading within PaliGemmaVLM
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("PaliGemmaVLM main test block - requires significant updates for proper testing.")

    # Example of how one might structure a basic config for testing:
    class DummyVLMAConfig:
        model_name_or_path = "google/paligemma-3b-pt-224" # Requires actual model for full test
        # To run without download, one would need to mock from_pretrained calls heavily.
        dtype = "torch.float32" # or "torch.bfloat16" if available
        freeze_vision_tower = False
        freeze_language_model = False

    class DummyVisionEncoderConfig:
        class DummyVisionTower:
            pass
        class DummyVisionResampler:
            type = "mlp"
        vision_tower = DummyVisionTower()
        vision_resampler = DummyVisionResampler()

    class DummyConfig:
        vlm_config = DummyVLMAConfig()
        vision_encoder_config = DummyVisionEncoderConfig()
        data = type('DataConfig', (), {'prompt_max_len': 77})() # Dummy for max_length if processor uses it

    config = DummyConfig()
    
    # Create a dummy processor and model for structural testing if actual model path is problematic for quick test
    # This is non-trivial to mock effectively for transformers.
    # For a real test, ensure config.vlm_config.model_name_or_path points to a valid PaliGemma model/processor.

    try:
        logger.info(f"Attempting to initialize PaliGemmaVLM with model: {config.vlm_config.model_name_or_path}")
        # To run this test, the model path must be valid or from_pretrained calls mocked.
        # For CI/CD or environments without large model downloads, this is tricky.
        # Assuming the model path is valid for now.
        # Check if the model path exists to provide a more graceful skip/warning
        import os
        if not os.path.exists(config.vlm_config.model_name_or_path):
            logger.warning(f"Model path {config.vlm_config.model_name_or_path} not found. Skipping PaliGemmaVLM instantiation test.")
        else:
            paligemma_vlm = PaliGemmaVLM(config, model_logger=logger)
            logger.info("PaliGemmaVLM initialized successfully (structurally).")

            # --- Test forward pass (requires dummy data from new VLADataset format) ---
            batch_size = 2
            seq_len = 4
            img_c, img_h, img_w = 3, 224, 224 # Assuming default image size
            if hasattr(paligemma_vlm.processor, 'image_processor') and hasattr(paligemma_vlm.processor.image_processor, 'size'):
                 size_info = paligemma_vlm.processor.image_processor.size
                 img_h = size_info.get('height', img_h) if isinstance(size_info, dict) else size_info[0] if isinstance(size_info, (list,tuple)) else img_h
                 img_w = size_info.get('width', img_w) if isinstance(size_info, dict) else size_info[1] if isinstance(size_info, (list,tuple)) else img_w


            dummy_image_1_batch = torch.randn(batch_size, seq_len, img_c, img_h, img_w, 
                                              dtype=paligemma_vlm.dtype if isinstance(paligemma_vlm.dtype, torch.dtype) else torch.float32)
            dummy_raw_prompts = ["this is a test prompt" for _ in range(batch_size)]
            dummy_vlm_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
            dummy_vlm_mask[0, -1] = False # Example of a padded step for one item

            logger.info("Testing PaliGemmaVLM forward pass...")
            # Move model to CPU for this test if no GPU assertion
            paligemma_vlm.to('cpu') # Ensure model is on CPU for test unless GPU is assumed/checked
            dummy_image_1_batch = dummy_image_1_batch.to('cpu')
            dummy_vlm_mask = dummy_vlm_mask.to('cpu')

            output_embeddings = paligemma_vlm(dummy_image_1_batch, dummy_raw_prompts, dummy_vlm_mask)
            logger.info(f"Output embeddings shape: {output_embeddings.shape}") # Expected: (B, S, D_vlm)
            expected_shape = (batch_size, seq_len, paligemma_vlm.output_embedding_dim)
            assert output_embeddings.shape == expected_shape, f"Output shape mismatch! Got {output_embeddings.shape}, expected {expected_shape}"
            logger.info("PaliGemmaVLM forward pass test completed (structurally).")

    except Exception as e:
        logger.error(f"Error during PaliGemmaVLM test: {e}", exc_info=True) 