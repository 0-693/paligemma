# Integrated VLA model implementation will be here 

import torch
import torch.nn as nn
import logging
from types import SimpleNamespace
# Assuming paligemma_vlm.py and action_head.py are in the same directory or accessible in PYTHONPATH
from .paligemma_vlm import PaliGemmaVLM
# from .action_head import ActionHead
from .action_head.flow_matching import FlowmatchingActionHead
# from ..configs.vla_config import VLAConfig # If loading config directly, else passed in

class VLAModel(nn.Module):
    def __init__(self, config, model_logger=None):
        """
        Integrated Vision-Language-Action (VLA) Model.
        Args:
            config (dict): Configuration dictionary for PaliGemmaVLM and ActionHead.
            model_logger (logging.Logger, optional): Logger for model operations.
        """
        super().__init__()
        self.config = config # Store the full config if needed elsewhere in VLAModel
        self.logger = model_logger if model_logger else logging.getLogger(__name__)
        self.device_param = nn.Parameter(torch.empty(0)) # For getting model device

        # Initialize PaliGemma VLM backbone
        # PaliGemmaVLM expects the part of the config that contains vlm_config, vision_encoder_config etc.
        self.paligemma_vlm = PaliGemmaVLM(config=config.model, model_logger=self.logger) # Pass config.model
        
        # Determine VLM output embedding dimension
        vlm_output_dim = self.paligemma_vlm.output_embedding_dim
        self.logger.info(f"VLAModel: PaliGemmaVLM initialized. Output embedding dimension: {vlm_output_dim}")

        # 构造miravla风格的action_head_config
        ah_cfg = config.model.action_head_config
        action_head_config = SimpleNamespace(
            embed_dim=ah_cfg.get('embed_dim', self.paligemma_vlm.output_embedding_dim),
            hidden_dim=ah_cfg.get('hidden_dim', 1024),
            action_dim=ah_cfg.get('action_dim', ah_cfg.get('num_action_dims', 7)),
            horizon=ah_cfg.get('horizon', 1),
            per_action_dim=ah_cfg.get('per_action_dim', ah_cfg.get('num_action_dims', 7)),
            state_dim=ah_cfg.get('state_dim', 7),
            state_hidden_dim=ah_cfg.get('state_hidden_dim', 512),
            num_heads=ah_cfg.get('num_heads', 8),
            num_layers=ah_cfg.get('num_layers', 4),
            dropout=ah_cfg.get('dropout', 0.0),
            num_inference_timesteps=ah_cfg.get('num_inference_timesteps', 50),
            num_categories=ah_cfg.get('num_categories', 1)
        )
        self.action_head = FlowmatchingActionHead(config=action_head_config)
        self.logger.info(f"VLAModel: ActionHead initialized.")

        # 自动推断per_action_dim，兼容老配置
        if not hasattr(config.model.action_head_config, 'per_action_dim') or config.model.action_head_config.get('per_action_dim', None) is None:
            # 兼容OmegaConf和普通dict
            num_action_dims = config.model.action_head_config.get('num_action_dims', None)
            if num_action_dims is None and hasattr(config.model.action_head_config, 'num_action_dims'):
                num_action_dims = config.model.action_head_config.num_action_dims
            config.model.action_head_config['per_action_dim'] = num_action_dims

    def get_vl_embeddings(self, image_1_batch, raw_prompt_texts_batch, vlm_attention_mask_batch, image_2_batch=None):
        """
        获取VLM输出的fused_tokens（多模态嵌入），与miravla风格一致。
        """
        return self.paligemma_vlm(
            image_1_batch=image_1_batch,
            raw_prompt_texts_batch=raw_prompt_texts_batch,
            vlm_attention_mask_batch=vlm_attention_mask_batch,
            image_2_batch=image_2_batch
        )

    def predict_action(self, full_fused_context: torch.Tensor, current_step_state: torch.Tensor = None, current_step_actions_gt: torch.Tensor = None):
        """
        Predicts action for a single conceptual step using the full VLM sequence as context.
        Args:
            full_fused_context (torch.Tensor): Full VLM embeddings (B, T, E) to be used as context.
            current_step_state (torch.Tensor, optional): State for the current step (B, D_state).
            current_step_actions_gt (torch.Tensor, optional): Ground truth action for the current step (B, D_action_per_step).
                                                             This assumes action_head's action_dim is D_action_per_step (e.g. horizon=1).
        Returns:
            torch.Tensor: Predicted action (or velocity if training) for the current step (B, D_action_output).
        """
        if current_step_actions_gt is None:  # Inference
            return self.action_head.get_action(fused_tokens=full_fused_context, state=current_step_state)
        else:  # Training
            # FlowmatchingActionHead.forward returns (pred_velocity, noise)
            # The trainer's _compute_loss currently expects the predicted action (or velocity).
            pred_velocity, _noise = self.action_head(
                fused_tokens=full_fused_context,
                state=current_step_state,
                actions_gt=current_step_actions_gt
            )
            return pred_velocity

    def forward(self, image_1_batch, raw_prompt_texts_batch, vlm_attention_mask_batch, 
                  state_batch=None, image_2_batch=None, actions_gt_seq=None):
        """
        Main forward pass. For multi-step action prediction, predicts one multi-step action sequence per sample.
        Args:
            image_1_batch (torch.Tensor): (B, S, C, H, W) or (B, C, H, W) - image sequences or single images
            raw_prompt_texts_batch (list[str]): List of B prompts.
            vlm_attention_mask_batch (torch.Tensor): (B, S) or (B,) - attention mask
            state_batch (torch.Tensor, optional): (B, S, D_state) or (B, D_state) - state sequences or single states  
            image_2_batch (torch.Tensor, optional): (B, S, C, H, W) or (B, C, H, W) - optional second images
            actions_gt_seq (torch.Tensor, optional): (B, action_dim) - flattened multi-step actions for training
        Returns:
            torch.Tensor: Predicted multi-step actions (B, action_dim).
        """
        # For multi-step action prediction, we use the most recent observation (last frame)
        # to predict future action sequence
        
        # Handle both sequence and single image formats
        if len(image_1_batch.shape) == 5:  # (B, S, C, H, W) - take last frame
            image_1_batch = image_1_batch[:, -1]  # (B, C, H, W)
        elif len(image_1_batch.shape) == 4:  # (B, C, H, W) - already single images
            pass
        else:
            raise ValueError(f"Unexpected image_1_batch shape: {image_1_batch.shape}")
            
        if image_2_batch is not None:
            if len(image_2_batch.shape) == 5:  # (B, S, C, H, W) - take last frame
                image_2_batch = image_2_batch[:, -1]  # (B, C, H, W)
            elif len(image_2_batch.shape) == 4:  # (B, C, H, W) - already single images
                pass
            else:
                raise ValueError(f"Unexpected image_2_batch shape: {image_2_batch.shape}")
        
        # Handle state batch - take current state (last frame's corresponding state)
        if state_batch is not None:
            if len(state_batch.shape) == 3:  # (B, S, D_state) - take last frame's state
                current_state = state_batch[:, -1]  # (B, D_state)
            elif len(state_batch.shape) == 2:  # (B, D_state) - already single states
                current_state = state_batch
            else:
                raise ValueError(f"Unexpected state_batch shape: {state_batch.shape}")
        else:
            current_state = None
        
        # Add sequence dimension for VLM processing (single observation)
        image_1_batch = image_1_batch.unsqueeze(1)  # (B, 1, C, H, W)
        if image_2_batch is not None:
            image_2_batch = image_2_batch.unsqueeze(1)  # (B, 1, C, H, W)
        
        # Get VLM embeddings
        full_fused_context_seq = self.get_vl_embeddings(
            image_1_batch=image_1_batch,
            raw_prompt_texts_batch=raw_prompt_texts_batch,
            vlm_attention_mask_batch=vlm_attention_mask_batch,
            image_2_batch=image_2_batch
        ) # Shape: (B, T_vlm, E)

        B, T_vlm, E = full_fused_context_seq.shape
        
        # Predict the multi-step action sequence using current observation
        if actions_gt_seq is None:  # Inference
            action_pred = self.action_head.get_action(
                fused_tokens=full_fused_context_seq,
                state=current_state
            )
        else:  # Training
            # Pass the full multi-step action sequence to the action head
            pred_velocity, _noise = self.action_head(
                fused_tokens=full_fused_context_seq,
                state=current_state,
                actions_gt=actions_gt_seq
            )
            action_pred = pred_velocity
        
        return action_pred

    def to(self, *args, **kwargs):
        # Override .to() to ensure submodules are also moved correctly.
        # And also ensure self.device_param is on the right device.
        super().to(*args, **kwargs)
        # After moving the VLAModel, its device_param will reflect the target device.
        # Submodules like paligemma_vlm and action_head should be moved by super().to().
        # We also need to make sure the internal paligemma_model within paligemma_vlm is moved.
        # The PaliGemmaVLM.to() method should handle its own internal model.
        # If paligemma_vlm itself doesn't have a .to() or doesn't pass it to its internal model, that needs fixing.
        # For now, assume nn.Module's default .to() handles registered submodules.
        self.logger.info(f"VLAModel moved to device: {self.device_param.device}")
        return self

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # This test requires a valid config object similar to what main_train.py would create.
    # It also requires the PaliGemma model files to be accessible at the path specified in the config.

    logger.info("Starting VLAModel conceptual test...")

    # Dummy config (simplified, use a proper OmegaConf or AttrDict loaded from YAML for real test)
    class DummyActionHeadConfig:
        use_state_input = True
        num_action_dims = 7
        num_action_bins = 256
        mlp_hidden_dims = [512]
        dropout_prob = 0.1
        input_dim = 2048 + 7 # Example: VLM_dim (2048) + state_dim (7)

    class DummyVLMConfig:
        model_name_or_path = "./weight/paligemma-3b-pt-224" # Needs to be a valid path for PaliGemma
        dtype = "torch.float32" # or "torch.bfloat16"
        freeze_vision_tower = False
        freeze_language_model = False
    
    class DummyVisionEncoderConfig: # Copied from paligemma_vlm test
        class DummyVisionTower:
            pass
        class DummyVisionResampler:
            type = "mlp"
        vision_tower = DummyVisionTower()
        vision_resampler = DummyVisionResampler()

    class DummyDataConfig:
        state_dim = 7 # Important for action head if use_state_input=True
        # other data params like prompt_max_len, etc.

    class DummyFullConfig:
        model = type('ModelConfig', (object,), {
            'vlm_config': DummyVLMConfig(),
            'action_head_config': DummyActionHeadConfig(),
            'vision_encoder_config': DummyVisionEncoderConfig() # PaliGemmaVLM needs this
        })()
        data = DummyDataConfig()
        # other top-level configs like training, optimizer if needed by sub-parts not tested here

    config = DummyFullConfig()

    # Set a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        import os
        if not os.path.exists(config.model.vlm_config.model_name_or_path):
            logger.warning(f"Model path {config.model.vlm_config.model_name_or_path} not found. Skipping VLAModel instantiation and test.")
        else:
            vla_model = VLAModel(config, model_logger=logger).to(device)
            logger.info("VLAModel initialized and moved to device successfully.")

            # Dummy input data based on new DataLoader output and PaliGemmaVLM input requirements
            batch_size = 2
            seq_len = 4
            img_c, img_h, img_w = 3, 224, 224 # Example image dims
            state_dim_val = config.data.state_dim

            dummy_image_1 = torch.randn(batch_size, seq_len, img_c, img_h, img_w, device=device, dtype=vla_model.paligemma_vlm.dtype if isinstance(vla_model.paligemma_vlm.dtype, torch.dtype) else torch.float32)
            dummy_raw_prompts = ["test prompt one" for _ in range(batch_size)]
            dummy_vlm_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
            dummy_vlm_mask[0, -1] = False # Mask one step for testing
            dummy_state = torch.randn(batch_size, seq_len, state_dim_val, device=device, dtype=dummy_image_1.dtype)

            logger.info("Testing VLAModel forward pass...")
            action_output = vla_model(
                image_1_batch=dummy_image_1,
                raw_prompt_texts_batch=dummy_raw_prompts,
                vlm_attention_mask_batch=dummy_vlm_mask,
                state_batch=dummy_state
            )

            logger.info(f"Action output shape: {action_output.shape}")
            expected_shape = (batch_size, seq_len, 
                              config.model.action_head_config.num_action_dims, 
                              config.model.action_head_config.num_action_bins)
            assert action_output.shape == expected_shape, f"Output shape mismatch! Got {action_output.shape}, expected {expected_shape}"
            logger.info("VLAModel forward pass test completed successfully.")

    except Exception as e:
        logger.error(f"Error during VLAModel test: {e}", exc_info=True)

    print("\nVLAModel integration test completed.")