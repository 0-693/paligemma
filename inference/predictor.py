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
from utils.config_utils import OmegaConfAttrDict
from torchvision import transforms

class VLAPredictor:
    def __init__(self, checkpoint_path, config=None, device=None, logger=None):
        """
        Predictor for the Vision-Language-Action Model.
        Args:
            checkpoint_path (str): Path to the model checkpoint file (.pth or .pth.tar).
            config (dict, optional): Model configuration dictionary. If None, will try to load from checkpoint.
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
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 统一处理配置，确保是OmegaConfAttrDict类型
        if config is None:
            raise ValueError("Configuration must be provided to VLAPredictor.")
        
        if isinstance(config, dict) and not isinstance(config, OmegaConfAttrDict):
            self.logger.warning("Configuration was a dict, converting to OmegaConfAttrDict for compatibility.")
            self.config = OmegaConfAttrDict(config)
        else:
            self.config = config

        # --- 新增：加载归一化统计数据 ---
        self.norm_stats = None
        normalization_stats_path = self.config.data.get('normalization_stats_path', None)
        if normalization_stats_path and os.path.exists(normalization_stats_path):
            try:
                with open(normalization_stats_path, 'r') as f:
                    self.norm_stats = json.load(f)
                self.logger.info(f"Successfully loaded normalization stats from {normalization_stats_path}")
            except Exception as e:
                self.logger.error(f"Error loading normalization stats: {e}", exc_info=True)
        else:
            self.logger.warning(f"Normalization stats file not found at path: {normalization_stats_path}. Action denormalization will be skipped.")

        # --- 修正：模型初始化 ---
        # 确定模型数据类型
        model_dtype_str = self.config.model.vlm_config.get('dtype', 'torch.float32')
        if model_dtype_str == 'torch.float16':
            self.model_dtype = torch.float16
        elif model_dtype_str == 'torch.bfloat16':
            self.model_dtype = torch.bfloat16
        else:
            self.model_dtype = torch.float32
        
        # 初始化模型
        # 添加关键调试信息
        self.logger.info("=== 模型配置调试信息 ===")
        self.logger.info(f"action_head_config: {self.config.model.action_head_config}")
        self.logger.info(f"horizon: {self.config.model.action_head_config.get('horizon', 'NOT_SET')}")
        self.logger.info(f"action_dim: {self.config.model.action_head_config.get('action_dim', 'NOT_SET')}")
        self.logger.info(f"max_seq_len: {self.config.data.get('max_seq_len', 'NOT_SET')}")
        
        self.model = VLAModel(
            config=self.config,
            model_logger=self.logger
        ).to(self.device)
        
        # 加载模型参数
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 处理参数键名
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        self.model.load_state_dict(new_state_dict, strict=False)
        self.logger.info("Loaded state_dict with strict=False.")

        # --- 新增：明确的成功加载日志 ---
        num_loaded_params = sum(p.numel() for p in new_state_dict.values())
        total_model_params = sum(p.numel() for p in self.model.state_dict().values())
        self.logger.info(f"Successfully loaded {len(new_state_dict)} keys ({num_loaded_params/1e6:.2f}M params) into the model ({total_model_params/1e6:.2f}M params).")

        self.model.eval()
        self.logger.info("Model loaded successfully and set to eval mode.")

        # --- 核心修正：从模型中获取唯一的、正确的处理器 ---
        self.processor = self.model.paligemma_vlm.processor
        self.logger.info("Sourced the one true processor directly from the VLM submodule.")

        # --- 修正模型属性获取路径 ---
        self.num_action_bins = self.config.model.action_head_config.num_action_bins
        self.action_bounds = self.config.data.get('action_bounds', [-1.0, 1.0])

    def preprocess_images(self, image_1, image_2=None):
        """
        重构: 使用模型自带的图像处理器将PIL图像转换为模型所需的张量。
        """
        def process_single_image(img):
            if img is None:
                return None
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            # 使用 self.processor 的 image_processor 组件来处理图像
            processed_inputs = self.processor.image_processor(images=img, return_tensors="pt")
            return processed_inputs['pixel_values']

        # 处理主图像，返回 (1, C, H, W) 的张量
        processed_image_1_tensor = process_single_image(image_1)
        # 为其增加一个序列维度，以匹配模型 (B, T, C, H, W) 的输入格式
        batch_image_1 = processed_image_1_tensor.unsqueeze(0)

        batch_image_2 = None
        if image_2 is not None:
            processed_image_2_tensor = process_single_image(image_2)
            batch_image_2 = processed_image_2_tensor.unsqueeze(0)

        return batch_image_1, batch_image_2

    def preprocess_single_item_direct(self, image_1, prompt_text, image_2=None, state_vector=None, max_seq_len=None):
        """
        重构: 该方法现在只负责收集原始数据，并将其打包成字典。
        所有预处理将由模型内部完成。
        
        Args:
            image_1: 主相机图像
            prompt_text: 文本指令
            image_2: 可选的第二个相机图像
            state_vector: 机器人状态向量
            max_seq_len: 序列长度，用于创建正确的时间维度
        """
        # 从配置获取序列长度，如果未提供参数的话
        if max_seq_len is None:
            max_seq_len = self.config.data.get('max_seq_len', 4)
        
        # 将原始PIL Image包装成批次 (B, T, C, H, W) -> (1, T, C, H, W)
        # 注意：这里我们为单次推理复制图像到每个时间步
        image_1_tensor = transforms.ToTensor()(image_1).unsqueeze(0)  # (1, C, H, W)
        # 复制到所有时间步: (1, T, C, H, W)
        image_1_tensor = image_1_tensor.unsqueeze(1).repeat(1, max_seq_len, 1, 1, 1)
        
        image_2_tensor = None
        if image_2 is not None:
            image_2_tensor = transforms.ToTensor()(image_2).unsqueeze(0)  # (1, C, H, W)
            image_2_tensor = image_2_tensor.unsqueeze(1).repeat(1, max_seq_len, 1, 1, 1)  # (1, T, C, H, W)

        # 将原始文本包装成列表
        raw_prompt_texts_batch = [prompt_text]

        # 处理状态向量，复制到所有时间步
        batch_state = None
        if state_vector is not None and self.config.model.action_head_config.get('use_state_input', False):
            state_tensor = torch.tensor(state_vector, dtype=torch.float32)

            if self.norm_stats and 'state' in self.norm_stats:
                state_stats = self.norm_stats['state']
                min_vals = torch.tensor(state_stats['min'], dtype=torch.float32)
                max_vals = torch.tensor(state_stats['max'], dtype=torch.float32)
                
                # 将状态归一化到 [-1, 1] 区间
                normalized_state_tensor = 2 * (state_tensor - min_vals) / (max_vals - min_vals).clamp(min=1e-8) - 1
                
                self.logger.info("输入的状态向量已根据训练数据进行归一化。")
                # 复制状态到所有时间步: (1, T, state_dim)
                batch_state = normalized_state_tensor.unsqueeze(0).unsqueeze(0).repeat(1, max_seq_len, 1).to(self.device, dtype=self.model_dtype)
            else:
                self.logger.warning("未找到状态归一化统计数据。将使用原始状态向量。如果模型在归一化状态上训练，这可能导致性能不佳。")
                # 复制状态到所有时间步: (1, T, state_dim)
                batch_state = state_tensor.unsqueeze(0).unsqueeze(0).repeat(1, max_seq_len, 1).to(self.device, dtype=self.model_dtype)
        
        # 创建VLM注意力掩码 (B, T) - 所有时间步都有效
        batch_vlm_mask = torch.ones((1, max_seq_len), dtype=torch.bool)
        
        # 返回一个包含正确键值的字典
        return {
            'image_1_batch': image_1_tensor.to(self.device, dtype=self.model_dtype),
            'image_2_batch': image_2_tensor.to(self.device, dtype=self.model_dtype) if image_2_tensor is not None else None,
            'raw_prompt_text': raw_prompt_texts_batch,
            'vlm_attention_mask': batch_vlm_mask.to(self.device),
            'state_batch': batch_state
        }

    def predict(self, batch):
        """
        此方法接收一个数据批次，执行必要的预处理（特别是文本分词），
        然后调用模型进行推理。
        """
        # Ensure the model is in evaluation mode
        self.model.eval()

        # The core prediction logic
        with torch.no_grad():
            # Construct a clean input dictionary for the model, similar to the trainer.
            # This avoids passing unexpected arguments to the model's forward pass.
            model_input = {}

            # 1. Handle image inputs (from 'image_1' provided by dataloader, as in trainer)
            if 'image_1' in batch:
                model_input['image_1_batch'] = batch['image_1']
            elif 'pixel_values' in batch: # Fallback for other potential data formats
                model_input['image_1_batch'] = batch['pixel_values']
            elif 'image_1_batch' in batch: # For single-item inference
                model_input['image_1_batch'] = batch['image_1_batch']

            if 'image_2' in batch:
                model_input['image_2_batch'] = batch['image_2']
            elif 'pixel_values_2' in batch: # Fallback for other potential data formats
                model_input['image_2_batch'] = batch['pixel_values_2']
            elif 'image_2_batch' in batch:
                 model_input['image_2_batch'] = batch['image_2_batch']

            # 2. Handle state input
            if 'state' in batch:
                model_input['state_batch'] = batch['state']
            elif 'state_batch' in batch:
                model_input['state_batch'] = batch['state_batch']

            # 3. Handle VLM attention mask
            if 'vlm_attention_mask' in batch:
                model_input['vlm_attention_mask_batch'] = batch['vlm_attention_mask']

            # 4. Handle text input (tokenizing raw text)
            raw_texts = None
            if 'prompt' in batch: # From dataset
                raw_texts = batch['prompt']
            elif 'raw_prompt_text' in batch: # From single-item inference
                raw_texts = batch['raw_prompt_text']

            if raw_texts:
                if isinstance(raw_texts, str):
                    raw_texts = [raw_texts]
                
                # The VLAModel's forward method expects raw text, which it passes to PaliGemmaVLM.
                # Let's align with the trainer and VLAModel's forward signature.
                model_input['raw_prompt_texts_batch'] = raw_texts
            
            # 5. Handle ground truth actions (not needed for prediction mode, but good practice to filter)
            # 'actions_gt_seq' is the key expected by VLAModel.forward
            if 'action' in batch:
                 model_input['actions_gt_seq'] = batch['action']


            # Filter out any None values before passing to the model
            final_model_input = {k: v for k, v in model_input.items() if v is not None}
            
            # Now, call the model with the prepared input dictionary.
            # The 'mode' argument is specific to our predictor logic, not the model itself.
            predictions = self.model(**final_model_input)
        
        # --- DEBUG: 打印模型原始输出 ---
        self.logger.info(f"模型原始输出 (反归一化前): {predictions.detach().cpu().numpy()}")
        
        # Post-process predictions
        # The model now returns the predicted action sequence directly.
        action_pred_normalized = predictions
        
        action_pred_denormalized = action_pred_normalized # Fallback value
        if self.norm_stats and 'action' in self.norm_stats:
            min_vals = self.norm_stats['action']['min']
            max_vals = self.norm_stats['action']['max']
            if not isinstance(min_vals, torch.Tensor):
                min_vals = torch.tensor(min_vals, device=action_pred_denormalized.device)
            if not isinstance(max_vals, torch.Tensor):
                max_vals = torch.tensor(max_vals, device=action_pred_denormalized.device)
            
            action_min = min_vals.clone().detach().to(action_pred_denormalized.device)
            action_max = max_vals.clone().detach().to(action_pred_denormalized.device)

            # Denormalize: from [-1, 1] to [min, max]
            action_pred_denormalized = denormalize(action_pred_normalized, action_min, action_max)
            # self.logger.info("Action denormalized using loaded stats.") # Too verbose for every step
        else:
            self.logger.warning("Normalization stats not found or invalid, returning model's raw [-1, 1] action predictions.")

        return {
            "action": action_pred_denormalized,  # 返回完整的action sequence
            "predicted_actions_continuous": action_pred_denormalized,
            "vlm_attention_mask": model_input.get('vlm_attention_mask_batch', 
                torch.ones((action_pred_denormalized.shape[0], action_pred_denormalized.shape[1]), 
                          dtype=torch.bool, device=action_pred_denormalized.device))
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
        single_item_preprocessed = predictor.preprocess_single_item_direct(
            image_1=Image.open(dummy_img_path1).convert("RGB"),
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