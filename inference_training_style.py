#!/usr/bin/env python3
"""
完全按照训练过程编写的推理脚本
复制训练代码中的验证流程，确保推理与训练时的模型调用方式完全一致
"""

import torch
import numpy as np
import time
import argparse
import os
from PIL import Image
import sys
import yaml
import json
import logging
from tqdm import tqdm

# 导入训练相关的模块
from model.vla_model import VLAModel
from data.loader import VLADataset, vla_collate_fn
from torch.utils.data import DataLoader
from utils.misc import setup_logging, load_checkpoint
from utils.config_utils import OmegaConfAttrDict

class InferenceTrainingStyle:
    def __init__(self, checkpoint_path, config, device=None, logger=None):
        """
        按照训练风格的推理器
        Args:
            checkpoint_path (str): 模型检查点路径
            config: 配置对象
            device: 设备
            logger: 日志器
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"使用设备: {self.device}")
        
        # 初始化模型（完全按照训练代码的方式）
        self.logger.info("正在初始化VLAModel...")
        self.model = VLAModel(config=config, model_logger=self.logger).to(self.device)
        self.logger.info(f"VLAModel初始化完成并移动到设备: {self.device}")
        
        # 加载检查点 (使用与训练/测试一致的 load_checkpoint 工具函数)
        self.logger.info(f"正在加载检查点: {checkpoint_path}")
        try:
            # 注意: load_checkpoint 通常返回 start_epoch 和 best_metric，
            # 对于推理，我们主要关心模型权重的加载。
            # 它内部处理了 'module.' 前缀和 state_dict 的提取。
            # 它还会将模型移到正确的设备。
            # 现在 load_checkpoint 接受 strict 参数，默认为 True。
            # 为了允许 'action_head.single_action_proj' 等键不匹配，我们传递 strict=False。
            load_checkpoint(self.model, filename=checkpoint_path, device=self.device, strict=False)
            self.logger.info(f"检查点 {checkpoint_path} 加载完成 (使用 utils.misc.load_checkpoint, strict=False)")
            
            # load_checkpoint 修改后会打印 missing/unexpected keys，如果 strict=False

        except Exception as e:
            self.logger.error(f"使用 load_checkpoint 加载检查点失败: {e}", exc_info=True)
            raise
        
        # 设置混合精度参数（与训练器完全一致）
        model_dtype = self.model.paligemma_vlm.dtype
        self.use_amp = (model_dtype == torch.float16 and self.device.type == 'cuda')
        self.use_bfloat16 = (model_dtype == torch.bfloat16 and self.device.type == 'cuda')
        self.logger.info(f"混合精度设置: use_amp={self.use_amp}, use_bfloat16={self.use_bfloat16}, model_dtype={model_dtype}")
        
        # 设置为评估模式
        self.model.eval()
        
    def _compute_loss(self, action_pred, action_labels, vlm_attention_mask):
        """
        计算连续动作的MSE损失（与训练器完全一致）
        Args:
            action_pred (torch.Tensor): (B, S, num_action_dims) 或 (B, S, D)
            action_labels (torch.Tensor): (B, S, num_action_dims) 或 (B, S, D)
            vlm_attention_mask (torch.Tensor): (B, S) boolean mask for valid sequence steps.
        """
        # 只对有效帧做MSE
        mask = vlm_attention_mask.unsqueeze(-1).float()  # (B, S, 1)
        mse = (action_pred - action_labels) ** 2
        masked_mse = mse * mask
        loss = masked_mse.sum() / mask.sum().clamp(min=1)
        return loss
    
    def normalize_state(self, state_vector):
        """
        对状态向量进行归一化，使用与训练时完全相同的归一化方法
        """
        # 加载归一化统计信息
        norm_stats_path = self.config.data.get('normalization_stats_path', None)
        if not norm_stats_path:
            self.logger.warning("未提供normalization_stats_path，state将不进行归一化")
            return state_vector
            
        try:
            with open(norm_stats_path, 'r') as f:
                norm_stats = json.load(f)
                
            if 'state' not in norm_stats or 'min' not in norm_stats['state'] or 'max' not in norm_stats['state']:
                self.logger.warning("归一化统计文件中缺少state的min/max信息")
                return state_vector
                
            state_min = np.array(norm_stats['state']['min'], dtype=np.float32)
            state_max = np.array(norm_stats['state']['max'], dtype=np.float32)
            
            # 应用归一化公式: normalized = 2 * (x - min) / (max - min) - 1 
            # 这与训练时使用的normalize函数相同
            normalized_state = 2.0 * (state_vector - state_min) / (state_max - state_min) - 1.0
            
            self.logger.info(f"State归一化完成:")
            self.logger.info(f"  原始state: {state_vector.tolist()}")
            self.logger.info(f"  归一化state: {normalized_state.tolist()}")
            self.logger.info(f"  使用min: {state_min.tolist()}")
            self.logger.info(f"  使用max: {state_max.tolist()}")
            
            return normalized_state
            
        except Exception as e:
            self.logger.error(f"加载归一化统计文件失败: {e}")
            return state_vector
    
    def process_input_data(self, image_1_path, prompt, state_vector, image_2_path=None):
        """
        直接处理输入数据，不通过parquet文件，复制VLADataset的处理逻辑
        """
        from transformers import AutoProcessor, SiglipImageProcessor
        from PIL import Image
        
        # 1. 归一化state
        normalized_state = self.normalize_state(state_vector)
        
        # 2. 处理图像（复制VLADataset的图像处理逻辑）
        siglip_model_name = getattr(self.config.data, 'siglip_model_name', 'google/siglip-base-patch16-224')
        siglip_processor = SiglipImageProcessor.from_pretrained(siglip_model_name)
        
        # 处理主图像
        pil_img1 = Image.open(image_1_path).convert("RGB")
        tensor_img1 = siglip_processor(images=pil_img1, return_tensors="pt").pixel_values.squeeze(0)
        
        # 处理腕部图像（如果有）
        tensor_img2 = None
        if image_2_path:
            pil_img2 = Image.open(image_2_path).convert("RGB")
            tensor_img2 = siglip_processor(images=pil_img2, return_tensors="pt").pixel_values.squeeze(0)
        
        # 3. 创建与训练时一致的批次数据
        max_seq_len = self.config.data.max_seq_len
        
        # 扩展到序列长度并添加batch维度
        image_1_batch = tensor_img1.unsqueeze(0).unsqueeze(0).repeat(1, max_seq_len, 1, 1, 1)  # (1, max_seq_len, C, H, W)
        
        if tensor_img2 is not None:
            image_2_batch = tensor_img2.unsqueeze(0).unsqueeze(0).repeat(1, max_seq_len, 1, 1, 1)  # (1, max_seq_len, C, H, W)
        else:
            # 创建零填充的图像
            C, H, W = tensor_img1.shape
            image_2_batch = torch.zeros(1, max_seq_len, C, H, W, dtype=tensor_img1.dtype)
        
        # 状态处理 - 扩展到序列长度并添加batch维度
        state_tensor = torch.tensor(normalized_state, dtype=torch.float32)
        state_batch = state_tensor.unsqueeze(0).unsqueeze(0).repeat(1, max_seq_len, 1)  # (1, max_seq_len, state_dim)
        
        # VLM注意力掩码 - 对于推理，我们只使用第一个时间步
        vlm_attention_mask = torch.zeros(1, max_seq_len, dtype=torch.bool)
        vlm_attention_mask[0, 0] = True  # 只有第一个时间步是有效的
        
        # 原始提示文本
        raw_prompt_texts_batch = [prompt]
        
        return {
            'image_1_batch': image_1_batch,
            'image_2_batch': image_2_batch,
            'state_batch': state_batch,
            'vlm_attention_mask': vlm_attention_mask,
            'raw_prompt_texts_batch': raw_prompt_texts_batch,
            'original_state': state_vector,
            'normalized_state': normalized_state
        }
    
    def predict_single_item(self, image_1_path, prompt, state_vector, image_2_path=None):
        """
        对单个样本进行推理，直接处理输入数据而不通过parquet文件
        """
        self.logger.info("直接处理输入数据进行推理...")
        
        # 处理输入数据（包括state归一化）
        processed_data = self.process_input_data(image_1_path, prompt, state_vector, image_2_path)
        
        try:
            results = []
            
            # 完全复制训练验证代码的推理流程
            with torch.no_grad():
                self.logger.info("开始模型推理...")
                
                # 将数据移动到设备
                image_1_batch = processed_data['image_1_batch'].to(self.device)
                image_2_batch = processed_data['image_2_batch'].to(self.device) 
                state_batch = processed_data['state_batch'].to(self.device)
                vlm_attention_mask = processed_data['vlm_attention_mask'].to(self.device)
                raw_prompt_texts_batch = processed_data['raw_prompt_texts_batch']
                
                self.logger.info(f"输入形状: image_1={image_1_batch.shape}, vlm_mask={vlm_attention_mask.shape}")
                self.logger.info(f"state={state_batch.shape}")
                self.logger.info(f"原始state: {processed_data['original_state'].tolist()}")
                self.logger.info(f"归一化state: {processed_data['normalized_state'].tolist()}")
                
                # 完全按照训练验证代码进行模型推理
                with torch.cuda.amp.autocast(
                    enabled=(self.use_amp or self.use_bfloat16), 
                    dtype=self.model.paligemma_vlm.dtype if self.device.type == 'cuda' else None
                ):
                    action_pred = self.model(
                        image_1_batch=image_1_batch,
                        raw_prompt_texts_batch=raw_prompt_texts_batch,
                        vlm_attention_mask_batch=vlm_attention_mask,
                        state_batch=state_batch,
                        image_2_batch=image_2_batch
                    )
                    
                self.logger.info(f"预测完成: action_pred shape={action_pred.shape}")
                
                # 收集结果
                results.append({
                    'action_pred': action_pred.cpu(),
                    'vlm_attention_mask': vlm_attention_mask.cpu(),
                    'raw_prompt_text': raw_prompt_texts_batch,
                    'state_input_original': processed_data['original_state'],
                    'state_input_normalized': processed_data['normalized_state']
                })
                    
        except Exception as e:
            self.logger.error(f"推理过程中发生错误: {e}", exc_info=True)
            raise
                
        return results

def denormalize_action(action, norm_stats):
    """
    将归一化的动作反归一化到原始范围
    """
    action_min = np.array(norm_stats['action']['min'])
    action_max = np.array(norm_stats['action']['max'])
    
    # 确保形状匹配
    if action.ndim > 1:
        shape_for_broadcast = [1] * (action.ndim - 1) + [len(action_min)]
        action_min = action_min.reshape(shape_for_broadcast)
        action_max = action_max.reshape(shape_for_broadcast)
    
    # 反归一化公式: x = 0.5 * (norm_x + 1.0) * (max - min) + min
    denorm_action = 0.5 * (action + 1.0) * (action_max - action_min) + action_min
    return denorm_action

def main(args):
    # 设置日志
    logger = setup_logging(name="InferenceTrainingStyle")
    logger.info("初始化训练风格的推理工具...")

    # --- 1. 配置加载（与训练脚本完全一致） ---
    config_path = args.config_path or "configs/vla_config.yaml"
    logger.info(f"加载配置文件: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = OmegaConfAttrDict(config_dict)
        
        logger.info("=== 配置信息 ===")
        logger.info(f"max_seq_len: {config.data.max_seq_len}")
        logger.info(f"action_head_config.horizon: {config.model.action_head_config.horizon}")
        logger.info(f"action_head_config.action_dim: {config.model.action_head_config.action_dim}")
        logger.info(f"action_head_config.num_action_dims: {config.model.action_head_config.num_action_dims}")
        
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return

    # --- 2. 初始化推理器（按照训练代码风格） ---
    try:
        inference_engine = InferenceTrainingStyle(
            checkpoint_path=args.checkpoint_path,
            config=config,
            device=args.device,
            logger=logger
        )
        logger.info("推理引擎初始化成功")
    except Exception as e:
        logger.error(f"初始化推理引擎失败: {e}", exc_info=True)
        return

    # --- 3. 准备输入数据 ---
    try:
        # 验证文件存在
        if not os.path.exists(args.image1_path):
            raise FileNotFoundError(f"主图像文件不存在: {args.image1_path}")
        if args.image2_path and not os.path.exists(args.image2_path):
            raise FileNotFoundError(f"腕部图像文件不存在: {args.image2_path}")
            
        state_vector = np.array(args.state, dtype=np.float32)
        
        logger.info(f"输入参数:")
        logger.info(f"  图像1: {args.image1_path}")
        logger.info(f"  图像2: {args.image2_path}")
        logger.info(f"  指令: '{args.prompt}'")
        logger.info(f"  原始状态: {state_vector.tolist()}")
        
    except Exception as e:
        logger.error(f"准备输入数据失败: {e}")
        return

    # --- 4. 执行推理（完全按照训练验证流程） ---
    try:
        logger.info("开始执行推理...")
        start_time = time.time()
        
        results = inference_engine.predict_single_item(
            image_1_path=args.image1_path,
            prompt=args.prompt,
            state_vector=state_vector,
            image_2_path=args.image2_path
        )
        
        end_time = time.time()
        logger.info(f"推理完成，耗时: {end_time - start_time:.3f} 秒")
        
        # --- 5. 处理和显示结果 ---
        if results:
            result = results[0]  # 单样本推理
            action_pred = result['action_pred']
            vlm_mask = result['vlm_attention_mask']
            original_state = result['state_input_original']
            normalized_state = result['state_input_normalized']
            
            print("\n" + "="*60)
            print("              训练风格推理结果")
            print("="*60)
            print(f"  语言指令: '{args.prompt}'")
            print(f"  输入状态(原始): {original_state.tolist()}")
            print(f"  输入状态(归一化): {normalized_state.tolist()}")
            print(f"  配置信息: horizon={config.model.action_head_config.horizon}, action_dim={config.model.action_head_config.action_dim}")
            print("-" * 60)
            
            # 转换为numpy并显示结果
            action_pred_np = action_pred.numpy()
            vlm_mask_np = vlm_mask.numpy()
            
            print(f"  预测动作形状: {action_pred_np.shape}")
            print(f"  注意力掩码形状: {vlm_mask_np.shape}")
            
            # 加载归一化统计信息进行反归一化
            norm_stats = None
            if config.data.get('normalization_stats_path'):
                try:
                    with open(config.data.normalization_stats_path, 'r') as f:
                        norm_stats = json.load(f)
                    logger.info(f"成功加载归一化统计信息")
                except Exception as e:
                    logger.warning(f"加载归一化统计信息失败: {e}")
            
            # 显示每个时间步的预测
            for b in range(action_pred_np.shape[0]):
                print(f"\nBatch {b + 1}:")
                for t in range(action_pred_np.shape[1]):
                    if vlm_mask_np[b, t]:  # 只显示有效的时间步
                        action_step = action_pred_np[b, t]
                        print(f"  Step {t + 1} (归一化): {action_step.tolist()}")
                        
                        # 反归一化
                        if norm_stats and 'action' in norm_stats:
                            action_denorm = denormalize_action(action_step, norm_stats)
                            print(f"  Step {t + 1} (反归一化): {action_denorm.tolist()}")
            
            print("="*60)
            
            # --- 6. 保存结果 ---
            if args.output_file:
                try:
                    output_dict = {
                        'prompt': args.prompt,
                        'input_state_original': original_state.tolist(),
                        'input_state_normalized': normalized_state.tolist(),
                        'predicted_actions_normalized': action_pred_np.tolist(),
                        'vlm_attention_mask': vlm_mask_np.tolist(),
                        'config_info': {
                            'horizon': config.model.action_head_config.horizon,
                            'action_dim': config.model.action_head_config.action_dim,
                            'max_seq_len': config.data.max_seq_len,
                            'num_action_dims': config.model.action_head_config.num_action_dims
                        },
                        'training_style_inference': True,
                        'normalization_stats': norm_stats,
                        'state_normalization_applied': True
                    }
                    
                    # 添加反归一化结果
                    if norm_stats and 'action' in norm_stats:
                        output_dict['predicted_actions_denormalized'] = denormalize_action(action_pred_np, norm_stats).tolist()
                    
                    with open(args.output_file, 'w') as f:
                        json.dump(output_dict, f, indent=2)
                    logger.info(f"结果已保存至: {args.output_file}")
                    
                except Exception as e:
                    logger.error(f"保存结果失败: {e}")
        else:
            logger.error("推理失败：没有返回结果")
            
    except Exception as e:
        logger.error(f"执行推理失败: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="训练风格的VLA模型推理工具，完全复制训练验证流程",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 输入参数
    parser.add_argument('--image1_path', type=str, required=True, 
                       help='主相机图像的文件路径')
    parser.add_argument('--prompt', type=str, required=True, 
                       help='自然语言指令')
    parser.add_argument('--state', type=float, nargs=7, required=True, 
                       help='7维的机器人状态向量，用空格分隔')
    parser.add_argument('--image2_path', type=str, default=None, 
                       help='(可选) 腕部相机图像的文件路径')

    # 模型和配置参数
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                       help='模型检查点路径')
    parser.add_argument('--config_path', type=str, default="configs/vla_config.yaml", 
                       help='模型配置文件路径')
    
    # 其他参数
    parser.add_argument('--device', type=str, default=None, 
                       help='运行设备 (cuda:0 或 cpu)')
    parser.add_argument('--output_file', type=str, default='inference_training_style_results.json', 
                       help='预测结果输出文件路径')
    
    args = parser.parse_args()
    
    # 验证文件路径
    if not os.path.exists(args.image1_path):
        parser.error(f"主图像文件不存在: {args.image1_path}")
    if args.image2_path and not os.path.exists(args.image2_path):
        parser.error(f"腕部图像文件不存在: {args.image2_path}")
    if not os.path.exists(args.checkpoint_path):
        parser.error(f"模型检查点文件不存在: {args.checkpoint_path}")
    if args.config_path and not os.path.exists(args.config_path):
        parser.error(f"配置文件不存在: {args.config_path}")

    main(args)
