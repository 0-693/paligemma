#!/usr/bin/env python3
"""
重写的离线推理脚本，支持新的多步动作预测架构
- 支持单观测输入预测horizon步动作
- 使用与训练一致的模型架构和数据处理流程
- 支持权重加载和动作反归一化
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
from utils.misc import setup_logging, load_checkpoint
from utils.config_utils import OmegaConfAttrDict

class VLAInference:
    def __init__(self, checkpoint_path, config, device=None, logger=None):
        """
        支持多步动作预测的VLA推理器
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
        
        # 获取多步动作预测参数
        self.horizon = config.model.action_head_config.horizon
        self.per_action_dim = config.model.action_head_config.per_action_dim
        self.action_dim = config.model.action_head_config.action_dim
        self.state_dim = config.model.action_head_config.state_dim
        
        self.logger.info(f"动作预测配置: horizon={self.horizon}, per_action_dim={self.per_action_dim}, action_dim={self.action_dim}")
        
        # 初始化模型
        self.logger.info("正在初始化VLAModel...")
        self.model = VLAModel(config=config, model_logger=self.logger).to(self.device)
        self.logger.info(f"VLAModel初始化完成并移动到设备: {self.device}")
        
        # 加载检查点
        self.logger.info(f"正在加载检查点: {checkpoint_path}")
        try:
            load_checkpoint(self.model, filename=checkpoint_path, device=self.device, strict=False)
            self.logger.info(f"检查点 {checkpoint_path} 加载完成")
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}", exc_info=True)
            raise
        
        # 设置混合精度参数
        model_dtype = self.model.paligemma_vlm.dtype
        self.use_amp = (model_dtype == torch.float16 and self.device.type == 'cuda')
        self.use_bfloat16 = (model_dtype == torch.bfloat16 and self.device.type == 'cuda')
        self.logger.info(f"混合精度设置: use_amp={self.use_amp}, use_bfloat16={self.use_bfloat16}, model_dtype={model_dtype}")
        
        # 设置为评估模式
        self.model.eval()
        
        # 加载归一化统计信息
        self.norm_stats = None
        norm_stats_path = self.config.data.get('normalization_stats_path', None)
        if norm_stats_path and os.path.exists(norm_stats_path):
            try:
                with open(norm_stats_path, 'r') as f:
                    self.norm_stats = json.load(f)
                self.logger.info(f"成功加载归一化统计信息: {norm_stats_path}")
            except Exception as e:
                self.logger.error(f"加载归一化统计信息失败: {e}")
                
        # 初始化图像处理器
        from transformers import SiglipImageProcessor
        siglip_model_name = getattr(self.config.data, 'siglip_model_name', 'google/siglip-base-patch16-224')
        self.siglip_processor = SiglipImageProcessor.from_pretrained(siglip_model_name)
        self.logger.info(f"图像处理器初始化完成: {siglip_model_name}")
    
    def normalize_state(self, state_vector):
        """对状态向量进行归一化"""
        if self.norm_stats is None or 'state' not in self.norm_stats:
            self.logger.warning("未找到状态归一化统计信息，使用原始状态")
            return state_vector
            
        try:
            state_min = np.array(self.norm_stats['state']['min'], dtype=np.float32)
            state_max = np.array(self.norm_stats['state']['max'], dtype=np.float32)
            
            # 归一化: normalized = 2 * (x - min) / (max - min) - 1
            normalized_state = 2.0 * (state_vector - state_min) / (state_max - state_min) - 1.0
            
            self.logger.debug(f"状态归一化: {state_vector} -> {normalized_state}")
            return normalized_state
            
        except Exception as e:
            self.logger.error(f"状态归一化失败: {e}")
            return state_vector
    
    def denormalize_actions(self, normalized_actions):
        """将归一化的动作反归一化到原始范围"""
        if self.norm_stats is None or 'action' not in self.norm_stats:
            self.logger.warning("未找到动作归一化统计信息，返回归一化的动作")
            return normalized_actions
            
        try:
            action_min = np.array(self.norm_stats['action']['min'], dtype=np.float32)
            action_max = np.array(self.norm_stats['action']['max'], dtype=np.float32)
            
            # 反归一化: x = (normalized + 1) * (max - min) / 2 + min
            denormalized_actions = (normalized_actions + 1.0) * (action_max - action_min) / 2.0 + action_min
            
            return denormalized_actions
            
        except Exception as e:
            self.logger.error(f"动作反归一化失败: {e}")
            return normalized_actions
    
    def process_images(self, image_1_path, image_2_path=None):
        """处理输入图像"""
        # 处理主图像
        try:
            pil_img1 = Image.open(image_1_path).convert("RGB")
            tensor_img1 = self.siglip_processor(images=pil_img1, return_tensors="pt").pixel_values.squeeze(0)
        except Exception as e:
            self.logger.error(f"处理图像1失败: {e}")
            raise
        
        # 处理第二张图像（如果有）
        tensor_img2 = None
        if image_2_path:
            try:
                pil_img2 = Image.open(image_2_path).convert("RGB")
                tensor_img2 = self.siglip_processor(images=pil_img2, return_tensors="pt").pixel_values.squeeze(0)
            except Exception as e:
                self.logger.warning(f"处理图像2失败: {e}，使用零填充")
        
        if tensor_img2 is None:
            # 创建零填充的图像
            C, H, W = tensor_img1.shape
            tensor_img2 = torch.zeros(C, H, W, dtype=tensor_img1.dtype)
        
        return tensor_img1, tensor_img2
    
    def predict(self, image_1_path, prompt, state_vector, image_2_path=None):
        """
        预测多步动作
        Args:
            image_1_path (str): 主图像路径
            prompt (str): 文本提示
            state_vector (list/np.array): 机器人状态向量
            image_2_path (str, optional): 第二张图像路径
            
        Returns:
            dict: 包含预测结果的字典
        """
        self.logger.info("开始预测...")
        
        # 转换状态向量
        if isinstance(state_vector, list):
            state_vector = np.array(state_vector, dtype=np.float32)
        
        # 验证状态维度
        if len(state_vector) != self.state_dim:
            raise ValueError(f"状态向量维度不匹配: 期望 {self.state_dim}, 得到 {len(state_vector)}")
        
        # 归一化状态
        normalized_state = self.normalize_state(state_vector)
        
        # 处理图像
        tensor_img1, tensor_img2 = self.process_images(image_1_path, image_2_path)
        
        # Prepare batch data (single sample)
        image_1_batch = tensor_img1.unsqueeze(0).to(self.device)  # (1, C, H, W)
        image_2_batch = tensor_img2.unsqueeze(0).to(self.device)  # (1, C, H, W)
        state_batch = torch.tensor(normalized_state, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, state_dim)
        raw_prompt_texts_batch = [prompt]
        
        # Create VLM attention mask (single observation, so mask is [True])
        vlm_attention_mask_batch = torch.tensor([[True]], dtype=torch.bool).to(self.device)  # (1, 1)
        
        self.logger.info(f"输入形状: image_1={image_1_batch.shape}, image_2={image_2_batch.shape}, state={state_batch.shape}")
        self.logger.info(f"VLM注意力掩码: {vlm_attention_mask_batch.shape}")
        self.logger.info(f"提示文本: {prompt}")
        self.logger.info(f"原始状态: {state_vector.tolist()}")
        self.logger.info(f"归一化状态: {normalized_state.tolist()}")
        
        # 模型推理
        with torch.no_grad():
            try:
                with torch.cuda.amp.autocast(
                    enabled=(self.use_amp or self.use_bfloat16), 
                    dtype=self.model.paligemma_vlm.dtype if self.device.type == 'cuda' else None
                ):
                    # 调用模型进行预测
                    action_pred = self.model(
                        image_1_batch=image_1_batch,
                        raw_prompt_texts_batch=raw_prompt_texts_batch,
                        vlm_attention_mask_batch=vlm_attention_mask_batch,
                        state_batch=state_batch,
                        image_2_batch=image_2_batch
                    )
                    
                self.logger.info(f"模型输出形状: {action_pred.shape}")
                
                # 转换为numpy并移到CPU
                action_pred_np = action_pred.cpu().numpy().squeeze(0)  # 移除批次维度: (action_dim,)
                
                # 验证输出维度
                if action_pred_np.shape[0] != self.action_dim:
                    raise ValueError(f"模型输出维度不匹配: 期望 {self.action_dim}, 得到 {action_pred_np.shape[0]}")
                
                # 重塑为多步动作格式: (action_dim,) -> (horizon, per_action_dim)
                multi_step_actions_normalized = action_pred_np.reshape(self.horizon, self.per_action_dim)
                
                # 反归一化每一步动作
                multi_step_actions = []
                for step_idx in range(self.horizon):
                    step_action_normalized = multi_step_actions_normalized[step_idx]
                    step_action_denormalized = self.denormalize_actions(step_action_normalized)
                    multi_step_actions.append(step_action_denormalized)
                
                multi_step_actions = np.array(multi_step_actions)  # (horizon, per_action_dim)
                
                self.logger.info(f"预测完成: {self.horizon}步动作, 每步{self.per_action_dim}维")
                
                # 返回详细结果
                results = {
                    'multi_step_actions': multi_step_actions,  # (horizon, per_action_dim) 反归一化后的动作
                    'multi_step_actions_normalized': multi_step_actions_normalized,  # (horizon, per_action_dim) 归一化的动作
                    'flattened_actions': action_pred_np,  # (action_dim,) 原始模型输出
                    'horizon': self.horizon,
                    'per_action_dim': self.per_action_dim,
                    'input_state_original': state_vector,
                    'input_state_normalized': normalized_state,
                    'prompt': prompt
                }
                
                # 打印预测结果
                self.logger.info("=" * 60)
                self.logger.info("预测结果:")
                self.logger.info(f"输入状态: {state_vector.tolist()}")
                self.logger.info(f"提示文本: {prompt}")
                self.logger.info(f"预测 {self.horizon} 步动作:")
                for step_idx in range(self.horizon):
                    self.logger.info(f"  步骤 {step_idx+1}: {multi_step_actions[step_idx].tolist()}")
                self.logger.info("=" * 60)
                
                return results
                
            except Exception as e:
                self.logger.error(f"模型推理失败: {e}", exc_info=True)
                raise

def main(args):
    # 设置日志
    logger = setup_logging(name="VLAInference")
    logger.info("初始化VLA多步动作预测推理工具...")

    # --- 1. 配置加载 ---
    config_path = args.config_path or "configs/vla_config_ddp.yaml"
    logger.info(f"加载配置文件: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = OmegaConfAttrDict(config_dict)
        
        logger.info("=== 配置信息 ===")
        logger.info(f"horizon: {config.model.action_head_config.horizon}")
        logger.info(f"per_action_dim: {config.model.action_head_config.per_action_dim}")
        logger.info(f"action_dim: {config.model.action_head_config.action_dim}")
        logger.info(f"state_dim: {config.model.action_head_config.state_dim}")
        
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return

    # --- 2. 初始化推理器 ---
    try:
        inference_engine = VLAInference(
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

    # --- 4. 执行推理 ---
    try:
        logger.info("开始执行推理...")
        start_time = time.time()
        
        results = inference_engine.predict(
            image_1_path=args.image1_path,
            prompt=args.prompt,
            state_vector=state_vector,
            image_2_path=args.image2_path
        )
        
        end_time = time.time()
        logger.info(f"推理完成，耗时: {end_time - start_time:.3f} 秒")
        
        # --- 5. 处理和显示结果 ---
        if results:
            multi_step_actions = results['multi_step_actions']
            horizon = results['horizon']
            per_action_dim = results['per_action_dim']
            
            print("\n" + "="*80)
            print("                    VLA 多步动作预测结果")
            print("="*80)
            print(f"  语言指令: '{args.prompt}'")
            print(f"  输入状态: {results['input_state_original'].tolist()}")
            print(f"  预测配置: {horizon} 步，每步 {per_action_dim} 维")
            print("-" * 80)
            
            # 显示每步预测的动作
            for step_idx in range(horizon):
                action_step = multi_step_actions[step_idx]
                print(f"  步骤 {step_idx+1}: {action_step.tolist()}")
            
            print("="*80)
            
            # --- 6. 保存结果 ---
            if args.output_file:
                try:
                    output_dict = {
                        'prompt': args.prompt,
                        'input_state': results['input_state_original'].tolist(),
                        'multi_step_actions': multi_step_actions.tolist(),
                        'config_info': {
                            'horizon': horizon,
                            'per_action_dim': per_action_dim,
                            'action_dim': results['flattened_actions'].shape[0],
                            'state_dim': len(results['input_state_original'])
                        },
                        'inference_type': 'multi_step_action_prediction',
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
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
        description="VLA多步动作预测推理工具，支持单观测输入预测horizon步动作",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 输入参数
    parser.add_argument('--image1_path', type=str, required=True, 
                       help='主相机图像的文件路径')
    parser.add_argument('--prompt', type=str, required=True, 
                       help='自然语言指令')
    parser.add_argument('--state', type=float, nargs='+', required=True, 
                       help='机器人状态向量，用空格分隔（维度根据配置确定）')
    parser.add_argument('--image2_path', type=str, default=None, 
                       help='(可选) 腕部相机图像的文件路径')

    # 模型和配置参数
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                       help='模型检查点路径')
    parser.add_argument('--config_path', type=str, default="configs/vla_config_ddp.yaml", 
                       help='模型配置文件路径')
    
    # 其他参数
    parser.add_argument('--device', type=str, default=None, 
                       help='运行设备 (cuda:0 或 cpu)')
    parser.add_argument('--output_file', type=str, default='vla_inference_results.json', 
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
