import torch
import numpy as np
import time
import argparse
import os
from PIL import Image
import sys
import yaml
from omegaconf import OmegaConf

from inference.predictor import VLAPredictor
from utils.misc import setup_logging

def get_default_config():
    """
    获取与 main_eval.py 和训练时兼容的默认配置。
    这确保了模型结构的一致性。
    """
    return {
        'model': {
            'vlm_config': {
                'model_name_or_path': "./weight/paligemma-3b-pt-224",
                'use_aux_camera': False,
                'dtype': 'torch.bfloat16',
                'num_image_tokens': 256,
            },
            'vision_resampler_config': {
                'type': 'mlp',
                'output_dim': 2048,
                'mlp_projector': {'hidden_dim': None}
            },
            'action_head_config': {
                'use_state_input': True,
                'state_dim': 7, 
                'num_action_dims': 7, 
                'num_action_bins': 256, 
                'hidden_layers_config': [1024, 512],
                'dropout_prob': 0.1
            }
        },
        'data': {
            'tokenizer_name_or_path': "./weight/paligemma-3b-pt-224",
            'image_processor_name_or_path': "./weight/paligemma-3b-pt-224",
            'action_bounds': [-1.0, 1.0],
            'normalization_stats_path': "normalization_stats.json"
        }
    }

def main(args):
    # 设置日志
    logger = setup_logging(name="VLAInferenceTool")
    logger.info("初始化VLA单次推理工具...")

    # --- 1. 配置加载 ---
    base_config = OmegaConf.create(get_default_config())
    config = base_config
    if args.config_path:
        try:
            user_config = OmegaConf.load(args.config_path)
            logger.info(f"从用户配置文件加载配置: {args.config_path}")
            config = OmegaConf.merge(base_config, user_config)
            logger.info("用户配置已成功合并。")
        except Exception as e:
            logger.error(f"加载或合并配置文件时出错: {e}", exc_info=True)
            return
    else:
        logger.info("未提供配置文件，使用默认配置。")

    # --- 2. 初始化预测器 ---
    try:
        predictor = VLAPredictor(
            checkpoint_path=args.checkpoint_path,
            config=config,
            device=args.device,
            logger=logger
        )
        logger.info("模型加载成功。")
    except Exception as e:
        logger.error(f"初始化预测器时失败: {e}", exc_info=True)
        return

    # --- 3. 加载和准备输入 ---
    try:
        logger.info(f"加载主图像: {args.image1_path}")
        image_1 = Image.open(args.image1_path).convert("RGB")
        
        image_2 = None
        if args.image2_path:
            logger.info(f"加载腕部图像: {args.image2_path}")
            image_2 = Image.open(args.image2_path).convert("RGB")
        elif config.model.vlm_config.get('use_aux_camera', False):
             logger.warning("模型配置需要腕部相机图像，但未通过 --image2_path 提供。")

        prompt = args.prompt
        state_vector = np.array(args.state, dtype=np.float32)

        logger.info(f"使用指令: '{prompt}'")
        logger.info(f"使用状态向量: {state_vector.tolist()}")

    except FileNotFoundError as e:
        logger.error(f"输入文件未找到: {e}")
        return
    except Exception as e:
        logger.error(f"加载输入时出错: {e}", exc_info=True)
        return

    # --- 4. 执行预测 ---
    try:
        logger.info("正在执行模型预测...")
        start_time = time.time()
        
        # 使用 predictor 的辅助函数准备输入
        preprocessed_input = predictor.preprocess_single_item_direct(
            image_1=image_1,
            prompt_text=prompt,
            image_2=image_2,
            state_vector=state_vector
        )
        
        # 模型推理
        predictions = predictor.predict(preprocessed_input)
        action = predictions['action']
        
        end_time = time.time()
        logger.info(f"预测完成，耗时: {end_time - start_time:.3f} 秒。")

        # --- 5. 显示结果 ---
        print("\n" + "="*35)
        print("         模型预测结果")
        print("="*35)
        print(f"  语言指令: '{prompt}'")
        print(f"  输入状态: {state_vector.tolist()}")
        print("-" * 35)
        # 将numpy数组转换为列表以便打印
        predicted_action_list = action.tolist()
        print(f"  预测动作 (7-dim): {predicted_action_list}")
        print("="*35)

    except Exception as e:
        logger.error(f"执行预测时出错: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VLA模型单次推理工具。给定图像、指令和状态，预测动作。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # 输入参数
    parser.add_argument('--image1_path', type=str, required=True, help='主相机图像的文件路径。')
    parser.add_argument('--prompt', type=str, required=True, help='自然语言指令。')
    parser.add_argument('--state', type=float, nargs=7, required=True, help='7维的机器人状态向量，以7个浮点数表示，用空格分隔。')
    parser.add_argument('--image2_path', type=str, default=None, help='(可选) 腕部相机图像的文件路径。')

    # 模型和配置参数
    parser.add_argument('--checkpoint_path', type=str, required=True, help='模型检查点路径 (.pth.tar 或 .pth)。')
    parser.add_argument('--config_path', type=str, default=None, help='可选的模型YAML配置文件路径。')
    
    # 其他参数
    parser.add_argument('--device', type=str, default=None, help='运行设备 (例如 "cuda:0" 或 "cpu"), 不指定则自动检测。')
    
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