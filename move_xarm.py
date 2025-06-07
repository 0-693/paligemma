import torch
import numpy as np
import cv2
import time
import argparse
from xarm_scripts import XArmAPI
from inference.predictor import VLAPredictor
from utils.misc import setup_logging

def init_robot_and_cameras(ip_address):
    """初始化机械臂和相机"""
    # 初始化机械臂
    arm = XArmAPI(ip_address)
    arm.motion_enable(enable=True)
    arm.set_gripper_enable(enable=True)
    arm.set_mode(6)
    arm.set_state(0)
    
    # 初始化相机
    base_camera = cv2.VideoCapture("/dev/video0")  # 基座相机
    wrist_camera = cv2.VideoCapture("/dev/video6")  # 手腕相机
    
    if not base_camera.isOpened() or not wrist_camera.isOpened():
        raise RuntimeError("无法打开一个或多个相机")
        
    return arm, base_camera, wrist_camera

def capture_images(base_camera, wrist_camera):
    """捕获两个相机的图像"""
    # 捕获基座相机图像
    ret_base, base_img = base_camera.read()
    if not ret_base:
        raise RuntimeError("无法从基座相机捕获图像")
    
    # 捕获手腕相机图像
    ret_wrist, wrist_img = wrist_camera.read()
    if not ret_wrist:
        raise RuntimeError("无法从手腕相机捕获图像")
    
    # 调整图像大小为224x224（根据模型要求）
    base_img_resized = cv2.resize(base_img, (224, 224))
    wrist_img_resized = cv2.resize(wrist_img, (224, 224))
    
    # 将BGR转换为RGB格式
    base_img_rgb = cv2.cvtColor(base_img_resized, cv2.COLOR_BGR2RGB)
    wrist_img_rgb = cv2.cvtColor(wrist_img_resized, cv2.COLOR_BGR2RGB)
    
    return base_img_rgb, wrist_img_rgb

def get_robot_state(arm):
    """获取机械臂当前状态"""
    # 获取关节角度（弧度）
    _, curr_angles_rad = arm.get_servo_angle(is_radian=True)
    # 获取夹爪位置
    _, curr_gripper_state = arm.get_gripper_position()
    # 组合成完整的状态向量
    curr_state = np.append(curr_angles_rad[:-1], curr_gripper_state)  # 去掉最后一个关节，加上夹爪状态
    return curr_state

def execute_action(arm, action, speed=8):
    """执行预测的动作"""
    # 分离关节角度和夹爪动作
    joint_action = action[:-1]  # 前6个值是关节角度
    gripper_action = action[-1]  # 最后一个值是夹爪动作
    
    # 设置关节角度
    arm.set_servo_angle(angle=joint_action, speed=speed, is_radian=True)
    # 设置夹爪位置
    arm.set_gripper_position(pos=gripper_action, wait=False)
    
    # 等待动作完成
    time.sleep(1/30)  # 30Hz的执行频率

def main(args):
    # 设置日志
    logger = setup_logging(name="XArmController")
    logger.info("初始化系统...")
    
    try:
        # 初始化机械臂和相机
        arm, base_camera, wrist_camera = init_robot_and_cameras(args.ip)
        logger.info(f"成功连接到机械臂（IP: {args.ip}）和相机")
        
        # 初始化预测器
        predictor = VLAPredictor(
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            logger=logger
        )
        logger.info("模型加载成功")
        
        while True:
            # 获取用户输入的prompt
            prompt = input("\n请输入指令 (输入'q'退出): ")
            if prompt.lower() == 'q':
                break
                
            try:
                # 捕获图像
                base_img, wrist_img = capture_images(base_camera, wrist_camera)
                
                # 获取机械臂当前状态
                current_state = get_robot_state(arm)
                
                # 准备模型输入（直接使用图像数据）
                preprocessed_input = predictor.preprocess_single_item_direct(
                    image_1=base_img,
                    prompt_text=prompt,
                    image_2=wrist_img,
                    state_vector=current_state,
                    max_seq_len=1,
                    prompt_max_len=128
                )
                
                # 模型推理并获取反归一化后的动作
                predictions = predictor.predict(preprocessed_input)
                action = predictions['action']  # 直接获取反归一化后的动作
                
                # 执行动作
                logger.info("执行预测的动作...")
                execute_action(arm, action)
                logger.info("动作执行完成")
                
            except Exception as e:
                logger.error(f"执行过程中出错: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"系统初始化失败: {str(e)}")
    finally:
        # 清理资源
        if 'base_camera' in locals():
            base_camera.release()
        if 'wrist_camera' in locals():
            wrist_camera.release()
        cv2.destroyAllWindows()
        logger.info("系统已关闭")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XArm控制器与VLA模型集成")
    parser.add_argument('--ip', type=str, default='192.168.1.222',
                        help='机械臂的IP地址')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--device', type=str, default=None,
                        help='运行设备 (例如 "cuda:0" 或 "cpu")')
    
    args = parser.parse_args()
    main(args)
