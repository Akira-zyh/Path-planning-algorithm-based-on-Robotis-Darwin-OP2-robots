import os
import sys
from controller import Robot, Camera, LED
from managers import RobotisOp2VisionManager, RobotisOp2GaitManager


# 初始化机器人
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# 初始化 VisionManager
vision_manager = RobotisOp2VisionManager(robot)

# 初始化 GaitManager
gait_manager = RobotisOp2GaitManager(robot, "")

# 初始化摄像头
camera = robot.getDevice('Camera')
camera.enable(timestep)

# 初始化 LED
head_led = robot.getLED('HeadLed')
head_led.set(0xff0000)  # 设置头部 LED 为红色

# 主循环
while robot.step(timestep) != -1:
    # 获取摄像头图像
    image = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()

    # 使用 VisionManager 处理图像并寻找目标（例如彩色球）
    target_position = vision_manager.find_ball(image, width, height)
    
    # 根据目标位置调整机器人的运动
    if target_position:
        # 如果检测到目标，调整步态参数以朝向目标
        x_amplitude = target_position.x / width * 2.0 - 1.0  # 将目标位置转换为步态参数
        gait_manager.setXAmplitude(x_amplitude)
        gait_manager.setYAmplitude(0.0)  # 保持 Y 方向步态参数为 0
        gait_manager.setBalanceEnable(True)
    else:
        # 如果未检测到目标，保持原地踏步
        gait_manager.setXAmplitude(0.0)
        gait_manager.setYAmplitude(0.0)
        gait_manager.setBalanceEnable(True)

    # 执行步态管理器的步态
    gait_manager.step(timestep)
