"""init controller."""

import os
import sys

from matplotlib import pyplot as plt
libraryPath = os.path.join(os.environ.get("WEBOTS_HOME"), 'projects', 'robots', 'robotis', 'darwin-op',  'libraries', 'python39')
libraryPath = libraryPath.replace('/', os.sep)
sys.path.append(libraryPath)
from controller import Robot, Keyboard
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager
import cv2
import torch
from torchvision import transforms
import numpy as np


NMOTORS = 20
motorNames = [
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR",
    "ArmLowerL", "PelvYR", "PelvYL", "PelvR", "PelvL", "LegUpperR",
    "LegUpperL", "LegLowerR", "LegLowerL", "AnkleR", "AnkleL", "FootR",
    "FootL", "Neck", "Head"
]

class Humanoid(Robot):
    def __init__(self):
        super().__init__()
        self.timeStep = int(self.getBasicTimeStep())
        
        # 初始化LED（参考网页1）
        self.led_head = self.getDevice("HeadLed")
        self.led_eye = self.getDevice("EyeLed")
        self.led_head.set(0xFFFF00)
        self.led_eye.set(0xFF0400)
        
        # 传感器初始化（网页4示例）
        self.accelerometer = self.getDevice("Accelerometer")
        self.accelerometer.enable(self.timeStep)
        self.gyro = self.getDevice("Gyro")
        self.gyro.enable(self.timeStep)
        self.camera = self.getDevice("camera")
        self.camera.enable(self.timeStep)
        self.camera.recognitionEnable(self.timeStep)
        self.camera.enableRecognitionSegmentation()
        self.compass = self.getDevice("compass")
        self.compass.enable(self.timeStep)
        self.gps = self.getDevice('gps')
        self.gps.enable(self.timeStep)
        self.imu = self.getDevice('imu')
        self.imu.enable(self.timeStep)
        # self.distancesensor = self.getDevice("distance sensor")
        # self.distancesensor.enable(self.timeStep)
        self.lidar = self.getDevice("lidar")
        self.lidar.enablePointCloud()
        self.lidar.enable(self.timeStep)
        self.max_range = 4
        self.angle_bins = np.linspace(0, 360, 361)
        self.target_position = [1.0, 0.0, -3.0]
        # 电机和位置传感器初始化（网页2方法）
        self.motors = []
        self.position_sensors = []
        for name in motorNames:
            motor = self.getDevice(name)
            sensor = self.getDevice(name + "S")
            sensor.enable(self.timeStep)
            self.motors.append(motor)
            self.position_sensors.append(sensor)

        # 键盘控制（网页6事件处理）
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.timeStep)
        self.fall_down_count = 0
        self.fall_up_count = 0
        # 动作管理模块（网页1关键配置）
        self.motion_manager = RobotisOp2MotionManager(self)
        self.gait_manager = RobotisOp2GaitManager(self, "config.ini")
        self.my_step()  # 首次更新传感器值
        
        # 初始化动作
        self.motion_manager.playPage(9)  # 初始姿势
        self.motors[-1].setPosition(0.8)
        self.wait(200)
        
    def my_step(self):
        if self.step(self.timeStep) == -1:
            sys.exit(0)
    
    def wait(self, ms):
        start_time = self.getTime()
        while (self.getTime() - start_time) * 1000 < ms:
            self.my_step()
    
    # def image_processing(self, index):
    #     image_data = self.camera.getImage()
    #     image = np.frombuffer(image_data, np.uint8).reshape(
    #         (self.camera.getHeight(), self.camera.getWidth(), 4)  # RGBA格式，4通道
    #     )
    #     cv2.imwrite(os.path.join("./TRAIN_DATA/", 'img2_{index}.png'.format(index=index)), image)
    
    def _calculate_goal_position(self, current_pos, imu_orientation):
        dx = self.target_position[0] - current_pos[0]
        dz = self.target_position[2] - current_pos[2]
        yaw = imu_orientation[2]
        rotated_x = dx * np.cos(yaw) + dz * np.sin(yaw)
        rotated_y = -dx * np.sin(yaw) + dz * np.cos(yaw)
        distance = np.sqrt(rotated_x**2 + rotated_y**2)
        angle = np.atan2(rotated_y, rotated_x)
        
        return [distance, angle]
        
    
    def visualize_point_cloud(self, point_cloud):
        # 提取点云的 x 和 z 坐标
        x = [point.x for point in point_cloud]
        z = [point.z for point in point_cloud]
        
        # 绘制点云
        plt.scatter(x, z, s=1)
        plt.xlim(-self.max_range, self.max_range)
        plt.ylim(-self.max_range, self.max_range)
        plt.axis('equal')
        plt.show()
        
    def get_observations(self):
        current_position = self.gps.getValues()
        imu_orientation = self.imu.getRollPitchYaw()
        return {
            # 'rgb': self.camera.getImage(),
            'current_position': current_position,
            'target_polar': self._calculate_goal_position(current_position, imu_orientation)
        }
    
    def set_target(self, x, y):
        self.target_position = [x, y]
    
    def execute_action(self, action):
        self.check_if_fallen()
        if action == 0:
            self.gait_manager.setXAmplitude(1.0)
        elif action == 1:
            self.gait_manager.setXAmplitude(-1.0)
        elif action == 2:
            self.gait_manager.setAAmplitude(-0.5)
        elif action == 3:
            self.gait_manager.setAAmplitude(0.5)
            
    def get_polar_data(self):
        point_cloud = self.lidar.getPointCloud()
        polar_data = []
        for point in point_cloud:
            r = np.sqrt(point.x**2 + point.z**2)
            theta = np.arctan2(point.z, point.x)
            polar_data.append((r, theta))
        if not point_cloud:
            print("Lidar点云数据为空，请检查设备是否启用")
            return
        return polar_data

    def check_if_fallen(self):
        acc_tolerance = 80.0
        acc_step = 50
        
        # 获取加速度计数据
        acc_values = self.accelerometer.getValues()
        y_acc = acc_values[1]
        
        if y_acc < 512.0 - acc_tolerance:
            self.fall_up_count += 1
        else:
            self.fall_up_count = 0
            
        if y_acc > 512.0 + acc_tolerance:
            self.fall_down_count += 1
        else:
            self.fall_down_count = 0
            
        # 跌倒恢复动作
        if self.fall_up_count > acc_step:
            print("Fall up detected. getting up...")
            self.motion_manager.playPage(10)  # 前滚翻恢复
            self.motion_manager.playPage(9)
            self.motors[-1].setPosition(0.7)
            print("Done!")
            self.fall_up_count = 0
        elif self.fall_down_count > acc_step:
            print("Fall down detected. getting up...")
            self.motion_manager.playPage(11)  # 后滚翻恢复
            self.motion_manager.playPage(9)
            self.motors[-1].setPosition(0.7)
            print("Done!")
            self.fall_down_count = 0

    def run(self):
        self.gait_manager.start()
        
        while True:
            self.check_if_fallen()
            # self.visualize_point_cloud(self.lidar.getPointCloud())
            data = self.lidar.getPointCloud()
            # 重置步态参数
            self.gait_manager.setXAmplitude(0.0)
            self.gait_manager.setAAmplitude(0.0)
            self.gait_manager.stop()
            
            
            # random_action = np.random.randint(0, 4)
            # self.execute_action(random_action)
            # print(random_action)
            
            # 步态更新
            self.gait_manager.step(self.timeStep)
            self.my_step()


            
# 主程序入口
if __name__ == "__main__":
    controller = Humanoid()
    controller.run()


