"""init controller."""

import os
import sys
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
        self.time_step = int(self.getBasicTimeStep())
        
        # 初始化LED（参考网页1）
        self.led_head = self.getDevice("HeadLed")
        self.led_eye = self.getDevice("EyeLed")
        self.led_head.set(0xFFFF00)
        self.led_eye.set(0xFF0400)
        
        # 传感器初始化（网页4示例）
        self.accelerometer = self.getDevice("Accelerometer")
        self.accelerometer.enable(self.time_step)
        self.gyro = self.getDevice("Gyro")
        self.gyro.enable(self.time_step)
        self.camera = self.getDevice("Camera")
        self.camera.enable(2*self.time_step)
        self.compass = self.getDevice("compass")
        self.compass.enable(self.time_step)
        self.gps = self.getDevice('gps')
        self.gps.enable(self.time_step)
        self.imu = self.getDevice('imu')
        self.imu.enable(self.time_step)
        self.target_position = [1.0, 0.0, -3.0]
        # 电机和位置传感器初始化（网页2方法）
        self.motors = []
        self.position_sensors = []
        for name in motorNames:
            motor = self.getDevice(name)
            sensor = self.getDevice(name + "S")
            sensor.enable(self.time_step)
            self.motors.append(motor)
            self.position_sensors.append(sensor)

        # 键盘控制（网页6事件处理）
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.time_step)
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
        print("Camera width:", self.camera.getWidth())
        print("Camera height:", self.camera.getHeight())
        
    def my_step(self):
        if self.step(self.time_step) == -1:
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
        
        
    def get_observations(self):
        current_position = self.gps.getValues()
        imu_orientation = self.imu.getRollPitchYaw()
        return {
            'rgb': self.camera.getImage(),
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
    
    def get_image(self):
        key = self.keyboard.getKey()
        if key == Keyboard.DOWN:
            image = self.camera.getImage()
            t = self.getTime()
            image_name = os.path.join("./data/easy1/", 'easy_{}_.png'.format(t))
            image = np.frombuffer(image, np.uint8).reshape(
                (self.camera.getHeight(), self.camera.getWidth(), 4) 
            )
            cv2.imwrite(image_name, image)
            print("image-{} got".format(t))
            
    
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
        print("Press ARROW DOWN to take image")
        # image_data = self.camera.getImage()
        # image = np.frombuffer(image_data, np.uint8).reshape(
        #     (self.camera.getHeight(), self.camera.getWidth(), 4)  # RGBA格式，4通道
        # )
        # image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        # img_name = 'img-{width}x{height}.png'.format(width=self.camera.getWidth(), height=self.camera.getHeight())
        # cv2.imwrite("bgr "+img_name, image_bgr)
        # cv2.imwrite("rgb "+img_name, image)
        # i = 0
        # j = 0
        self.gait_manager.start()
        
        while True:
            self.check_if_fallen()
            # print(self.get_observations()['current_position'], self.get_observations()['target_polar'])
            # 重置步态参数
            self.gait_manager.setXAmplitude(0.0)
            self.gait_manager.setAAmplitude(0.0)
            self.gait_manager.stop()
            
            self.get_image()
            
            # random_action = np.random.randint(0, 4)
            # self.execute_action(random_action)
            # print(random_action)
            
            # 步态更新
            self.gait_manager.step(self.time_step)
            self.my_step()
            
# 主程序入口
if __name__ == "__main__":
    controller = Humanoid()
    controller.run()


