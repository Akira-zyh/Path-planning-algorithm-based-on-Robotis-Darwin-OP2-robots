from controller import Robot, Keyboard
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import navigation

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
        
        # Initialize LEDs
        self.led_head = self.getDevice("HeadLed")
        self.led_eye = self.getDevice("EyeLed")
        self.led_head.set(0xFF00FF)
        self.led_eye.set(0x0006FF)
        
        # Initialize sensors
        self.accelerometer = self.getDevice("Accelerometer")
        self.accelerometer.enable(self.time_step)
        self.gyro = self.getDevice("Gyro")
        self.gyro.enable(self.time_step)
        self.camera = self.getDevice("Camera")
        self.camera.enable(2 * self.time_step)
        self.compass = self.getDevice("compass")
        self.compass.enable(self.time_step)
        self.imu = self.getDevice("imu")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice('gps')
        self.gps.enable(self.time_step)
        
        # Initialize motors and position sensors
        self.motors = []
        self.position_sensors = []
        for name in motorNames:
            motor = self.getDevice(name)
            sensor = self.getDevice(name + "S")
            sensor.enable(self.time_step)
            self.motors.append(motor)
            self.position_sensors.append(sensor)

        # Initialize keyboard
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.time_step)
        
        # Initialize navigation system
        self.navigation = navigation.MaplessNavigation(self)
        
        # Initialize fall detection counter
        self.fall_up_count = 0
        self.fall_down_count = 0
    
    def my_step(self):
        if self.step(self.time_step) == -1:
            sys.exit(0)
    
    def wait(self, ms):
        start_time = self.getTime()
        while (self.getTime() - start_time) * 1000 < ms:
            self.my_step()
    
    def execute_action(self, action):
        """Update motor position according to action

        Args:
            action (_type_): _description_
        """
        for i in range(len(self.motors)):
            self.motors[i].setPosition(action[i])
    
    def reset(self):
        """Reset robot to initial position
        """
        for motor in self.motors:
            motor.setPosition(0.0)
        self.wait(200)
    
    def get_inertial_unit_data(self):
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        return np.array([roll, pitch, yaw])
    
    def check_if_fallen(self):
        acc_tolerance = 80.0
        acc_step = 100
        
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
            self.reset()
            print("Done!")
            self.fall_up_count = 0
        elif self.fall_down_count > acc_step:
            print("Fall down detected. getting up...")
            self.reset()
            print("Done!")
            self.fall_down_count = 0
    
    def run(self):
        print("-------Walk example of ROBOTIS OP2(Python)-------")
        print("Press SPACE to start/stop walking")
        print("Use ARROWS to control movement")
        
        self.my_step()  # 首次更新传感器值
        
        # 启动导航系统
        self.navigation.run()

if __name__ == "__main__":
    controller = Humanoid()
    controller.run()