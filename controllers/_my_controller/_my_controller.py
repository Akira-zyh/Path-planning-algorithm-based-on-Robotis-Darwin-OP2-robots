# 需确保Python版本为3.7+，并在PyCharm中配置Webots库路径（参考网页3）
from controller import Robot, Keyboard
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager
import sys
import time
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
        self.led_head.set(0xFF00FF)
        self.led_eye.set(0x0006FF)
        
        # 传感器初始化（网页4示例）
        self.accelerometer = self.getDevice("Accelerometer")
        self.accelerometer.enable(self.time_step)
        self.gyro = self.getDevice("Gyro")
        self.gyro.enable(self.time_step)
        self.camera = self.getDevice("Camera")
        self.camera.enable(2*self.time_step)
        self.compass = self.getDevice("compass")
        self.compass.enable(self.time_step)
        self.imu= self.getDevice("imu")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice('gps')
        self.gps.enable(self.time_step)
        
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
        
        # 动作管理模块（网页1关键配置）
        self.motion_manager = RobotisOp2MotionManager(self)
        self.gait_manager = RobotisOp2GaitManager(self, "config.ini")
        
    def my_step(self):
        if self.step(self.time_step) == -1:
            sys.exit(0)
    
    def wait(self, ms):
        start_time = self.getTime()
        while (self.getTime() - start_time) * 1000 < ms:
            self.my_step()
    
    def run(self):
        print("-------Walk example of ROBOTIS OP2(Python)-------")
        print("Press SPACE to start/stop walking")
        print("Use ARROWS to control movement")
        
        self.my_step()  # 首次更新传感器值
        
        # 初始化动作（网页1示例）
        self.motion_manager.playPage(9)  # 初始姿势
        self.motors[-1].setPosition(0.7)
        self.wait(200)
        print("Camera width:", self.camera.getWidth())
        print("Camera height:", self.camera.getHeight())
        # image_data = self.camera.getImage()
        # image = np.frombuffer(image_data, np.uint8).reshape(
            # (self.camera.getHeight(), self.camera.getWidth(), 4)  # RGBA格式，4通道
        # )
        # image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        # cv2.imwrite('camera_image.png', image_bgr)
        
        is_walking = False
        while True:
            self.check_if_fallen()
            
            # 重置步态参数
            self.gait_manager.setXAmplitude(0.0)
            self.gait_manager.setAAmplitude(0.0)
            
            # 处理键盘输入（网页2事件循环）
            key = self.keyboard.getKey()
            if key != -1:
                if key == ord(' '):
                    if is_walking:
                        self.gait_manager.stop()
                        is_walking = False
                        self.wait(200)
                    else:
                        self.gait_manager.start()
                        is_walking = True
                        self.wait(200)
                elif key == Keyboard.UP:
                    self.gait_manager.setXAmplitude(1.0)
                elif key == Keyboard.DOWN:
                    self.gait_manager.setXAmplitude(-1.0)
                elif key == Keyboard.RIGHT:
                    self.gait_manager.setAAmplitude(-0.5)
                elif key == Keyboard.LEFT:
                    self.gait_manager.setAAmplitude(0.5)
            
            # 步态更新（网页4主循环逻辑）
            self.gait_manager.step(self.time_step)
            self.my_step()
    
    def check_if_fallen(self):
        acc_tolerance = 80.0
        acc_step = 100
        
        # 获取加速度计数据（网页4传感器读取方法）
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
            
        # 跌倒恢复动作（网页1检查逻辑）
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

# 主程序入口（参考网页2启动方式）
if __name__ == "__main__":
    controller = Humanoid()
    controller.run()