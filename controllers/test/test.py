from controller import Robot, Motor, Accelerometer, Gyro, LED
from managers import RobotisOp2MotionManager, RobotisOp2GaitManager  # 假设存在对应的 Python 管理器类
import sys
import time

NMOTORS = 20

motor_names = [
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR",
    "ArmLowerL", "PelvYR", "PelvYL", "PelvR", "PelvL", "LegUpperR",
    "LegUpperL", "LegLowerR", "LegLowerL", "AnkleR", "AnkleL", "FootR",
    "FootL", "Neck", "Head"
]

class Walk(Robot):
    def __init__(self):
        super().__init__()  # 初始化父类 Robot
        self.mTimeStep = int(self.getBasicTimeStep())

        # 初始化 LED
        self.head_led = self.getDevice("HeadLed")
        self.eye_led = self.getDevice("EyeLed")
        self.head_led.set(0xFF0000)
        self.eye_led.set(0x00FF00)
        self.camera = self.getDevice("Camera")
        self.camera.enable(2*self.mTimeStep)

        # 初始化传感器
        self.accelerometer = self.getDevice("Accelerometer")
        self.accelerometer.enable(self.mTimeStep)
        self.gyro = self.getDevice("Gyro")
        self.gyro.enable(self.mTimeStep)

        # 初始化电机和位置传感器
        self.motors = []
        self.position_sensors = []
        for name in motor_names:
            motor = self.getDevice(name)
            self.motors.append(motor)
            sensor = self.getDevice(name + "S")
            sensor.enable(self.mTimeStep)
            self.position_sensors.append(sensor)

        # 键盘输入
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.mTimeStep)

        # 初始化运动管理器
        self.mMotionManager = RobotisOp2MotionManager(self)
        self.mGaitManager = RobotisOp2GaitManager(self, "config.ini")

        # 跌倒检测计数器
        self.fup = 0
        self.fdown = 0

    def myStep(self):
        ret = self.step(self.mTimeStep)
        if ret == -1:
            sys.exit(0)  # 仿真结束正常退出

    def wait(self, ms):
        start_time = self.getTime()
        while (self.getTime() - start_time) * 1000 < ms:
            self.myStep()
            
    def walk_straight(self):
        start_time = self.getTime()
        while self.getTime() - start_time < 1.0:
            self.checkIfFallen()
            self.mGaitManager.setXAmplitude(1.0)
            self.mGaitManager.setAAmplitude(0.0)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
            
    def walk_backwards(self):
        start_time = self.getTime()
        while self.getTime() - start_time < 1.0:
            self.checkIfFallen()
            self.mGaitManager.setXAmplitude(-1.0)
            self.mGaitManager.setAAmplitude(0.0)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
    
    def turn_left(self):
        start_time = self.getTime()
        while self.getTime() - start_time < 1.0:
            self.checkIfFallen()
            self.mGaitManager.setXAmplitude(0.0)
            self.mGaitManager.setAAmplitude(0.5)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
    
    def turn_right(self):
        start_time = self.getTime()
        while self.getTime() - start_time < 1.0:
            self.checkIfFallen()
            self.mGaitManager.setXAmplitude(0.0)
            self.mGaitManager.setAAmplitude(-0.5)
            self.mGaitManager.step(self.mTimeStep)
            self.myStep()
    
    def run(self):
        print("-------Walk example of ROBOTIS OP2-------")
        print("This example illustrates Gait Manager")
        print("Press the space bar to start/stop walking")
        print("Use the arrow keys to move the robot while walking")

        self.myStep()  # 首次更新传感器


        # 启动步态管理器
        self.mGaitManager.start()
        is_walking = False
        self.mMotionManager.playPage(4)
        self.motors[19].setPosition(0.6)
        self.wait(50)
        for _ in range(0, 4):
            self.walk_backwards()
        for _ in range(0, 5):
            self.turn_right()
        # 阶段 1: 向前走 2 秒
        for _ in range(0, 100):
            self.walk_straight()

        self.mGaitManager.stop()
        sys.exit(0)

    def checkIfFallen(self):
        acc_tolerance = 80.0
        acc_step = 10
        acc = self.accelerometer.getValues()

        if acc[1] < 512.0 - acc_tolerance:
            self.fup += 1
        else:
            self.fup = 0

        if acc[1] > 480.0 + acc_tolerance:
            self.fdown += 1
        else:
            self.fdown = 0

        if self.fup > acc_step:
            print("Forward fall detected! Executing recovery...")
            self.mMotionManager.playPage(10)
            self.mMotionManager.playPage(9)
            self.fup = 0
        elif self.fdown > acc_step:
            print("Backward fall detected! Executing recovery...")
            self.mMotionManager.playPage(11)
            self.mMotionManager.playPage(9)
            self.fdown = 0
            
        #

if __name__ == "__main__":
    controller = Walk()
    controller.run()
    sys.exit(1)