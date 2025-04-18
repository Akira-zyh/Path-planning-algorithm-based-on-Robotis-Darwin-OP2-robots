from deepbots.supervisor.controllers.robot_supervisor import RobotSupervisor
# from deepbots.supervisor.controllers.supervisor_env import RobotSupervisorEnv
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
import numpy as np
import sys

NMOTORS = 20
motorNames = [
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR",
    "ArmLowerL", "PelvYR", "PelvYL", "PelvR", "PelvL", "LegUpperR",
    "LegUpperL", "LegLowerR", "LegLowerL", "AnkleR", "AnkleL", "FootR",
    "FootL", "Neck", "Head"
]

def normalize_to_range(value, min_val, max_val, new_min, new_max, clip=False):
    """
    Normalize value to a specified new range by supplying the current range.

    :param value: value to be normalized
    :param min_val: value's min value, value ∈ [min_val, max_val]
    :param max_val: value's max value, value ∈ [min_val, max_val]
    :param new_min: normalized range min value
    :param new_max: normalized range max value
    :param clip: whether to clip normalized value to new range or not
    :return: normalized value ∈ [new_min, new_max]
    """
    value = float(value)
    min_val = float(min_val)
    max_val = float(max_val)
    new_min = float(new_min)
    new_max = float(new_max)

    if clip:
        return np.clip((new_max - new_min) / (max_val - min_val) * (value - max_val) + new_max, new_min, new_max)
    else:
        return (new_max - new_min) / (max_val - min_val) * (value - max_val) + new_max

class NavigationEnv(RobotSupervisor):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=np.array([0.0] * 10 + [-1.0 , -1.0]),
                                     high=np.array([10.0] * 10 + [1.0 , 1.0]),
                                     dtype=np.float64)
        self.action_space = spaces.Discrete(4)
        self.robot = self.supervisor.getSelf()
        self.led_head = self.supervisor.getDevice("HeadLed")
        self.led_eye = self.supervisor.getDevice("EyeLed")
        self.led_head.set(0xFFFF00)
        self.led_eye.set(0xFF0400)
        self.motors = []
        self.position_sensors = []
        for name in motorNames:
            motor = self.supervisor.getDevice(name)
            sensor = self.supervisor.getDevice(name + "S")
            sensor.enable(self.timestep)
            self.motors.append(motor)
            self.position_sensors.append(sensor)
        self.accelerometer = self.supervisor.getDevice("Accelerometer")
        self.accelerometer.enable(self.timestep)
        self.gyro = self.supervisor.getDevice("Gyro")
        self.gyro.enable(self.timestep)
        # self.camera = self.supervisor.getDevice("camera")
        # self.camera.enable(self.timestep)
        # self.camera.recognitionEnable(self.timestep)
        # self.camera.enableRecognitionSegmentation()
        self.compass = self.supervisor.getDevice("compass")
        self.compass.enable(self.timestep)
        self.lidar = self.supervisor.getDevice("lidar")
        self.lidar.enablePointCloud()
        self.lidar.enable(self.timestep)
        self.gps = self.supervisor.getDevice('gps')
        self.gps.enable(self.timestep)
        self.imu = self.supervisor.getDevice('imu')
        self.imu.enable(self.timestep)
        self.motion_manager = RobotisOp2MotionManager(self.supervisor)
        self.gait_manager = RobotisOp2GaitManager(self.supervisor, "config.ini")
        self.steps_per_episode = 500
        self.episode_score = 0
        self.episode_score_list = []
        # robot_velocity = self.supervisor.getVelocity()[0]
        self.my_step()
        target_position = self.supervisor.getFromDef("TARGET").getPosition()
        robot_position = self.gps.getValues()
        distance_to_target = np.linalg.norm(np.array(target_position) - np.array(robot_position))
        print("Distance to target: ", distance_to_target)
        print("Robot Position: ", self.gps.getValues())
        print("Target Position: ", self.supervisor.getFromDef("TARGET").getPosition())

    def my_step(self):
        if self.step(self.timestep) == -1:
            sys.exit(0)
    def get_observations(self):
        lidar_data = self.lidar.getRangeImage()
        lidar_pointData = self.lidar.getPointCloud()
        # print(len(lidar_data))
        simplified_lidar_data = [np.mean(lidar_data[i::10]) for i in range(10)]
        # print(simplified_lidar_data)
        normalized_lidar_data = [normalize_to_range(d, 0.0, 10.0, -1.0, 1.0) for d in simplified_lidar_data]
        # print(normalized_lidar_data)
        # target_position = self.supervisor.getFromDef("TARGET").getPosition()
        # robot_position = self.gps.getValues()
        # distance_to_target = np.linalg.norm(np.array(target_position) - np.array(robot_position))
        # normalized_distance = normalize_to_range(distance_to_target, 0.0, 10.0, -1.0, 1.0)
        
        # robot_velocity = self.robot.getVelocity()[0]
        # normalized_velocity = normalize_to_range(robot_velocity, -0.5, 0.5, -1.0, 1.0)
        
        # return normalized_lidar_data + [normalized_distance, normalized_velocity]
        pass
        
    def get_reward(self, action=None):
        reward = 0.1
        lidar_data = self.lidar.getRangeImage() 
        min_distance = min(lidar_data)
        if min_distance < 0.5:
            reward -= 1.0
        else:
            reward += 0.1
        
        target_position = self.supervisor.getFromDef("TARGET").getPosition()
        robot_position  = self.robot.getPosition()
        distance = np.linalg.norm(np.array(target_position) - np.array(robot_position))
        if distance < 0.5:
            reward += 10.0
    
        return reward
        
    def is_done(self):
        lidar_data = self.lidar.getRangeImage()
        min_distance = min(lidar_data)
        if min_distance < 0.3:
            return True

        target_position = self.supervisor.getFromDef("TARGET").getPosition()
        robot_position = self.gps.getValues()
        distance = np.linalg.norm(np.array(target_position) - np.array(robot_position))
        if distance < 0.5:
            return True
        
        return False
    
    def apply_action(self, action):
        if action == 0:
            self.gait_manager.setXAmplitude(1.0)
        elif action == 1:
            self.gait_manager.setXAmplitude(-1.0)
        elif action == 2:
            self.gait_manager.setAAmplitude(-0.5)
        elif action == 3:
            self.gait_manager.setAAmplitude(0.5)
        self.my_step()
    
    def get_info(self):
        return None
    
    def run(self):
        while True:
            self.my_step()
            data = self.lidar.getPointCloud()
            self.apply_action(0)
            
    

env = NavigationEnv()
env.run()