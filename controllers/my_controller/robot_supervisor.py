import random
import numpy as np
from controller import Supervisor, Keyboard
from gym.spaces import Discrete, Box
from robot_supervisor_env import RobotSupervisorEnv
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager

motorNames = [
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR",
    "ArmLowerL", "PelvYR", "PelvYL", "PelvR", "PelvL", "LegUpperR",
    "LegUpperL", "LegLowerR", "LegLowerL", "AnkleR", "AnkleL", "FootR",
    "FootL", "Neck", "Head"
]
LIDAR_RESOLUTION = 64
LIDAR_RANGE = 5.0
MAX_ENVIRONMENT_DISTANCE = 10.0
STEP_PENALTY = -0.1
COLLISION_PENALTY = -50.0
TARGET_REWARD = 1000.0
FALL_PENALTY = -100.0


PROGRESS_REWARD_WEIGHT = 1.0
COLLISION_DISTANCE_THRESHOLD = 0.15
TARGET_DISTANCE_THRESHOLD = 0.3
FALL_THRESHOLD = 1.0


class NavigationRobotSupervisor(RobotSupervisorEnv):
    def __init__(self, description, seed=None):
        super().__init__()
        
        # ================ General Setup ==================
        random.seed(seed) if seed is not None else random.seed()
        self.experiment_decription = description
        self.viewpoint = self.getFromDef("VIEWPOINT")
        self.viewpoint_position = self.viewpoint.getField("position").getSFVec3f()
        self.viewpoint_orientation = self.viewpoint.getField("orientation").getSFRotation()
        
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)

        # ================ Robot Setup ===================
        
        self.robot = self.getSelf()
        self.motors = []
        self.position_sensors = []
        for name in motorNames:
            motor = self.getDevice(name)
            sensor = self.getDevice(name + "S")
            sensor.enable(self.timestep)
            self.motors.append(motor)
            self.position_sensors.append(sensor)
        self.fall_up_count = 0
        self.fall_down_count = 0
        self.led_head = self.getDevice("HeadLed")
        self.led_eye = self.getDevice("EyeLed")
        self.led_head.set(0xFFFF00)
        self.led_eye.set(0xFF0400) 
        self.gait_manager = RobotisOp2GaitManager(self, "config.ini")
        self.motion_manager = RobotisOp2MotionManager(self)
        self.motion_manager.playPage(9)
        self.accelerometer = self.getDevice("Accelerometer")
        self.accelerometer.enable(self.timestep)
        self.gait_manager.start()
        self.gait_manager.setBalanceEnable(True)
        self.gyro = self.getDevice("Gyro")
        self.gyro.enable(self.timestep)
        self.lidar = self.getDevice("lidar")
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        self.distance_sensors = []
        sensor_names = ["front_ds", "left_front_ds", "right_front_ds"]
        for name in sensor_names:
            sensor = self.getDevice(name)
            if sensor is not None:
                sensor.enable(self.timestep)
                self.distance_sensors.append(sensor)
            else:
                print(f"Warning: Distance sensor {name} not found!")
        
        # ================ Env Setup =================== 
        
        self.target_node = self.getFromDef("TARGET")
        self.target_position = np.array(self.target_node.getPosition()[:2])
        print("Initial target pos: ", self.target_position)
        
        OBSERVATION_SPACE = LIDAR_RESOLUTION + 2
        low_bounds = np.concatenate([np.full(LIDAR_RESOLUTION, 0.0), np.array([0.0, -1.0])], dtype=np.float32)
        high_bounds = np.concatenate([np.full(LIDAR_RESOLUTION, 1.0), np.array([1.0, 1.0])], dtype=np.float32)
        self.observation_space = Box(low_bounds, high_bounds, shape=(OBSERVATION_SPACE, ), dtype=np.float32)
        self.action_space = Discrete(4)
        
        self.previous_distance_to_target = float('inf')
        self.current_distance_to_target = float('inf')
        self.current_relative_angle_to_target = 0.0
        
        # ================ Robot Initial Pos =================== 
        self.initial_robot_trans_field = self.robot.getField('translation')
        self.initial_robot_rot_field = self.robot.getField('rotation')
        self.initial_robot_pos_value = self.initial_robot_trans_field.getSFVec3f()[:2]
        self.initial_robot_rot_value = self.initial_robot_rot_field.getSFRotation()
        self.info = {}
        
        print(f"Initial robot pos:  [{self.initial_robot_pos_value[0]:.1f}, {self.initial_robot_pos_value[1]:.1f}]")
        # ================ Test ===================
        position = self.robot.getPosition()
        orientation_matrix = self.robot.getOrientation() 
        print(orientation_matrix)
        
        
    def get_observations(self):
        range_image = self.lidar.getRangeImage()
        
        if range_image:
            lidar_state = np.array(range_image, dtype=np.float32)
            lidar_state[np.isinf(lidar_state)] = LIDAR_RANGE
            lidar_state[np.isnan(lidar_state)] = LIDAR_RANGE
            normalized_lidar_state = lidar_state / LIDAR_RANGE
        else:
            # return normalized max range if no range image is available
            normalized_lidar_state = np.full(LIDAR_RESOLUTION, 1.0, dtype=np.float32)
        
        position = self.robot.getPosition()[:2]
        orientation_matrix = self.robot.getOrientation()
        
        yaw = np.arctan2(orientation_matrix[1], orientation_matrix[0])
        
        robot_pos_np = np.array(position)
        target_vector = self.target_position - robot_pos_np
        
        distance = np.linalg.norm(target_vector)
        
        angle_to_target_world = np.arctan2(target_vector[1], target_vector[0])
        relative_angle = angle_to_target_world - yaw
        relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))
        
        normalized_distance = distance / MAX_ENVIRONMENT_DISTANCE
        normalized_angle = relative_angle / np.pi
        
        current_state = np.concatenate([normalized_lidar_state, np.array([normalized_distance, normalized_angle], dtype=np.float32)])

        self.current_distance_to_target = distance
        self.current_relative_angle_to_target = relative_angle
        
        return current_state
        
    def get_reward(self, action):
        reward = STEP_PENALTY
        info = self.get_info()
        collision = self.is_done() and info.get('collision', False)
        target_reached = self.is_done() and info.get('target_reached', False)
        fell_down = info.get('fell_down', False)
        if collision:
            reward = COLLISION_PENALTY
        elif target_reached:
            reward = TARGET_REWARD
        elif fell_down:
            reward += FALL_PENALTY
        else:
            reward += PROGRESS_REWARD_WEIGHT * (self.previous_distance_to_target - self.current_distance_to_target)
            # Optional: Penalize large angular velocity if straight movement is preferred
            # reward -= 0.005 * abs(action[1]) # action is the action from the previous step
        self.previous_distance_to_target = self.current_distance_to_target
        
        return reward

    def is_done(self):
        collision = False
        
        for sensor in self.distance_sensors:
            if sensor.getValue() < COLLISION_DISTANCE_THRESHOLD:
                collision = True
                break
                
        target_reached = self.current_distance_to_target < TARGET_DISTANCE_THRESHOLD
        if target_reached:
            print("Target reached!")
        fell_down = False
        
        # TODO: check if robot fell down instead of check the value of accelerometer
        if self.check_if_fallen():
            fell_down = True
        else: fell_down = False
        
        done = collision or target_reached or fell_down
        
        self.info = {
            'collision': collision,
            'target_reached': target_reached,
            'fell_down': fell_down,
            'distance_to_target': self.current_distance_to_target,
            'angle_to_target': self.current_relative_angle_to_target
        }
        return done

    def check_if_fallen(self) -> bool:
        acc_tolerance = 80.0
        acc_step = 100
        
        # 获取加速度计数据
        acc_values = self.accelerometer.getValues()
        y_acc = acc_values[1]
        
        if y_acc < 512.0 - acc_tolerance:
            self.fall_up_count += 10
        else:
            self.fall_up_count = 0
            
        # if self.robot.getField("translation").getSFVec3f()[2] < 0.1:
        if y_acc > 512.0 + acc_tolerance:
            self.fall_down_count += 10
        else:
            self.fall_down_count = 0
            
        # 跌倒恢复动作
        if self.fall_up_count > acc_step:
            self.motion_manager.playPage(10)  # 前滚翻恢复
            self.motion_manager.playPage(9)
            self.fall_up_count = 0
            return True
        elif self.fall_down_count > acc_step:
            self.motion_manager.playPage(11)  # 后滚翻恢复
            self.motion_manager.playPage(9)
            self.fall_down_count = 0
            return True
        return False
    
    def apply_action(self, action):
        # self.gait_manager.start()
        if action == 0: # Stop walking
            # self.gait_manager.stop()
            pass
        elif action == 1: # Start walking forward
            # Set gait parameters for forward walk (adjust amplitudes as needed)
            self.gait_manager.setXAmplitude(1.0) # Forward step length
            self.gait_manager.setYAmplitude(0.0)  # Sideways step length
            self.gait_manager.setAAmplitude(0.0)  # Angular step
            # self.gait_manager.start()
        elif action == 2: # Start turning left
            self.gait_manager.setXAmplitude(0.3)
            self.gait_manager.setYAmplitude(0.0)
            self.gait_manager.setAAmplitude(0.5) # Left turn amplitude (positive)
            # self.gait_manager.start()
        elif action == 3: # Start turning right
            self.gait_manager.setXAmplitude(0.3)
            self.gait_manager.setYAmplitude(0.0)
            self.gait_manager.setAAmplitude(-0.5) # Right turn amplitude (negative)
            # self.gait_manager.start()
        # TODO: Add more actions for other gaits or motions if defined in action space.
    
    def step(self, action):
        self.apply_action(action)
        super(Supervisor, self).step(self.timestep)
        self.gait_manager.step(self.timestep)
        next_state = self.get_observations()
        done = self.is_done()
        reward = self.get_reward(action)
        info = self.get_info()
        return next_state, reward, done, info
        
    def reset(self):
        print("Resetting environment")
        self.gait_manager.stop()
        
        self.initial_robot_trans_field.setSFVec3f(self.initial_robot_pos_value)
        self.initial_robot_rot_field.setSFRotation(self.initial_robot_rot_value)
        self.simulationResetPhysics()
        
        self.motion_manager.playPage(9)
        while self.motion_manager.isMotionPlaying(): self.step(self.timestep)
        try:
            self.motion_manager.stopMotion()
        except AttributeError:
            print("AttributeError: MotionManager has no attribute stopMotion")
        
        self.gait_manager.start()
        self.step(self.timestep)
        self.previous_distance_to_target = float('inf')
        initial_state = self.get_observations()
        self.previous_distance_to_target = self.current_distance_to_target
        
        print(f"Environment reset complete. Robot position resetted")

        return initial_state
    
    def get_info(self):
        return self.info
    