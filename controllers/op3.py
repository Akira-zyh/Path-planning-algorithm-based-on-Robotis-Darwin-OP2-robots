from controller import Supervisor, Motor, PositionSensor, Camera, Accelerometer, Gyro, GPS, TouchSensor
import numpy as np
import math
import sys
import os

import cv2
from PIL import Image
import stable_baselines3 as sb3
from stable_baselines3 import PPO
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
import gym
from gym import spaces
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

TIME_STEP = 32
MAX_SIMULATION_TIME = 60000
MAX_STEP_PER_EPISODE = int(MAX_SIMULATION_TIME / TIME_STEP)

MOTOR_NAMES = ['ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
                     'ArmLowerR', 'ArmLowerL', 'PelvYR', 'PelvYL', 'PelvR',
                     'PelvL', 'LegUpperR', 'LegUpperL', 'LegLowerR', 'LegLowerL',
                     'AnkleR', 'AnkleL', 'FootR', 'FootL', 'Neck', 'Head', 'GraspL', 'GraspR']

GPS_NAMES  = ['left_gps1', 'right_gps1', 'left_gps2', 'right_gps2']

TOUCH_SENSOR_NAMES = ['touch_grasp_L1', 'touch_grasp_L1_1', 'touch_grasp_L1_2',
                      'touch_grasp_L2', 'touch_grasp_L2_1',
                      'touch_grasp_R1', 'touch_grasp_R1_1', 'touch_grasp_R1_2',
                      'touch_grasp_R2', 'touch_grasp_R2_1',
                      'touch_foot_L1', 'touch_foot_L2', 'touch_foot_R1', 'touch_foot_R2',
                      'touch_arm_L1', 'touch_arm_R1', 'touch_leg_L1', 'touch_leg_L2',
                      'touch_leg_R1', 'touch_leg_R2']

# TOUCH_SENSOR_NAMES = ['touch_grasp_L1', 'touch_grasp_L1_1', 'touch_grasp_L1_2',
#                       'touch_grasp_L2', 'touch_grasp_L2_1',
#                       'touch_grasp_R1', 'touch_grasp_R1_1', 'touch_grasp_R1_2',
#                       'touch_grasp_R2', 'touch_grasp_R2_1'
#                       ]

NUM_JOINTS = len(MOTOR_NAMES)
NUM_GPS = len(GPS_NAMES)
NUM_TOUCH_SENSORS = len(TOUCH_SENSOR_NAMES)
GPS_DATA_SIZE = NUM_GPS * 3
TOUCH_DATA_SIZE = NUM_TOUCH_SENSORS
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_CHANNELS = 3

VECTOR_OBS_SIZE = NUM_JOINTS + NUM_GPS + NUM_TOUCH_SENSORS

ACTION_SPACE_SIZE = 2
LOG_DIR = './training_logs/'
SAVE_FREQ = 1000
TOTAL_TIMESTEPS = 1000000
MODEL_SAVE_PATH = './trained_models/robotis_op3_grasping'

DISTANCE_REWARD_WEIGHT = -50.0
GRASP_SUCCESS_REWARD = 100.0
COLLISION_REWARD = -10.0
FALL_PENALTY = -50.0
JOINT_LIMIT_PENALTY = -10.0
MAINTAIN_STABILITY_REWARD = 1.0

LADDER_RUNG_NAMES = ['stick5', 'stick6', 'stick7', 'stick8', 'stick9', 'stick10']
TARGET_RUNG_INDEX = 0


class RobotisOP3GraspingEnv(gym.Env):
    def __init__(self):
        super(RobotisOP3GraspingEnv, self).__init__()
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.motors = [self.robot.getDevice(name) for name in MOTOR_NAMES]
        self.position_sensors = [self.robot.getDevice(name + 'S') for name in MOTOR_NAMES]
        self.acclerometer = self.robot.getDevice('Accelerometer')
        self.gyro = self.robot.getDevice('Gyro')
        self.camera = self.robot.getDevice('Camera')
        self.gps_sensors = [self.robot.getDevice(name) for name in GPS_NAMES]
        self.touch_sensors = [self.robot.getDevice(name) for name in TOUCH_SENSOR_NAMES]
        self.touch = [
            self.touch_sensors[TOUCH_SENSOR_NAMES.index('touch_grasp_L1')], 
            self.touch_sensors[TOUCH_SENSOR_NAMES.index('touch_grasp_R1')]
            ]
        self.touch_collision = [
            self.touch_sensors[TOUCH_SENSOR_NAMES.index('touch_arm_L1')],
            self.touch_sensors[TOUCH_SENSOR_NAMES.index('touch_arm_R1')],
            self.touch_sensors[TOUCH_SENSOR_NAMES.index('touch_leg_L1')],
            self.touch_sensors[TOUCH_SENSOR_NAMES.index('touch_leg_L2')],
            self.touch_sensors[TOUCH_SENSOR_NAMES.index('touch_leg_R1')],
            self.touch_sensors[TOUCH_SENSOR_NAMES.index('touch_leg_R2')]
            ]

        
        
        for sensor in self.position_sensors:
            sensor.enable(self.timestep)
        if self.acclerometer:
            self.acclerometer.enable(self.timestep)
        if self.gyro:
            self.gyro.enable(self.timestep)
        if self.camera:
            self.camera.enable(self.timestep)
        for sensor in self.gps_sensors:
            if sensor:
                sensor.enable(self.timestep)
            else:
                print(f"[ERROR] GPS sensor not found: {sensor}")
        for sensor in self.touch_sensors:
            if sensor:
                sensor.enable(self.timestep)
            else:
                print(f"[ERROR] Touch sensor not found: {sensor}")
            
        self.ladder_rungs = [self.robot.getFromDef(name) for name in LADDER_RUNG_NAMES]
        self.rungs_translation = [self.robot.getFromDef(name).getField('translation').getSFVec3f() for name in LADDER_RUNG_NAMES]
        if any(rung is None for rung in self.ladder_rungs):
            print("[ERROR]: One or more ladder rungs not found!")
        
        self.robot_base_node = self.robot.getFromDef('Base_Link')
        if self.robot_base_node is None:
            print("[ERROR]: Robot base node not found! [Base_Link] is used to calculate the ralative position of the ladder rungs.")

        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(CAMERA_HEIGHT, CAMERA_WIDTH, CAMERA_CHANNELS), dtype=np.uint8),
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(VECTOR_OBS_SIZE,), dtype=np.float32)
        })

        self.current_step = 0
        self.last_joint_positions = None
        self.target_rung_index = TARGET_RUNG_INDEX
        
        self.joint_limits = [
            [-3.14, 3.14], [-3.14, 2.85], [-0.68, 2.3], [-2.25, 0.77], [-1.65, 1.16], [-1.18, 1.63],
            [-2.42, 0.66], [-0.69, 2.5], [-1.01, 1.01], [-1, 0.93], [-1.77, 0.45], [-0.5, 1.68],
            [-0.02, 2.25], [-2.25, 0.03], [-1.24, 1.38], [-1.39, 1.22], [-0.68, 1.04], [-1.02, 0.6],
            [-1.81, 1.81], [-0.36, 0.94], [-1.0, 1.0], [-1.0, 1.0]
        ]
        
        self.initial_robot_position = [0.0, -0.96, 0.27]
        self.initial_robot_rotation = [-0.43007, 0.407788, 0.80545, 1.74776]

    def apply_action(self, action):
        if action == 0:
            # [TODO]
            self.motors[MOTOR_NAMES.index('GraspL')].setPosition(-1.0)
            self.motors[MOTOR_NAMES.index('GraspR')].setPosition(-1.0)
        elif action == 1:
            # [TODO]
            self.motors[MOTOR_NAMES.index('GraspL')].setPosition(1.0)
            self.motors[MOTOR_NAMES.index('GraspR')].setPosition(1.0)
        # [TODO]
        elif action == 2:
            pass
    def step(self, action):
        self.apply_action(action)
        self.robot.step(self.timestep)
        self.current_step += 1
        observation = self._get_observation()
        
        reward, done, grasp_achieved = self._compute_reward()
        
        if self.current_step >= MAX_STEP_PER_EPISODE:
            done = True
        
        info = {
            'grasp_achieved': grasp_achieved
        }

        return observation, reward, done, False, info
    
    def _initialize_pose(self):
        a = 0.5
        b = 0.8
        self.motors[MOTOR_NAMES.index('GraspL')].setPosition(1.0)
        self.motors[MOTOR_NAMES.index('GraspR')].setPosition(1.0)
        self.motors[MOTOR_NAMES.index('Neck')].setPosition(-0.2) 
        self.motors[MOTOR_NAMES.index('Head')].setPosition(0.0)
        self.motors[MOTOR_NAMES.index('PelvYL')].setPosition(0.0)
        self.motors[MOTOR_NAMES.index('PelvYR')].setPosition(0.0)
        self.motors[MOTOR_NAMES.index('LegUpperL')].setPosition(0.0)
        self.motors[MOTOR_NAMES.index('LegUpperR')].setPosition(0.0)
        self.motors[MOTOR_NAMES.index('LegLowerL')].setPosition(0.0)
        self.motors[MOTOR_NAMES.index('LegLowerR')].setPosition(0.0)
        self.motors[MOTOR_NAMES.index('AnkleL')].setPosition(0.0)
        self.motors[MOTOR_NAMES.index('AnkleR')].setPosition(0.0)
        self.motors[MOTOR_NAMES.index('FootL')].setPosition(0.0)
        self.motors[MOTOR_NAMES.index('FootR')].setPosition(0.0)
        self.motors[MOTOR_NAMES.index('PelvR')].setPosition(0.0)
        self.motors[MOTOR_NAMES.index('PelvL')].setPosition(0.0)
        self.motors[MOTOR_NAMES.index('ShoulderL')].setPosition(a)
        self.motors[MOTOR_NAMES.index('ShoulderR')].setPosition(-a)
        self.motors[MOTOR_NAMES.index('ArmUpperL')].setPosition(0.0)
        self.motors[MOTOR_NAMES.index('ArmUpperR')].setPosition(0.0)
        self.motors[MOTOR_NAMES.index('ArmLowerL')].setPosition(-b)
        self.motors[MOTOR_NAMES.index('ArmLowerR')].setPosition(b)
        for motor in self.motors:
            motor.setVelocity(1.0)
            
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.robot_base_node:
            self.robot_base_node.getField("translation").setSFVec3f(self.initial_robot_position)
            self.robot_base_node.getField("rotation").setSFVec3f(self.initial_robot_rotation)
            self.robot.step(self.timestep)

        self._initialize_pose()
        
        self.target_rung_index = TARGET_RUNG_INDEX
        
        self.robot.step(self.timestep * 20)
        observation = self._get_observation()
        
        self.current_step = 0
        self.last_joint_positions = [sensor.getValue() for sensor in self.position_sensors]

        info = {}
        return observation, info
    
    def _get_observation(self):
        joint_positions = [sensor.getValue() for sensor in self.position_sensors]

        accel_data = self.acclerometer.getValues() if self.acclerometer else [0.0, 0.0, 0.0]
        gyro_data = self.gyro.getValues() if self.gyro else [0.0, 0.0, 0.0]
        
        # robot_gps_pos = self.gps_sensors[0].getValues() if self.gps_sensors else [0.0, 0.0, 0.0]
        robot_gps_pos = [0.0, 0.0, 0.0]
        if self.gps_sensors:
            valid_gps_readings = []
            for sensor in self.gps_sensors:
                valid_gps_readings.append(sensor.getValues())
            
            if valid_gps_readings:
                avg_pos = np.mean(valid_gps_readings, axis=0)
                robot_gps_pos = avg_pos.tolist()
            else:
                print("[WARN]: No valid GPS readings available.")
        
        touch_data = [sensor.getValue() if sensor else 0.0 for sensor in self.touch_sensors]

        img_name = f'img{self.current_step}.png'
        self.camera.saveImage(img_name, 100)
        image = np.array(Image.open(f"./{img_name}"))
        grayscale_image = image[:, :, 0:1]
        target_rung_relative_pos = [0.0, 0.0, 0.0]

        if self.ladder_rungs and self.target_rung_index < len(self.ladder_rungs) and self.ladder_rungs[self.target_rung_index]:
            rung_pos = self.ladder_rungs[self.target_rung_index].getPosition()
            target_rung_relative_pos = [rung_pos[i] - robot_gps_pos[i] for i in range(3)]
            
        vector_obs = np.concatenate([
            joint_positions,
            accel_data,
            gyro_data,
            robot_gps_pos,
            touch_data,
            target_rung_relative_pos,
        ]).astype(np.float32)

        observation = {
            "image": grayscale_image,
            "vector": vector_obs
        }

        return observation

    def _calculate_reward(self):
        reward = 0.0
        done = False
        grasp_achieved = False
        
        robot_gps_pos = [0.0, 0.0, 0.0]
        if self.gps_sensors:
            valid_gps_readings = []
            for sensor in self.gps_sensors:
                if sensor and sensor.getValues():
                    valid_gps_readings.append(sensor.getValues())

            if valid_gps_readings:
                avg_pos = np.mean(valid_gps_readings, axis=0)
                robot_gps_pos = avg_pos.tolist()
            else:
                print("[WARN]: No valid GPS readings available for reward calculation.")
        
        target_rung_pos = [0.0, 0.0, 0.0]
        if self.ladder_rungs and self.target_rung_index < len(self.ladder_rungs) and self.ladder_rungs[self.target_rung_index]:
            target_rung_pos = self.ladder_rungs[self.target_rung_index].getPosition()
        distance_to_rung = np.linalg.norm(np.array(robot_gps_pos) - np.array(target_rung_pos))
        reward += distance_to_rung * DISTANCE_REWARD_WEIGHT
        
        left_grasp_touch = \
            self.touch_sensors[TOUCH_SENSOR_NAMES.index("touch_grasp_L1")].getValue() if self.touch_sensors[TOUCH_SENSOR_NAMES.index('touch_grasp_L1')] is not None else 0.0 \
            and self.touch_sensors[TOUCH_SENSOR_NAMES.index("touch_grasp_L1_1")].getValue() if self.touch_sensors[TOUCH_SENSOR_NAMES.index('touch_grasp_L1_1')] is not None else 0.0 \
            and self.touch_sensors[TOUCH_SENSOR_NAMES.index("touch_grasp_L1_2")].getValue() if self.touch_sensors[TOUCH_SENSOR_NAMES.index('touch_grasp_L1_2')] is not None else 0.0 \
        
        right_grasp_touch = \
            self.touch_sensors[TOUCH_SENSOR_NAMES.index("touch_grasp_R1")].getValue() if self.touch_sensors[TOUCH_SENSOR_NAMES.index('touch_grasp_R1')] is not None else 0.0 \
            and self.touch_sensors[TOUCH_SENSOR_NAMES.index("touch_grasp_R1_1")].getValue() if self.touch_sensors[TOUCH_SENSOR_NAMES.index('touch_grasp_R1_1')] is not None else 0.0 \
            and self.touch_sensors[TOUCH_SENSOR_NAMES.index("touch_grasp_R1_2")].getValue() if self.touch_sensors[TOUCH_SENSOR_NAMES.index('touch_grasp_R1_2')] is not None else 0.0 \
        
        if left_grasp_touch > 0.5 and right_grasp_touch > 0.5:
            if distance_to_rung < 0.2:
                reward += GRASP_SUCCESS_REWARD
                grasp_achieved = True
                done = True
            else:
                reward += -0.01
        
        if self.acclerometer:
            accel = self.acclerometer.getValues()
            if abs(accel[0]) > 8.0 or abs(accel[1]) > 8.0:
                reward += FALL_PENALTY
                done = True
        
        current_joint_positions = [sensor.getValue() for sensor in self.position_sensors]
        for i in range(NUM_JOINTS):
            if current_joint_positions[i] < self.joint_limits[i][0] or current_joint_positions[i] > self.joint_limits[i][1]:
                reward += JOINT_LIMIT_PENALTY
                done = True
        
        if self.acclerometer:
            accel = self.acclerometer.getValues()
            stability_reward = MAINTAIN_STABILITY_REWARD * max(0, 1.0 - abs(accel[2] + 9.8) / 9.8)
            reward += stability_reward

        if self.last_joint_positions is not None:
            joint_movement_diff = np.linalg.norm(np.array(current_joint_positions) - np.array(self.last_joint_positions))
            if joint_movement_diff > 0.5:
                reward += -1.0
                pass
        
        self.last_joint_positions = current_joint_positions
        
        return reward, done, grasp_achieved
    
    def render():
        pass
    
    def close():
        pass 
        
# -------------------------- Train and Test --------------------------

def run():
    robot = Supervisor()
    env = RobotisOP3GraspingEnv(robot)
    vec_env = DummyVecEnv([lambda: RobotisOP3GraspingEnv(robot)])

    model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR)

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=MODEL_SAVE_PATH,
        name_prefix="op3_ladder_grasping_ppo"
    )

    print(f"[INFO] Starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    print("[INFO] Training finished")

    model.save(f"{MODEL_SAVE_PATH}_final")
    print(f"[INFO] Final model saved to {MODEL_SAVE_PATH}_final.zip")

    print("[INFO] Starting testing...")
    loaded_model = PPO.load(f"{MODEL_SAVE_PATH}_final")

    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    
    while robot.step(TIME_STEP) != -1 and not done:
        action, _states = loaded_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        env.render()
        print(f"[INFO] Step: {env.current_step}, Reward: {reward}, Done: {done}, Grasp Achieved: {info.get('grasp_achieved', False)}")

    print(f"[INFO] Testing finished. Total reward: {total_reward}")

    env.close()

# -------------------------- Raise Leg Function --------------------------
# [TODO]
def _raise_leg_L1(motors):
    motor1 = motors[MOTOR_NAMES.index("LegLowerL")]
    motor1.setPosition(0.7)
    motor1.setVelocity(1.0)
    motor2 = motors[MOTOR_NAMES.index("AnkleL")]
    motor2.setPosition(-0.5)
    motor2.setVelocity(1.0)
    print("[INFO] Stage 1 finished.")
          
def _raise_leg_L2(motors):
    motor0 = motors[MOTOR_NAMES.index("LegUpperL")]
    motor0.setPosition(-1.65)
    motor1 = motors[MOTOR_NAMES.index("LegLowerL")]
    motor1.setPosition(2.2)
    motor1.setVelocity(2.0)
    motor2 = motors[MOTOR_NAMES.index("AnkleL")]
    motor2.setPosition(-0.85)
    motor2.setVelocity(2.0)
    print("[INFO] Stage 2 finished.")

def _raise_leg_L3(motors):
    motor1 = motors[MOTOR_NAMES.index("LegLowerL")]
    motor1.setPosition(1.8)
    motor1.setVelocity(1.0)
    motor2 = motors[MOTOR_NAMES.index("AnkleL")]
    motor2.setPosition(-0.45)
    motor2.setVelocity(1.0)
    print("[INFO] Stage 3 finished.")

def _raise_leg_L4(motors):
    motor0 = motors[MOTOR_NAMES.index("LegUpperL")]
    motor0.setPosition(-1.4)
    motor0.setVelocity(1.0)
    motor1 = motors[MOTOR_NAMES.index("LegLowerL")]
    motor1.setPosition(1.55)
    motor1.setVelocity(1.0)
    motor2 = motors[MOTOR_NAMES.index("AnkleL")]
    motor2.setPosition(-0.45)
    motor2.setVelocity(1.0)
    print("[INFO] Stage 4 finished.")

def _raise_leg_L5(motors):
    motor1 = motors[MOTOR_NAMES.index("LegLowerL")]
    motor1.setPosition(1.45)
    motor1.setVelocity(1.0)
    motor2 = motors[MOTOR_NAMES.index("AnkleL")]
    motor2.setPosition(-0.45)
    motor2.setVelocity(1.0)
    print("[INFO] Stage 5 finished.")

def _raise_leg_L6(motors):
    motor0 = motors[MOTOR_NAMES.index("LegUpperL")]
    motor0.setPosition(-1.65)
    motor0.setVelocity(1.0)
    motor1 = motors[MOTOR_NAMES.index("LegLowerL")]
    motor1.setPosition(2.2)
    motor1.setVelocity(2.0)
    motor2 = motors[MOTOR_NAMES.index("AnkleL")]
    motor2.setPosition(-0.85)
    motor2.setVelocity(2.0)
    print("[INFO] Stage 6 finished.")

def _raise_leg_L7(motors):
    motor0 = motors[MOTOR_NAMES.index("LegLowerL")]
    motor0.setPosition(1.8)
    motor0.setVelocity(1.0)
    motor1 = motors[MOTOR_NAMES.index("AnkleL")]
    motor1.setPosition(-0.8)
    motor1.setVelocity(1.0)
    motor2 = motors[MOTOR_NAMES.index("AnkleR")]
    motor2.setPosition(-0.8)
    motor2.setVelocity(1.0)
    motor3 = motors[MOTOR_NAMES.index("ArmLowerL")]
    motor3.setPosition(0.2)
    motor3.setVelocity(1.0)
    motor3 = motors[MOTOR_NAMES.index("ShoulderL")]
    motor3.setPosition(-0.4)
    motor3.setVelocity(1.0)
    motor4 = motors[MOTOR_NAMES.index("ArmLowerR")]
    motor4.setPosition(-0.2)
    motor4.setVelocity(1.0)
    motor3 = motors[MOTOR_NAMES.index("ShoulderR")]
    motor3.setPosition(0.4)
    motor3.setVelocity(1.0)
    print("[INFO] Stage 7 finished.")

def _raise_leg_L8(motors):
    motor1 = motors[MOTOR_NAMES.index("LegLowerL")]
    motor1.setPosition(1.6)
    motor1.setVelocity(1.0)
    motor2 = motors[MOTOR_NAMES.index("AnkleL")]
    motor2.setPosition(-0.45)
    motor2.setVelocity(1.0)
    print("[INFO] Stage 8 finished.")

# -------------

def _raise_leg_R1(motors):
    motor1 = motors[MOTOR_NAMES.index("LegLowerR")]
    motor1.setPosition(-0.7)
    motor1.setVelocity(1.0)
    motor2 = motors[MOTOR_NAMES.index("AnkleR")]
    motor2.setPosition(-0.35)
    motor2.setVelocity(1.0)
    print("[INFO] Stage 1 finished.")

def _raise_leg_R2(motors):
    motor0 = motors[MOTOR_NAMES.index("LegUpperR")]
    motor0.setPosition(1.65)
    motor0.setVelocity(1.0)
    motor1 = motors[MOTOR_NAMES.index("LegLowerR")]
    motor1.setPosition(-2.2)
    motor1.setVelocity(2.0)
    motor2 = motors[MOTOR_NAMES.index("AnkleR")]
    motor2.setPosition(-0.85)
    motor2.setVelocity(2.0)
    print("[INFO] Stage 2 finished.")

def _raise_leg_R3(motors):
    motor1 = motors[MOTOR_NAMES.index("LegLowerR")]
    motor1.setPosition(-1.8)
    motor1.setVelocity(1.0)
    motor2 = motors[MOTOR_NAMES.index("AnkleR")]
    motor2.setPosition(-0.45)
    motor2.setVelocity(1.0)
    print("[INFO] Stage 3 finished.")

def _raise_leg_R4(motors):
    motor0 = motors[MOTOR_NAMES.index("LegUpperR")]
    motor0.setPosition(1.4)
    motor0.setVelocity(1.0)
    motor1 = motors[MOTOR_NAMES.index("LegLowerR")]
    motor1.setPosition(-1.55)
    motor1.setVelocity(1.0)
    motor2 = motors[MOTOR_NAMES.index("AnkleR")]
    motor2.setPosition(-0.45)
    motor2.setVelocity(1.0)
    print("[INFO] Stage 4 finished.")

def _raise_leg_R5(motors):
    motor1 = motors[MOTOR_NAMES.index("LegLowerR")]
    motor1.setPosition(-1.45)
    motor1.setVelocity(1.0)
    motor2 = motors[MOTOR_NAMES.index("AnkleR")]
    motor2.setPosition(-0.45)
    motor2.setVelocity(1.0)
    print("[INFO] Stage 5 finished.")

def _raise_leg_R6(motors):
    motor0 = motors[MOTOR_NAMES.index("LegUpperR")]
    motor0.setPosition(1.65)
    motor0.setVelocity(1.0)
    motor1 = motors[MOTOR_NAMES.index("LegLowerR")]
    motor1.setPosition(-2.2)
    motor1.setVelocity(2.0)
    motor2 = motors[MOTOR_NAMES.index("AnkleR")]
    motor2.setPosition(-0.85)
    motor2.setVelocity(2.0)
    print("[INFO] Stage 6 finished.")

def _raise_leg_R7(motors):
    motor0 = motors[MOTOR_NAMES.index("LegLowerR")]
    motor0.setPosition(-1.8)
    motor0.setVelocity(1.0)
    motor1 = motors[MOTOR_NAMES.index("AnkleR")]
    motor1.setPosition(-0.8)
    motor1.setVelocity(1.0)
    motor3 = motors[MOTOR_NAMES.index("ArmLowerL")]
    motor3.setPosition(-0.2)
    motor3.setVelocity(1.0)
    motor3 = motors[MOTOR_NAMES.index("ShoulderL")]
    motor3.setPosition(-0.4)
    motor3.setVelocity(1.0)
    motor4 = motors[MOTOR_NAMES.index("ArmLowerR")]
    motor4.setPosition(-0.2)
    motor4.setVelocity(1.0)
    motor3 = motors[MOTOR_NAMES.index("ShoulderR")]
    motor3.setPosition(0.4)
    motor3.setVelocity(1.0)
    print("[INFO] Stage 7 finished.")

def _raise_leg_R8(motors):
    motor1 = motors[MOTOR_NAMES.index("LegLowerR")]
    motor1.setPosition(-1.6)
    motor1.setVelocity(1.0)
    motor2 = motors[MOTOR_NAMES.index("AnkleR")]
    motor2.setPosition(-0.45)
    motor2.setVelocity(1.0)
    print("[INFO] Stage 8 finished.")
# --------------

def _wait_steps(robot: Supervisor, step: int):
    timer = 0
    while robot.step(32) != -1:
        timer += 32
        if timer >= step:
            break

def raise_legL(robot:Supervisor, motors):
    _raise_leg_L1(motors=motors)
    _wait_steps(robot, 1000)
    _raise_leg_L2(motors=motors)
    _wait_steps(robot, 2000)
    _raise_leg_L3(motors=motors)
    _wait_steps(robot, 2000)
    _raise_leg_L4(motors=motors)
    _wait_steps(robot, 1000)
    _raise_leg_L5(motors=motors)
    _wait_steps(robot, 2000)
    _raise_leg_L6(motors=motors)
    _wait_steps(robot, 2000)
    _raise_leg_L7(motors=motors)
    _wait_steps(robot, 2000)
    _raise_leg_L8(motors=motors)
    _wait_steps(robot, 2000)

def raise_legR(robot:Supervisor, motors):
    _raise_leg_R1(motors=motors)
    _wait_steps(robot, 1000)
    _raise_leg_R2(motors=motors)
    _wait_steps(robot, 2000)
    _raise_leg_R3(motors=motors)
    _wait_steps(robot, 2000)
    _raise_leg_R4(motors=motors)
    _wait_steps(robot, 1000)
    _raise_leg_R5(motors=motors)
    _wait_steps(robot, 2000)
    _raise_leg_R6(motors=motors)
    _wait_steps(robot, 2000)
    _raise_leg_R7(motors=motors)
    _wait_steps(robot, 2000)
    _raise_leg_R8(motors=motors)
    _wait_steps(robot, 2000)

# -------------------------- Stand up  Function --------------------------

def stand_up(robot:Supervisor, motors):
    motor0 = motors[MOTOR_NAMES.index("LegLowerL")]
    motor0.setPosition(1.1) 
    motor0.setVelocity(1.0)
    motor1 = motors[MOTOR_NAMES.index("LegUpperL")]
    motor1.setPosition(-1.45) 
    motor1.setVelocity(1.0)
    motor12 = motors[MOTOR_NAMES.index("LegLowerR")]
    motor12.setPosition(-1.1) 
    motor12.setVelocity(1.0)
    motor2 = motors[MOTOR_NAMES.index("LegUpperR")]
    motor2.setPosition(1.45) 
    motor2.setVelocity(1.0)
    motor3 = motors[MOTOR_NAMES.index("ArmLowerL")]
    motor3.setPosition(-0.1) 
    motor3.setVelocity(1.0)
    motor4 = motors[MOTOR_NAMES.index("ArmLowerR")]
    motor4.setPosition(0.1)
    motor4.setVelocity(1.0)
    motor5 = motors[MOTOR_NAMES.index("AnkleL")]
    motor5.setPosition(0.25) 
    motor5.setVelocity(1.0)
    motor6 = motors[MOTOR_NAMES.index("AnkleR")]
    motor6.setPosition(-0.25) 
    motor6.setVelocity(1.0)
    motor7 = motors[MOTOR_NAMES.index("ShoulderL")]
    motor7.setPosition(-0.05) 
    motor7.setVelocity(1.0)
    motor8 = motors[MOTOR_NAMES.index("ShoulderR")]
    motor8.setPosition(0.05) 
    motor8.setVelocity(1.0)
    motor9 = motors[MOTOR_NAMES.index("GraspL")]
    motor9.setPosition(0) 
    motor9.setVelocity(1.0)
    motor10 = motors[MOTOR_NAMES.index("GraspR")]
    motor10.setPosition(0) 
    motor10.setVelocity(1.0)


def adjust(robot:Supervisor, motors):
    motor3 = motors[MOTOR_NAMES.index("ArmLowerL")]
    motor3.setPosition(-0.7) 
    motor3.setVelocity(1.0)
    motor4 = motors[MOTOR_NAMES.index("ArmLowerR")]
    motor4.setPosition(0.7)
    motor4.setVelocity(1.0)
    motor5 = motors[MOTOR_NAMES.index("AnkleL")]
    motor5.setPosition(0.55) 
    motor5.setVelocity(1.0)
    motor6 = motors[MOTOR_NAMES.index("AnkleR")]
    motor6.setPosition(0.55) 
    motor6.setVelocity(1.0)
    motor7 = motors[MOTOR_NAMES.index("ShoulderL")]
    motor7.setPosition(0.0) 
    motor7.setVelocity(1.0)
    motor8 = motors[MOTOR_NAMES.index("ShoulderR")]
    motor8.setPosition(0.0) 
    motor8.setVelocity(1.0) 

# -------------------------- Grasp Release Function --------------------------

def realse_graspL(robot:Supervisor, motors):
    motor1 = motors[MOTOR_NAMES.index("GraspL")]
    motor1.setPosition(2.0)
    motor1.setVelocity(2.0)
    _wait_steps(robot, 1000)
    
def realse_graspR(robot:Supervisor, motors):
    motor1 = motors[MOTOR_NAMES.index("GraspR")]
    motor1.setPosition(2.0)
    motor1.setVelocity(2.0)
    _wait_steps(robot, 1000)

# -------------------------- Raise Arm Function --------------------------

def _raise_armR1(robot:Supervisor, motors):
    motor4 = motors[MOTOR_NAMES.index("ArmLowerR")]
    motor4.setPosition(0.8)
    motor4.setVelocity(1.0)
    # motor2 = motors[MOTOR_NAMES.index("ArmUpperR")]
    # motor2.setPosition(-0.65)
    # motor2.setVelocity(1.0)
    motor3 = motors[MOTOR_NAMES.index("ShoulderR")]
    motor3.setPosition(-0.5)
    motor3.setVelocity(1.0)

def _raise_armR2(robot:Supervisor, motors):
    motor4 = motors[MOTOR_NAMES.index("ArmLowerR")]
    motor4.setPosition(1.0)
    motor4.setVelocity(1.0)
    motor2 = motors[MOTOR_NAMES.index("Head")]
    motor2.setPosition(0.4)
    motor2.setVelocity(1.0)
    motor3 = motors[MOTOR_NAMES.index("ShoulderR")]
    motor3.setPosition(-0.2)
    motor3.setVelocity(1.0)

def raise_armR(robot:Supervisor, motors):
    _raise_armR1(robot, motors)
    _wait_steps(robot, 1000)
    _raise_armR2(robot, motors)
    _wait_steps(robot, 1000)

# -------------------------- Test Function --------------------------  
# [TODO]
def test():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    motors = [robot.getDevice(name) for name in MOTOR_NAMES]
    for motor in motors:
        motor.setVelocity(0.5)
    position_sensors = [robot.getDevice(name + 'S') for name in MOTOR_NAMES]
    camera = robot.getDevice('Camera')
    camera.enable(timestep)
    
    for sensor in position_sensors:
        sensor.enable(timestep)

    robot.step(timestep)
    # camera_data = np.array(camera.getImage(), dtype=np.uint8).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))
    # camera_data = np.array(camera.getImageArray(), dtype=np.uint8).reshape((CAMERA_WIDTH, CAMERA_HEIGHT, 3))
    # if camera_data is not None:
        # cv2.imwrite('camera_data.png', camera_data)
    # camera.saveImage('camera_data2.png', 100)
    robot.step(timestep)
    # img_name = "./camera_data2.png"
    # image = np.array(Image.open(f"./{img_name}").resize((128, 128)).convert('L')) / 255.0
    # print(torch.tensor(image))
    motors[MOTOR_NAMES.index('GraspR')].setPosition(-0.5)
    motors[MOTOR_NAMES.index('GraspL')].setPosition(-0.5)
    robot.step(timestep)
    realse_graspR(robot=robot, motors=motors)
    raise_armR(robot=robot, motors=motors)
    raise_legL(robot=robot, motors=motors)
    raise_legR(robot=robot, motors=motors)
    realse_graspL(robot=robot, motors=motors)
    stand_up(robot=robot, motors=motors)
    raise_armR(robot=robot, motors=motors)
    adjust(robot=robot, motors=motors)
    _wait_steps(robot, 1000)

    while robot.step(timestep) != -1:
        # print("[TEST] ", position_sensors[MOTOR_NAMES.index('ShoulderL')].getValue()
        pass



# -------------------------------- Test code --------------------------------
if __name__ == "__main__":
    # env = RobotisOP3GraspingEnv()
    # env._initialize_pose()
    # env._get_observation()
    # ---------------------------- Functional test ----------------------------
    test()
    
    
