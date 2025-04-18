import random
from warnings import warn
import numpy as np
from gymnasium.spaces import Box, Discrete
from deepbots.supervisor.controllers.robot_supervisor import RobotSupervisor
from utilities import normalize_to_range, get_angle_from_target, get_distance_from_target
from controller import Supervisor, Keyboard
class NavigationRobotSupervisor(RobotSupervisor):
    def __init__(self, description, maximum_episode_steps=500, step_window=1, seconds_window=0, add_action_to_obs=True,
                 reset_on_collisions=0, manual_control=False, on_target_threshold=0.1,
                 max_ds_range=100.0, ds_type="generic", ds_n_rays=1, ds_aperture=0.1,
                 ds_resolution=-1, ds_noise=0.0, ds_denial_list=None,
                 target_distance_weight=1.0, target_angle_weight=1.0, dist_sensors_weight=1.0,
                 target_reach_weight=1.0, collision_weight=1.0, smoothness_weight=1.0, speed_weight=1.0,
                 map_width=7, map_height=7, cell_size=None, seed=None):
        super().__init__()
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.environment_description = description
        self.manual_control = manual_control
        
        self.viewpoint = self.supervisor.getFromDef("VIEWPOINT")
        self.viewpoint_position = self.viewpoint.getField("position").getSFVec3f()
        self.viewpoint_orientation = self.viewpoint.getField("orientation").getSFRotation()
        # print(self.viewpoint_position)
        # print(self.viewpoint_orientation)
        
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)
        
        if ds_denial_list is None:
            ds_denial_list = []
        else:
            self.ds_denial_list = ds_denial_list
        
        self.robot = self.supervisor.getSelf()
        self.number_of_distance_sensors = 13
        
        self.action_space = Discrete(5)
        
        self.add_action_to_obs = add_action_to_obs
        self.step_window = step_window
        self.seconds_window = seconds_window
        self.obs_list = []
        # Distance to target, angle to target, touch left, touch right
        single_obs_low = [0.0, -1.0, 0.0, 0.0]

        if self.add_action_to_obs:
            single_obs_low.extend([0.0 for _ in range(self.action_space.n)])
        single_obs_low.extend([0.0 for _ in range(self.number_of_distance_sensors)])
        
        self.single_obs_size = len(single_obs_low)
        obs_low = []
        obs_high = []
        for _ in range(self.step_window + self.seconds_window):
            obs_low.extend(single_obs_low)
            obs_high.extend(single_obs_low)
            self.obs_list.extend((0.0 for _ in range(self.single_obs_size)))
        
        self.obs_memory = [[0.0 for _ in range(self.single_obs_size)]
                           for _ in range((self.step_window * int(np.ceil(1000 / self.timestep))) + self.seconds_window)]
        self.observation_counter_limit = int(np.ceil(1000 / self.timestep))
        self.observation_counter = self.observation_counter_limit
        
        self.observation_space = Box(low=np.array(obs_low), high=np.array(obs_high), dtype=np.float64)

        self.distance_sensors = []
        self.ds_max = []
        self.ds_type = ds_type
        self.ds_n_rays = ds_n_rays
        self.ds_aperture = ds_aperture
        self.ds_resolution = ds_resolution
        self.ds_noise = ds_noise
        robot_children = self.robot.getField('bodySlot')
        robot_child = robot_children.getMFNode(2)
        print(robot_child.getType())
            
        ds_group = robot_child.getField("children")
        print(ds_group.getCount()) 
        for i in range(self.number_of_distance_sensors):
            self.distance_sensors.append(self.supervisor.getDevice(f"ds{i}"))
            self.distance_sensors[-1].enable(self.timestep)
            # ds_node = ds_group.getMFNode(i)
            # lookup_table = ds_node.getField("lookupTable")

            # lookup_table.removeMF(0)
            # lookup_table.removeMF(lookup_table.getCount() - 1)
            # lookup_table.insertMFVec3f(0, [0.0, max_ds_range / 100.0, 0.0])
            # lookup_table.insertMFVec3f(1, [0.25 * max_ds_range / 100.0, 0.25 * max_ds_range, self.ds_noise])
            # lookup_table.insertMFVec3f(2, [0.5 * max_ds_range / 100.0, 0.5 * max_ds_range, self.ds_noise])
            # lookup_table.insertMFVec3f(3, [0.75 * max_ds_range / 100.0, 0.75 * max_ds_range, self.ds_noise])
            # lookup_table.insertMFVec3f(4, [max_ds_range / 100.0, max_ds_range, 0.0])
            
            # ds_node.getField("type").setSFString(self.ds_type)
            # ds_node.getField("numberOfRays").setSFInt32(self.ds_n_rays)
            # ds_node.getField("aperture").setSFFloat(self.ds_aperture)
            # ds_node.getField("resolution").setSFFloat(self.ds_resolution)
            self.ds_max.append(max_ds_range)
            
        self.touch_sensor_left = self.supervisor.getDevice('ts_left')
        self.touch_sensor_left.enable(self.timestep)
        self.touch_sensor_right = self.supervisor.getDevice('ts_right')
        self.touch_sensor_right.enable(self.timestep)
        self.NMOTORS = 20
        self.motorNames = [
            "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR",
            "ArmLowerL", "PelvYR", "PelvYL", "PelvR", "PelvL", "LegUpperR",
            "LegUpperL", "LegLowerR", "LegLowerL", "AnkleR", "AnkleL", "FootR",
            "FootL", "Neck", "Head"
        ]
        self.motors = []
        self.position_sensors = []
        for name in self.motorNames:
            motor = self.supervisor.getDevice(name)
            sensor = self.supervisor.getDevice(name + "S")
            sensor.enable(self.timeStep)
            self.motors.append(motor)
            self.position_sensors.append(sensor)
        # 传感器初始化（网页4示例）
        self.accelerometer = self.supervisor.getDevice("Accelerometer")
        self.accelerometer.enable(self.timeStep)
        self.gyro = self.supervisor.getDevice("Gyro")
        self.gyro.enable(self.timeStep)
        # self.camera = self.supervisor.getDevice("camera")
        # self.camera.enable(self.timeStep)
        # self.camera.recognitionEnable(self.timeStep)
        # self.camera.enableRecognitionSegmentation()
        # self.compass = self.supervisor.getDevice("compass")
        # self.compass.enable(self.timeStep)
        # self.gps = self.supervisor.getDevice('gps')
        # self.gps.enable(self.timeStep)
        # self.imu = self.supervisor.getDevice('imu')
        # self.imu.enable(self.timeStep)

        self.target = self.supervisor.getFromDef("TARGET")
        self.target.getField("rotation").setSFRotation([0.0, 0.0, 1.0, 0.0])

        self.on_target_threshold = on_target_threshold
        self.initial_target_distance = 0.0
        self.initial_target_angle = 0.0
        self.current_tar_d = 0.0
        self.previous_tar_d = 0.0
        self.currrent_tar_a = 0.0
        self.previous_tar_a = 0.0
        self.current_dist_sensors = [0.0 for _ in range(len(self.distance_sensors))]
        self.previous_dist_sensors = [0.0 for _ in range(len(self.distance_sensors))]
        self.current_touch_sensors = [0.0, 0.0]
        self.current_position = [0, 0]
        self.previous_position = [0, 0]
        self.current_rotation = 0.0
        self.previous_rotation = 0.0
        self.current_rotation_change = 0.0
        self.previous_rotation_change = 0.0
        
        self.current_timestep = 0
        self.collisions_counter = 0
        self.reset_on_collisions = reset_on_collisions
        self.maximum_episode_steps = maximum_episode_steps
        self.done_reason = ""
        self.reset_count = -1
        self.reach_target_count = 0
        self.colision_termination_count = 0
        self.timeout_count = 0
        

    def apply_action(self, action):
        return super().apply_action(action)

    def get_info(self):
        return super().get_info()
    
    def get_observations(self):
        return super().get_observations()
    
    def get_reward(self, action):
        return super().get_reward(action)
    
    def is_done(self):
        return super().is_done()
    
env = NavigationRobotSupervisor(description="")
