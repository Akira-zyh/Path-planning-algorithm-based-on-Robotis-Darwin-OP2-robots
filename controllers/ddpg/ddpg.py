import random
from warnings import warn
import numpy as np
from gymnasium.spaces import Box, Discrete
from deepbots.supervisor.controllers.robot_supervisor import RobotSupervisor
from utilities import normalize_to_range, get_angle_from_target, get_distance_from_target
from controller import Supervisor, Keyboard

class Grid:
    def __init__(self, width, height, origin, cell_size):
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.origin = origin
        self.cell_size = cell_size
        
    def size(self):
        return len(self.grid[0]), len(self.grid[1])
    
    def add_cell(self, x, y, node, z=None):
        if self.grid[y][x] is None and self.is_in_range(x, y):
            self.grid[y][x] = node
            if z is None:
                node.getField("translation").setSFVec3f(
                    [
                        self.get_world_coordinates(x, y)[0], 
                        self.get_world_coordinates(x, y)[1], 
                        node.getPosition()[2]
                    ]
                )
            else:
                node.getField("translation").setSFVec3f(
                    [
                        self.get_world_coordinates(x, y)[0], 
                        self.get_world_coordinates(x, y)[1],
                        z
                    ]
                )
            return True
        return False

    def remove_cell(self, x, y):
        if self.is_in_range(x, y):
            self.grid[y][x] = None
        else:
            warn("Can't remove cell outside grid range")
    
    def get_cell(self, x, y):
        if self.is_in_range(x, y):
            return self.grid[y][x]
        else:
            warn("Can't get cell outside grid range")
            
    def get_neighbourhood(self, x, y):
        if self.is_in_range(x, y):
            neighbourhood_coords = [
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
                (x + 1, y + 1),
                (x - 1, y - 1),
                (x - 1, y + 1),
                (x + 1, y - 1)
            ]
            neighbourhood_nodes = []
            for nc in neighbourhood_coords:
                if self.is_in_range(nc[0], nc[1]):
                    neighbourhood_nodes.append(self.get_cell(nc[0], nc[1]))
            return neighbourhood_nodes
        else:
            warn("Can't get neighbourhood of cell outside of gride range")
            return None
    
    def is_empty(self, x, y):
        if self.is_in_range(x, y):
            if self.grid[y][x]:
                return False
            else:
                return True
        else: 
            warn("Coordinates provided are outside of grid range")
            return None

    def empty(self):
        self.grid = [[None for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))]

    def add_random(self, node, z=None):
        x = random.randint(0, len(self.grid[0]) - 1)
        y = random.randint(0, len(self.grid) - 1)
        if self.grid[y][x] is None:
            return self.add_cell(x, y, node, z=z)
        else:
            self.add_random(node, z=z)
    
    def add_near(self, x, y, node, min_distance=1, max_distance=1):
        for tries in range(self.size()[0] * self.size()[1]):
            coords = self.get_random_neighbour(x, y, min_distance, max_distance)
            if coords and self.add_cell(coords[0], coords[1], node):
                return True
        return False

    def get_random_neighour(self, x, y, d_min, d_max):
        neighbours = []
        rows = self.size()[0]
        cols = self.size()[1]
        for i in range(-d_min, d_max + 1):
            for j in range(-d_min, d_max + 1):
                if i == 0 and j == 0:
                    continue
                if 0 <= x + i < rows and 0 <= y + j < cols:
                    distance = abs(x + i - x) + abs(y + j - y)
                    if d_min <= distance <= d_max:
                        neighbours.append((x + i, y + j))
        if len(neighbours) == 0:
            return None
        return random.choice(neighbours) 
    
    def get_world_coordinates(self, x, y):
        if self.is_in_range(x, y):
            world_x = self.origin[0] + x * self.cell_size[0]
            world_y = self.origin[1] - y * self.cell_size[1]
            return world_x, world_y
        else:
            return None
    
    def get_grid_coordinates(self, world_x, world_y):
        x = round((world_x - self.origin[0]) / self.cell_size[0])
        y = -round((world_y - self.origin[1]) / self.cell_size[1])
        if self.is_in_range(x, y):
            return x, y
        else:
            return None, None

    def find_by_name(self, name):
        for y in range(len(self.grid)):
            for x in range(len(self.grid[0])):
                if self.grid[y][x] and self.grid[y][x].getField("name").getSFString() == name:
                    return x, y
        return None, None

    def is_in_range(self, x, y):
        if (0 <= x < len(self.grid[0])) and (0 <= y < len(self.grid)):
            return True
        return False

    def bfs_path(self, start, goal):
        start = tuple(start)
        goal = tuple(goal)
        queue = [(start, [start])]
        visited = set()
        visited.add(start)
        while queue:
            coords, path = queue.pop(0)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                x, y = coords
                x2, y2 = x + dx, y + dy
                if self.is_in_range(x2, y2) and (x2, y2) not in visited:
                    if self.grid[x2][y2] is not None and (x2, y2) == goal:
                        return path + [(x2, y2)]
                    elif self.grid[y2][x2] is None:
                        visited.add((x2, y2))
                        queue.append(((x2, y2), path + [(x2, y2)]))
        return None
                        

class NavigationRobotSupervisor(RobotSupervisor):
    def __init__(self, description, maximum_episode_steps=500, step_window=1, seconds_window=0, add_action_to_obs=True,
                 reset_on_collisions=0, manual_control=False, on_target_threshold=0.1,
                 max_ds_range=100.0, ds_type="generic", ds_n_rays=1, ds_aperture=0.1,
                 ds_resolution=-1, ds_noise=0.0, ds_denial_list=None,
                 target_distance_weight=1.0, target_angle_weight=1.0, dist_sensors_weight=1.0,
                 target_reach_weight=1.0, collision_weight=1.0, smoothness_weight=1.0, speed_weight=1.0, fall_down_weight=1.0,
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
        self.de_thresholds = [8.0, 8.0, 8.0, 10.15, 14.7, 13.15, 12.7, 13.15, 14.7, 10.15, 8.0, 8.0, 8.0]
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
            
        self.touch_sensor_left = self.supervisor.getDevice('ts-left')
        self.touch_sensor_left.enable(self.timestep)
        self.touch_sensor_right = self.supervisor.getDevice('ts-right')
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
            sensor.enable(self.timestep)
            self.motors.append(motor)
            self.position_sensors.append(sensor)
        # 传感器初始化（网页4示例）
        self.accelerometer = self.supervisor.getDevice("Accelerometer")
        self.accelerometer.enable(self.timestep)
        self.gyro = self.supervisor.getDevice("Gyro")
        self.gyro.enable(self.timestep)
        # self.camera = self.supervisor.getDevice("camera")
        # self.camera.enable(self.timestep)
        # self.camera.recognitionEnable(self.timestep)
        # self.camera.enableRecognitionSegmentation()
        # self.compass = self.supervisor.getDevice("compass")
        # self.compass.enable(self.timestep)
        # self.gps = self.supervisor.getDevice('gps')
        # self.gps.enable(self.timestep)
        # self.imu = self.supervisor.getDevice('imu')
        # self.imu.enable(self.timestep)

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
        self.min_distance_reached = float("inf")
        self.min_dist_reached_list = []
        self.smoothnes_list = []
        self.episode_accumated_reward = 0.0
        self.touched_obstacle_left = False
        self.touched_obstacle_right = False
        self.mask = [True for _ in range(self.action_space.n)]
        self.trigger_done = False
        self.just_reset = True
        
        self.reward_weight_dict = {
            "dist_tar": target_distance_weight,
            "ang_tar": target_angle_weight,
            "dist_sensors": dist_sensors_weight,
            "tar_reach": target_reach_weight,
            "collision": collision_weight,
            "smoothless_weight": smoothness_weight,
            "collision": collision_weight,
            "speed_weight": speed_weight,
            "fall_down": fall_down_weight,
        }

        self.map_width, self.map_height = map_width, map_height
        if cell_size is None:
            self.cell_size = [0.5, 0.5]
        origin = [-(self.map_width // 2) * self.cell_size[0], (self.map_height // 2) * self.cell_size[1]]
        self.map = Grid(self.map_width, self.map_height, origin, self.cell_size)

        self.all_obstacles = []
        self.all_obstacles_starting_positions = []
        for childNodeIndex in range(self.supervisor.getFromDef("OBSTACLES").getField("children").getCount()):
            child = self.supervisor.getFromDef("OBSTACLES").getField("children").getMFNode(childNodeIndex)
            self.all_obstacles.append(child)
            self.all_obstacles_starting_positions.append(child.getField("translation").getSFVec3f())
        
        self.walls = []
        self.walls_starting_positions = []
        for childNodeIndex in range(self.supervisor.getFromDef("WALLS").getField("children").getCount()):
            child = self.supervisor.getFromDef("WALLS").getField("children").getMFNode(childNodeIndex)
            self.walls.append(child)
            self.walls_starting_positions.append(child.getField("translation").getSFVec3f())
        
        self.all_path_nodes = []
        self.all_path_nodes_starting_positions = []
        for childNodeIndex in range(self.supervisor.getFromDef("PATH").getField("children").getCount()):
            child = self.supervisor.getFromDef("PATH").getField("children").getMFNode(childNodeIndex)
            self.all_path_nodes.append(child)
            self.all_path_nodes_starting_positions.append(child.getField("translation").getSFVec3f())
            
        self.current_difficulty = {}
        self.number_of_obstacles = 0
        if self.number_of_obstacles > len(self.all_obstacles):
            warn(f"\n \nNumber of obstacles set is greater than the number of obstacles that exist in the "
                 f"world ({self.number_of_obstacles} > {len(self.all_obstacles)}).\n"
                 f"Number of obstacles is set to {len(self.all_obstacles)}.\n ")
            self.number_of_obstacles = len(self.all_obstacles)
        self.path_to_target = []
        self.min_target_dist = 1
        self.max_target_dist = 1

    def set_reward_weight_dict(self, target_distance_weight, target_angle_weight, dist_sensors_weight, 
                               target_reach_weight, collision_weight, smoothness_weight, fall_down_weight):
        self.reward_weight_dict = {'target_distance_weight': target_distance_weight,
                                    'target_angle_weight': target_angle_weight,
                                    'dist_sensors_weight': dist_sensors_weight,
                                    'target_reach_weight': target_reach_weight,
                                    'collision_weight': collision_weight,
                                    'smoothness_weight': smoothness_weight,
                                    "fall_down_weight": fall_down_weight
        }
        
    def set_maximum_episode_steps(self,new_value):
        self.maximum_episode_steps = new_value
    
    def set_difficulty(self, difficulty_dict, key=None):
        self.current_difficulty = difficulty_dict
        self.number_of_obstacles = difficulty_dict['number_of_obstacles']
        self.min_target_dist = difficulty_dict['min_target_dist']
        self.max_target_dist = difficulty_dict['max_target_dist']
        if key is not None:
            print(f"Changed difficulty to {key}, {difficulty_dict}")
        else:
            print("Changed difficulty", difficulty_dict)
    
    def get_action_mask(self):
        self.mask = [True for _ in range(self.action_space.n)]
        
        reading_under_threshold = [0.0 for _ in range(self.number_of_distance_sensors)]
        detecting_obstacle = [False for _ in range(self.number_of_distance_sensors)]
        front_under_half_threshold = False
        for i in range(len(self.current_dist_sensors)):
            if self.current_dist_sensors[i] <= self.ds_max[i] / 2:
                detecting_obstacle[i] = True
            if self.current_dist_sensors[i] < self.ds_thresholds[i]:
                reading_under_threshold[i] = self.ds_thresholds[i] - self.current_dist_sensors[i]
                if i in [4, 5, 6, 7, 8] and self.current_dist_sensors[i] < (self.ds_thresholds[i] / 2):
                    front_under_half_threshold = True

        reading_under_threshold_left = reading_under_threshold[0:5]
        reading_under_threshold_right = reading_under_threshold[8:13]

        if any(self.current_touch_sensors):
            self.mask[0] = False
            self.mask[1] = True
            if self.current_touch_sensors[0]:
                self.touched_obstacle_left = True
            if self.current_touch_sensors[1]:
                self.touched_obstacle_right = True
        elif not any(reading_under_threshold):
            self.touched_obstacle_left = False
            self.touched_obstacle_right = False

        if self.touched_obstacle_left or self.touched_obstacle_right:
            self.mask[0] = False
            self.mask[1] = True 

            if self.touched_obstacle_left and not self.touched_obstacle_right:
                self.mask[2] = False
                self.mask[3] = True
            if self.touched_obstacle_right and not self.touched_obstacle_left:
                self.mask[3] = False
                self.mask[2] = True
        else:
            if front_under_half_threshold:
                self.mask[0] = False
            
            if not any(detecting_obstacle) and abs(self.currrent_tar_a) < 0.1:
                self.mask[2] = self.mask[3] = False

            angle_threshold = 0.1

            if not any(reading_under_threshold_right):
                if self.currrent_tar_a <= - angle_threshold or any(reading_under_threshold_left):
                    self.mask[2] = False
                
            if not any(reading_under_threshold_left):
                if self.currrent_tar_a >= angle_threshold or any(reading_under_threshold_right):
                    self.mask[3] = False
            
            if any(reading_under_threshold_left) and any(reading_under_threshold_right):
                sum_left = sum(reading_under_threshold_left)
                sum_right = sum(reading_under_threshold_right)
                if sum_left - sum_right < -5.0:
                    self.mask[2] = True
                elif sum_left - sum_right > 5.0:
                    self.mask[3] = True
                else:
                    self.touched_obstacle_right = self.touched_obstacle_left = True
        return self.mask

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
