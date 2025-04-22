import random
from warnings import warn
import numpy as np
from gym.spaces import Box, Discrete
from robot_supervisor_env import RobotSupervisorEnv
from utilities import normalize_to_range, get_distance_from_target, get_angle_from_target
from controller import Supervisor, Keyboard
from managers import RobotisOp2MotionManager, RobotisOp2GaitManager

NMOTORS = 20
motorNames = [
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR",
    "ArmLowerL", "PelvYR", "PelvYL", "PelvR", "PelvL", "LegUpperR",
    "LegUpperL", "LegLowerR", "LegLowerL", "AnkleR", "AnkleL", "FootR",
    "FootL", "Neck", "Head"
]

class FindAndAvoidV2RobotSupervisor(RobotSupervisorEnv):
    def __init__(self, description, maximum_episode_steps, step_window=1, seconds_window=0, add_action_to_obs=True,
                 reset_on_collisions=0, manual_control=False, on_target_threshold=0.1,
                 max_ds_range=100.0, ds_type="generic", ds_n_rays=1, ds_aperture=0.1,
                 ds_resolution=-1, ds_noise=0.0, ds_denial_list=None,
                 target_distance_weight=1.0, target_angle_weight=1.0, dist_sensors_weight=1.0,
                 target_reach_weight=1.0, collision_weight=1.0, smoothness_weight=1.0, speed_weight=1.0,
                 map_width=7, map_height=7, cell_size=None, seed=None):
        super().__init__()
        
        ################################################################################################################
        # General
        
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.experiment_desc = description
        self.manual_control = manual_control
        
        self.viewpoint = self.getFromDef("VIEWPOINT")
        self.viewpoint_position = self.viewpoint.getField("position").getSFVec3f()
        self.viewpoint_orientation = self.viewpoint.getField("orientation").getSFRotation()
        
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)
        
        if ds_denial_list is None:
            self.ds_denial_list = []
        else:
            self.ds_denial_list = ds_denial_list
            
        ################################################################################################################
        # Robot setup  
        
        self.robot = self.getSelf()
        
        self.number_of_distance_sensors = 13
        self.led_head = self.getDevice("HeadLed")
        self.led_eye = self.getDevice("EyeLed")
        self.led_head.set(0xFFFF00)
        self.led_eye.set(0xFF0400)      
        self.motors = []
        self.position_sensors = []
        for name in motorNames:
            motor = self.getDevice(name)
            sensor = self.getDevice(name + "S")
            sensor.enable(self.timestep)
            self.motors.append(motor)
            self.position_sensors.append(sensor)
        self.accelerometer = self.getDevice("Accelerometer")
        self.accelerometer.enable(self.timestep)
        self.gyro = self.getDevice("Gyro")
        self.gyro.enable(self.timestep)
        self.camera = self.getDevice("Camera")
        self.camera.enable(2*self.timestep)
        # self.compass = self.getDevice("compass")
        # self.compass.enable(self.timestep)
        # self.gps = self.getDevice('gps')
        # self.gps.enable(self.timestep)
        # self.imu = self.getDevice('imu')
        # self.imu.enable(self.timestep)
        self.motion_manager = RobotisOp2MotionManager(self)
        self.gait_manager = RobotisOp2GaitManager(self, "config.ini")
        
        self.action_space = Discrete(5)

        self.add_action_to_obs = add_action_to_obs
        self.step_window = step_window
        self.seconds_window = seconds_window
        self.obs_list = []
        # Distance to target, angle to target, gait manager xamplitude, motor speed aamplitude, touch left, touch right
        single_obs_low = [0.0, -1.0, -1.0, -1.0, 0.0, 0.0]
        # Add action one-hot vector
        if self.add_action_to_obs:
            single_obs_low.extend([0.0 for _ in range(self.action_space.n)])
        # Append distance sensor values
        single_obs_low.extend([0.0 for _ in range(self.number_of_distance_sensors)])
        
        # Set up corresponding observation high values
        single_obs_high = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        if self.add_action_to_obs:
            single_obs_high.extend([1.0 for _ in range(self.action_space.n)])
        single_obs_high.extend([1.0 for _ in range(self.number_of_distance_sensors)])
        
        # Expand sizes depending on step window and seconds window
        self.single_obs_size = len(single_obs_low)
        obs_low = []
        obs_high = []
        for _ in range(self.step_window + self.seconds_window):
            obs_low.extend(single_obs_low)
            obs_high.extend(single_obs_high)
            self.obs_list.extend([0.0 for _ in range(self.single_obs_size)])
        # Memory is used for creating the windows in get_observation()
        self.obs_memory = [[0.0 for _ in range(self.single_obs_size)]
                           for _ in range((self.seconds_window * int(np.ceil(1000 / self.timestep))) +
                                          self.step_window)]
        self.observation_counter_limit = int(np.ceil(1000 / self.timestep))
        self.observation_counter = self.observation_counter_limit

        # Finally initialize space
        self.observation_space = Box(low=np.array(obs_low),
                                     high=np.array(obs_high),
                                     dtype=np.float64)
        
        # Set up sensors
        self.distance_sensors = []
        self.ds_max = []
        self.ds_type = ds_type
        self.ds_n_rays = ds_n_rays
        self.ds_aperture = ds_aperture
        self.ds_resolution = ds_resolution
        self.ds_noise = ds_noise
        # The minimum distance sensor thresholds, under which there is an obstacle obstructing forward movement
        # Note that these values are highly dependent on how the sensors are placed on the robot
        self.ds_thresholds = [8.0, 8.0, 8.0, 10.15, 14.7, 13.15,
                              12.7,
                              13.15, 14.7, 10.15, 8.0, 8.0, 8.0]
        # Loop through the ds_group node to set max sensor values, initialize the devices, set the type, etc.
        robot_children = self.robot.getField("bodySlot")
        for childNodeIndex in range(robot_children.getCount()):
            robot_child = robot_children.getMFNode(childNodeIndex)
            if robot_child.getTypeName() == "Group":
                ds_group = robot_child.getField("children")
                for i in range(self.number_of_distance_sensors):
                    self.distance_sensors.append(self.getDevice(f"distance sensor({str(i)})"))
                    self.distance_sensors[-1].enable(self.timestep)  # NOQA
                    ds_node = ds_group.getMFNode(i)
                    ds_node.getField("lookupTable").setMFVec3f(4, [max_ds_range / 100.0, max_ds_range])
                    ds_node.getField("lookupTable").setMFVec3f(3, [0.75 * max_ds_range / 100.0, 0.75 * max_ds_range,
                                                                   self.ds_noise])
                    ds_node.getField("lookupTable").setMFVec3f(2, [0.5 * max_ds_range / 100.0, 0.5 * max_ds_range,
                                                                   self.ds_noise])
                    ds_node.getField("lookupTable").setMFVec3f(1, [0.25 * max_ds_range / 100.0, 0.25 * max_ds_range,
                                                                   self.ds_noise])
                    ds_node.getField("type").setSFString(self.ds_type)
                    ds_node.getField("numberOfRays").setSFInt32(self.ds_n_rays)
                    ds_node.getField("aperture").setSFFloat(self.ds_aperture)
                    ds_node.getField("resolution").setSFFloat(self.ds_resolution)
                    self.ds_max.append(max_ds_range)  # NOQA

        # Touch sensors are used to determine when the robot collides with an obstacle
        self.touch_sensor_left = self.getDevice("touch sensor left")
        self.touch_sensor_left.enable(self.timestep)  # NOQA
        self.touch_sensor_right = self.getDevice("touch sensor right")
        self.touch_sensor_right.enable(self.timestep)  # NOQA

        # Set up motors
        # self.left_motor = self.getDevice("left_wheel")
        # self.right_motor = self.getDevice("right_wheel")
        # self.motor_speeds = [0.0, 0.0]
        # self.set_velocity(self.motor_speeds[0], self.motor_speeds[1])
        # self.motion_manager.playPage(1, False)
        # self.motion_manager.step(self.timestep)
        
        # self.gait_manager.start()
        # self.gait_manager.setBalanceEnable(True)
        # self.gait_manager.setXAmplitude(1.0)
        # self.gait_manager.setAAmplitude(0.5)
        # self.step(self.timestep)
        
        # Grab target node
        self.target = self.getFromDef("TARGET")
        self.target.getField("rotation").setSFRotation([0.0, 0.0, 1.0, 0.0])
        
        self.on_target_threshold = on_target_threshold  # Threshold that defines whether robot is considered "on target"
        self.initial_target_distance = 0.0
        self.initial_target_angle = 0.0
        self.current_tar_d = 0.0  # Distance to target
        self.previous_tar_d = 0.0
        self.current_tar_a = 0.0  # Angle to target in respect to the facing angle of the robot
        self.previous_tar_a = 0.0
        self.current_dist_sensors = [0.0 for _ in range(len(self.distance_sensors))]  # Latest distance sensor values
        self.previous_dist_sensors = [0.0 for _ in range(len(self.distance_sensors))]
        self.current_touch_sensors = [0.0, 0.0]
        self.current_position = [0, 0]  # World position
        self.previous_position = [0, 0]
        self.current_rotation = 0.0  # Facing angle
        self.previous_rotation = 0.0
        self.current_rotation_change = 0.0  # Latest facing angle change
        self.previous_rotation_change = 0.0
        
        # Various episode/training metrics, etc.
        self.current_timestep = 0
        self.collisions_counter = 0  # Counter of collisions during the episode
        self.reset_on_collisions = reset_on_collisions  # Upper limit of number of collisions before reset
        self.maximum_episode_steps = maximum_episode_steps  # Steps before timeout
        self.done_reason = ""  # Used to terminate episode and print the reason the episode is done
        self.reset_count = -1  # How many resets of the env overall, -1 to disregard the very first reset
        self.reach_target_count = 0  # How many times the target was reached
        self.collision_termination_count = 0  # How many times an episode was terminated due to collisions
        self.timeout_count = 0  # How many times an episode timed out
        self.min_distance_reached = float("inf")  # The current episode minimum distance to target reached
        self.min_dist_reached_list = []  # Used to store latest minimum distances reached, used as training metric
        self.smoothness_list = []  # Used to store the episode smoothness rewards, used as training metric
        self.episode_accumulated_reward = 0.0  # The reward accumulated in the current episode
        self.touched_obstacle_left = False
        self.touched_obstacle_right = False
        self.mask = [True for _ in range(self.action_space.n)]  # The action mask
        self.trigger_done = False  # Used to trigger the done condition
        self.just_reset = True  # Whether the episode was just reset
        
        self.reward_weight_dict = {"dist_tar": target_distance_weight, "ang_tar": target_angle_weight,
                                   "dist_sensors": dist_sensors_weight, "tar_reach": target_reach_weight,
                                   "collision": collision_weight, "smoothness_weight": smoothness_weight,
                                   "speed_weight": speed_weight}
        
        ################################################################################################################
        # Map stuff
        self.map_width, self.map_height = map_width, map_height
        if cell_size is None:
            self.cell_size = [0.5, 0.5]
        # Center map to (0, 0)
        origin = [-(self.map_width // 2) * self.cell_size[0], (self.map_height // 2) * self.cell_size[1]]
        self.map = Grid(self.map_width, self.map_height, origin, self.cell_size)

        # Obstacle references and starting positions used to reset them
        self.all_obstacles = []
        self.all_obstacles_starting_positions = []
        for childNodeIndex in range(self.getFromDef("OBSTACLES").getField("children").getCount()):
            child = self.getFromDef("OBSTACLES").getField("children").getMFNode(childNodeIndex)  # NOQA
            self.all_obstacles.append(child)
            self.all_obstacles_starting_positions.append(child.getField("translation").getSFVec3f())

        # Wall references
        self.walls = [self.getFromDef("WALL_1"), self.getFromDef("WALL_2")]
        self.walls_starting_positions = [self.getFromDef("WALL_1").getField("translation").getSFVec3f(),
                                         self.getFromDef("WALL_2").getField("translation").getSFVec3f()]

        # Path node references and starting positions used to reset them
        self.all_path_nodes = []
        self.all_path_nodes_starting_positions = []
        for childNodeIndex in range(self.getFromDef("PATH").getField("children").getCount()):
            child = self.getFromDef("PATH").getField("children").getMFNode(childNodeIndex)  # NOQA
            self.all_path_nodes.append(child)
            self.all_path_nodes_starting_positions.append(child.getField("translation").getSFVec3f())

        self.current_difficulty = {}
        self.number_of_obstacles = 0  # The number of obstacles to use, set from set_difficulty method
        if self.number_of_obstacles > len(self.all_obstacles):
            warn(f"\n \nNumber of obstacles set is greater than the number of obstacles that exist in the "
                 f"world ({self.number_of_obstacles} > {len(self.all_obstacles)}).\n"
                 f"Number of obstacles is set to {len(self.all_obstacles)}.\n ")
            self.number_of_obstacles = len(self.all_obstacles)

        self.path_to_target = []  # The map cells of the path
        # The min and max (manhattan) distances of the target length allowed, set from set_difficulty method
        self.min_target_dist = 1
        self.max_target_dist = 1
        
        
class Grid:
    """
    The grid map used to place all objects in the arena and find the paths.

    Partially coded by OpenAI's ChatGPT.
    """

    def __init__(self, width, height, origin, cell_size):
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.origin = origin
        self.cell_size = cell_size

    def size(self):
        return len(self.grid[0]), len(self.grid)

    def add_cell(self, x, y, node, z=None):
        if self.grid[y][x] is None and self.is_in_range(x, y):
            self.grid[y][x] = node
            if z is None:
                node.getField("translation").setSFVec3f(
                    [self.get_world_coordinates(x, y)[0], self.get_world_coordinates(x, y)[1], node.getPosition()[2]])
            else:
                node.getField("translation").setSFVec3f(
                    [self.get_world_coordinates(x, y)[0], self.get_world_coordinates(x, y)[1], z])
            return True
        return False

    def remove_cell(self, x, y):
        if self.is_in_range(x, y):
            self.grid[y][x] = None
        else:
            warn("Can't remove cell outside grid range.")

    def get_cell(self, x, y):
        if self.is_in_range(x, y):
            return self.grid[y][x]
        else:
            warn("Can't return cell outside grid range.")
            return None

    def get_neighbourhood(self, x, y):
        if self.is_in_range(x, y):
            neighbourhood_coords = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
                                    (x + 1, y + 1), (x - 1, y - 1),
                                    (x - 1, y + 1), (x + 1, y - 1)]
            neighbourhood_nodes = []
            for nc in neighbourhood_coords:
                if self.is_in_range(nc[0], nc[1]):
                    neighbourhood_nodes.append(self.get_cell(nc[0], nc[1]))
            return neighbourhood_nodes
        else:
            warn("Can't get neighbourhood of cell outside grid range.")
            return None

    def is_empty(self, x, y):
        if self.is_in_range(x, y):
            if self.grid[y][x]:
                return False
            else:
                return True
        else:
            warn("Coordinates provided are outside grid range.")
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
        # Make sure the randomly selected cell is not occupied
        for tries in range(self.size()[0] * self.size()[1]):
            coords = self.get_random_neighbor(x, y, min_distance, max_distance)
            if coords and self.add_cell(coords[0], coords[1], node):
                return True  # Return success, the node was added
        return False  # Failed to insert near

    def get_random_neighbor(self, x, y, d_min, d_max):
        neighbors = []
        rows = self.size()[0]
        cols = self.size()[1]
        for i in range(-d_max, d_max + 1):
            for j in range(-d_max, d_max + 1):
                if i == 0 and j == 0:
                    continue
                if 0 <= x + i < rows and 0 <= y + j < cols:
                    distance = abs(x + i - x) + abs(y + j - y)
                    if d_min <= distance <= d_max:
                        neighbors.append((x + i, y + j))
        if len(neighbors) == 0:
            return None
        return random.choice(neighbors)

    def get_world_coordinates(self, x, y):
        if self.is_in_range(x, y):
            world_x = self.origin[0] + x * self.cell_size[0]
            world_y = self.origin[1] - y * self.cell_size[1]
            return world_x, world_y
        else:
            return None, None

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
                if self.grid[y][x] and self.grid[y][x].getField("name").getSFString() == name:  # NOQA
                    return x, y
        return None

    def is_in_range(self, x, y):
        if (0 <= x < len(self.grid[0])) and (0 <= y < len(self.grid)):
            return True
        return False

    def bfs_path(self, start, goal):
        start = tuple(start)
        goal = tuple(goal)
        queue = [(start, [start])]  # (coordinates, path to coordinates)
        visited = set()
        visited.add(start)
        while queue:
            coords, path = queue.pop(0)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (-1, -1), (1, -1), (-1, 1)]:  # neighbors
                x, y = coords
                x2, y2 = x + dx, y + dy
                if self.is_in_range(x2, y2) and (x2, y2) not in visited:
                    if self.grid[y2][x2] is not None and (x2, y2) == goal:
                        return path + [(x2, y2)]
                    elif self.grid[y2][x2] is None:
                        visited.add((x2, y2))
                        queue.append(((x2, y2), path + [(x2, y2)]))
        return None


robot = FindAndAvoidV2RobotSupervisor(description="", maximum_episode_steps=500)
