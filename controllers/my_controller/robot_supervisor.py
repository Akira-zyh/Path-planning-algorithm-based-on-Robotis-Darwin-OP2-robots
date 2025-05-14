import random
from time import sleep
import numpy as np
from controller import Supervisor, Keyboard
from gym.spaces import Discrete, Box
from robot_supervisor_env import RobotSupervisorEnv
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager
import math # 导入 math 模块用于计算姿态角


# 定义机器人电机名称

motorNames = [
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR",
    "ArmLowerL", "PelvYR", "PelvYL", "PelvR", "PelvL", "LegUpperR",
    "LegUpperL", "LegLowerR", "LegLowerL", "AnkleR", "AnkleL", "FootR",
    "FootL", "Neck", "Head"
]


# 传感器和环境参数

LIDAR_RESOLUTION = 64
LIDAR_RANGE = 1  # Lidar最大测量距离
MAX_ENVIRONMENT_DISTANCE = 10.0 # 环境最大距离，用于归一化目标距离
N_SIMULATION_STEPS_PER_ACTION = 500
OBSTACLE_NUM = 30

# 奖励函数参数
STEP_PENALTY = -0.1  # 每一步的惩罚，鼓励尽快到达目标
COLLISION_PENALTY = -50.0 # 碰撞惩罚
TARGET_REWARD = 10000.0 # 到达目标奖励
FALL_PENALTY = -200.0 # 跌倒惩罚
BALANCE_REWARD_WEIGHT = 0.1 # 平衡奖励权重
PROGRESS_REWARD_WEIGHT = 150.0 # 接近目标奖励权重 (增加权重)
ANGULAR_VELOCITY_PENALTY_WEIGHT = 0.001 # 角速度惩罚权重，鼓励直线行走 (增加权重)
STILL_PENALTY = -1.0 # 静止惩罚，鼓励移动 (增加惩罚)
SURVIVAL_REWARD = 0.1 # 每一步存活奖励，鼓励机器人保持活跃
LIDAR_PROXIMITY_PENALTY_WEIGHT = 15.0 # Lidar近距离惩罚权重 (新增)
LIDAR_PROXIMITY_THRESHOLD = 1 # Lidar近距离惩罚阈值：当Lidar距离小于此值时开始惩罚 (新增)



# 阈值

COLLISION_DISTANCE_THRESHOLD = 0.15 # 碰撞距离阈值
TARGET_DISTANCE_THRESHOLD = 0.3 # 到达目标距离阈值
FALL_PITCH_THRESHOLD = 0.8 # 俯仰角跌倒阈值 (弧度，约45.8度)
FALL_ROLL_THRESHOLD = 0.8 # 滚转角跌倒阈值 (弧度，约45.8度)
# OBSTACLE_AVOIDANCE_LIDAR_THRESHOLD = 0.7 # Lidar避障阈值 (用于规则避障，现已移除规则)


# 步态控制参数
TARGET_TURN_GAIN = 0.2 # 根据目标角度调整转向幅度的增益
FORWARD_AMPLITUDE = 0.7 # 前进动作的步长幅度
TURN_AMPLITUDE = 0.5 # 转向动作的角速度幅度
TURN_FORWARD_AMPLITUDE = 0.1 # 转向时前进的步长幅度

class NavigationRobotSupervisor(RobotSupervisorEnv):

    def __init__(self, description, seed=None):
        super().__init__()

        # ================ General Setup ==================

        random.seed(seed) if seed is not None else random.seed()
        self.experiment_decription = description
        self.viewpoint = self.getFromDef("VIEWPOINT")
        # 存储初始视点位置和方向，用于重置
        self.initial_viewpoint_position = self.viewpoint.getField("position").getSFVec3f()
        self.initial_viewpoint_orientation = self.viewpoint.getField("orientation").getSFRotation()
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)

        # ================ Robot Setup ===================

        self.robot = self.getSelf()
        self.motors = []
        self.position_sensors = []
        # 获取并启用所有电机和位置传感器
        for name in motorNames:
            motor = self.getDevice(name)
            sensor = self.getDevice(name + "S")
            sensor.enable(self.timestep)
            self.motors.append(motor)
            self.position_sensors.append(sensor)

        self.obstacle_nodes = []
        for obstacle_node in range(self.getFromDef("OBSTACLES").getField("children").getCount()):
            obstacle = self.getFromDef("OBSTACLES").getField("children").getMFNode(obstacle_node)
            self.obstacle_nodes.append(obstacle)

        # 跌倒计数器，用于判断是否需要执行恢复动作

        self.fall_up_count = 0
        self.fall_down_count = 0
        self.side_fall_count = 0


        # 获取并设置LED颜色

        self.led_head = self.getDevice("HeadLed")
        self.led_eye = self.getDevice("EyeLed")
        self.led_head.set(0xFFFF00)
        self.led_eye.set(0xFF0400) 


        # 初始化步态管理器和运动管理器

        self.gait_manager = RobotisOp2GaitManager(self, "config.ini")
        self.motion_manager = RobotisOp2MotionManager(self)


        # 播放初始站立动作 (page 9)

        self.motion_manager.playPage(9)
        # 等待动作播放完成
        while self.motion_manager.isMotionPlaying():
             super(Supervisor, self).step(self.timestep)

        # 获取并启用加速度计、陀螺仪和Lidar
        self.accelerometer = self.getDevice("Accelerometer")
        self.accelerometer.enable(self.timestep)
        # self.camera = self.getDevice("camera")
        # self.camera.enable(self.timestep)

        self.gyro = self.getDevice("Gyro")
        self.gyro.enable(self.timestep)

        self.lidar = self.getDevice("lidar")
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud() 
        self.sector_min = []
        # 获取并启用距离传感器

        self.distance_sensors = []
        sensor_names = ["front_ds", "left_front_ds", "right_front_ds"]
        for name in sensor_names:
            sensor = self.getDevice(name)
            if sensor is not None:
                sensor.enable(self.timestep)
                self.distance_sensors.append(sensor)
            else:
                print(f"Warning: Distance sensor {name} not found!")


        # 启动步态管理器并启用平衡控制

        self.gait_manager.start()
        self.gait_manager.setBalanceEnable(True)

         

        # ================ Env Setup ===================

        # 获取目标节点及其位置

        self.target_node = self.getFromDef("TARGET")
        self.target_position = np.array(self.target_node.getPosition()[:2]) # 只取x, y坐标
        print("Initial target pos: ", self.target_position)


        # 定义状态空间

        # 状态空间包含：Lidar数据 (LIDAR_RESOLUTION维), 归一化目标距离 (1维), 归一化目标相对角度 (1维),

        # 加速度计数据 (3维), 陀螺仪数据 (3维), 机器人姿态角 (俯仰和滚转，2维)

        # OBSERVATION_SPACE_DIM = LIDAR_RESOLUTION + 2 + 3 + 3 + 2
        OBSERVATION_SPACE_DIM = 72 + 3 + 3 + 3 + 2
        # low_bounds = np.concatenate([
        #     np.full(LIDAR_RESOLUTION, 0.0),  # Lidar 归一化距离 [0, 1]
        #     np.array([0.0, -1.0]),          # 归一化目标距离 [0, 1], 归一化目标相对角度 [-1, 1]
        #     np.full(3, -np.inf),            # 加速度计 (理论范围，实际可能有限)
        #     np.full(3, -np.inf),             # 陀螺仪 (理论范围，实际可能有限)
        #     np.full(2, -np.pi)              # 姿态角 (俯仰和滚转) [-pi, pi]
        # ], dtype=np.float32)

        # high_bounds = np.concatenate([
        #     np.full(LIDAR_RESOLUTION, 1.0),  # Lidar 归一化距离 [0, 1]
        #     np.array([1.0, 1.0]),           # 归一化目标距离 [0, 1], 归一化目标相对角度 [-1, 1]
        #     np.full(3, np.inf),             # 加速度计
        #     np.full(3, np.inf),              # 陀螺仪
        #     np.full(2, np.pi)               # 姿态角 (俯仰和滚转) [-pi, pi]
        # ], dtype=np.float32)
        low_bounds = np.concatenate([
            np.full(72, 0.0),          # Lidar [0,1]
            np.array([0.0, -1.0, -1.0]),             # 新增cos角度维度（距离[0,1], sin角度[-1,1], cos角度[-1,1]
            np.full(3, -np.inf),                     # 加速度计
            np.full(3, -np.inf),                     # 陀螺仪
            np.full(2, -np.pi)                       # 姿态角
        ], dtype=np.float32)

        high_bounds = np.concatenate([
            np.full(72, 1.0),          # Lidar
            np.array([1.0, 1.0, 1.0]),              # 对应新增的cos角度维度
            np.full(3, np.inf),                      # 加速度计
            np.full(3, np.inf),                      # 陀螺仪
            np.full(2, np.pi)                        # 姿态角
        ], dtype=np.float32)


        self.observation_space = Box(low_bounds, high_bounds, shape=(OBSERVATION_SPACE_DIM, ), dtype=np.float32)


        # 定义动作空间 (离散动作)

        # 0: Stop

        # 1: Forward (and turn towards target)

        # 2: Turn Left

        # 3: Turn Right

        self.action_space = Discrete(4)


        # 用于奖励计算的变量

        self.previous_distance_to_target = float('inf')
        self.current_distance_to_target = float('inf')
        self.current_relative_angle_to_target = 0.0
        self.previous_robot_position = None # 用于计算移动距离


        # ================ Robot Initial Pos ===================

        # 存储机器人初始位置和旋转，用于重置
        self.initial_robot_trans_field = self.robot.getField('translation')
        self.initial_robot_rot_field = self.robot.getField('rotation')
        self.initial_robot_pos_value = self.initial_robot_trans_field.getSFVec3f()
        self.initial_robot_rot_value = self.initial_robot_rot_field.getSFRotation()
        self.info = {} # 用于存储额外信息


        print(f"Initial robot pos:  [{self.initial_robot_pos_value[0]:.1f}, {self.initial_robot_pos_value[1]:.1f}]")


    def get_observations(self):

        # 获取Lidar数据

        range_image = self.lidar.getRangeImage()
        if range_image:
            lidar_state = np.array(range_image, dtype=np.float32)
            # 将无穷大和NaN值替换为最大Lidar距离
            lidar_state[np.isinf(lidar_state)] = LIDAR_RANGE
            lidar_state[np.isnan(lidar_state)] = LIDAR_RANGE
            # 归一化Lidar数据到 [0, 1]
            normalized_lidar_state = lidar_state / LIDAR_RANGE

        else:
            # 如果没有Lidar数据，返回全1的数组 (表示最大距离)
            normalized_lidar_state = np.full(LIDAR_RESOLUTION, 1.0, dtype=np.float32)
        
        sector_num = 8  # 将64个激光点划分为8个扇区
        sector_size = LIDAR_RESOLUTION // sector_num
        self.sector_min = [np.min(normalized_lidar_state[i*sector_size:(i+1)*sector_size]) 
                    for i in range(sector_num)]
        processed_lidar = np.concatenate([normalized_lidar_state, self.sector_min])
        # 获取机器人当前位置和方向

        position = self.robot.getPosition()
        orientation_matrix = self.robot.getOrientation()


        # 计算机器人当前的偏航角 (绕Z轴旋转角度)

        # 使用atan2从旋转矩阵中提取偏航角

        # yaw = np.arctan2(orientation_matrix[1], orientation_matrix[0])
        yaw = np.arctan2(orientation_matrix[3], orientation_matrix[0])


        # 计算机器人到目标的向量、距离和相对角度

        robot_pos_np = np.array(position[:2]) # 只取x, y坐标
        target_vector = self.target_position - robot_pos_np

        distance = np.linalg.norm(target_vector)


        # 计算目标在世界坐标系下的角度

        angle_to_target_world = np.arctan2(target_vector[1], target_vector[0])
        # 计算目标相对于机器人当前方向的相对角度

        relative_angle = yaw - angle_to_target_world  # 交换角度计算顺序
        # 将相对角度规范化到 [-pi, pi] 范围内

        relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))

        # 归一化目标距离和相对角度

        normalized_distance = distance / MAX_ENVIRONMENT_DISTANCE
        # normalized_angle = relative_angle / np.pi # 归一化到 [-1, 1]
        normalized_angle_sin = np.sin(relative_angle)
        normalized_angle_cos = np.cos(relative_angle)


        # 获取加速度计和陀螺仪数据
        acc_values = np.array(self.accelerometer.getValues(), dtype=np.float32)

        gyro_values = np.array(self.gyro.getValues(), dtype=np.float32)


        # 计算机器人的姿态角 (俯仰角和滚转角)

        # 假设机器人坐标系与Webots世界坐标系对齐，且加速度计测量的是重力加速度在机器人坐标系各轴上的分量

        # 俯仰角 (pitch): 绕机器人X轴旋转，影响加速度计Y和Z读数

        # 滚转角 (roll): 绕机器人Y轴旋转，影响加速度计X和Z读数

        # 这里使用简单的 atan2 计算方法，可能需要根据实际机器人坐标系进行调整
        roll = math.atan2(-acc_values[0], acc_values[2])

        pitch = math.atan2(acc_values[1], math.sqrt(acc_values[0]**2 + acc_values[2]**2))

        attitude_angles = np.array([pitch, roll], dtype=np.float32)



        # 构建最终的状态向量

        # current_state = np.concatenate([
        #     normalized_lidar_state,
        #     np.array([normalized_distance, normalized_angle], dtype=np.float32),
        #     acc_values,
        #     gyro_values,
        #     attitude_angles # 添加姿态角到状态空间
        # ])
        current_state = np.concatenate([
            # normalized_lidar_state,
            processed_lidar,
            np.array([normalized_distance, normalized_angle_sin, normalized_angle_cos], dtype=np.float32),
            acc_values,
            gyro_values,
            attitude_angles # 添加姿态角到状态空间
        ])


        # 更新当前距离和相对角度，用于奖励计算

        self.current_distance_to_target = distance
        self.current_relative_angle_to_target = relative_angle
        self.current_robot_position = robot_pos_np # 更新当前机器人位置

        return current_state


    def get_reward(self, action):

        # 初始化奖励

        reward = STEP_PENALTY # 每一步给予小额惩罚
        reward += SURVIVAL_REWARD # 每一步存活奖励

        # 获取当前回合状态信息
        obstacle_threshold = 0.3  # 30% Lidar距离阈值
        danger_sectors = [i for i, val in enumerate(self.sector_min) 
                        if val < obstacle_threshold]
        
        # 动作方向与障碍物方位匹配奖励（网页7模糊逻辑简化）
        if action == 2:  # 左转
            reward += 0.5 if 0 or 1 or 2 in danger_sectors else -0.2  # 右侧有障碍时奖励
        elif action == 3:  # 右转
            reward += 0.5 if 5 or 6 or 7 in danger_sectors else -0.2  # 左侧有障碍时奖励
        info = self.get_info()
        collision = info.get('collision', False)
        target_reached = info.get('target_reached', False)
        fell_down = info.get('fell_down', False)

        # 根据回合结束状态给予大额奖励或惩罚

        if collision:
            reward += COLLISION_PENALTY
        elif target_reached:

            reward += TARGET_REWARD

        elif fell_down:

            reward += FALL_PENALTY

        else:

            # 如果回合未结束，计算基于进度的奖励

            # 奖励接近目标

            reward += PROGRESS_REWARD_WEIGHT * (self.previous_distance_to_target - self.current_distance_to_target)


            # 奖励保持平衡 (基于姿态角和陀螺仪)

            # 惩罚大的俯仰角和滚转角，惩罚大的角速度
            # acc_values = self.accelerometer.getValues()
            # gyro_values = self.gyro.getValues()
            # roll = math.atan2(-acc_values[0], acc_values[2])
            # pitch = math.atan2(acc_values[1], math.sqrt(acc_values[0]**2 + acc_values[2]**2))
            # balance_penalty = BALANCE_REWARD_WEIGHT * (abs(pitch) + abs(roll)) # 惩罚大的姿态角
            # angular_velocity_penalty = ANGULAR_VELOCITY_PENALTY_WEIGHT * np.sum(np.abs(gyro_values)) # 惩罚大的角速度
            # reward -= (balance_penalty + angular_velocity_penalty)

            # 新增：基于Lidar最近距离的惩罚，鼓励远离障碍物

            lidar_range_image = self.lidar.getRangeImage()

            if lidar_range_image:

                min_lidar_distance = np.min(lidar_range_image)

                # 只在距离小于阈值时进行惩罚

                if min_lidar_distance < LIDAR_PROXIMITY_THRESHOLD:

                    # 使用 1 / distance 的形式，距离越近惩罚越大

                    # 为了避免除以零，可以加一个很小的数，或者限制最小距离

                    proximity_penalty = LIDAR_PROXIMITY_PENALTY_WEIGHT / (min_lidar_distance + 0.01) # 加0.01避免除以零

                    reward -= proximity_penalty



            # 惩罚静止 (如果当前动作不是停止)

            if action != 0 and self.previous_robot_position is not None:

                 movement_distance = np.linalg.norm(self.current_robot_position - self.previous_robot_position)

                 if movement_distance < 0.02: # 如果移动距离很小，认为是静止 (阈值略微增加)

                     reward += STILL_PENALTY


        # 更新前一步到目标的距离和机器人位置

        self.previous_distance_to_target = self.current_distance_to_target

        self.previous_robot_position = self.current_robot_position


        return reward


    def is_done(self):

        collision = False

        # 检查距离传感器是否检测到碰撞

        for sensor in self.distance_sensors:

            if sensor is not None and sensor.getValue() < COLLISION_DISTANCE_THRESHOLD:

                collision = True

                # print("Collision detected by distance sensor!")

                break


        # 检查Lidar是否有近距离障碍物 (作为额外的碰撞或危险信号)

        lidar_range_image = self.lidar.getRangeImage()

        if lidar_range_image:

             # 检查Lidar前方区域是否有障碍物

             center_index = LIDAR_RESOLUTION // 2

             front_indices_start = max(0, center_index - LIDAR_RESOLUTION // 12) # 检查前方约 +/- 15度范围

             front_indices_end = min(LIDAR_RESOLUTION, center_index + LIDAR_RESOLUTION // 12)

             for i in range(front_indices_start, front_indices_end):

                 if lidar_range_image[i] < COLLISION_DISTANCE_THRESHOLD:

                     collision = True

                     # print("Collision detected by Lidar!")

                     break



        # 检查是否到达目标

        target_reached = self.current_distance_to_target < TARGET_DISTANCE_THRESHOLD

        if target_reached:

            print("Target reached!")


        # 检查是否跌倒

        fell_down = self.check_if_fallen()

        if fell_down:

            print("Robot fell down!")


        # 判断回合是否结束

        # done = collision or target_reached or fell_down
        done =  target_reached


        # 更新 info 字典

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
        SIDE_FALL_ANGLE = 0.8
        CONSECUTIVE_CYCLES = 5
        

        # 获取加速度计数据

        acc_values = self.accelerometer.getValues()
        # roll = math.atan2(-acc_values[0], acc_values[1])
        y_acc = acc_values[1]
        
        # if abs(roll) > SIDE_FALL_ANGLE:
        #     self.side_fall_count = min(self.side_fall_count + 1, CONSECUTIVE_CYCLES)
        # else:
        #     self.side_fall_count = max(self.side_fall_count - 1, 0)

        # if self.side_fall_count >= CONSECUTIVE_CYCLES:
        #     self._convert_side_fall(roll)

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

    # def _convert_side_fall(self, roll):

    #     if roll > 0.5:

    #         self.motion_manager.playPage(12)  # 左侧侧翻恢复

    #     elif roll < -0.5:

    #         self.motion_manager.playPage(13)  # 右侧侧翻恢复

    #     self.side_fall_count = 0

    #     self.gait_manager.setXAmplitude(1.5)  # 增大前进步幅
    #     sleep(0.5)  # 保持动作持续时间
        
    #     # 4. 触发标准前倒检测（复用原有前倒恢复）
    #     self.fall_up_count = 101 

    def apply_action(self, action):

        # 根据DRL代理输出的动作控制机器人步态

        # Gait manager 持续运行，我们通过设置其幅度参数来控制步态


        # 移除规则避障逻辑，代理将自行决定如何避障

        # Lidar-based Obstacle Avoidance Logic (Rule-based override)

        # Check Lidar readings in the forward direction

        # lidar_range_image = self.lidar.getRangeImage()

        # obstacle_ahead = False

        # if lidar_range_image:

        #     center_index = LIDAR_RESOLUTION // 2

        #     front_indices_start = max(0, center_index - LIDAR_RESOLUTION // 12)

        #     front_indices_end = min(LIDAR_RESOLUTION, center_index + LIDAR_RESOLUTION // 12)

        #     for i in range(front_indices_start, front_indices_end):

        #         if lidar_range_image[i] < OBSTACLE_AVOIDANCE_LIDAR_THRESHOLD:

        #             obstacle_ahead = True

        #             break


        # 根据动作设置步态管理器参数
        # print(action)
        front_sectors = self.sector_min[3:5]  # 前向4个扇区
        if np.min(front_sectors) < 0.4:  # 前方障碍物检测
            # 强制转向策略（网页2 A*启发式简化）
            if min(self.sector_min[0], self.sector_min[1], self.sector_min[2]) > min(self.sector_min[5], self.sector_min[6], self.sector_min[7]):  # 左侧更通畅
                action = 3  # you转
            else:
                action = 2  # zuo转
        if action == 0: # Stop walking

            self.gait_manager.setXAmplitude(0.0)

            self.gait_manager.setYAmplitude(0.0)

            self.gait_manager.setAAmplitude(0.0)

            # print("Action: Stop")


        elif action == 1: # Forward (and turn towards target)

            # 不再检查 obstacle_ahead，代理自行学习避障

            self.gait_manager.setXAmplitude(FORWARD_AMPLITUDE) # 前进步长

            self.gait_manager.setYAmplitude(0.0)  # 侧向步长为0

            # 根据目标相对角度计算转向幅度

            # **修正转向方向**

            # turn_amplitude = self.current_relative_angle_to_target * TARGET_TURN_GAIN

            # # 限制转向幅度在合理范围内

            # clamped_turn_amplitude = np.clip(turn_amplitude, -TURN_AMPLITUDE, TURN_AMPLITUDE)

            # self.gait_manager.setAAmplitude(clamped_turn_amplitude)

            turn_amplitude = self.current_relative_angle_to_target * TARGET_TURN_GAIN
            # 根据Webots转向方向调整符号
            self.gait_manager.setAAmplitude(-turn_amplitude)
            # print(f"Action: Forward. Relative angle: {self.current_relative_angle_to_target:.2f}, Turn amplitude: {clamped_turn_amplitude:.2f}")



        elif action == 2: # Turn Left

            # 执行固定幅度的左转

            self.gait_manager.setXAmplitude(TURN_FORWARD_AMPLITUDE) # 转向时小步前进

            self.gait_manager.setYAmplitude(0.0)

            self.gait_manager.setAAmplitude(-TURN_AMPLITUDE) # 向左转

            # print("Action: Turn Left")


        elif action == 3: # Turn Right

            # 执行固定幅度的右转

            self.gait_manager.setXAmplitude(TURN_FORWARD_AMPLITUDE) # 转向时小步前进

            self.gait_manager.setYAmplitude(0.0)

            self.gait_manager.setAAmplitude(TURN_AMPLITUDE) # 向右转

            # print("Action: Turn Right")


        # TODO: 如果需要，可以添加更多动作，例如侧向移动等


    def step(self, action):

        # 在每个仿真步中执行的操作
        self.apply_action(action) # 根据动作设置步态参数

        # # 推进Webots仿真一步

        # super(Supervisor, self).step(self.timestep)

        # # 推进步态管理器一步

        # # self.gait_manager.step(self.timestep)
        # self.gait_manager.step(self.timestep)
        for _ in range(int(N_SIMULATION_STEPS_PER_ACTION / self.timestep)):
            # 推进Webots仿真一步
            super(Supervisor, self).step(self.timestep)
            # 推进步态管理器一步，它会根据之前设置的参数生成电机目标
            self.gait_manager.step(self.timestep)

            # 在每个小的仿真步检查回合是否结束
            # 如果发生碰撞或跌倒，立即停止当前“行走步”的执行
            done = self.is_done() # is_done 会更新 self.info
            if done:
                break # 提前跳出内部仿真循环
        # 获取下一个状态
        
         
        next_state = self.get_observations()

        # 判断回合是否结束

        done = self.is_done()

        # 计算奖励

        reward = self.get_reward(action)

        # 获取额外信息

        info = self.get_info()


        return next_state, reward, done, info


    def reset(self):

        # 重置环境到初始状态

        print("Resetting environment")

        # 停止步态管理器

        self.gait_manager.stop()


        # 重置机器人位置和旋转

        self.initial_robot_trans_field.setSFVec3f(self.initial_robot_pos_value)

        self.initial_robot_rot_field.setSFRotation(self.initial_robot_rot_value)


        # 重置仿真物理状态

        self.simulationResetPhysics()


        # 播放初始站立动作

        self.motion_manager.playPage(9)

        # 等待动作播放完成

        while self.motion_manager.isMotionPlaying():

             super(Supervisor, self).step(self.timestep)


        # 重新启动步态管理器

        self.gait_manager.start()

        # # 推进一步仿真，让机器人进入稳定状态并获取初始观测

        # super(Supervisor, self).step(self.timestep)

        # self.gait_manager.step(self.timestep)

        for _ in range(int(N_SIMULATION_STEPS_PER_ACTION / self.timestep)):
            # 推进Webots仿真一步
            super(Supervisor, self).step(self.timestep)
            # 推进步态管理器一步，它会根据之前设置的参数生成电机目标
            self.gait_manager.step(self.timestep)

            # 在每个小的仿真步检查回合是否结束
            # 如果发生碰撞或跌倒，立即停止当前“行走步”的执行
            done = self.is_done() # is_done 会更新 self.info
            if done:
                break # 提前跳出内部仿真循环
        
         

        # 获取初始状态

        initial_state = self.get_observations()

        # 更新前一步到目标的距离和机器人位置

        self.previous_distance_to_target = self.current_distance_to_target

        self.previous_robot_position = self.current_robot_position


        print(f"Environment reset complete. Robot position resetted")
        # new_target_x = random.uniform(-5, 5)
        # new_target_y = random.uniform(-5, 5)
        # for i in range(OBSTACLE_NUM):
        #     self.obstacle_nodes[i].getField("translation").setSFVec3f([random.uniform(-5, 5), random.uniform(-5, 5), 0])
        # self.target_node.getField("translation").setSFVec3f([new_target_x, new_target_y, 0])
        # self.target_position = np.array([new_target_x, new_target_y])

        new_target_x = random.uniform(-5, 5)
        new_target_y = random.uniform(-5, 5)
         
        # Get the robot's current position after reset and initial stand motion
        robot_reset_pos = np.array(self.robot.getPosition()[:2])

        # 生成障碍物坐标（排除目标点周围0.3米和机器人周围0.5米）
        OBSTACLE_SAFETY_MARGIN = 0.3 # Define the safety margin around the robot and target
        for i in range(OBSTACLE_NUM):
            while True:
                # 生成候选坐标
                x = random.uniform(-5, 5)
                y = random.uniform(-5, 5)
                
                candidate_pos = np.array([x, y])

                # 计算与目标的欧氏距离
                distance_to_target = np.linalg.norm(candidate_pos - np.array([new_target_x, new_target_y]))
                
                # 计算与机器人初始位置的欧氏距离
                distance_to_robot = np.linalg.norm(candidate_pos - robot_reset_pos)

                # 检查是否与目标或机器人初始位置距离过近
                if distance_to_target > OBSTACLE_SAFETY_MARGIN and distance_to_robot > OBSTACLE_SAFETY_MARGIN:
                    self.obstacle_nodes[i].getField("translation").setSFVec3f([x, y, 0.5])
                    break

        # 设置目标点位置
        self.target_node.getField("translation").setSFVec3f([new_target_x, new_target_y, 0])
        self.target_position = np.array([new_target_x, new_target_y])
        
        return initial_state
            


    def get_info(self):

        # 返回包含回合状态的字典

        return self.info 