# 需确保Python版本为3.7+，并在PyCharm中配置Webots库路径
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from controller import Robot, Keyboard
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager
import sys
import time

# 硬件配置参数
NMOTORS = 20
motorNames = [
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR",
    "ArmLowerL", "PelvYR", "PelvYL", "PelvR", "PelvL", "LegUpperR",
    "LegUpperL", "LegLowerR", "LegLowerL", "AnkleR", "AnkleL", "FootR",
    "FootL", "Neck", "Head"
]
target_position = [0.0, 2.0, 0.0]  # 目标坐标(x,y,z)

class DQN:
    def __init__(self, walk_controller):
        self.walker = walk_controller  
        self.state_dim = 256  # 融合传感器维度
        self.action_dim = 4    # 动作空间：前进/后退/左转/右转
        self.policy_net = self._build_network()
        self.target_net = self._build_network()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.target_update = 1000
        self.steps = 0  # 新增训练步数计数器

    def _build_network(self):
        """构建深度Q网络（网页1网络结构）"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )

    def get_state(self):
        """多传感器数据融合（网页4状态空间设计）"""
        # 惯性传感器数据
        acc = np.array(self.walker.accelerometer.getValues())
        gyro = np.array(self.walker.gyro.getValues())
        
        # 视觉数据预处理
        img = self.walker.camera.getImageArray()
        img_gray = np.mean(img, axis=2).reshape(-1)[:64]  # 降维至64维
        
        # 定位数据
        compass = np.array(self.walker.compass.getValues())
        imu = np.array(self.walker.imu.getRollPitchYaw())
        
        return np.concatenate([acc, gyro, img_gray, compass, imu]).astype(np.float32)

    def plan_action(self, state):
        """ε-贪婪策略生成动作（网页2探索策略）"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                return self.policy_net(state_tensor).argmax().item()
                
    def compute_reward(self):
        """动态奖励函数（网页3多目标优化）"""
        # 定位奖励
        current_pos = self.walker.gps.getValues()[:2]
        distance_reward = -np.linalg.norm(np.array(current_pos) - np.array(target_position[:2]))
        
        # 能耗惩罚
        power_cost = 0
        for m in self.walker.motors:
            velocity = m.getVelocity() 
            torque = m.getAvailableTorque()
            power_cost += abs(velocity * torque) if torque else 0
        
        # 稳定性奖励
        imu_data = np.abs(self.walker.imu.getRollPitchYaw())
        stability_reward = -np.sum(imu_data)
        
        return distance_reward * 0.5 - power_cost * 0.3 + stability_reward * 0.2
        
    def train(self):
        """经验回放训练（网页5训练流程）"""
        if len(self.memory) < self.batch_size:
            return
            
        # 随机采样批次
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Q值计算
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0](@ref).detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 反向传播
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 参数更新
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        self.steps += 1
        
        # 目标网络同步（网页6参数更新机制）
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

class Humanoid(Robot):
    def __init__(self):
        super().__init__()
        self.time_step = int(self.getBasicTimeStep())
        
        # 初始化LED
        self.led_head = self.getDevice("HeadLed")
        self.led_eye = self.getDevice("EyeLed")
        self.led_head.set(0xFF00FF)  # 紫色头部LED
        self.led_eye.set(0x0006FF)   # 蓝色眼睛LED
        
        # 传感器初始化（网页7传感器配置）
        self.accelerometer = self.getDevice("Accelerometer")
        self.accelerometer.enable(self.time_step)
        self.gyro = self.getDevice("Gyro")
        self.gyro.enable(self.time_step)
        self.camera = self.getDevice("Camera")
        self.camera.enable(2*self.time_step)
        self.compass = self.getDevice("compass")
        self.compass.enable(self.time_step)
        self.imu = self.getDevice("imu")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice('gps')
        self.gps.enable(self.time_step)
        
        # 电机控制初始化
        self.motors = []
        self.position_sensors = []
        for name in motorNames:
            motor = self.getDevice(name)
            sensor = self.getDevice(name + "S")
            sensor.enable(self.time_step)
            self.motors.append(motor)
            self.position_sensors.append(sensor)
        
        # 运动控制模块
        self.motion_manager = RobotisOp2MotionManager(self)
        self.gait_manager = RobotisOp2GaitManager(self, "config.ini")
        self.planner = DQN(self)
        
        # 跌倒检测计数器
        self.fall_up_count = 0
        self.fall_down_count = 0

    def my_step(self):
        """步进仿真（网页1步进控制）"""
        if self.step(self.time_step) == -1:
            sys.exit(0)
    
    def wait(self, ms):
        """等待指定毫秒"""
        start_time = self.getTime()
        while (self.getTime() - start_time) * 1000 < ms:
            self.my_step()
    
    def check_target_reached(self):
        """目标到达检测（网页2终止条件）"""
        current_pos = self.gps.getValues()
        return np.linalg.norm(np.array(current_pos) - np.array(target_position)) < 0.5
    
    def reset_environment(self):
        """环境重置（网页3训练流程）"""
        # 重置机器人姿势
        self.motion_manager.playPage(9)  
        self.motors[-1].setPosition(0.7)  # 头部复位
        self.wait(200)
        
        # 重置传感器数据
        self.accelerometer.enable(self.time_step)
        self.gps.enable(self.time_step)
        
    def run(self):
        print("------- Humanoid DQN Path Planning -------")
        self.my_step()  # 初始化传感器
        
        # 初始姿势设置
        self.reset_environment()
        episode = 0
        
        while True:
            state = self.planner.get_state()
            action = self.planner.plan_action(state)
            
            # 动作映射（网页4控制接口）
            self.gait_manager.start()
            if action == 0:   # 前进
                self.gait_manager.setXAmplitude(1.0)
            elif action == 1: # 后退
                self.gait_manager.setXAmplitude(-1.0)
            elif action == 2: # 左转
                self.gait_manager.setAAmplitude(0.5)
            elif action == 3: # 右转
                self.gait_manager.setAAmplitude(-0.5)
                
            # 执行步态
            self.gait_manager.step(self.time_step)
            self.my_step()
            
            # 获取新状态
            next_state = self.planner.get_state()
            reward = self.planner.compute_reward()
            done = self.check_target_reached()
            
            # 存储经验
            self.planner.memory.append((state, action, reward, next_state, done))
            
            # 训练网络
            self.planner.train()
            
            # 回合结束处理
            if done:
                print(f"Episode {episode} Reward: {reward:.2f}")
                episode += 1
                self.reset_environment()
                
            self.check_if_fallen()

    def check_if_fallen(self):
        """跌倒检测与恢复（网页5安全机制）"""
        acc_tolerance = 80.0
        acc_step = 100
        
        acc_values = self.accelerometer.getValues()
        y_acc = acc_values[1](@ref)
        
        if y_acc < 512.0 - acc_tolerance:
            self.fall_up_count += 1
        else:
            self.fall_up_count = 0
            
        if y_acc > 512.0 + acc_tolerance:
            self.fall_down_count += 1
        else:
            self.fall_down_count = 0
            
        # 触发恢复动作
        if self.fall_up_count > acc_step:
            print("Fall recovery triggered")
            self.motion_manager.playPage(10)
            self.motion_manager.playPage(9)
            self.fall_up_count = 0
        elif self.fall_down_count > acc_step:
            print("Fall recovery triggered")
            self.motion_manager.playPage(11)
            self.motion_manager.playPage(9)
            self.fall_down_count = 0

if __name__ == "__main__":
    robot = Humanoid()
    
    # 训练模式配置
    try:
        robot.run()
    except KeyboardInterrupt:
        # 保存模型参数
        torch.save(robot.planner.policy_net.state_dict(), 'dqn_model.pth')
        print("Model saved before exit")