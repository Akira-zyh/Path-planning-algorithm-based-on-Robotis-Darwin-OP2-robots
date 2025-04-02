# Webots仿真环境要求：R2023b及以上版本，安装PyTorch
from controller import Robot, Motor, Accelerometer, Gyro, LED, GPS, Compass
from managers import RobotisOp2MotionManager, RobotisOp2GaitManager
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# 硬件配置参数
NMOTORS = 20
MOTOR_NAMES = [
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR",
    "ArmLowerL", "PelvYR", "PelvYL", "PelvR", "PelvL", "LegUpperR",
    "LegUpperL", "LegLowerR", "LegLowerL", "AnkleR", "AnkleL", "FootR",
    "FootL", "Neck", "Head"
]

class DQN(nn.Module):
    """深度Q网络架构定义"""
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    """DQN算法智能体封装"""
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=1e-4, weight_decay=1e-5)
        
        self.memory = deque(maxlen=100000)
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.update_freq = 1000
        self.step_count = 0

    def select_action(self, state, training=True):
        """动作选择策略"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.policy_net.net[-1].out_features-1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                return self.policy_net(state_t).argmax().item()

    def store_transition(self, transition):
        """存储转移经验"""
        self.memory.append(transition)

    def update_network(self):
        """网络参数更新"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # 从记忆库采样
        transitions = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (~dones) * self.gamma * next_q
        
        # 计算损失
        loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)
        
        # 反向传播优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新目标网络
        if self.step_count % self.update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 探索率衰减
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.step_count += 1
        
        return loss.item()

class HumanoidController(Robot):
    """人形机器人主控制器"""
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        
        # 硬件初始化
        self._init_sensors()
        self._init_actuators()
        self._init_position_sensors()
        self._init_managers()
        
        # 导航参数
        self.target_position = np.array([5.0, 0.0])  # X-Y平面目标位置
        
        # DQN智能体
        state_dim = self._get_state().shape[0]
        action_dim = 4  # 前进/后退/左转/右转
        self.agent = DQNAgent(state_dim, action_dim)
        
        # 训练参数
        self.episode_rewards = []
        self.current_episode = 0
        self.max_episode_steps = 1000

    def _init_sensors(self):
        """初始化传感器系统"""
        # 运动传感器
        self.accelerometer = self.getDevice("Accelerometer")
        self.accelerometer.enable(self.timestep)
        self.gyro = self.getDevice("Gyro")
        self.gyro.enable(self.timestep)
        
        # 导航传感器
        self.gps = self.getDevice("gps")
        self.gps.enable(self.timestep)
        self.compass = self.getDevice("compass")
        self.compass.enable(self.timestep)
        
        # 视觉传感器
        self.camera = self.getDevice("Camera")
        self.camera.enable(2*self.timestep)

    def _init_actuators(self):
        """初始化执行器系统"""
        self.motors = [self.getDevice(name) for name in MOTOR_NAMES]
        for motor in self.motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

    def _init_position_sensors(self):
        """启用所有关节位置传感器"""
        self.position_sensors = []
        for name in MOTOR_NAMES:
            sensor = self.getDevice(name + "S")
            sensor.enable(self.timestep)
            self.position_sensors.append(sensor)

    def _init_managers(self):
        """初始化运动管理模块"""
        self.motion_mgr = RobotisOp2MotionManager(self)
        self.gait_mgr = RobotisOp2GaitManager(self, "config.ini")
        self.gait_mgr.start()
        self.motion_mgr.playPage(9)  # 初始站立姿势

    def _get_state(self):
        """获取当前状态观测"""
        # 位置和方向
        position = np.array(self.gps.getValues()[:2])
        compass_val = self.compass.getValues()
        heading = math.atan2(compass_val[0], compass_val[1])
        
        # 目标相对信息
        target_vec = self.target_position - position
        distance = np.linalg.norm(target_vec)
        angle_to_target = math.atan2(target_vec[1], target_vec[0]) - heading
        
        # 运动状态
        accel = self.accelerometer.getValues()
        gyro = self.gyro.getValues()
        
        # 状态归一化
        return np.concatenate([
            position / 10.0,  # 假设场景尺寸在10米范围内
            [heading / math.pi], 
            target_vec / 10.0,
            [distance / 10.0, angle_to_target / math.pi],
            np.array(accel) / 9.81,
            np.array(gyro) / (2*math.pi)
        ]).astype(np.float32)

    def _get_reward(self, prev_state, new_state):
        """计算即时奖励"""
        # 进度奖励
        progress = (prev_state[4] - new_state[4]) * 10.0
        
        # 方向奖励
        angle_penalty = abs(new_state[5]) * 0.5
        
        # 能量消耗惩罚
        accel_norm = np.linalg.norm(new_state[6:9])
        energy_penalty = accel_norm * 0.1
        
        # 任务完成奖励
        success_bonus = 100.0 if new_state[4] < 0.5 else 0.0
        
        # 跌倒惩罚
        fall_penalty = -100.0 if self._check_fall() else 0.0
        
        return progress - angle_penalty - energy_penalty + success_bonus + fall_penalty

    def _check_fall(self):
        """跌倒检测算法"""
        acc_z = self.accelerometer.getValues()[2]
        return acc_z < 3.0  # 当Z轴加速度小于3m/s²时判定为跌倒

    def _execute_action(self, action):
        """执行动作到运动系统"""
        action_map = {
            0: (1.0, 0.0),   # 前进
            1: (-1.0, 0.0),  # 后退
            2: (0.0, 0.5),   # 左转
            3: (0.0, -0.5)   # 右转
        }
        x_amp, a_amp = action_map[action]
        self.gait_mgr.setXAmplitude(x_amp)
        self.gait_mgr.setAAmplitude(a_amp)
        self.gait_mgr.step(self.timestep)
        self.step(self.timestep)

    def _reset_episode(self):
        """重置训练回合"""
        self.simulationReset()
        self.wait(100)
        self.gait_mgr.start()
        self.motion_mgr.playPage(9)
        return self._get_state()

    def train(self, total_episodes=1000):
        """主训练循环"""
        for ep in range(total_episodes):
            state = self._reset_episode()
            total_reward = 0.0
            done = False
            step = 0
            
            while not done and step < self.max_episode_steps:
                # 选择并执行动作
                action = self.agent.select_action(state)
                self._execute_action(action)
                
                # 获取新状态和奖励
                new_state = self._get_state()
                reward = self._get_reward(state, new_state)
                done = new_state[4] < 0.5 or self._check_fall()
                
                # 存储经验
                transition = (state, action, reward, new_state, done)
                self.agent.store_transition(transition)
                
                # 更新网络
                loss = self.agent.update_network()
                
                # 记录统计信息
                total_reward += reward
                state = new_state
                step += 1
                
                # 监控输出
                if step % 100 == 0:
                    print(f"Ep {ep} Step {step} | R:{total_reward:.1f} | ε:{self.agent.epsilon:.3f} | Loss:{loss:.3f}")
            
            # 回合结束处理
            self.episode_rewards.append(total_reward)
            self.current_episode +=1
            
            # 保存模型检查点
            if ep % 50 == 0 or total_reward > max(self.episode_rewards):
                torch.save({
                    'policy_net': self.agent.policy_net.state_dict(),
                    'target_net': self.agent.target_net.state_dict(),
                    'optimizer': self.agent.optimizer.state_dict(),
                    'epsilon': self.agent.epsilon,
                    'rewards': self.episode_rewards
                }, f"checkpoint_ep{ep}.pth")
            
            print(f"Episode {ep} completed | Total Reward: {total_reward:.1f} | Avg Reward: {np.mean(self.episode_rewards[-10:]):.1f}")

def main():
    controller = HumanoidController()
    try:
        # 加载预训练检查点（如果存在）
        checkpoint = torch.load("latest_checkpoint.pth")
        controller.agent.policy_net.load_state_dict(checkpoint['policy_net'])
        controller.agent.target_net.load_state_dict(checkpoint['target_net'])
        controller.agent.optimizer.load_state_dict(checkpoint['optimizer'])
        controller.agent.epsilon = checkpoint['epsilon']
        controller.episode_rewards = checkpoint['rewards']
        print("成功加载预训练模型")
    except FileNotFoundError:
        print("未找到检查点，开始新训练")
    
    # 启动训练
    controller.train(total_episodes=1000)

if __name__ == "__main__":
    main()