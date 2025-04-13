import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from controller import Robot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRGB(nn.Module):
    def __init__(self):
        super(EncoderRGB, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 60 * 60, 128)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc(x))
        return x

class EncoderNonVisual(nn.Module):
    def __init__(self):
        super(EncoderNonVisual, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 128)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

class DynamicsModel(nn.Module):
    def __init__(self):
        super(DynamicsModel, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

class PolicyModel(nn.Module):
    def __init__(self):
        super(PolicyModel, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 4)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

class RewardPredictor(nn.Module):
    def __init__(self):
        super(RewardPredictor, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TerminationPredictor(nn.Module):
    def __init__(self):
        super(TerminationPredictor, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class ValuePredictor(nn.Module):
    def __init__(self):
        super(ValuePredictor, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MaplessNavigation:
    def __init__(self, robot):
        self.robot = robot
        self.time_step = robot.time_step
        
        # 初始化模型
        self.encoder_rgb = EncoderRGB().to(device)
        self.encoder_non_visual = EncoderNonVisual().to(device)
        self.dynamics_model = DynamicsModel().to(device)
        self.policy = PolicyModel().to(device)
        self.reward_predictor = RewardPredictor().to(device)
        self.termination_predictor = TerminationPredictor().to(device)
        self.value_predictor = ValuePredictor().to(device)
        
        # 定义优化器
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.reward_optimizer = optim.Adam(self.reward_predictor.parameters(), lr=1e-4)
        self.termination_optimizer = optim.Adam(self.termination_predictor.parameters(), lr=1e-4)
        self.value_optimizer = optim.Adam(self.value_predictor.parameters(), lr=1e-4)
        
    def get_observations(self):
        # 获取RGB图像
        camera = self.robot.camera
        image_data = camera.getImage()
        image = np.frombuffer(image_data, np.uint8).reshape(
            (camera.getHeight(), camera.getWidth(), 4)
        )
        image = image[:, :, :3]  # 只取RGB通道
        
        # 获取非视觉观测
        acc_values = self.robot.accelerometer.getValues()
        gyro_values = self.robot.gyro.getValues()
        compass_values = self.robot.compass.getValues()
        imu_values = self.robot.get_inertial_unit_data()
        gps_values = self.robot.gps.getValues()
        
        non_visual_obs = np.concatenate([
            acc_values, gyro_values, compass_values, imu_values, gps_values
        ])
        
        return image, non_visual_obs
    
    def fuse_observations(self, rgb_image, non_visual_obs):
        # 使用自编码器融合观测
        rgb_tensor = torch.tensor(rgb_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        non_visual_tensor = torch.tensor(non_visual_obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        encoded_rgb = self.encoder_rgb(rgb_tensor)
        encoded_non_visual = self.encoder_non_visual(non_visual_tensor)
        
        # 融合潜在状态
        fused_state = torch.cat([encoded_rgb, encoded_non_visual], dim=1)
        
        return fused_state
    
    def plan_action(self, fused_state):
        # 使用策略网络预测动作
        action = self.policy(fused_state)
        
        # 使用动力学模型预测下一个状态
        next_state = self.dynamics_model(fused_state)
        
        return action, next_state
    
    def update_models(self, state, action, reward, next_state, termination):
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        predicted_action = self.policy(state)
        policy_loss = nn.MSELoss()(action, predicted_action)
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # 更新奖励预测器
        self.reward_optimizer.zero_grad()
        predicted_reward = self.reward_predictor(next_state)
        reward_loss = nn.MSELoss()(reward, predicted_reward)
        reward_loss.backward()
        self.reward_optimizer.step()
        
        # 更新终止预测器
        self.termination_optimizer.zero_grad()
        predicted_termination = self.termination_predictor(next_state)
        termination_loss = nn.MSELoss()(termination, predicted_termination)
        termination_loss.backward()
        self.termination_optimizer.step()
        
        # 更新价值预测器
        self.value_optimizer.zero_grad()
        predicted_value = self.value_predictor(next_state)
        value_loss = nn.MSELoss()(reward, predicted_value)
        value_loss.backward()
        self.value_optimizer.step()
    
    def run(self):
        while True:
            # 获取观测
            rgb_image, non_visual_obs = self.get_observations()
            
            # 融合观测
            fused_state = self.fuse_observations(rgb_image, non_visual_obs)
            
            # 规划动作
            action, next_state = self.plan_action(fused_state)
            
            # 执行动作
            self.robot.execute_action(action.detach().cpu().numpy())
            
            # 获取奖励和终止状态
            reward = self.reward_predictor(next_state)
            termination = self.termination_predictor(next_state)
            
            # 更新模型
            self.update_models(fused_state, action, reward, next_state, termination)
            
            # 检查终止条件
            if termination.item() > 0.5:
                self.robot.reset()
            
            # 步进
            self.robot.my_step()