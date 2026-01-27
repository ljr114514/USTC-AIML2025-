from utils import *
import numpy as np
import torch
import torch.nn as nn
from typing import *
import sys
import argparse

# --- 设备初始化 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='args')
parser.add_argument('--num_episodes', type=int, default=5000, help='number of episodes')
parser.add_argument('--checkpoint', type=int, default=500, help='the interval of saving models')
parser.add_argument('--use_wandb', action='store_true', help='use wandb for experiment tracking')
parser.add_argument('--wandb_project', type=str, default='aiml_gobang', help='wandb project name')
parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name')
args = parser.parse_args()

num_episodes = args.num_episodes
checkpoint = args.checkpoint

# ==========================================
# 模型组件定义
# ==========================================

class ResBlock(nn.Module):
    """
    残差块：保持深度网络训练的稳定性
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class Actor(nn.Module):
    def __init__(self, board_size: int, lr=1e-4):
        super().__init__()
        self.board_size = board_size
        
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.linear_blocks = nn.Sequential(
            nn.Linear(board_size * board_size, board_size ** 2) 
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: Union[np.ndarray, torch.Tensor]):
        # 统一转为 Tensor 并处理维度
        if isinstance(x, np.ndarray):
            output = torch.as_tensor(x, dtype=torch.float32, device=device)
        else:
            output = x.to(torch.float32).to(device)

        if len(output.shape) == 2:
            output = output.unsqueeze(0).unsqueeze(0)
        elif len(output.shape) == 3:
            output = output.unsqueeze(1)
        
        x_features = self.conv_blocks(output)
        x_flat = self.flatten(x_features)
        logits = self.linear_blocks(x_flat)

        # 使用 reshape 解决内存不连续导致的 view 报错
        board_flat = output.reshape(output.size(0), -1)
        illegal_mask = (board_flat != 0) 
        logits = logits.masked_fill(illegal_mask, -1e15)

        return torch.softmax(logits, dim=1)

class Critic(nn.Module):
    def __init__(self, board_size: int, lr=1e-4):
        super().__init__()
        self.board_size = board_size

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.linear_blocks = nn.Sequential(
            nn.Linear(board_size * board_size, board_size ** 2)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: Union[np.ndarray, torch.Tensor], action: np.ndarray):
        indices = torch.tensor(
            [_position_to_index(self.board_size, r, c) for r, c in action],
            device=device, dtype=torch.long
        )

        if isinstance(x, np.ndarray):
            output = torch.as_tensor(x, dtype=torch.float32, device=device)
        else:
            output = x.to(torch.float32).to(device)

        if len(output.shape) == 2:
            output = output.unsqueeze(0).unsqueeze(0)
        elif len(output.shape) == 3:
            output = output.unsqueeze(1)

        x_features = self.conv_blocks(output)
        x_flat = self.flatten(x_features)
        q_all = self.linear_blocks(x_flat)
        return q_all.gather(1, indices.unsqueeze(1)).squeeze(1)

# ==========================================
# 主模型框架
# ==========================================

class GobangModel(nn.Module):
    def __init__(self, board_size: int, bound: int):
        super().__init__()
        self.bound = bound
        self.board_size = board_size
        self.actor = Actor(board_size)
        self.critic = Critic(board_size)
        
        # 学习率调度器
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor.optimizer, step_size=1000, gamma=0.5)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic.optimizer, step_size=1000, gamma=0.5)
        
        self.total_steps = 0 
        self.to(device)

    def forward(self, x, action):
        return self.actor(x), self.critic(x, action)

    def optimize(self, policy, qs, actions, rewards, next_qs, gamma, eps=1e-6):
        self.total_steps += 1
        
        # 1. Critic Loss
        targets = rewards + gamma * next_qs.detach()
        critic_loss = nn.MSELoss()(qs, targets)

        # 2. Advantage 计算与标准化
        raw_advantage = (targets - qs).detach()
        advantage = (raw_advantage - raw_advantage.mean()) / (raw_advantage.std() + 1e-8)

        # 3. Actor Loss
        indices = torch.tensor(
            [_position_to_index(self.board_size, r, c) for r, c in actions],
            device=device, dtype=torch.long
        )
        aimed_policy = policy[torch.arange(len(indices), device=device), indices]
        pg_loss = -torch.mean(torch.log(aimed_policy + eps) * advantage)

        # 动态熵权重：5000轮内从0.05衰减到0.01
        initial_entropy_coef = 0.05
        min_entropy_coef = 0.01
        entropy_coef = max(min_entropy_coef, initial_entropy_coef - self.total_steps * (initial_entropy_coef - min_entropy_coef) / 5000)

        entropy = -torch.sum(policy * torch.log(policy + eps), dim=1).mean()
        actor_loss = pg_loss - entropy_coef * entropy

        # 4. 反向传播
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        
        # 每100步更新调度器
        if self.total_steps % 100 == 0:
            self.actor_scheduler.step()
            self.critic_scheduler.step()

        return actor_loss.item(), critic_loss.item()

# ==========================================
# 训练执行入口
# ==========================================

if __name__ == "__main__":
    print(f"Current device is {device}.")

    try:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            mode="online" if args.use_wandb else "disabled",
            config={
                "num_episodes": num_episodes,
                "checkpoint": checkpoint,
                "board_size": 12,
                "bound": 5,
                "model_type": "ResNet-AC"
            }
        )
    except ImportError:
        pass

    agent = GobangModel(board_size=12, bound=5).to(device)

    # 调用 utils.py 中的训练函数
    train_model(
        agent,
        num_episodes=num_episodes,
        checkpoint=checkpoint
    )

    try:
        wandb.finish()
    except:
        pass