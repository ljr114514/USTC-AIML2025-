
from utils import *
import numpy as np
import torch
import torch.nn as nn
from typing import *
import sys
import argparse

parser = argparse.ArgumentParser(description='args')
# Added default=1000 to prevent 'None' when arguments are missing
parser.add_argument('--num_episodes', type=int, default=1000, help='number of episodes')
parser.add_argument('--checkpoint', type=int, default=200, help='the interval of saving models')
parser.add_argument('--use_wandb', action='store_true', help='use wandb for experiment tracking (requires wandb installed)')
parser.add_argument('--wandb_project', type=str, default='aiml', help='wandb project name')
parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name')
args = parser.parse_args()
num_episodes = args.num_episodes
checkpoint = args.checkpoint


class Actor(nn.Module):
    """
    The actor is responsible for generating dependable policies to maximize the cumulative reward as much as possible.
    It takes a batch of arrays shaped either (B, 1, N, N) or (N, N) as input, and outputs a tensor shaped (B, N ** 2)
    as the generated policy.
    """

    def __init__(self, board_size: int, lr=1e-4):
        super().__init__()
        self.board_size = board_size
        """
        # Define your NN structures here. Torch modules have to be registered during the initialization process.
        # For example, you can define CNN structures as follows:
        ...
        """

        # BEGIN YOUR CODE
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        self.linear_blocks = nn.Sequential(
            nn.Linear(128 * board_size * board_size, board_size ** 2)
        )
        # END YOUR CODE

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: np.ndarray):
        if len(x.shape) == 2:
            output = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        else:
            output = torch.as_tensor(x, dtype=torch.float32, device=device)

        # BEGIN YOUR CODE
        x_features = self.conv_blocks(output)
        x_flat = self.flatten(x_features)
        logits = self.linear_blocks(x_flat)

        board_flat = output.view(output.size(0), -1)
        illegal_mask = board_flat != 0
        logits = logits.masked_fill(illegal_mask, -1e9)

        output = torch.softmax(logits, dim=1)
        # END YOUR CODE
        return output


class Critic(nn.Module):
    """
    The critic is responsible for generating dependable Q-values to fit the solution of Bellman Equations.
    """

    def __init__(self, board_size: int, lr=1e-4):
        super().__init__()
        self.board_size = board_size

        # BEGIN YOUR CODE
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        self.linear_blocks = nn.Sequential(
            nn.Linear(128 * board_size * board_size, board_size ** 2)
        )
        # END YOUR CODE

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: np.ndarray, action: np.ndarray):
        indices = torch.tensor(
            [_position_to_index(self.board_size, r, c) for r, c in action],
            device=device,
            dtype=torch.long
        )

        if len(x.shape) == 2:
            output = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        else:
            output = torch.as_tensor(x, dtype=torch.float32, device=device)

        # BEGIN YOUR CODE
        x_features = self.conv_blocks(output)
        x_flat = self.flatten(x_features)
        q_all = self.linear_blocks(x_flat)
        output = q_all.gather(1, indices.unsqueeze(1)).squeeze(1)
        # END YOUR CODE

        return output


class GobangModel(nn.Module):
    """
    Integrates Actor and Critic.
    """

    def __init__(self, board_size: int, bound: int):
        super().__init__()
        self.bound = bound
        self.board_size = board_size

        # BEGIN YOUR CODE
        self.actor = Actor(board_size)
        self.critic = Critic(board_size)
        # END YOUR CODE

        self.to(device)

    def forward(self, x, action):
        return self.actor(x), self.critic(x, action)

    def optimize(self, policy, qs, actions, rewards, next_qs, gamma, eps=1e-6):
        targets = rewards + gamma * next_qs.detach()
        critic_loss = nn.MSELoss()(qs, targets)

        indices = torch.tensor(
            [_position_to_index(self.board_size, r, c) for r, c in actions],
            device=device,
            dtype=torch.long
        )

        aimed_policy = policy[torch.arange(len(indices), device=device), indices]
        advantage = qs.detach() - qs.detach().mean()

        actor_loss = -torch.mean(torch.log(aimed_policy + eps) * advantage)

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        return actor_loss.item(), critic_loss.item()


if __name__ == "__main__":

    # ==============================
    # 统一、安全地初始化 wandb
    # ==============================
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
            }
        )
    except ImportError:
        pass

    agent = GobangModel(board_size=12, bound=5).to(device)

    train_model(
        agent,
        num_episodes=num_episodes,
        checkpoint=checkpoint
    )

    try:
        wandb.finish()
    except:
        pass



