import torch
from utils import *
from submission import GobangModel

board_size = 12
bound = 5


def get_opponent():
    """
    Load and return the trained white-player opponent model.
    """
    opponent = GobangModel(board_size=board_size, bound=bound)

    # 修改为你最终保存的对手模型路径
    opponent_path = "checkpoints/model_final.pth"

    state_dict = torch.load(opponent_path, map_location=device)
    opponent.load_state_dict(state_dict)

    opponent.to(device)
    opponent.eval()
    return opponent


__all__ = ['get_opponent']

