import torch
from utils import *
from submission import GobangModel

board_size = 12
bound = 5


def get_model():
    """
    Load and return the trained black-player model.
    """
    model = GobangModel(board_size=board_size, bound=bound)

    # 修改为你最终保存的模型路径
    model_path = "checkpoints/model_final.pth"

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


__all__ = ['get_model']
