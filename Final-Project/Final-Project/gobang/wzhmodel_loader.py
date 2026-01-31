import torch
from utils import *
from submissiona import GobangModel

board_size = 12
bound = 5


def get_model():
    """
    Load and return the trained black-player model.
    """

    
    model_type = "cnn"   # 如果你用的是 mlp / cnn，请改成对应的

    model = GobangModel(
        board_size=board_size,
        bound=bound,
        #model_type=model_type
    )

    model_path = "checkpoints/model_11999.pth"

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)   # strict=True 默认，结构一致就不会炸

    model.to(device)
    model.eval()
    
    return model


__all__ = ["get_model"]
