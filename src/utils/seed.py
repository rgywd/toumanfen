"""随机种子设置模块"""

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    设置随机种子以确保实验可复现

    Args:
        seed: 随机种子值
        deterministic: 是否使用确定性算法(可能影响性能)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # PyTorch 1.8+
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass

    except ImportError:
        pass  # PyTorch not installed


def get_random_state(seed: Optional[int] = None) -> np.random.RandomState:
    """
    获取NumPy随机状态对象

    Args:
        seed: 随机种子

    Returns:
        随机状态对象
    """
    return np.random.RandomState(seed)
