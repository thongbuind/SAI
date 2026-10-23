import random
import numpy as np
import torch

def seed_everything(seed: int):
    """Đặt seed cho python, numpy (RNG toàn cục) và torch (CPU + mọi GPU)."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
