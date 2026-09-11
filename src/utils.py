import os
import sys
import re
import gc
import random
import logging

import numpy as np
import pandas as pd
import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clear_vram(var_names=None):
    if var_names:
        for var_name in var_names:
            if var_name in globals():
                del globals()[var_name]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("VRAM cleared successfully!")


def mapk(actual, predicted, k=3):
    score = 0.0
    for i in range(len(actual)):
        for j in range(k):
            if predicted[i][j] == actual[i]:
                score += 1.0 / (j + 1)
                break
    return score / len(actual)


def normalize_matrix(mat):
    row_min = mat.min(axis=1, keepdims=True)
    row_max = mat.max(axis=1, keepdims=True)
    return (mat - row_min) / np.clip(row_max - row_min, 1e-9, None)
