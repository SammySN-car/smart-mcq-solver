"""
Utility functions for Smart MCQ Solver.
"""
import os
import re
import gc
import random
import logging
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clear_vram(variables: Dict[str, Any]) -> None:
    """Clear variables from memory and empty CUDA cache."""
    for name, var in variables.items():
        del var
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("VRAM cleared successfully")


def mapk(actual: List[str], predicted: List[List[str]], k: int = 3) -> float:
    """Calculate Mean Average Precision at K.
    
    Args:
        actual: List of ground truth answers.
        predicted: List of lists of predicted answers (ranked).
        k: Number of top predictions to consider.
    
    Returns:
        MAP@3 score.
    """
    score = 0.0
    for i in range(len(actual)):
        for j in range(k):
            if predicted[i][j] == actual[i]:
                score += 1.0 / (j + 1)
                break
    return score / len(actual)


def normalize_matrix(mat: np.ndarray) -> np.ndarray:
    """Normalize a matrix row-wise to [0, 1]."""
    row_min = mat.min(axis=1, keepdims=True)
    row_max = mat.max(axis=1, keepdims=True)
    return (mat - row_min) / np.clip(row_max - row_min, 1e-9, None)
