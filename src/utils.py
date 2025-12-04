"""
utils.py
--------
Purpose:
- Common utility functions (e.g., set_seed for reproducibility)
"""

import random
import os
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across Python, NumPy, and environment variables
    to ensure reproducible results.

    Args:
        seed (int): Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
