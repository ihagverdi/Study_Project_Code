from typing import Tuple

import numpy as np

def max_scaling(y_train, *arrays):
    """
    Scales y_train and any number of other arrays by the max of y_train.
    Returns scaled arrays followed by the scale factor.
    """
    y_max = np.max(y_train)
    scale = 1.0 if y_max == 0 else (1.0 / y_max)
    
    y_train_scaled = y_train * scale
    
    # Apply scale to all other arrays
    processed_arrays = [arr * scale for arr in arrays]
    
    return y_train_scaled, *processed_arrays, scale

def log1p_scaling(y_train, *arrays):
    """
    Applies log1p to y_train and any number of other arrays.
    Usage: y_train, y_val = log_scaling(y_train, y_val)
    """
    y_train_logged = np.log1p(y_train)
    
    # Apply log to all other arrays
    processed_arrays = [np.log1p(arr) for arr in arrays]
    
    return y_train_logged, *processed_arrays
