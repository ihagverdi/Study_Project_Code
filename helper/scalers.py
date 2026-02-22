import numpy as np

def max_scaling(y_train, *arrays):
    """
    Scales y_train and any number of other arrays by the max of y_train.
    Usage: y_train, y_val, y_test = max_scaling(y_train, y_val, y_test)
    """
    y_max = np.max(y_train)
    scale = 1 if y_max == 0 else (1.0 / y_max)
    
    y_train_scaled = y_train * scale
    
    # Apply scale to all other arrays
    processed_arrays = [arr * scale for arr in arrays]
    
    return (y_train_scaled, *processed_arrays)

def log_scaling(y_train, *arrays):
    """
    Applies log1p to y_train and any number of other arrays.
    Usage: y_train, y_val = log_scaling(y_train, y_val)
    """
    y_train_logged = np.log1p(y_train)
    
    # Apply log to all other arrays
    processed_arrays = [np.log1p(arr) for arr in arrays]
    
    return (y_train_logged, *processed_arrays)

def z_score_scaling(y_train, *arrays):
    """
    Standardizes y_train and any number of other arrays based on y_train statistics.
    Returns scaled arrays followed by mean and std.
    Usage: y_train, y_val, mean, std = z_score_scaling(y_train, y_val)
    """
    mean = np.mean(y_train)
    std = np.std(y_train)
    
    # Prevent division by zero
    if std == 0:
        std = 1.0

    y_train_scaled = (y_train - mean) / std
    
    # Apply standardization to all other arrays
    processed_arrays = [(arr - mean) / std for arr in arrays]
    
    # Return: (Train, Val, Test, ..., Mean, Std)
    return (y_train_scaled, *processed_arrays, mean, std)