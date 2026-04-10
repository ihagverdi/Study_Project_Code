from typing import Any, Dict, List
import torch
import numpy as np
from tabpfn_project.helper.utils import dict_to_cpu
from tabpfn_project.helper.y_scalers import log1p_scaling

def batch_predict_tabpfn(
    model: Any, 
    X_test: np.ndarray, 
    validation_batch_size: int
) -> List[Dict[str, Any]]:
    """
    Batch predictor for TabPFN model to prevent VRAM overflow.

    Args:
        model: The TabPFN model instance.
        X_test: Input features for testing, shape (B, D).
        validation_batch_size: Number of instances to process per batch.
    
    Returns:
        A list of dictionaries containing model predictions moved to CPU.
    """
    if validation_batch_size <= 0:
        raise ValueError("validation_batch_size must be a positive integer")

    n_instances = X_test.shape[0]
    tabpfn_preds = []
    with torch.inference_mode():
        for start in range(0, n_instances, validation_batch_size):
            X_batch = X_test[start : start + validation_batch_size]
            
            # Generate predictions with full distribution output
            preds = model.predict(X_batch, output_type="full")
            
            # Move tensors to CPU immediately to prevent GPU memory accumulation
            tabpfn_preds.append(dict_to_cpu(preds))
    return tabpfn_preds

def oracle_predict_tabpfn(
    model: Any, 
    y_test_original: np.ndarray,
    target_scale: str,
) -> List[Dict[str, Any]]:
    """
    Oracle predictor: Fits the model on the ground truth targets of each 
    individual instance and predicts for that instance.

    Args:
        model: The TabPFN model instance.
        y_test_scaled: Scaled targets, shape (B, O).
    
    Returns:
        A list of dictionaries containing oracle predictions moved to CPU.
    """
    assert target_scale in ['log', 'original'], "target_scale must be either 'log' or 'original'"
    y_test_scaled = (log1p_scaling(y_test_original)[0] if target_scale == 'log' else y_test_original)
    tabpfn_preds = []

    for inst in y_test_scaled:
        # Create dummy features (zeros) for the current instance
        X_temp = np.zeros((inst.shape[0], 1))
        
        # Fit the model specifically on the targets for this instance
        model.fit(X_temp, inst)

        with torch.inference_mode():
            # Predict using the first dummy feature entry as the representative for this instance
            preds = model.predict(X_temp[:1], output_type="full")
            tabpfn_preds.append(dict_to_cpu(preds))
    
    return tabpfn_preds
