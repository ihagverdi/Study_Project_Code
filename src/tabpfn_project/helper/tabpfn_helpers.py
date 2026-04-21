from typing import Any, Dict, List, Tuple
import torch
import numpy as np
from tabpfn_project.helper.preprocess import Z_score_features, del_constant_features
from tabpfn_project.helper.utils import dict_to_cpu, subsample_data
from tabpfn_project.helper.y_scalers import log1p_scaling
from tabpfn_project.scripts.model_handler import track_gpu_memory_and_time

class TabPFNEnsembleManager:
    """
    Abstracts the logic for generating diverse datasets and training 
    multiple TabPFN models.
    """
    @staticmethod
    def generate_datasets(
        X_train_flat: np.ndarray,
        X_test: np.ndarray,
        y_train_flat: np.ndarray,
        n_views: int, 
        target_scale: str,
        context_size: int,
        context_seed: int,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        
        datasets = []
        
        for i in range(n_views):
            # Create a unique seed for each ensemble member to ensure diversity
            member_seed = context_seed + i
            
            X_train_sub, y_train_sub = subsample_data(
                X_train_flat, y_train_flat, 
                context_size=context_size, seed=member_seed, with_replacement=True
            )

            X_train_sub, X_test_sub = del_constant_features(X_train_sub, X_test)
            # X_train_sub, X_test_sub = Z_score_features(X_train_sub, X_test_sub)
            y_train_sub_scaled = (log1p_scaling(y_train_sub)[0] if target_scale == 'log' else y_train_sub)
            
            datasets.append((X_train_sub, X_test_sub, y_train_sub_scaled))
                
        return datasets

    @staticmethod
    def train_and_predict(
        datasets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        device: torch.device,
        random_state: int
    ) -> Tuple[List[Any], Dict[str, List[Any]]]:
        
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        from tabpfn_project.globals import TABPFN_VAL_BATCH_SIZE
        
        all_preds = []
        mem_time_stats = {"fit": [], "predict":[]}
        
        for i, (X_train, X_test, y_train) in enumerate(datasets):
            model = TabPFNRegressor.create_default_for_version(
                ModelVersion.V2_5, 
                ignore_pretraining_limits=True, 
                device=device, 
                random_state=random_state
            )
            
            with track_gpu_memory_and_time(device) as f_stats:
                model.fit(X_train, y_train.ravel())
            mem_time_stats["fit"].append(f_stats)
            
            with track_gpu_memory_and_time(device) as p_stats:
                preds = batch_predict_tabpfn(model, X_test, validation_batch_size=TABPFN_VAL_BATCH_SIZE)
            mem_time_stats["predict"].append(p_stats)
            
            all_preds.append(preds)
            
        return all_preds, mem_time_stats

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
