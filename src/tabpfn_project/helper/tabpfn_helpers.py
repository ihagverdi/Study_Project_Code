from typing import Any, Dict, List, Tuple
import torch
import numpy as np
from tabpfn_project.helper.utils import dict_to_cpu
from tabpfn_project.helper.y_scalers import log1p_scaling
from tabpfn_project.scripts.model_handler import track_gpu_memory_and_time

class TabPFNEnsembleManager:
    """
    Abstracts the logic for generating diverse datasets and training 
    multiple TabPFN models.
    """
    @staticmethod
    def generate_datasets(
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        ensemble_size: int, 
        strategy: str, 
        jitter_val: float, 
        random_state: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        
        rng = np.random.default_rng(random_state)
        datasets = []
        N = X_train.shape[0]
        sample_size = N // ensemble_size
        
        for i in range(ensemble_size):
            if strategy == "subagging":
                # Implementation of rigorous sampling WITH replacement (Bootstrapping)
                indices = rng.choice(N, size=sample_size, replace=True)
                datasets.append((X_train[indices], y_train[indices]))
                
            elif strategy == "jitter_ensemble":
                # Ensure each model gets a unique geometric projection via independent noise
                from tabpfn_project.helper.utils import add_feature_jitter
                seed = int(rng.integers(0, 10**9))
                X_ens = add_feature_jitter(X=X_train, jitter_intensity=jitter_val, random_state=seed)
                datasets.append((X_ens, y_train.copy()))
                
            else:
                raise ValueError(f"Unknown ensemble strategy: {strategy}")
                
        return datasets

    @staticmethod
    def train_and_predict(
        datasets: List[Tuple[np.ndarray, np.ndarray]], 
        X_test: np.ndarray, 
        device: torch.device, 
        random_state: int
    ) -> Tuple[List[Any], Dict[str, List[Any]]]:
        
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        from tabpfn_project.helper.tabpfn_helpers import batch_predict_tabpfn
        from tabpfn_project.globals import TABPFN_VAL_BATCH_SIZE
        
        all_preds = []
        mem_time_stats = {"fit": [], "predict":[]}
        
        for i, (X_tr, y_tr) in enumerate(datasets):
            model = TabPFNRegressor.create_default_for_version(
                ModelVersion.V2_5, 
                ignore_pretraining_limits=True, 
                device=device, 
                random_state=random_state
            )
            
            with track_gpu_memory_and_time(device) as f_stats:
                model.fit(X_tr, y_tr.ravel())
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
