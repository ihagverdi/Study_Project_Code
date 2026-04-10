from typing import Tuple

import numpy as np

from tabpfn_project.experiment_config import ExperimentConfig
from tabpfn_project.helper.load_data import load_distnet_data
from tabpfn_project.helper.preprocess import del_constant_features
from tabpfn_project.helper.utils import subsample_features, subsample_flattened_data, subsample_targets_per_instance
from tabpfn_project.paths import DISTNET_DATA_DIR


def prepare_datasets(cfg: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Handles loading, flattening, and feature subsampling."""
    X_train, X_test, y_train, y_test = load_distnet_data(DISTNET_DATA_DIR, cfg.scenario, cfg.fold)

    instance_ids = np.arange(X_train.shape[0])

    # 1. Target Subsampling
    if cfg.num_samples_per_instance != 100:
        assert 1 <= cfg.num_samples_per_instance <= 100
        assert cfg.seed_samples_per_instance is not None
        y_train = subsample_targets_per_instance(y_train, cfg.num_samples_per_instance, cfg.seed_samples_per_instance)
    
    # 2. Flattening
    X_train_flat = np.repeat(X_train, repeats=cfg.num_samples_per_instance, axis=0)
    y_train_flat = y_train.reshape(-1, 1)

    instance_ids_flat = np.repeat(instance_ids, repeats=cfg.num_samples_per_instance)

    # 3. Context Subsampling
    if cfg.context_size is not None:
        assert cfg.seed_context_size is not None
        X_train_flat, y_train_flat, instance_ids_flat = subsample_flattened_data(
            X_train_flat, y_train_flat, instance_ids_flat, context_size=cfg.context_size, 
            seed=cfg.seed_context_size, subsample_method=cfg.subsample_method
        )
    
    # 4. Feature Cleaning
    X_train_flat, X_test = del_constant_features(X_train_flat, X_test)

    # 5. Feature Dropping / Agnostic mode
    drop_rate = 1.0 if cfg.feature_agnostic else cfg.feature_drop_rate
    if drop_rate and drop_rate > 0.0:
        seed = -1 if (cfg.feature_agnostic or drop_rate == 1.0) else cfg.seed_feature_drop_rate
        X_train_flat, X_test = subsample_features(X_train_flat, X_test, drop_rate=drop_rate, seed=seed)

    return X_train_flat, X_test, y_train_flat, y_test, instance_ids_flat
