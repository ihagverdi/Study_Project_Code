from typing import Tuple
import numpy as np
from tabpfn_project.experiment_config import ExperimentConfig
from tabpfn_project.helper.load_data import load_distnet_data
from tabpfn_project.helper.utils import subsample_features, subsample_data, subsample_targets_per_instance

def prepare_datasets(cfg: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares the datasets applying structural adjustments based on the provided configuration.

    Args:
        cfg (ExperimentConfig): The experiment configuration containing all necessary parameters.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (X_train_flat, X_test, y_train_flat, y_test, train_group_ids_flat)

    """
    X_train, X_test, y_train, y_test = load_distnet_data(cfg.scenario, cfg.fold)
    train_group_ids = np.arange(X_train.shape[0])

    # subsample targets per instance if specified
    if cfg.num_samples_per_instance != 100:
        assert 1 <= cfg.num_samples_per_instance <= 100, f"num_samples_per_instance must be between 1 and 100, got {cfg.num_samples_per_instance}"
        assert cfg.seed_samples_per_instance is not None
        if cfg.num_samples_per_instance < 100:
            y_train = subsample_targets_per_instance(y_train, cfg.num_samples_per_instance, cfg.seed_samples_per_instance)

    # subsample from unflattened data (on instance level) if specified
    if cfg.subsample_unflattened and cfg.context_size is not None:
        assert cfg.seed_context_size is not None
        X_train, y_train, train_group_ids = subsample_data(
            X_train, y_train, train_group_ids, context_size=cfg.context_size, 
            seed=cfg.seed_context_size, with_replacement=False
        )

    # flatten data for model input
    X_train_flat = np.repeat(X_train, repeats=cfg.num_samples_per_instance, axis=0)
    y_train_flat = y_train.reshape(-1, 1)
    train_group_ids_flat = np.repeat(train_group_ids, repeats=cfg.num_samples_per_instance)

    # subsample from flattened data (on sample level) if specified
    if not cfg.subsample_unflattened and cfg.context_size is not None:
        assert cfg.seed_context_size is not None
        X_train_flat, y_train_flat, train_group_ids_flat = subsample_data(
            X_train_flat, y_train_flat, train_group_ids_flat, context_size=cfg.context_size, 
            seed=cfg.seed_context_size, with_replacement=True
        )
    
    # feature dropping if specified
    if cfg.feature_drop_rate is not None or cfg.n_features_keep is not None:
    
        if cfg.feature_drop_rate is not None:
            assert 0.0 <= cfg.feature_drop_rate <= 1.0, \
                f"feature_drop_rate must be in [0.0, 1.0], got {cfg.feature_drop_rate}"
        
        if cfg.n_features_keep is not None:
            assert cfg.n_features_keep >= 0, \
                f"n_features_keep must be non-negative, got {cfg.n_features_keep}"

        assert cfg.seed_feature_drop_rate is not None, "Seed for feature dropping must be provided."

        X_train_flat, X_test = subsample_features(
            X_train_flat, 
            X_test, 
            drop_rate=cfg.feature_drop_rate, 
            n_features_keep=cfg.n_features_keep, 
            seed=cfg.seed_feature_drop_rate
        )

    return X_train_flat, X_test, y_train_flat, y_test, train_group_ids_flat
