from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    """Container for all hyperparameters to avoid long function signatures."""
    scenario: str
    model_name: str
    fold: int
    save_dir: str
    num_samples_per_instance: int = 100
    context_size: Optional[int] = None
    use_cpu: bool = False
    target_scale: Optional[str] = None
    subsample_method: str = 'flatten-random'
    early_stopping: bool = False
    seed_context_size: Optional[int] = None
    seed_feature_drop_rate: Optional[int] = None
    feature_drop_rate: Optional[float] = None
    val_batch_size: int = 1000
    seed_samples_per_instance: Optional[int] = None
    n_epochs: int = 1000
    batch_size: int = 16
    wc_time_limit: int = 3540
    do_hpo: bool = False
    hpo_time: Optional[int] = None
    feature_agnostic: bool = False
    oracle: bool = False
    remove_duplicates: bool = False