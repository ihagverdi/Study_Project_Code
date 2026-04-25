from dataclasses import dataclass
from typing import Optional

from tabpfn_project.helper.utils import TargetScale


@dataclass
class ExperimentConfig:
    """Container for all hyperparameters to avoid long function signatures."""
    scenario: str
    model_name: str
    fold: int
    save_dir: str
    target_scale: TargetScale
    num_samples_per_instance: int = 100
    context_size: Optional[int] = None
    use_cpu: bool = False
    subsample_unflattened: bool = False
    jitter_x: bool = False
    jitter_val: Optional[float] = None
    rand_extend_x: bool = False
    n_rand_cols: Optional[int] = None
    early_stopping: bool = False
    seed_context_size: Optional[int] = None
    seed_feature_drop_rate: Optional[int] = None
    feature_drop_rate: Optional[float] = None
    n_features_keep: Optional[int] = None
    seed_samples_per_instance: Optional[int] = None
    do_hpo: bool = False
    oracle: bool = False
    remove_duplicates: bool = False