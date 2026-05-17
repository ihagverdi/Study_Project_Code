import argparse
import pickle
import time

import numpy as np

# Project imports
from tabpfn_project.experiment_config import ExperimentConfig
from tabpfn_project.globals import (
    DISTNET_SCENARIOS, MODELS, TARGET_SCALES
)
from tabpfn_project.helper.utils import TargetScale, generate_experiment_id
from tabpfn_project.paths import RESULTS_DIR
from tabpfn_project.scripts.model_handler import BayesianDistNetHandler, DistNetHandler, GPHandler, LognormalHandler, RFHandler, TabPFNHandler
from tabpfn_project.scripts.prepare_data import prepare_datasets

def train_test_model(cfg: ExperimentConfig):
    save_dir = RESULTS_DIR / cfg.save_dir.lstrip('/\\')
    save_dir.mkdir(parents=True, exist_ok=True)

    X_train_flat, X_test, y_train_flat, y_test, train_group_ids_flat = prepare_datasets(cfg)
    rand_array = np.zeros(10)

    handlers = {
        'distnet': DistNetHandler(),
        'bayesian_distnet': BayesianDistNetHandler(),
        'tabpfn': TabPFNHandler(),
        'random_forest': RFHandler(),
        'lognormal': LognormalHandler(),
        'gp': GPHandler()
    }
    
    if cfg.model_name not in handlers:
        raise ValueError(f"Unsupported model: {cfg.model_name}")
    
    handler = handlers[cfg.model_name]
    res_dict, res_name = handler.run(cfg, X_train_flat, X_test, y_train_flat, y_test, train_group_ids_flat)

    meta_dir = save_dir / "metadata"
    meta_dir.mkdir(exist_ok=True)
    
    with open(meta_dir / res_name, 'wb') as f:
        pickle.dump(res_dict, f)
    print(f"Results saved to {meta_dir / res_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict algorithm runtime distribution.")
    parser.add_argument("--scenario", type=str, required=True, choices=DISTNET_SCENARIOS)
    parser.add_argument("--model", type=str, required=True, choices=MODELS)
    parser.add_argument("--remove_duplicates", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--fold", type=int, required=True, choices=range(10))
    parser.add_argument("--num_samples_per_instance", type=int, default=100)
    parser.add_argument("--target_scale", type=str, required=True, choices=TARGET_SCALES)
    parser.add_argument("--subsample_unflattened", action="store_true")
    parser.add_argument("--jitter_x", action="store_true")
    parser.add_argument("--jitter_val", type=float, default=None)
    parser.add_argument("--rand_extend_x", action="store_true")
    parser.add_argument("--n_rand_cols", type=int, default=None)
    parser.add_argument("--context_size", type=int, default=None)
    parser.add_argument("--feature_drop_rate", type=float, default=None)
    parser.add_argument("--n_features_keep", type=int, default=None)
    parser.add_argument("--seed_context_size", type=int, default=None)
    parser.add_argument("--seed_feature_drop_rate", type=int, default=None)
    parser.add_argument("--seed_samples_per_instance", type=int, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--load_dir", type=str, required=False, default=None)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--do_hpo", action="store_true")
    parser.add_argument("--rf_new_default", action="store_true")


    args = parser.parse_args()
    config = ExperimentConfig(
        scenario=args.scenario, model_name=args.model, fold=args.fold, save_dir=args.save_dir,
        num_samples_per_instance=args.num_samples_per_instance, context_size=args.context_size,
        use_cpu=args.use_cpu, target_scale=TargetScale.from_str(args.target_scale),
        subsample_unflattened=args.subsample_unflattened, early_stopping=args.early_stopping,
        seed_context_size=args.seed_context_size, seed_feature_drop_rate=args.seed_feature_drop_rate, feature_drop_rate=args.feature_drop_rate, seed_samples_per_instance=args.seed_samples_per_instance, do_hpo=args.do_hpo, oracle=args.oracle, remove_duplicates=args.remove_duplicates, jitter_x=args.jitter_x, rand_extend_x=args.rand_extend_x, n_rand_cols=args.n_rand_cols, jitter_val=args.jitter_val, n_features_keep=args.n_features_keep, rf_new_default=args.rf_new_default, load_dir=args.load_dir
    )
    
    start = time.perf_counter()
    train_test_model(config)
    print(f"✅ Experiment completed in {time.perf_counter() - start:.2f} seconds.")