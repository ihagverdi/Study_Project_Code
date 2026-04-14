import argparse
import pickle
import time

# Project imports
from tabpfn_project.experiment_config import ExperimentConfig
from tabpfn_project.globals import (
    DISTNET_SCENARIOS, MODELS,
    SUBSAMPLE_METHOD_CHOICES, TARGET_SCALES
)
from tabpfn_project.helper.utils import generate_experiment_id
from tabpfn_project.paths import RESULTS_DIR
from tabpfn_project.scripts.model_handler import DistNetHandler, LognormalHandler, RFHandler, TabPFNHandler
from tabpfn_project.scripts.preprocessing import prepare_datasets

def train_test_model(cfg: ExperimentConfig):
    # 1. Setup Directory
    save_dir = RESULTS_DIR / cfg.save_dir.lstrip('/\\')
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. Data Pipeline
    X_train, X_test, y_train, y_test, instance_ids = prepare_datasets(cfg)

    # 3. Model Execution
    handlers = {
        'distnet': DistNetHandler(),
        'tabpfn': TabPFNHandler(),
        'random_forest': RFHandler(),
        'lognormal': LognormalHandler()
    }
    
    if cfg.model_name not in handlers:
        raise ValueError(f"Unsupported model: {cfg.model_name}")
    
    handler = handlers[cfg.model_name]
    model_results = handler.run(cfg, X_train, X_test, y_train, y_test, instance_ids)

    # 4. Standardize Results Dictionary
    results_dict = {
        'model_name': cfg.model_name,
        'scenario': cfg.scenario,
        'fold': cfg.fold,
        'seed_context_size': cfg.seed_context_size,
        'seed_feature_drop_rate': cfg.seed_feature_drop_rate,
        'seed_samples_per_instance': cfg.seed_samples_per_instance,
        'feature_drop_rate': cfg.feature_drop_rate,
        'context_size': cfg.context_size,
        'target_scale': cfg.target_scale,
        'subsample_method': cfg.subsample_method,
        'num_samples_per_instance': cfg.num_samples_per_instance,
        'use_cpu': cfg.use_cpu,
        'save_dir': str(save_dir),
        'n_samples': X_train.shape[0],
        'n_features': X_train.shape[1],
        'instance_ids': instance_ids,
        'feature_agnostic': cfg.feature_agnostic,
        'remove_duplicates': cfg.remove_duplicates,
        'oracle': cfg.oracle,
        'do_hpo': cfg.do_hpo,
        'hpo_time': cfg.hpo_time,
        'hpo_results': {},
        **model_results
    }

    # 5. Save Metadata
    exp_id = generate_experiment_id(cfg)
    res_filename = f"{exp_id}_metadata.pkl"
    
    meta_dir = save_dir / "metadata"
    meta_dir.mkdir(exist_ok=True)
    
    with open(meta_dir / res_filename, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"Results saved to {meta_dir / res_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict algorithm runtime distribution.")
    parser.add_argument("--scenario", type=str, required=True, choices=DISTNET_SCENARIOS)
    parser.add_argument("--model", type=str, required=True, choices=MODELS)
    parser.add_argument("--feature_agnostic", action="store_true")
    parser.add_argument("--remove_duplicates", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--fold", type=int, required=True, choices=range(10))
    parser.add_argument("--num_samples_per_instance", type=int, default=100)
    parser.add_argument("--val_batch_size", type=int, default=1000)
    parser.add_argument("--target_scale", type=str, default=None, choices=TARGET_SCALES)
    parser.add_argument("--subsample_method", type=str, default='flatten-random', choices=SUBSAMPLE_METHOD_CHOICES)
    parser.add_argument("--context_size", type=int, default=None)
    parser.add_argument("--feature_drop_rate", type=float, default=None)
    parser.add_argument("--seed_context_size", type=int, default=None)
    parser.add_argument("--seed_feature_drop_rate", type=int, default=None)
    parser.add_argument("--seed_samples_per_instance", type=int, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--wc_time_limit", type=int, default=3540)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--do_hpo", action="store_true")
    parser.add_argument("--hpo_time", type=int, default=None)

    args = parser.parse_args()
    config = ExperimentConfig(
        scenario=args.scenario, model_name=args.model, fold=args.fold, save_dir=args.save_dir,
        num_samples_per_instance=args.num_samples_per_instance, context_size=args.context_size,
        use_cpu=args.use_cpu, target_scale=args.target_scale, subsample_method=args.subsample_method,
        early_stopping=args.early_stopping, seed_context_size=args.seed_context_size,
        seed_feature_drop_rate=args.seed_feature_drop_rate, feature_drop_rate=args.feature_drop_rate,
        val_batch_size=args.val_batch_size, n_epochs=args.n_epochs, batch_size=args.batch_size,
        wc_time_limit=args.wc_time_limit, seed_samples_per_instance=args.seed_samples_per_instance,
        do_hpo=args.do_hpo, hpo_time=args.hpo_time, feature_agnostic=args.feature_agnostic, oracle=args.oracle,
        remove_duplicates=args.remove_duplicates
    )
    
    start = time.perf_counter()
    train_test_model(config)
    print(f"✅ Experiment completed in {time.perf_counter() - start:.2f} seconds.")