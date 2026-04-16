from __future__ import annotations

import argparse
import pickle
from typing import Dict, Tuple, cast

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tabpfn_project.helper import data_source_release, load_data
from tabpfn_project.helper.bayesian_distnet import (
    CustomDataset,
    calculate_posthoc_metrics,
    device,
    exportData,
    parseConfig,
    preprocess_features,
    train,
)
from tabpfn_project.paths import DISTNET_DATA_DIR, RESULTS_DIR


def _select_kfold_split(num_instances: int, fold: int) -> Tuple[np.ndarray, np.ndarray]:
    """Replicate old_project KFold split order exactly."""
    idx = np.arange(num_instances)
    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    cntr = -1
    for train_idx, test_idx in kf.split(idx):
        # Legacy behavior: reset numpy seed in each split iteration.
        np.random.seed(2)
        cntr += 1
        if cntr != fold:
            continue
        return train_idx, test_idx

    raise ValueError(f"Fold {fold} is out of range for 10-fold CV.")


def _scale_test_fold_like_original(runtimes: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    """Replicate create_dataframes.getDataPerFold(mode='test') scaling for one fold."""
    y_tra_run = runtimes[train_idx]
    y_tst_run = runtimes[test_idx]
    y_max = np.max(y_tra_run)
    y_min = 0.0
    return (y_tst_run - y_min) / y_max


def reproduce_bayesian_distnet_results(
    scenario: str,
    fold: int,
    samples_per_instance: int,
    seed: int,
) -> Dict[str, object]:
    """Run the Bayesian DistNet reproduction pipeline and return metrics dictionary.

    This function mirrors the old experimental protocol:
    - data loading/cleanup from legacy loaders
    - 10-fold split with fixed random_state
    - train/validation split on unflattened train fold
    - legacy subsampling in CustomDataset/preprocess
    - legacy train loop via bayesian_distnet.train
    - post-hoc metrics via calculate_posthoc_metrics
    """

    if fold < 0 or fold > 9:
        raise ValueError("fold must be in [0, 9].")
    if samples_per_instance < 1 or samples_per_instance > 100:
        raise ValueError("samples_per_instance must be in [1, 100].")

    sc_dict = data_source_release.get_sc_dict(DISTNET_DATA_DIR)
    if scenario not in sc_dict:
        valid = ", ".join(sorted(sc_dict.keys()))
        raise ValueError(f"Unknown scenario '{scenario}'. Valid scenarios: {valid}")

    # Parse legacy Bayesian configuration exactly.
    training_config, model_config = parseConfig("BAYESIAN_LOGNORMAL")

    # Match old default device selection (cuda:0 when available, else cpu).
    device.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load cleaned scenario data with the same old retrieval settings.
    runtimes, features, _ = load_data.get_data(
        scenario=scenario,
        sc_dict=sc_dict,
        retrieve=sc_dict[scenario]["use"],
    )
    runtimes = np.asarray(runtimes)
    features = np.asarray(features)

    train_idx, test_idx = _select_kfold_split(runtimes.shape[0], fold)

    # Legacy train/valid split on unflattened train indices.
    split_ratio = float(cast(float, training_config["split_ratio"]))
    sp_r = 1.0 - split_ratio
    cut = int(len(train_idx) * sp_r)
    train_indices = train_idx[:cut]
    valid_indices = train_idx[cut:]

    # Build old-style flattened dataset and loaders (includes legacy subsampling by seed).
    dataset = CustomDataset(
        features=features,
        runtimes=runtimes,
        train_idx=train_indices,
        validate_idx=valid_indices,
        num_train_samples=samples_per_instance,
        seed=seed,
        lb=0,
        device_num=0,
    )
    train_sampler, valid_sampler = dataset.getTrainValidSubsetSampler()
    batch_size = int(cast(int, training_config["batch_size"]))
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    # Replicate old X_test preprocessing used by training exportData calls.
    X_train = features[train_indices, :]
    X_valid = features[valid_indices, :]
    X_test = features[test_idx, :]
    X_train, X_valid, X_test = preprocess_features(X_train, X_valid, X_test, scal="meanstd")

    model = train(
        num_features=int(dataset.getNumFeatures()),
        net_type="bayes_distnet",
        train_loader=train_loader,
        validation_loader=valid_loader,
        X_test=X_test,
        model_path=None,
        data_path_test=None,
        training_config=training_config,
        model_config=model_config,
    )

    # Old create_dataframes uses the latest periodically exported test predictions.
    artifacts = getattr(model, "_memory_training_artifacts", {})
    periodic_exports = artifacts.get("periodic_exports", {})
    if periodic_exports:
        export_epoch = max(periodic_exports.keys())
        model_params = periodic_exports[export_epoch]
    else:
        # Fallback should not normally trigger, but keeps output robust.
        export_epoch = -1
        model_params = exportData(X_test, None, model)

    # Scale test runtimes exactly as old create_dataframes mode='test'.
    y_test_scaled = _scale_test_fold_like_original(runtimes, train_idx, test_idx)

    metric_values = calculate_posthoc_metrics(
        model_params=model_params,
        fold_runtimes=y_test_scaled,
        dist="BAYESIAN_LOGNORMAL",
        net_type="bayes_distnet",
    )

    # Keep key names aligned with old reporting while also exposing NLL alias.
    result: Dict[str, object] = {
        "Scenario": scenario,
        "Num Samples": int(samples_per_instance),
        "LB": 0,
        "Seed": int(seed),
        "Fold": int(fold),
        "Mode": "test",
        "Model": "BayesianDistnet",
        "LLH": "Lognormal",
        "NLL": float(metric_values["NLLH"]),
        "NLLH": float(metric_values["NLLH"]),
        "P-KS": float(metric_values["P-KS"]),
        "D-KS": float(metric_values["D-KS"]),
        "D-B": float(metric_values["D-B"]),
        "KLD": float(metric_values["KLD"]),
        "Var": float(metric_values["Var"]),
        "Mass": float(metric_values["Mass"]),
        "export_epoch": int(export_epoch),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce Bayesian DistNet results using the original experimental protocol.",
    )
    parser.add_argument("--scenario", type=str, help="Scenario name, e.g., clasp_factoring")
    parser.add_argument("--fold", type=int, help="Fold index in [0, 9]")
    parser.add_argument("--samples_per_instance", type=int, help="Number of training samples per instance")
    parser.add_argument("--seed", type=int, help="Subsampling seed")
    parser.add_argument("--save_dir", type=str, required=True, help="Relative output directory under RESULTS_DIR")

    args = parser.parse_args()
    result = reproduce_bayesian_distnet_results(
        scenario=args.scenario,
        fold=args.fold,
        samples_per_instance=args.samples_per_instance,
        seed=args.seed,
    ); print(result)

    save_dir = RESULTS_DIR / args.save_dir.lstrip('/\\')
    save_dir.mkdir(parents=True, exist_ok=True)

    output_path = save_dir / (
        f"bayesian_distnet_results_{args.scenario}_"
        f"fold{args.fold}_samples{args.samples_per_instance}_seed{args.seed}.pkl"
    )
    # Use an explicit protocol for stable, reproducible serialization.
    with open(output_path, "wb") as f:
        pickle.dump(result, f, protocol=4)

    print(str(output_path))

if __name__ == "__main__":
    main()
