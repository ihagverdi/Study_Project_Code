import os
import argparse
import pickle
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import torch
import time
from helper.tabpfn_vs_distnet_helpers import data_source_release, load_data
from helper.tabpfn_vs_distnet_helpers.preprocess import preprocess_features

RANDOM_STATE=0  # From original distnet paper

def subsample_training_data(X_train_flat, y_train_flat, context_size, seed, subsample_method):
    """
    Subsamples the flattened dataset. Currently supports 'flatten-random' which randomly samples from the flattened training data.
    
    Args:
        X_train_flat: (n_instances, n_features)
        y_train_flat: (n_instances, 1)
        context_size: Total number of (X, y) pairs to return.
        seed: Random seed for reproducibility.
        subsample_method: Strategy for sampling ('flatten-random')
        
    Returns:
        X_out: (context_size, n_features)
        y_out: (context_size, 1)
    """
    rng = np.random.default_rng(seed)
    n_samples = X_train_flat.shape[0]
    
    if subsample_method == 'flatten-random':
        selected_indices = rng.choice(n_samples, size=context_size, replace=True)
        X_out = X_train_flat[selected_indices]
        y_out = y_train_flat[selected_indices]
        return X_out, y_out

def subsample_features(X_train, *arrays, drop_rate, seed):
    """
    Randomly samples a subset of features from the input arrays based on the specified drop rate.
    
    Args:
        X_train: (n_samples, n_features)
        *arrays: Additional arrays with shape (n_samples, n_features) to subsample features from
        drop_rate: Fraction of features to drop (0.0 to 1.0)
        seed: Random seed for reproducibility.

    Returns:
        Tuple of subsampled arrays, with the same order as input (X_train, *arrays)

    """

    rng = np.random.default_rng(seed=seed)
    n_features = X_train.shape[1]
    feature_idx = rng.choice(n_features, size=int(n_features * (1 - drop_rate)), replace=False)

    processed_arrays = [arr[:, feature_idx] for arr in arrays]

    return (X_train[:, feature_idx], *processed_arrays)

def train_test_model(
    model_name,
    scenario,
    fold,
    save_dir,
    num_samples_per_instance,
    context_size,
    use_cpu,
    target_scale,
    subsample_method,
    early_stopping,
    seed_context,
    seed_features,
    feature_drop_rate,
    val_batch_size,
    seed_samples_per_instance,
    n_epochs,
    batch_size,
    wc_time_limit,
):
    assert 0 <= fold <= 9, "Fold must be between 0 and 9"
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get scenario configuration and data
    sc_dict = data_source_release.get_sc_dict()
    data_dir = data_source_release.get_data_dir()
    
    if scenario not in sc_dict.keys():
        raise ValueError(f"Invalid scenario: {scenario}. Must be one of {list(sc_dict.keys())}")
    
    # Load data
    runtimes, features, _ = load_data.get_data(
        scenario=scenario, 
        data_dir=data_dir,
        sc_dict=sc_dict,
        retrieve=sc_dict[scenario]['use']
    )
    
    features = np.asarray(features)
    runtimes = np.asarray(runtimes)

    # Get CV splits
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    splits = list(kf.split(np.arange(features.shape[0])))
    train_idx, test_idx = splits[fold]  # process the specified fold

    #------------------------------------#
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = runtimes[train_idx], runtimes[test_idx]
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test), "X and y must have the same length."
    assert y_train.shape[1] == 100 and y_test.shape[1] == 100, "Data must have 100 runtime observations per instance."

    del features, runtimes  # free memory

    if num_samples_per_instance != 100:  # 100 is the full data
        print(f"Subsampling the training data to {num_samples_per_instance} samples per instance (without replacement).")
        assert 1 <= num_samples_per_instance <= 100, "num_samples_per_instance must be between 1 and 100"
        assert seed_samples_per_instance is not None, "seed_samples_per_instance must be provided"
        rng = np.random.default_rng(seed=seed_samples_per_instance)
        subsample_idx = rng.choice(y_train.shape[1], size=num_samples_per_instance, replace=False)
        y_train = y_train[:, subsample_idx]
    

    # Flatten the whole training set
    X_train_flat = np.repeat(X_train, repeats=num_samples_per_instance, axis=0)
    y_train_flat = y_train.reshape(-1, 1)

    if context_size is not None:
        assert seed_context is not None, "seed_context must be provided when context_size is specified."
        assert 0 < context_size <= X_train_flat.shape[0], "invalid context_size value."
        if context_size < X_train_flat.shape[0]:
            print(f"Subsampling the training data to context size {context_size} using method '{subsample_method}'")
            X_train_flat, y_train_flat = subsample_training_data(X_train_flat, y_train_flat, context_size=context_size, seed=seed_context, subsample_method=subsample_method)
            
    if feature_drop_rate is not None:
        assert seed_features is not None, "seed_features must be provided when feature_drop_rate > 0.0"
        assert 0.0 < feature_drop_rate < 1.0, "feature_drop_rate must be in (0.0, 1.0)"
        print(f"Sampling features with drop rate {feature_drop_rate}")
        X_train_flat, X_test = subsample_features(X_train_flat, X_test, drop_rate=feature_drop_rate, seed=seed_features)
    
    results_dict = None  # the ultimate dict to store after model fit&predict
    if model_name == 'distnet':
        from helper.tabpfn_vs_distnet_helpers.distnet_lognormal import DistNetModel
        from helper.tabpfn_vs_distnet_helpers.distnet_helpers import calculate_nllh_distnet
        from helper.tabpfn_vs_distnet_helpers.scalers import max_scaling

        assert target_scale in ['max'], "DistNet only supports 'max' scaling currently."

        N = X_train_flat.shape[0]
        early_stopping_patience = 50
        E_final = None
        if early_stopping and N < 512:
            # 5-fold CV to find the optimal epoch, then retrain on the full training set.
            print(f"[DistNet] N={N} < 512: running 5-fold CV to determine E_final.")
            kf_inner = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            best_epochs = []
            for cv_fold, (tr_idx, val_idx) in enumerate(kf_inner.split(X_train_flat), start=1):
                X_f_tr, X_f_val = X_train_flat[tr_idx], X_train_flat[val_idx]
                y_f_tr, y_f_val = y_train_flat[tr_idx], y_train_flat[val_idx]
                X_f_tr, X_f_val = preprocess_features(X_f_tr, X_f_val, scal="meanstd")
                y_f_tr, y_f_val = max_scaling(y_f_tr, y_f_val)
                fold_model = DistNetModel(
                    n_input_features=X_f_tr.shape[1],
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    wc_time_limit=wc_time_limit,
                    save_path=None,
                    X_valid=X_f_val,
                    y_valid=y_f_val,
                    early_stopping=True,
                    early_stopping_patience=early_stopping_patience,
                    random_state=RANDOM_STATE,
                )
                fold_model.train(X_f_tr, y_f_tr)
                best_epochs.append(fold_model.best_epoch)
                print(f"  [CV {cv_fold}/5] best_epoch={fold_model.best_epoch}")
            E_final = int(round(np.mean(best_epochs)))
            print(f"[DistNet] best_epochs={best_epochs} -> E_final={E_final}")

            # Retrain on the full training set for exactly E_final epochs (no hold-out).
            X_train_flat, X_test = preprocess_features(X_train_flat, X_test, scal="meanstd")
            y_train_flat, y_test = max_scaling(y_train_flat, y_test)
            model = DistNetModel(
                n_input_features=X_train_flat.shape[1],
                n_epochs=E_final,
                batch_size=batch_size,
                wc_time_limit=wc_time_limit,
                save_path=None,
                early_stopping=False,
                random_state=RANDOM_STATE,
            )

        elif early_stopping and N >= 512:
            # Direct early stopping on an 80/20 hold-out split.
            print(f"[DistNet] N={N} >= 512: using direct early stopping on an 80/20 hold-out split.")
            X_train_flat, X_valid_flat, y_train_flat, y_valid_flat = train_test_split(
                X_train_flat, y_train_flat, test_size=0.2, random_state=RANDOM_STATE
            )
            X_train_flat, X_valid_flat, X_test = preprocess_features(X_train_flat, X_valid_flat, X_test, scal="meanstd")
            y_train_flat, y_valid_flat, y_test = max_scaling(y_train_flat, y_valid_flat, y_test)
            model = DistNetModel(
                n_input_features=X_train_flat.shape[1],
                n_epochs=n_epochs,
                batch_size=batch_size,
                wc_time_limit=wc_time_limit,
                save_path=None,
                X_valid=X_valid_flat,
                y_valid=y_valid_flat,
                early_stopping=True,
                early_stopping_patience=early_stopping_patience,
                random_state=RANDOM_STATE,
            )

        else:
            # No early stopping: train on the full training set for n_epochs.
            X_train_flat, X_test = preprocess_features(X_train_flat, X_test, scal="meanstd")
            y_train_flat, y_test = max_scaling(y_train_flat, y_test)
            model = DistNetModel(
                n_input_features=X_train_flat.shape[1],
                n_epochs=n_epochs,
                batch_size=batch_size,
                wc_time_limit=wc_time_limit,
                save_path=None,
                early_stopping=False,
                random_state=RANDOM_STATE,
            )

        distnet_fit_time_start = time.time()
        model.train(X_train_flat, y_train_flat)
        distnet_fit_time = time.time() - distnet_fit_time_start

        # For direct early stopping, record the best epoch found.
        if early_stopping and N >= 512:
            E_final = model.best_epoch

        distnet_predict_time_start = time.time()
        y_pred = model.predict(X_test)
        distnet_predict_time = time.time() - distnet_predict_time_start

        nllh = calculate_nllh_distnet(y_test, y_pred, target_scale=target_scale)

        results_dict = {
            'model_name': 'distnet',
            'model_config': model.model.state_dict(),
            'scenario': scenario,
            'fold': fold,
            'seed_context': seed_context,
            'seed_features': seed_features,
            'seed_samples_per_instance': seed_samples_per_instance,
            'feature_drop_rate': feature_drop_rate,
            'context_size': context_size,
            'target_scale': target_scale,
            'subsample_method': subsample_method,
            'num_samples_per_instance': num_samples_per_instance,
            'use_cpu': use_cpu,
            'early_stopping': early_stopping,
            'early_stopping_patience': early_stopping_patience,
            'E_final': E_final,
            'save_dir': save_dir,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'wc_time_limit': wc_time_limit,
            'test_preds': y_pred,
            'random_state': RANDOM_STATE,
            'result_metrics': {
                'nllh': nllh,
                'fit_time': distnet_fit_time,
                'predict_time': distnet_predict_time,
            }
        }
        print(f"DistNet Test NLLH: {nllh:.4f}, fit and predict time: {(distnet_fit_time+distnet_predict_time):.2f} seconds.")


    elif model_name == 'tabpfn':
        from tabpfn import TabPFNRegressor
        from helper.tabpfn_vs_distnet_helpers.scalers import log_scaling, max_scaling, z_score_scaling
        from helper.tabpfn_vs_distnet_helpers.tabpfn_helpers import predict_and_calculate_nllh_tabpfn
        
        # Scale y (runtime) values
        args = None
        if target_scale == 'max':
            y_train_flat, y_test = max_scaling(y_train_flat, y_test)
        elif target_scale == 'log':
            y_train_flat, y_test = log_scaling(y_train_flat, y_test)
        elif target_scale == "z-score":
            y_train_flat, y_test, mean, std = z_score_scaling(y_train_flat, y_test)
            args = [mean, std]
        elif target_scale == "none":
            pass  # no scaling

        # no preprocessing of features for TabPFN
        # initialize and train model
        device = torch.device('cuda' if (torch.cuda.is_available() and not use_cpu) else 'cpu')
        print(f"TabPFN using device: {device}")

        model = TabPFNRegressor(device=device, random_state=RANDOM_STATE, ignore_pretraining_limits=True)

        tabpfn_fit_time_start = time.time()
        model.fit(X_train_flat, y_train_flat.ravel())
        tabpfn_fit_time = time.time() - tabpfn_fit_time_start

        # evaluate model
        nllh, tabpfn_preds, tabpfn_predict_time = predict_and_calculate_nllh_tabpfn(model, X_test, y_test, validation_batch_size=val_batch_size, target_scale=target_scale, args=args)

        results_dict = {
            'model_name': 'tabpfn',
            'model_config': model.__dict__,
            'scenario': scenario,
            'fold': fold,
            'seed_context': seed_context,
            'seed_features': seed_features,
            'seed_samples_per_instance': seed_samples_per_instance,
            'feature_drop_rate': feature_drop_rate,
            'context_size': context_size,
            'target_scale': target_scale,
            'subsample_method': subsample_method,
            'num_samples_per_instance': num_samples_per_instance,
            'use_cpu': use_cpu,
            'save_dir': save_dir,
            'val_batch_size': val_batch_size,
            'test_preds': tabpfn_preds,
            'random_state': RANDOM_STATE,
            'result_metrics': {
                'nllh': nllh,
                'fit_time': tabpfn_fit_time,
                'predict_time': tabpfn_predict_time,
            }
        }
        print(f"TabPFN Test NLLH: {nllh:.4f}, fit and predict time: {(tabpfn_fit_time+tabpfn_predict_time):.2f} seconds.")
        

    assert results_dict is not None, "results_dict is NONE"
    results_file_name = (f"{model_name}_{scenario}_{fold}_{seed_context}_{seed_features}_{seed_samples_per_instance}_{feature_drop_rate}_"
                         f"{context_size}_{target_scale}_{subsample_method}_{num_samples_per_instance}_{early_stopping}_{'cpu' if use_cpu else 'gpu'}.pkl")

    os.makedirs(os.path.join(save_dir, "metadata"), exist_ok=True)
    results_save_path = os.path.join(save_dir, "metadata", results_file_name)

    with open(results_save_path, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"Results saved to {results_save_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Model for Given Scenario (DistNet or TabPFN)')

    parser.add_argument('--scenario', type=str, required=True, help='Scenario name (e.g., lpg-zeno, clasp_factoring)')
    parser.add_argument('--model', type=str, required=True, help='model type to train (distnet or tabpfn)')
    parser.add_argument('--fold', type=int, required=True, help='Cross-validation fold index (0-9)')
    parser.add_argument('--num_samples_per_instance', type=int, default=100, help='Number of training samples per instance (1-100)')
    parser.add_argument('--val_batch_size', type=int, default=1000, help='Validation batch size for TabPFN (default: 1000)')
    parser.add_argument('--target_scale', type=str, required=True, help='Target scaling method (log, z-score, max, none)')
    parser.add_argument('--subsample_method', type=str, default=None, help='Sampling method (instance-wise, flatten-random)')
    parser.add_argument('--context_size', type=int, default=None, help='Number of flattened training samples (default: None, use all)')
    parser.add_argument('--feature_drop_rate', type=float, default=None, help='Feature drop rate (default: None, use all features)')
    parser.add_argument('--seed_context', type=int, default=None, help='Random seed for sampling context')
    parser.add_argument('--seed_features', type=int, default=None, help='Random seed for sampling features')
    parser.add_argument('--seed_samples_per_instance', type=int, default=None, help='Random seed for sampling samples per instance')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the results')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Max training epochs for DistNet (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=16, help='Mini-batch size for DistNet training (default: 16)')
    parser.add_argument('--wc_time_limit', type=int, default=60*59, help='Wall-clock time limit in seconds per DistNet training run (default: 3540)')
    parser.add_argument('--early_stopping', action='store_true', help='Enable adaptive epoch search (find_optimal_epochs) for DistNet')
    parser.add_argument('--use_cpu', action='store_true', help='add this flag to use CPU instead of GPU (if applicable)')

    args = parser.parse_args()
    
    print(f"🧪 Starting the experiment with the following configuration:\n{args}")
    start = time.time()
    train_test_model(
        model_name=args.model,
        scenario=args.scenario,
        fold=args.fold,
        save_dir=args.save_dir,
        num_samples_per_instance=args.num_samples_per_instance,
        context_size=args.context_size,
        use_cpu=args.use_cpu,
        target_scale=args.target_scale,
        subsample_method=args.subsample_method,
        early_stopping=args.early_stopping,
        seed_context=args.seed_context,
        seed_features=args.seed_features,
        feature_drop_rate=args.feature_drop_rate,
        val_batch_size=args.val_batch_size,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        wc_time_limit=args.wc_time_limit,
        seed_samples_per_instance=args.seed_samples_per_instance
    )
    end = time.time()
    print(f"✅ Experiment completed in {end - start:.2f} seconds.")