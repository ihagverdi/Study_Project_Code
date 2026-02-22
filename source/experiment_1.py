import os
import argparse
import pickle
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import torch
import time
from helper import data_source_release, load_data
from helper.preprocess import preprocess_features

RANDOM_STATE=0

def subsample_training_data(X_train_flat, y_train_flat, context_size, seed, subsample_method):
    """
    Subsamples the dataset to maximize instance diversity.
    
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
    Randomly drops features from the dataset based on the specified drop rate.
    
    Args:
        X_train: (n_samples, n_features)
        *arrays: Additional arrays with shape (n_samples, n_features) to subsample features from
        drop_rate: Fraction of features to drop (0.0 to 1.0)
        seed: Random seed for reproducibility.

    Returns:
        tuple(X_train_sub, X_valid_sub, X_test_sub)


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
    train_idx, test_idx = splits[fold]

    #------------------------------------#
    
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = runtimes[train_idx], runtimes[test_idx]

    if num_samples_per_instance != 100:  # 100 is the full data
        print(f"Subsampling the training data to {num_samples_per_instance} samples per instance (without replacement).")
        assert 1 <= num_samples_per_instance <= 100, "num_samples_per_instance must be between 1 and 100"
        rng = np.random.default_rng(seed=100)
        subsample_idx = rng.choice(runtimes.shape[1], size=num_samples_per_instance, replace=False)
        y_train = y_train[:, subsample_idx]
    

    # Flatten the whole training set
    X_train_flat = np.repeat(X_train, repeats=num_samples_per_instance, axis=0)
    y_train_flat = y_train.reshape(-1, 1)

    # context sampling
    if context_size is not None:
        assert seed_context is not None, "seed_context must be provided when context_size is specified."
        print(f"Subsampling the training data to context size {context_size} using method '{subsample_method}'")
        X_train_flat, y_train_flat = subsample_training_data(X_train_flat, y_train_flat, context_size=context_size, seed=seed_context, subsample_method=subsample_method)
            
    if feature_drop_rate is not None:
        assert seed_features is not None, "seed_features must be provided when feature_drop_rate > 0.0"
        assert feature_drop_rate > 0.0 and feature_drop_rate < 1.0, "feature_drop_rate must be in (0.0, 1.0)"
        print(f"Sampling features with drop rate {feature_drop_rate}")
        X_train_flat, X_test = subsample_features(X_train_flat, X_test, drop_rate=feature_drop_rate, seed=seed_features)
    
    # X_train shape: (n, d)
    # y_train shape: (n, 1)
    if model_name == 'distnet':
        from helper.distnet_lognormal import DistNetModel
        from helper.distnet_helpers import calculate_nllh
        from helper.scalers import max_scaling

        early_stopping = X_train_flat.shape[0] >= 256 and early_stopping  # at least 50 validation samples
        print(f"early stopping: {early_stopping}")
        # preprocess features
        if early_stopping:
            X_train_flat, X_valid_flat, y_train_flat, y_valid_flat = train_test_split(X_train_flat, y_train_flat, test_size=0.2, random_state=RANDOM_STATE)
            X_train_flat, X_valid_flat, X_test = preprocess_features(X_train_flat, X_valid_flat, X_test, scal="meanstd")
        else:
            X_train_flat, X_test = preprocess_features(X_train_flat, X_test, scal="meanstd")

        # scale the targets
        assert target_scale in ['max'], "DistNet only supports 'max' scaling currently."

        if target_scale == 'max':
            if early_stopping:
                y_train_flat, y_valid_flat, y_test = max_scaling(y_train_flat, y_valid_flat, y_test)
            else:
                y_train_flat, y_test = max_scaling(y_train_flat, y_test)

        # give the model a name and pass it a path to save
        distnet_model_name = f"model_{model_name}_{scenario}_{fold}_{seed_context}_{seed_features}_{feature_drop_rate}_{context_size}_{target_scale}_{subsample_method}_{num_samples_per_instance}.pt"
        os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
        distnet_save_path = os.path.join(save_dir, "models", distnet_model_name)
        # initialize and train model
        model = DistNetModel(n_input_features=X_train_flat.shape[1],
                            n_epohcs=n_epochs,
                            batch_size=batch_size,
                            wc_time_limit=wc_time_limit,
                            save_path=distnet_save_path,
                            X_valid=None if not early_stopping else X_valid_flat,
                            y_valid=None if not early_stopping else y_valid_flat,
                            early_stopping=early_stopping,
                            early_stopping_patience=50)  # patience is active only if early_stopping=True
        
        # train model
        model.train(X_train_flat, y_train_flat)
        # evaluate model
        y_pred = model.predict(X_test)
        nllh = calculate_nllh(y_test, y_pred, target_scale=target_scale)

        print(f"DistNet Test NLLH: {nllh:.4f}")

    elif model_name == 'tabpfn':
        from tabpfn import TabPFNRegressor
        from helper.scalers import log_scaling, max_scaling, z_score_scaling
        from helper.pfn_helpers import calculate_nllh_pfn
        
        # Scale y (runtime) values
        args = None
        if target_scale == 'max':
            y_train_flat, y_test = max_scaling(y_train_flat, y_test)
        elif target_scale == 'log':
            y_train_flat, y_test = log_scaling(y_train_flat, y_test)
        elif target_scale == "z-score":
            y_train_flat, y_test, mean, std = z_score_scaling(y_train_flat, y_test)
            args = [mean, std]

        # no preprocessing of features for TabPFN

        # initialize and train model
        if use_cpu:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"TabPFN using device: {device}")
            
        model = TabPFNRegressor(device=device, random_state=RANDOM_STATE, ignore_pretraining_limits=True)

        model.fit(X_train_flat, y_train_flat.ravel())

        # evaluate model
        nllh = calculate_nllh_pfn(model, X_test, y_test, validation_batch_size=val_batch_size, target_scale=target_scale, args=args)

        print(f"TabPFN Test NLLH: {nllh:.4f}")
        

    results_dict = {
    'metrics': {
        'nllh': nllh
    }}

    if model_name == 'distnet':
        results_dict['distnet'] = {}
        results_dict['distnet']['config'] = {
            'model_name': model_name,
            'scenario': scenario,
            'fold': fold,
            'seed_context': seed_context,
            'seed_features': seed_features,
            'feature_drop_rate': feature_drop_rate,
            'context_size': context_size,
            'target_scale': target_scale,
            'subsample_method': subsample_method,
            'num_samples_per_instance': num_samples_per_instance,
            'use_cpu': use_cpu,
            'early_stopping': early_stopping,
            'save_dir': save_dir,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'wc_time_limit': wc_time_limit
        }

    elif model_name == 'tabpfn':
        results_dict['tabpfn'] = {}
        results_dict['tabpfn']['config'] = {
            'model_name': model_name,
            'scenario': scenario,
            'fold': fold,
            'seed_context': seed_context,
            'seed_features': seed_features,
            'feature_drop_rate': feature_drop_rate,
            'context_size': context_size,
            'target_scale': target_scale,
            'subsample_method': subsample_method,
            'num_samples_per_instance': num_samples_per_instance,
            'use_cpu': use_cpu,
            'save_dir': save_dir,
            'val_batch_size': val_batch_size
        }

    # 2. Build filename and save
    results_file_name = f"{model_name}_{scenario}_{fold}_{seed_context}_{seed_features}_{feature_drop_rate}_{context_size}_{target_scale}_{subsample_method}_{num_samples_per_instance}_{'cpu' if use_cpu else 'gpu'}.pkl"
    os.makedirs(os.path.join(save_dir, "metadata"), exist_ok=True)
    results_save_path = os.path.join(save_dir, "metadata", results_file_name)

    with open(results_save_path, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"Results saved to {results_save_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Model for Given Scenario (DistNet or TabPFN)')

    parser.add_argument('--scenario', type=str, help='Scenario name (e.g., lpg-zeno, clasp_factoring)')
    parser.add_argument('--model', type=str, help='model type to train (distnet or tabpfn)')
    parser.add_argument('--fold', type=int, help='Cross-validation fold index (0-9)')
    parser.add_argument('--num_samples_per_instance', type=int, default=100, help='Number of training samples per instance (1-100)')
    parser.add_argument('--val_batch_size', type=int, default=1000, help='Validation batch size for TabPFN (default: 1000)')
    parser.add_argument('--target_scale', type=str, help='Target scaling method (log, z-score, max)')
    parser.add_argument('--subsample_method', type=str, help='Sampling method (instance-wise, flatten-random)')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping during training for DistNet')
    parser.add_argument('--context_size', type=int, default=None, help='Number of flattened training samples (default: None, use all)')
    parser.add_argument('--feature_drop_rate', type=float, default=None, help='Feature drop rate (default: None, use all features)')
    parser.add_argument('--seed_context', type=int, default=None, help='Random seed for sampling context')
    parser.add_argument('--seed_features', type=int, default=None, help='Random seed for sampling features')
    parser.add_argument('--save_dir', type=str, help='Directory to save the trained model')
    parser.add_argument('--use_cpu', action='store_true', help='add this flag to use CPU instead of GPU (if applicable)')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of training epochs for DistNet')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DistNet training')
    parser.add_argument('--wc_time_limit', type=int, default=3600, help='Wall clock time limit for DistNet training (seconds)')


    args = parser.parse_args()
    
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
        wc_time_limit=args.wc_time_limit
    )
    end = time.time()
    print(f"Total execution time: {end - start:.2f} seconds")