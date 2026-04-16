import contextlib
import gc
import pickle
import time
import numpy as np
import torch
from typing import Dict

# Project imports
from tabpfn_project.experiment_config import ExperimentConfig
from tabpfn_project.helper.preprocess import preprocess_features
from tabpfn_project.globals import (
    MIN_CLAMP_LLH, N_GRID_POINTS, RANDOM_STATE, 
)
from tabpfn_project.helper.utils import generate_experiment_id
from tabpfn_project.helper.y_scalers import max_scaling, log1p_scaling
from tabpfn_project.paths import RESULTS_DIR

from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical
from smac import HyperparameterOptimizationFacade, Scenario

@contextlib.contextmanager
def track_gpu_memory_and_time(device_input):
    is_cuda = False
    try:
        device = torch.device(device_input)
        if device.type == 'cuda' and torch.cuda.is_available():
            is_cuda = True
    except Exception:
        pass

    stats = {"baseline_mb": 0.0, "peak_mb": 0.0, "spike_mb": 0.0, "time_s": 0.0}
    if is_cuda:
        gc.collect() 
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        baseline_mem_bytes = torch.cuda.memory_allocated(device)
    
    start_time = time.perf_counter()
    yield stats
    
    if is_cuda:
        torch.cuda.synchronize(device)
        
    stats["time_s"] = time.perf_counter() - start_time
    if is_cuda:
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
        stats["baseline_mb"] = baseline_mem_bytes / (1024 ** 2)
        stats["peak_mb"] = peak_mem_bytes / (1024 ** 2)
        stats["spike_mb"] = (peak_mem_bytes - baseline_mem_bytes) / (1024 ** 2)

class BaseModelHandler:
    def run(self, cfg: ExperimentConfig, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, instance_ids: np.ndarray) -> Dict:
        raise NotImplementedError

class DistNetHandler(BaseModelHandler):
    def run(self, cfg: ExperimentConfig, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, instance_ids: np.ndarray):
        from tabpfn_project.helper.distnet_lognormal import DistNetModel
        from tabpfn_project.helper.calculate_metrics import calculate_metrics_distnet
        from sklearn.model_selection import GroupShuffleSplit
        
        assert cfg.target_scale in ['max', 'log'], "Invalid target_scale for DistNet. Only 'max' and 'log' are supported."
        assert instance_ids is not None, "instance_ids must be provided to prevent data leakage."

        device = torch.device('cuda' if (torch.cuda.is_available() and not cfg.use_cpu) else 'cpu')
        best_epoch, y_scale = None, None
        # Early Stopping Logic
        if cfg.early_stopping:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
            tr_idx, val_idx = next(gss.split(X_train, y_train, groups=instance_ids))
            
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            
            X_tr, X_val, X_test = preprocess_features(X_tr, X_val, X_test, scal="meanstd")
            y_tr, y_val, y_scale = (max_scaling(y_tr, y_val) if cfg.target_scale == 'max' else (*log1p_scaling(y_tr, y_val), None))
            model = DistNetModel(model_target_scale=cfg.target_scale, n_input_features=X_tr.shape[1], n_epochs=cfg.n_epochs, batch_size=cfg.batch_size, 
                                 wc_time_limit=cfg.wc_time_limit, X_valid=X_val, y_valid=y_val, 
                                 early_stopping=True, early_stopping_patience=50, random_state=RANDOM_STATE)
            X_train, y_train = X_tr, y_tr
        else:
            X_train, X_test = preprocess_features(X_train, X_test, scal="meanstd")
            y_train, y_scale = (max_scaling(y_train) if cfg.target_scale == 'max' else (*log1p_scaling(y_train), None))
            model = DistNetModel(model_target_scale=cfg.target_scale, n_input_features=X_train.shape[1], n_epochs=cfg.n_epochs, batch_size=cfg.batch_size, 
                                 wc_time_limit=cfg.wc_time_limit, early_stopping=False, random_state=RANDOM_STATE)

        fit_start = time.perf_counter()
        model.train(X_train, y_train)
        fit_time = time.perf_counter() - fit_start
        
        if cfg.early_stopping: best_epoch = model.best_epoch

        pred_start = time.perf_counter()
        y_pred = model.predict(X_test)
        pred_time = time.perf_counter() - pred_start

        metrics_sum, inst_sum = calculate_metrics_distnet(
            y_test, y_pred, device=device, target_scale=cfg.target_scale, y_scaler=y_scale, N_grid_points=N_GRID_POINTS
        ); print(f"DistNet Metrics Summary scale={cfg.target_scale}: {metrics_sum}")

        return {
            'y_test_preds': y_pred,
            'result_metrics': {'metrics_summary': metrics_sum, 'instance_summary': inst_sum},
            'model_specific_info': {
                'model_config': model.model.state_dict(), 'best_epoch': best_epoch, 'fit_time': fit_time, 
                'predict_time': pred_time, 'y_scale': y_scale, 'n_epochs': cfg.n_epochs
            }
        }

class BayesianDistNetHandler(BaseModelHandler):
    def run(self, cfg: ExperimentConfig, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, instance_ids: np.ndarray):
        from tabpfn_project.helper.bayesian_distnet import BayesianDistNetModel
        from tabpfn_project.helper.calculate_metrics import calculate_metrics_distnet
        from sklearn.model_selection import GroupShuffleSplit

        assert cfg.target_scale in ['max'], "Invalid target_scale for BayesianDistNet. Only 'max' is supported."
        assert cfg.early_stopping, "Early stopping must be enabled for BayesianDistNet."
        assert instance_ids is not None, "instance_ids must be provided to prevent data leakage."

        device = torch.device('cpu')

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        tr_idx, val_idx = next(gss.split(X_train, y_train, groups=instance_ids))
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        X_tr, X_val, X_test = preprocess_features(X_tr, X_val, X_test, scal="meanstd")

        print(f"Early stopping enabled for BayesianDistNet.")
        y_tr, y_val, y_scale = max_scaling(y_tr, y_val)

        model = BayesianDistNetModel(
            n_input_features=X_tr.shape[1],
            X_valid=X_val,
            y_valid=y_val,
        )

        fit_start = time.perf_counter()
        model.train(X_tr, y_tr)
        fit_time = time.perf_counter() - fit_start

        best_epoch = model.best_epoch

        pred_start = time.perf_counter()
        y_pred = model.predict(X_test)
        pred_time = time.perf_counter() - pred_start

        metrics_sum, inst_sum = calculate_metrics_distnet(
            y_test, y_pred, device=device, target_scale='max', y_scaler=y_scale, N_grid_points=N_GRID_POINTS
        )
        
        print(f"BayesianDistNet Metrics Summary scale=max: {metrics_sum}")

        legacy_posthoc = {}
        try:
            scaled_y_test = y_test * y_scale
            model.predict_samples(X_test)
            legacy_posthoc = model.evaluate_posthoc_metrics(scaled_y_test)
        except Exception as exc:
            legacy_posthoc = {'error': str(exc)}

        if model.model is None:
            raise RuntimeError("BayesianDistNet model is not initialized after training.")

        return {
            'y_test_preds': y_pred,
            'result_metrics': {'metrics_summary': metrics_sum, 'instance_summary': inst_sum},
            'model_specific_info': {
                'model_config': model.model.state_dict(),
                'best_epoch': best_epoch,
                'fit_time': fit_time,
                'predict_time': pred_time,
                'y_scale': y_scale,
                'n_epochs': model.n_epochs,
                'legacy_posthoc_metrics': legacy_posthoc,
            }
        }

class TabPFNHandler(BaseModelHandler):
    def run(self, cfg: ExperimentConfig, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, instance_ids: np.ndarray):
        from tabpfn import TabPFNRegressor
        from tabpfn_project.helper.tabpfn_helpers import batch_predict_tabpfn, oracle_predict_tabpfn
        from tabpfn_project.helper.calculate_metrics import calculate_metrics_tabpfn
        from tabpfn.constants import ModelVersion

        assert cfg.target_scale in ['log', 'original']
        y_train_scaled = (log1p_scaling(y_train)[0] if cfg.target_scale == 'log' else y_train)
        
        device = torch.device('cuda' if (torch.cuda.is_available() and not cfg.use_cpu) else 'cpu')
        model = TabPFNRegressor.create_default_for_version(ModelVersion.V2_5, ignore_pretraining_limits=True, device=device, random_state=RANDOM_STATE)

        mem_stats = {"fit": {}, "predict": {}}
        
        if not cfg.oracle:
            with track_gpu_memory_and_time(device) as stats:
                model.fit(X_train, y_train_scaled.ravel())
            mem_stats["fit"] = stats

            with track_gpu_memory_and_time(device) as stats:
                preds = batch_predict_tabpfn(model, X_test, validation_batch_size=cfg.val_batch_size)
        else:
            with track_gpu_memory_and_time(device) as stats:
                preds = oracle_predict_tabpfn(model, y_test, target_scale=cfg.target_scale)

        mem_stats["predict"] = stats
        metrics_sum, inst_sum = calculate_metrics_tabpfn(
            y_test, preds, device=device, target_scale=cfg.target_scale, N_grid_points=N_GRID_POINTS
        ); print(f"TabPFN Metrics Summary scale={cfg.target_scale}: {metrics_sum}")

        # Save Predictions file separately as per original logic
        self._save_preds(cfg, preds)

        return {
            'y_test_preds': None,
            'result_metrics': {'metrics_summary': metrics_sum, 'instance_summary': inst_sum},
            'model_specific_info': {'mem_time_stats': mem_stats, 'val_batch_size': cfg.val_batch_size}
        }

    def _save_preds(self, cfg, preds):
        exp_id = generate_experiment_id(cfg)
        filename = f"tabpfn_{exp_id}_test_preds.pkl"
        
        path = RESULTS_DIR / cfg.save_dir.lstrip('/\\') / "tabpfn_preds_full"
        path.mkdir(parents=True, exist_ok=True)
        with open(path / filename, 'wb') as f:
            pickle.dump(preds, f)

class LognormalHandler(BaseModelHandler):
    def run(self, cfg: ExperimentConfig, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, instance_ids: np.ndarray):
        from tabpfn_project.helper.calculate_metrics import calculate_metrics_lognormal
        device = torch.device('cuda' if (torch.cuda.is_available() and not cfg.use_cpu) else 'cpu')

        if cfg.oracle:
            y_test_t = torch.as_tensor(y_test, dtype=torch.float32, device=device)
            log_y = torch.log(y_test_t)
            mu, sigma = torch.mean(log_y, dim=1, keepdim=True), torch.std(log_y, dim=1, keepdim=True)
        else:
            y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
            log_y = torch.log(y_train_t)
            mu, sigma = torch.mean(log_y), torch.std(log_y)
            
        dist = torch.distributions.LogNormal(loc=mu, scale=sigma)
        metrics_sum, inst_sum = calculate_metrics_lognormal(y_test, dist, device=device, N_grid_points=N_GRID_POINTS)
        print(f"Lognormal Metrics Summary scale={cfg.target_scale}: {metrics_sum}")

        return {
            'y_test_preds': [mu, sigma],
            'result_metrics': {'metrics_summary': metrics_sum, 'instance_summary': inst_sum},
            'model_specific_info': {}
        }

class RFHandler(BaseModelHandler):
    def run(self, cfg: ExperimentConfig, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, instance_ids: np.ndarray):
        from tabpfn_project.helper.random_forest import RuntimePredictionRandomForest
        from sklearn.model_selection import GroupKFold
        from tabpfn_project.helper.calculate_metrics import calculate_metrics_random_forest
        
        device = torch.device("cuda" if (torch.cuda.is_available() and not getattr(cfg, 'use_cpu', False)) else "cpu")
        assert cfg.target_scale == 'log', "Target scale must be 'log'"
        
        do_hpo = getattr(cfg, 'do_hpo', False)
        
        if do_hpo:
            cs = ConfigurationSpace()
            cs.add([
                Integer("n_estimators", (10, 50), default=10),
                Float("max_features", (0.4, 1.0), default=0.5),
                Integer("min_samples_split", (2, 5), log=True, default=5),
                Float("var_min", (1e-5, 1e-1), log=True, default=0.01),
                Categorical("bootstrap", [True, False], default=False)
            ])

            def smac_objective(config, seed=0) -> float:
                """
                Inner optimization loop via GroupKFold.
                Predictions and evaluations are done directly on all validation observations.
                """
                gkf = GroupKFold(n_splits=3)
                metrics_per_fold =[]
                
                for in_train_idx, in_val_idx in gkf.split(X_train, y_train, groups=instance_ids):
                    X_in_train_raw_split = X_train[in_train_idx]
                    X_in_val_raw_split = X_train[in_val_idx]
                    
                    y_in_train = y_train[in_train_idx]
                    y_in_val = y_train[in_val_idx]
                    
                    # 1. Preprocess purely on inner split (Leakage-Safe)
                    X_in_train_scaled, X_in_val_scaled = preprocess_features(
                        X_in_train_raw_split, X_in_val_raw_split, scal="meanstd"
                    )
                    
                    # Apply log1p scaling to inner train targets
                    y_in_train_scaled = log1p_scaling(y_in_train)[0]
                    
                    # 2. Fit Candidate Model
                    model = RuntimePredictionRandomForest(
                        random_state=seed, 
                        n_estimators=config["n_estimators"],
                        max_features=config["max_features"],
                        min_samples_split=config["min_samples_split"],
                        bootstrap=config["bootstrap"],
                        var_min=config["var_min"],
                    )
                    model.fit(X_in_train_scaled, y_in_train_scaled)
                    
                    # 3. Direct Prediction on ALL validation rows
                    means, vars_ = model.predict(X_in_val_scaled)
                    
                    m_t = torch.as_tensor(means, dtype=torch.float32, device=device)
                    v_t = torch.as_tensor(vars_, dtype=torch.float32, device=device)
                    
                    # 4. Compute NLL strictly per prediction & aggregate (No bias correction)
                    z_in_val = torch.log1p(torch.as_tensor(y_in_val, dtype=torch.float32, device=device))
                    
                    dist = torch.distributions.Normal(loc=m_t, scale=torch.sqrt(v_t))
                    llh = dist.log_prob(z_in_val).clamp(min=MIN_CLAMP_LLH)
                    
                    # Compute mean NLL over the validation set
                    nllh = -llh.mean()
                    metrics_per_fold.append(nllh.item())
                    
                return float(np.mean(metrics_per_fold))

            # SMAC3 requires n_trials, use a large limit so that hpo_time serves as the actual cap
            n_trials = getattr(cfg, 'hpo_n_trials', 999999)
            hpo_time = getattr(cfg, 'hpo_time', 3600)

            scenario = Scenario(
                configspace=cs,
                deterministic=True,
                n_trials=n_trials,
                walltime_limit=hpo_time,
                n_workers=1,
            )
            
            smac = HyperparameterOptimizationFacade(scenario=scenario, target_function=smac_objective)
            print(f"Starting SMAC3 HPO (Time budget: {hpo_time}s)...")
            best_config = smac.optimize()
            best_params = dict(best_config)
            print(f"HPO Complete. Best hyperparameters: {best_params}")

        else:
            # HPO is skipped, default to specified base parameters
            best_params = {
                "n_estimators": 10,
                "max_features": 0.5,
                "min_samples_split": 5,
                "bootstrap": False,
                "var_min": 0.01
            }
            print(f"HPO disabled. Using default parameters: {best_params}")

        # --- Final Model Training ---
        # Outer preprocessing step applied safely now that HPO is resolved
        X_train_final, X_test_final = preprocess_features(X_train, X_test, scal="meanstd")
        y_train_final = log1p_scaling(y_train)[0]

        final_model = RuntimePredictionRandomForest(
            random_state=RANDOM_STATE,
            n_estimators=best_params["n_estimators"],
            max_features=best_params["max_features"],
            min_samples_split=best_params["min_samples_split"],
            bootstrap=best_params["bootstrap"], 
            var_min=best_params["var_min"],
        )

        start = time.perf_counter()
        final_model.fit(X_train_final, y_train_final)
        fit_time = time.perf_counter() - start

        # Predict strictly on test
        pred_start = time.perf_counter()
        means, vars_ = final_model.predict(X_test_final)
        pred_time = time.perf_counter() - pred_start

        metrics_sum, inst_sum = calculate_metrics_random_forest(
            y_test, (means, vars_), device=device, N_grid_points=N_GRID_POINTS
        )
        
        print(f"="*20)
        for k,v in metrics_sum.items():
            print(f"{k}: {v:.4f}")
        print(f"="*20)

        return {
            'y_test_preds': [means, vars_],
            'result_metrics': {'metrics_summary': metrics_sum, 'instance_summary': inst_sum},
            'model_specific_info': {
                'fit_time': fit_time, 
                'predict_time': pred_time,
                'best_hyperparameters': best_params
            }
        }