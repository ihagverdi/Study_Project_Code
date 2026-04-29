import pickle
import time
import numpy as np
import torch
from typing import Dict

# Project imports
from tabpfn_project.experiment_config import ExperimentConfig
from tabpfn_project.helper.preprocess import preprocess_feats
from tabpfn_project.globals import (
    LLH_EPSILON, N_GRID_POINTS, RANDOM_STATE 
)
from tabpfn_project.helper.utils import TargetScale, generate_experiment_id, sample_k_per_instance, track_gpu_memory_and_time
from tabpfn_project.helper.y_scalers import max_scaling, log1p_scaling
from tabpfn_project.paths import RESULTS_DIR

from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Constant, EqualsCondition
from smac import HyperparameterOptimizationFacade, Scenario

class BaseModelHandler:
    def run(self, cfg: ExperimentConfig, X_train_flat: np.ndarray, X_test: np.ndarray, y_train_flat: np.ndarray, y_test: np.ndarray, train_group_ids_flat: np.ndarray) -> Dict:
        raise NotImplementedError

class DistNetHandler(BaseModelHandler):
    def run(self, cfg: ExperimentConfig, X_train_flat: np.ndarray, X_test: np.ndarray, y_train_flat: np.ndarray, y_test: np.ndarray, train_group_ids_flat: np.ndarray) -> Dict:
        from tabpfn_project.helper.distnet_lognormal import DistNetModel
        from tabpfn_project.helper.calculate_metrics import calculate_metrics_distnet
        from sklearn.model_selection import GroupShuffleSplit
        from tabpfn_project.globals import DISTNET_N_EPOCHS, DISTNET_BATCH_SIZE, DISTNET_WCT, DISTNET_ES_PATIENCE
        
        assert cfg.target_scale.value in ['max', 'log'], "Invalid target_scale for DistNet. Only 'max' and 'log' are supported."
        device = torch.device('cuda' if (torch.cuda.is_available() and not cfg.use_cpu) else 'cpu')
        best_epoch, y_scaler = None, None
        X_val, y_val, y_train_scaled = None, None, None
        # Early Stopping Logic
        if cfg.early_stopping:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
            tr_idx, val_idx = next(gss.split(X_train_flat, y_train_flat, groups=train_group_ids_flat))
            
            X_tr, X_val = X_train_flat[tr_idx], X_train_flat[val_idx]
            y_tr, y_val = y_train_flat[tr_idx], y_train_flat[val_idx]
            ids_tr, ids_val = train_group_ids_flat[tr_idx], train_group_ids_flat[val_idx]

            X_tr, X_val, X_test = preprocess_feats(X_tr, X_val, X_test)

            if cfg.target_scale == TargetScale.LOG:
                y_tr_scaled, y_val_scaled = log1p_scaling(y_tr, y_val)
            elif cfg.target_scale == TargetScale.MAX:
                y_tr_scaled, y_val_scaled, y_scaler = max_scaling(y_tr, y_val)
                
            model = DistNetModel(model_target_scale=cfg.target_scale, n_input_features=X_tr.shape[1], n_epochs=DISTNET_N_EPOCHS, batch_size=DISTNET_BATCH_SIZE, 
                                 wc_time_limit=DISTNET_WCT, X_valid=X_val, y_valid=y_val_scaled, 
                                 early_stopping=True, early_stopping_patience=DISTNET_ES_PATIENCE, random_state=RANDOM_STATE)
            X_train_flat, y_train_scaled, train_group_ids_flat = X_tr, y_tr_scaled, ids_tr
        else:
            X_train_flat, X_test = preprocess_feats(X_train_flat, X_test)

            if cfg.target_scale == TargetScale.LOG:
                y_train_scaled = log1p_scaling(y_train_flat)[0]
            elif cfg.target_scale == TargetScale.MAX:
                y_train_scaled, y_scaler = max_scaling(y_train_flat)

            model = DistNetModel(model_target_scale=cfg.target_scale, n_input_features=X_train_flat.shape[1], n_epochs=DISTNET_N_EPOCHS, batch_size=DISTNET_BATCH_SIZE, wc_time_limit=DISTNET_WCT, early_stopping=False, random_state=RANDOM_STATE)

        fit_start = time.perf_counter()
        model.train(X_train_flat, y_train_scaled)
        fit_time = time.perf_counter() - fit_start
        
        if cfg.early_stopping: best_epoch = model.best_epoch

        pred_start = time.perf_counter()
        y_pred = model.predict(X_test)
        pred_time = time.perf_counter() - pred_start

        metrics_sum, inst_sum = calculate_metrics_distnet(
            y_test, y_pred, device=device, target_scale=cfg.target_scale, y_scaler=y_scaler, N_grid_points=N_GRID_POINTS
        )
        
        return {
            'y_test_preds': y_pred,
            'result_metrics': {'metrics_summary': metrics_sum, 'instance_summary': inst_sum},
            'model_specific_info': {
                'model_config': model.model.state_dict(), 'best_epoch': best_epoch, 'fit_time': fit_time, 
                'predict_time': pred_time, 'y_scaler': y_scaler, 'n_samples': X_train_flat.shape[0], 'n_features': X_train_flat.shape[1], 'train_group_ids': train_group_ids_flat, 'n_samples_val': X_val.shape[0] if X_val is not None else None, 'n_features_val': X_val.shape[1] if X_val is not None else None
            }
        }

class BayesianDistNetHandler(BaseModelHandler):
    def run(self, cfg: ExperimentConfig, X_train_flat: np.ndarray, X_test: np.ndarray, y_train_flat: np.ndarray, y_test: np.ndarray, train_group_ids_flat: np.ndarray):
        from tabpfn_project.helper.bayesian_distnet import BayesianDistNetModel
        from tabpfn_project.helper.calculate_metrics import calculate_metrics_distnet
        from sklearn.model_selection import GroupShuffleSplit

        assert cfg.target_scale == TargetScale.MAX, "BayesianDistNet currently only supports 'max' target scale."
        device = torch.device('cuda' if (torch.cuda.is_available() and not cfg.use_cpu) else 'cpu')

        if cfg.early_stopping:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
            tr_idx, val_idx = next(gss.split(X_train_flat, y_train_flat, groups=train_group_ids_flat))
            X_tr, X_val = X_train_flat[tr_idx], X_train_flat[val_idx]
            y_tr, y_val = y_train_flat[tr_idx], y_train_flat[val_idx]

            X_tr, X_val, X_test = preprocess_feats(X_tr, X_val, X_test)
            y_tr, y_val, y_scaler = max_scaling(y_tr, y_val)
            handler = BayesianDistNetModel(n_input_features=X_tr.shape[1], device=device, X_valid=X_val, y_valid=y_val, early_stopping=True)
        else:
            X_tr, X_test = preprocess_feats(X_train_flat, X_test)
            y_tr, y_scaler = max_scaling(y_train_flat)
            handler = BayesianDistNetModel(n_input_features=X_tr.shape[1], device=device, early_stopping=False)

        fit_start = time.perf_counter()
        handler.train(X_tr, y_tr)
        fit_time = time.perf_counter() - fit_start

        best_epoch = handler.best_epoch

        pred_start = time.perf_counter()
        y_pred = handler.predict(X_test)
        pred_time = time.perf_counter() - pred_start

        metrics_sum, inst_sum = calculate_metrics_distnet(
            y_test, y_pred, device=device, target_scale=cfg.target_scale, y_scaler=y_scaler, N_grid_points=N_GRID_POINTS
        )
        
        print(f"BayesianDistNet Metrics Summary scale={cfg.target_scale.value}: {metrics_sum}")

        return {
            'y_test_preds': y_pred,
            'result_metrics': {'metrics_summary': metrics_sum, 'instance_summary': inst_sum},
            'model_specific_info': {
                'model_config': handler.model.state_dict(),
                'best_epoch': best_epoch,
                'fit_time': fit_time,
                'predict_time': pred_time,
                'y_scaler': y_scaler,
                'n_epochs': handler.n_epochs,
                'n_samples': X_tr.shape[0],
                'n_features': X_tr.shape[1],
                'n_samples_val': X_val.shape[0] if cfg.early_stopping else None,
                'n_features_val': X_val.shape[1] if cfg.early_stopping else None,
            }
        }

class TabPFNHandler(BaseModelHandler):
    def run(self, cfg: ExperimentConfig, X_train_flat: np.ndarray, X_test: np.ndarray, y_train_flat: np.ndarray, y_test: np.ndarray, train_group_ids_flat: np.ndarray):
        from tabpfn import TabPFNRegressor
        from tabpfn_project.globals import TABPFN_VAL_BATCH_SIZE
        from tabpfn_project.helper.tabpfn_helpers import batch_predict_tabpfn, oracle_predict_tabpfn
        from tabpfn_project.helper.calculate_metrics import calculate_metrics_tabpfn
        from tabpfn_project.helper.utils import add_feature_jitter, append_random_columns
        from tabpfn.constants import ModelVersion

        device = torch.device('cuda' if (torch.cuda.is_available() and not cfg.use_cpu) else 'cpu')
        mem_stats = {"fit": {}, "predict": {}}
        all_preds = []
        y_scaler = None  # max-scaler, active only if target_scale='max'

        if not cfg.oracle:
            if cfg.jitter_x:
                assert cfg.jitter_val is not None, "jitter_val must be provided when jitter_x is enabled."
                print(f"Applying baseline feature jitter with intensity {cfg.jitter_val} to training data.")
                X_train_flat = add_feature_jitter(X_train_flat, jitter_intensity=cfg.jitter_val, random_state=RANDOM_STATE)
                
            if cfg.rand_extend_x:
                assert cfg.n_rand_cols is not None, "n_rand_cols must be provided when rand_extend_x is enabled."
                print(f"Applying random feature extension with {cfg.n_rand_cols} columns.")
                X_train_flat, X_test = append_random_columns(X_train_flat, X_test, n_random_cols=cfg.n_rand_cols, random_state=RANDOM_STATE)

        if cfg.oracle:
            print("Running TabPFN ORACLE model.")
            model = TabPFNRegressor.create_default_for_version(ModelVersion.V2_5, ignore_pretraining_limits=True, device=device, random_state=RANDOM_STATE)
            with track_gpu_memory_and_time(device) as stats:
                preds, y_scaler = oracle_predict_tabpfn(model, y_test, target_scale=cfg.target_scale)
            all_preds = [preds]
            mem_stats["predict"] = stats

        else:
            print("Running SINGLE TabPFN model.")

            if cfg.remove_duplicates:
                print(f"Removing duplicates: keeping 1 sample per instance in training data to prevent leakage.")
                X_train_flat, y_train_flat, train_group_ids_flat = sample_k_per_instance(X_train_flat, y_train_flat, train_group_ids_flat, k=1, seed=RANDOM_STATE)

            X_train_flat, X_test = preprocess_feats(X_train_flat, X_test)

            y_train_scaled = y_train_flat
            if cfg.target_scale == TargetScale.LOG:
                y_train_scaled = log1p_scaling(y_train_flat)[0]
            elif cfg.target_scale == TargetScale.MAX:
                y_train_scaled, y_scaler = max_scaling(y_train_flat)
            
            model = TabPFNRegressor.create_default_for_version(ModelVersion.V2_5, ignore_pretraining_limits=True, device=device, random_state=RANDOM_STATE)
            with track_gpu_memory_and_time(device) as stats:
                model.fit(X_train_flat, y_train_scaled.ravel())
            mem_stats["fit"] = stats

            with track_gpu_memory_and_time(device) as stats:
                preds = batch_predict_tabpfn(model, X_test, validation_batch_size=TABPFN_VAL_BATCH_SIZE)
            all_preds = [preds]
            mem_stats["predict"] = stats
    
        metrics_sum, inst_sum = calculate_metrics_tabpfn(
                *all_preds, y_test_original=y_test, device=device, target_scale=cfg.target_scale, y_scaler=y_scaler, N_grid_points=N_GRID_POINTS
            )
        
        for k, v in metrics_sum.items():
            print(f"{k}: {v:.4f}")
        print(mem_stats)

        self._save_preds(cfg, all_preds)

        return {
            'y_test_preds': None,
            'result_metrics': {'metrics_summary': metrics_sum, 'instance_summary': inst_sum},
            'model_specific_info': {'mem_time_stats': mem_stats, 'y_scaler': y_scaler, 'n_samples': X_train_flat.shape[0], 'n_features': X_train_flat.shape[1], 'train_group_ids': train_group_ids_flat}
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
    def run(self, cfg: ExperimentConfig, X_train_flat: np.ndarray, X_test: np.ndarray, y_train_flat: np.ndarray, y_test: np.ndarray, train_group_ids_flat: np.ndarray):
        from tabpfn_project.helper.random_forest import RuntimePredictionRandomForest
        from sklearn.model_selection import GroupKFold
        from tabpfn_project.helper.calculate_metrics import calculate_metrics_random_forest
        from tabpfn_project.globals import MAX_HPO_TRIALS, MAX_HPO_WCT

        device = torch.device("cuda" if (torch.cuda.is_available() and not getattr(cfg, 'use_cpu', False)) else "cpu")
        assert cfg.target_scale == TargetScale.LOG, "Target scale must be 'log'"
        
        best_params = dict()
        num_unique_configs = None
        num_finished_trials = None
        num_submitted_trials = None
        if cfg.do_hpo:
            smac_folder_name = f"rf_hpo_{generate_experiment_id(cfg)}"
            cs = ConfigurationSpace()
            # TabArena inspired defaults for RF HPO Search Space
            cs.add([
                Constant("n_estimators", 50),
                Float("max_features", (0.4, 1.0), default=0.5),
                Float("max_samples", (0.5, 1.0), default=1.0),
                Float("min_impurity_decrease", (1e-5, 1e-3), log=True),
                Integer("min_samples_split", (2, 5), log=True, default=5),
                Float("var_min", (1e-5, 1e-1), log=True, default=0.01),
                Categorical("bootstrap", [True, False], default=False)
            ])
            condition = EqualsCondition(cs["max_samples"], cs["bootstrap"], True)
            cs.add(condition)

            def smac_objective(config, seed) -> float:
                """
                Inner optimization loop via GroupKFold.
                Predictions and evaluations are done directly on all validation observations.
                """
                bootstrap_val = config.get("bootstrap", False)
                max_samples_val = config.get("max_samples", None)
                if not bootstrap_val:
                    max_samples_val = None  # Ensure max_samples is ignored when bootstrap=False

                gkf = GroupKFold(n_splits=3)
                metrics_per_fold =[]
                for in_train_idx, in_val_idx in gkf.split(X_train_flat, y_train_flat, groups=train_group_ids_flat):
                    X_in_train_raw_split = X_train_flat[in_train_idx]
                    X_in_val_raw_split = X_train_flat[in_val_idx]
                    
                    y_in_train = y_train_flat[in_train_idx]
                    y_in_val = y_train_flat[in_val_idx]
                    
                    X_in_train_scaled, X_in_val_scaled = preprocess_feats(
                        X_in_train_raw_split, X_in_val_raw_split
                    )
                    
                    # Apply log1p scaling to inner train targets
                    y_in_train_scaled = log1p_scaling(y_in_train)[0]
                    
                    # 2. Fit Candidate Model
                    model = RuntimePredictionRandomForest(
                        random_state=seed, 
                        n_estimators=config["n_estimators"],
                        max_features=config["max_features"],
                        min_samples_split=config["min_samples_split"],
                        bootstrap=bootstrap_val,
                        var_min=config["var_min"],
                        max_samples=max_samples_val,
                        min_impurity_decrease=config["min_impurity_decrease"]
                    )
                    model.fit(X_in_train_scaled, y_in_train_scaled)
                    
                    # 3. Direct Prediction on ALL validation rows
                    means, vars = model.predict(X_in_val_scaled)
                    
                    m_t = torch.as_tensor(means, dtype=torch.float32, device=device)
                    v_t = torch.as_tensor(vars, dtype=torch.float32, device=device)
                    
                    z_in_val = torch.log1p(torch.as_tensor(y_in_val, dtype=torch.float32, device=device))
                    
                    dist = torch.distributions.Normal(loc=m_t, scale=torch.sqrt(v_t))

                    clamp_val = torch.log(torch.tensor(LLH_EPSILON, device=device))
                    llh = dist.log_prob(z_in_val).clamp(min=clamp_val)
                    
                    nllh = -llh.mean()
                    metrics_per_fold.append(nllh.item())
                    
                return float(np.mean(metrics_per_fold))

            scenario = Scenario(
                name=smac_folder_name,
                configspace=cs,
                n_trials=MAX_HPO_TRIALS,
                walltime_limit=MAX_HPO_WCT,
                seed=RANDOM_STATE,
                n_workers=1,
            )
            
            smac = HyperparameterOptimizationFacade(scenario=scenario, target_function=smac_objective, overwrite=False)
            print(f"Starting SMAC3 HPO...")
            best_config = smac.optimize()
            best_params = dict(best_config)
            print(f"HPO Complete. Best hyperparameters: {best_params}")
            num_unique_configs = len(smac.runhistory.get_configs())
            num_finished_trials = smac.runhistory.finished
            num_submitted_trials = smac.runhistory.submitted
            print(f"HPO Trials - Submitted: {num_submitted_trials}, Finished: {num_finished_trials}, Unique Configs Evaluated: {num_unique_configs}")

        else:
            # Default RF parameters for runtime prediction (with log-scaling)
            best_params = {
                "n_estimators": 50,
                "max_features": 0.5,
                "min_samples_split": 5,
                "var_min": 1e-6,
                "bootstrap": False,
                "max_samples": None,
                "min_impurity_decrease": 0.0,
            }
            print(f"HPO disabled. Using default parameters: {best_params}")

        # --- Final Model Training ---
        X_train_final, X_test_final = preprocess_feats(X_train_flat, X_test)
        y_train_final = log1p_scaling(y_train_flat)[0]

        bootstrap_val = best_params.get("bootstrap", False)
        max_samples_val = best_params.get("max_samples", None)
        if not bootstrap_val:
            max_samples_val = None  # Ensure max_samples is ignored when bootstrap=False

        final_model = RuntimePredictionRandomForest(
                        random_state=RANDOM_STATE, 
                        n_estimators=best_params["n_estimators"],
                        max_features=best_params["max_features"],
                        min_samples_split=best_params["min_samples_split"],
                        var_min=best_params["var_min"],
                        min_impurity_decrease=best_params["min_impurity_decrease"],
                        bootstrap=bootstrap_val,
                        max_samples=max_samples_val,
                    )

        start = time.perf_counter()
        final_model.fit(X_train_final, y_train_final)
        fit_time = time.perf_counter() - start

        # Predict strictly on test
        pred_start = time.perf_counter()
        means, vars = final_model.predict(X_test_final)
        pred_time = time.perf_counter() - pred_start

        metrics_sum, inst_sum = calculate_metrics_random_forest(
            y_test, (means, vars), device=device, N_grid_points=N_GRID_POINTS
        )

        return {
            'y_test_preds': [means, vars],
            'result_metrics': {'metrics_summary': metrics_sum, 'instance_summary': inst_sum},
            'model_specific_info': {
                'fit_time': fit_time, 
                'predict_time': pred_time,
                'best_hyperparameters': best_params,
                'num_unique_configs': num_unique_configs,
                'num_finished_trials': num_finished_trials,
                'num_submitted_trials': num_submitted_trials,
                'n_samples': X_train_final.shape[0],
                'n_features': X_train_final.shape[1],
            }
        }
