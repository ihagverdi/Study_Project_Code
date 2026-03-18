import time
import numpy as np

# NGBoost & QRF & Scikit-Learn Ecosystem
from ngboost import NGBRegressor
from quantile_forest import RandomForestQuantileRegressor
from ngboost.distns import LogNormal
from ngboost.scores import LogScore
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# SMAC3 & ConfigSpace Imports
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario

from tabpfn_project.globals import RANDOM_STATE
import time
import numpy as np
from sklearn.model_selection import train_test_split

def train_evaluate_qrf(X_train_flat: np.ndarray, 
                        y_train_flat: np.ndarray, 
                        X_test: np.ndarray, 
                        do_hpo: bool,
                        hpo_time: int,):
    """
    Trains a robust Quantile Random Forest.
    Optionally performs SMAC3-based Hyperparameter Optimization.
    args:
    X_train_flat : np.ndarray
        Flattened feature matrix of shape (n_instances * n_obs_per_instance, n_features).
    y_train_flat : np.ndarray
        Flattened (scaled) target vector of shape (n_instances * n_obs_per_instance,).
    X_test : np.ndarray
        Test feature matrix of shape (n_test_instances, n_features).
    do_hpo : bool
        Whether to perform hyperparameter optimization.
    hpo_time : int
        Time limit for hyperparameter optimization in seconds.

    returns: y_pred_quantiles, best_params, fit_time, test_time

    """
    
    print("[INFO] Initializing Quantile Random Forest pipeline")
    y_train_eval = y_train_flat

    # Base configuration dictionary (overwritten if do_hpo is True)
    best_params = {
        "n_estimators": 100,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "max_samples": None,
        "max_depth": None,
        "criterion": "squared_error"
    }

    # 2. SMAC3 Hyperparameter Optimization
    if do_hpo:
        print(f"[INFO] Starting SMAC3 HPO for QRF (Time limit: {hpo_time}s)...")
        # Split data for HPO validation to avoid data leakage
        X_tr_hpo, X_val_hpo, y_tr_hpo, y_val_hpo = train_test_split(
            X_train_flat, y_train_eval, test_size=0.15, random_state=RANDOM_STATE
        )

        cs = ConfigurationSpace()
        cs.add([
            Integer("n_estimators", (100, 1000)),
            Integer("min_samples_leaf", (1, 200)),
            Float("max_features", (0.1, 1.0)),
            Float("max_samples", (0.1, 1.0)),
            Integer("max_depth", (5, 50)),
            Categorical("criterion", ["squared_error", "friedman_mse"])
        ])

        def qrf_objective(config: Configuration, seed: int) -> float:
            model = RandomForestQuantileRegressor(
                n_estimators=config["n_estimators"],
                min_samples_leaf=config["min_samples_leaf"],
                max_features=config["max_features"],
                max_samples=config["max_samples"],
                max_depth=config["max_depth"],
                criterion=config["criterion"],
                bootstrap=True, # Required for max_samples to take effect
                random_state=seed, 
                n_jobs=-1
            )
            model.fit(X_tr_hpo, y_tr_hpo)
            
            # Predict dense quantiles for fast HPO loss calculation
            hpo_quantiles = np.arange(0.0001, 1.0, 0.0001).tolist()
            y_pred = model.predict(X_val_hpo, quantiles=hpo_quantiles)
            
            # Compute Average Pinball Loss (Quantile Loss)
            avg_pinball_loss = 0.0
            for i, q in enumerate(hpo_quantiles):
                error = y_val_hpo - y_pred[:, i]
                avg_pinball_loss += np.mean(np.maximum(q * error, (q - 1) * error))
            return avg_pinball_loss / len(hpo_quantiles)

        scenario = Scenario(configspace=cs, deterministic=True, n_trials=999999, walltime_limit=hpo_time)
        smac = HPOFacade(scenario, qrf_objective, overwrite=True)
        
        incumbent = smac.optimize()
        best_params = dict(incumbent)
        print(f"[INFO] HPO Completed. Best Config: {best_params}")

    # 3. Final Model Training
    print("[INFO] Building Final QRF Model...")
    final_model = RandomForestQuantileRegressor(
        n_estimators=best_params["n_estimators"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features=best_params["max_features"],
        max_samples=best_params["max_samples"],
        max_depth=best_params["max_depth"],
        criterion=best_params["criterion"],
        bootstrap=True,
        random_state=RANDOM_STATE, 
        n_jobs=-1
    )
    
    start_fit = time.perf_counter()
    final_model.fit(X_train_flat, y_train_eval)
    fit_time = time.perf_counter() - start_fit
    print(f"[INFO] Final Training Completed in {fit_time:.2f} seconds.")
    
    # 4. Inference & Evaluation
    print("[INFO] Beginning Inference & Evaluation on Unseen Test Data...")
    # Predict dense quantiles to construct the CDF
    quantiles = np.arange(0.0001, 1.0, 0.0001).tolist()
    start_test = time.perf_counter()
    y_pred_quantiles = final_model.predict(X_test, quantiles=quantiles)
    test_time = time.perf_counter() - start_test
    

    print(f"[INFO] Evaluation Completed in {test_time:.2f} seconds.")

    return y_pred_quantiles, best_params, fit_time, test_time

def train_evaluate_ngboost(X_train_flat: np.ndarray, 
                            y_train_flat: np.ndarray, 
                            X_test: np.ndarray, 
                            do_hpo: bool,
                            hpo_time: int):
    """
    Trains a robust NGBoost model to predict parametric runtime distributions (LogNormal).
    Optionally performs SMAC3-based Hyperparameter Optimization.
    args:
    X_train_flat : np.ndarray
        Flattened feature matrix of shape (n_instances * n_obs_per_instance, n_features).
    y_train_flat : np.ndarray
        Flattened (scaled) runtime targets of shape (n_instances * n_obs_per_instance,).
    X_test : np.ndarray
        Unflattened feature matrix of shape (n_test_instances, n_features).
    do_hpo : bool
        Whether to perform SMAC3 Hyperparameter Optimization.
    hpo_time : int
        Time limit for HPO in seconds.
    returns test_dists.params, best_params, fit_time, test_time
    """
    print(f"[INFO] Initializing NGBoost pipeline")

    y_train_eval = y_train_flat
    # Base configuration dictionary (will be overwritten if do_hpo is True)
    best_params = {
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.01,
        "minibatch_frac": 1.0,
        "col_sample": 1.0
    }

    # 2. SMAC3 Hyperparameter Optimization
    if do_hpo:
        print(f"[INFO] Starting SMAC3 HPO for NGBoost (Time limit: {hpo_time}s)...")
        # Isolate a validation set specifically for HPO to avoid testing data leakage
        X_tr_hpo, X_val_hpo, y_tr_hpo, y_val_hpo = train_test_split(
            X_train_flat, y_train_eval, test_size=0.15, random_state=RANDOM_STATE
        )

        # Define Search Space
        cs = ConfigurationSpace()
        cs.add([
            Integer("n_estimators", (100, 1500)),
            Integer("max_depth", (2, 5)),
            Float("learning_rate", (0.001, 0.1), log=True),
            Float("minibatch_frac", (0.5, 1.0)),
            Float("col_sample", (0.5, 1.0)),
        ])

        def ngboost_objective(config: Configuration, seed: int) -> float:
            base_tree = DecisionTreeRegressor(
                criterion="friedman_mse", min_samples_split=2, min_samples_leaf=1,
                max_depth=config["max_depth"], splitter="best", random_state=seed
            )
            model = NGBRegressor(
                Dist=LogNormal, Score=LogScore, Base=base_tree,
                n_estimators=config["n_estimators"], # Passed from ConfigSpace directly
                learning_rate=config["learning_rate"],
                minibatch_frac=config["minibatch_frac"], col_sample=config["col_sample"],
                random_state=seed, verbose=False
            )
            # Fit without early stopping. SMAC explicitly evaluates the exact iteration count.
            # This guarantees that SMAC learns the correct interplay between learning_rate and n_estimators.
            model.fit(X_tr_hpo, y_tr_hpo)
            
            # Evaluate via Negative Log-Likelihood (NLL)
            preds = model.pred_dist(X_val_hpo)
            return -preds.logpdf(y_val_hpo).mean()

        scenario = Scenario(configspace=cs, deterministic=True, n_trials=999999, walltime_limit=hpo_time)
        smac = HPOFacade(scenario, ngboost_objective, overwrite=True)
        
        incumbent = smac.optimize()
        best_params = dict(incumbent)
        print(f"[INFO] HPO Completed. Best Config: {best_params}")

    # 3. Final Model Configuration & Training
    print("[INFO] Building Final NGBoost Model...")
    final_base_learner = DecisionTreeRegressor(
        criterion="friedman_mse", min_samples_split=2, min_samples_leaf=1,
        max_depth=best_params["max_depth"], splitter="best", random_state=RANDOM_STATE
    )
    final_model = NGBRegressor(
        Dist=LogNormal, Score=LogScore, Base=final_base_learner,
        n_estimators=best_params["n_estimators"], 
        learning_rate=best_params["learning_rate"],
        minibatch_frac=best_params["minibatch_frac"], col_sample=best_params["col_sample"],
        verbose=True, verbose_eval=100, tol=1e-4, random_state=RANDOM_STATE
    )

    start_fit = time.perf_counter()
    final_model.fit(X_train_flat, y_train_eval)
    fit_time = time.perf_counter() - start_fit
    print(f"[INFO] Final Training Completed in {fit_time:.2f} seconds.")
    
    # 4. Inference & Evaluation
    print("[INFO] Beginning Inference & Evaluation on Unseen Test Data...")
    start_test = time.perf_counter()
    test_dists = final_model.pred_dist(X_test)
    test_time = time.perf_counter() - start_test
    print(f"[INFO] Evaluation Completed in {test_time:.2f} seconds.")
    
    return test_dists.params, best_params, fit_time, test_time