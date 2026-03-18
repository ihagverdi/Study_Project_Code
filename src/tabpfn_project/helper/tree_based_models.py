import time
import numpy as np

# NGBoost & QRF & Scikit-Learn Ecosystem
from ngboost import NGBRegressor
from quantile_forest import RandomForestQuantileRegressor
from ngboost.distns import LogNormal
from ngboost.scores import LogScore
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import torch

# SMAC3 & ConfigSpace Imports
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario

from tabpfn_project.globals import RANDOM_STATE
import time
import numpy as np
from sklearn.model_selection import train_test_split
import math

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
            hpo_quantiles = np.arange(0.001, 1.0, 0.001).tolist()
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
    quantiles = np.arange(0.001, 1.0, 0.001).tolist()
    start_test = time.perf_counter()
    y_pred_quantiles = final_model.predict(X_test, quantiles=quantiles)
    test_time = time.perf_counter() - start_test
    

    print(f"[INFO] Evaluation Completed in {test_time:.2f} seconds.")

    return y_pred_quantiles, best_params, fit_time, test_time

def calculate_all_distribution_metrics_qrf_logspace_pchip(
    y_test_orig,
    preds, 
    *,
    device, 
    y_scaler=None, 
    N_grid_points,
):
    """
    Evaluates QRF natively in Z-space using a Grafted-PCHIP formulation
    and an exact matching NLLH correction.
    """
    y_test_orig = torch.as_tensor(y_test_orig, dtype=torch.float32, device=device)
    z_test_orig = torch.log1p(y_test_orig)
    z_preds = torch.as_tensor(preds, dtype=torch.float32, device=device)
    
    # 1. Enforce strict monotonicity (prevents h=0 division if trees predict point-masses)
    eps_tensor = torch.linspace(0, 1e-4, 999, device=device).unsqueeze(0)
    z_preds_strict = torch.cummax(z_preds, dim=1)[0] + eps_tensor

    # =========================================================
    # 2. PCHIP COEFFICIENTS CALCULATION
    # =========================================================
    q_tensor = torch.linspace(0.001, 0.999, 999, device=device)
    dq = 0.001
    
    h = z_preds_strict[:, 1:] - z_preds_strict[:, :-1]  # shape (B, 998)
    m = dq / h                                          # Secant slopes
    
    # Harmonic mean for interior knot derivatives
    w1 = 2 * h[:, 1:] + h[:, :-1]
    w2 = h[:, 1:] + 2 * h[:, :-1]
    
    d = torch.zeros_like(z_preds_strict)
    d[:, 1:-1] = (w1 + w2) / (w1 / m[:, :-1] + w2 / m[:, 1:])
    d[:, 0] = m[:, 0]     # Left boundary derivative
    d[:, -1] = m[:, -1]   # Right boundary derivative

    # =========================================================
    # 3. UNIFIED PCHIP EVALUATOR (Yields Exact CDF and PDF)
    # =========================================================
    def eval_pchip_cdf_pdf(Z_eval):
        """ Evaluates continuous CDF and PDF analytically over Z_eval using Hermite Splines and Exp Tails. """
        idx = torch.searchsorted(z_preds_strict, Z_eval)
        
        mask_left = Z_eval <= z_preds_strict[:, 0:1]
        mask_right = Z_eval > z_preds_strict[:, -1:]
        mask_in = ~(mask_left | mask_right)

        # Mapping to interval indices [0, 997]
        k = idx.clamp(1, 998) - 1  
        
        # Gather coefficients for the interior evaluation
        h_k = torch.gather(h, 1, k)
        m_k = torch.gather(m, 1, k)
        d_k = torch.gather(d, 1, k)
        d_kp1 = torch.gather(d, 1, k+1)
        x_k = torch.gather(z_preds_strict, 1, k)
        y_k = q_tensor[k]
        y_kp1 = q_tensor[k+1]
        
        t = (Z_eval - x_k) / h_k
        t2, t3 = t**2, t**3
        
        # Exact Analytical Interior CDF and its Derivative (PDF)
        cdf_in = y_k*(2*t3 - 3*t2 + 1) + y_kp1*(-2*t3 + 3*t2) + h_k*d_k*(t3 - 2*t2 + t) + h_k*d_kp1*(t3 - t2)
        pdf_in = m_k*(6*t - 6*t2) + d_k*(1 - 4*t + 3*t2) + d_kp1*(-2*t + 3*t2)
        
        # Grafted Analytical Left Tail (Exponential integrating to 0.001)
        x_0, d_0 = z_preds_strict[:, 0:1], d[:, 0:1]
        lambda_L = d_0 / 0.001
        cdf_left = 0.001 * torch.exp(lambda_L * (Z_eval - x_0))
        pdf_left = d_0 * torch.exp(lambda_L * (Z_eval - x_0))
        
        # Grafted Analytical Right Tail (Exponential integrating to 0.001)
        x_end, d_end = z_preds_strict[:, -1:], d[:, -1:]
        lambda_R = d_end / 0.001
        cdf_right = 1.0 - 0.001 * torch.exp(-lambda_R * (Z_eval - x_end))
        pdf_right = d_end * torch.exp(-lambda_R * (Z_eval - x_end))
        
        cdf_vals = torch.where(mask_left, cdf_left, torch.where(mask_right, cdf_right, cdf_in))
        pdf_vals = torch.where(mask_left, pdf_left, torch.where(mask_right, pdf_right, pdf_in))
        
        # Absolute safety to ensure log operates properly 
        pdf_vals = pdf_vals.clamp(min=1e-87)
        
        return cdf_vals, pdf_vals

    # =========================================================
    # 4. BOUNDS & PIECEWISE INTEGRATION GRID
    # =========================================================
    min_z_emp = z_test_orig.min(dim=1, keepdim=True)[0]
    max_z_emp = z_test_orig.max(dim=1, keepdim=True)[0]
    z_range = (max_z_emp - min_z_emp).clamp(min=1e-5)
    
    core_start = min_z_emp - 0.05 * z_range
    core_end = max_z_emp + 0.05 * z_range

    global_start = torch.minimum(core_start - 0.5 * z_range, z_preds_strict[:, 0].unsqueeze(1))
    global_end = torch.maximum(core_end + 0.5 * z_range, z_preds_strict[:, -1].unsqueeze(1))

    left_pts, core_pts, right_pts = int(N_grid_points * 1/6), int(N_grid_points * 2/3), int(N_grid_points * 1/6)

    steps_left = torch.linspace(0, 1, left_pts, device=device).view(1, -1)
    z_grid_left = global_start + steps_left * (core_start - global_start)
    steps_core = torch.linspace(0, 1, core_pts + 1, device=device).view(1, -1)[:, 1:]
    z_grid_core = core_start + steps_core * (core_end - core_start)
    steps_right = torch.linspace(0, 1, right_pts + 1, device=device).view(1, -1)[:, 1:]
    z_grid_right = core_end + steps_right * (global_end - core_end)

    z_grid = torch.cat([z_grid_left, z_grid_core, z_grid_right], dim=1)  # shape (B, N_grid_points)

    # =========================================================
    # 5. CDF EVALUATION (Wasserstein, KS, CRPS)
    # =========================================================
    indicator = (z_test_orig.unsqueeze(1) <= z_grid.unsqueeze(2)).float()  
    F_emp = indicator.mean(dim=2)
    
    F_model, _ = eval_pchip_cdf_pdf(z_grid)
    
    cdf_diff = F_model - F_emp
    abs_cdf_diff = torch.abs(cdf_diff)
    
    all_w1 = torch.trapezoid(abs_cdf_diff, x=z_grid, dim=1)  
    all_ks = torch.max(abs_cdf_diff, dim=1)[0]                   

    cvm_distance = torch.trapezoid(cdf_diff ** 2, x=z_grid, dim=1)
    z_sorted = torch.sort(z_test_orig, dim=1)[0]
    diffs = z_sorted[:, 1:] - z_sorted[:, :-1]  
    
    N = z_test_orig.shape[1]
    i = torch.arange(1, N, device=device).float()
    weights = (i / N) * (1.0 - i / N)           
    
    empirical_spread = torch.sum(weights * diffs, dim=1) 
    all_crps = cvm_distance + empirical_spread

    # =========================================================
    # 6. VECTORIZED NLLH (Evaluated strictly via TabPFN logic)
    # =========================================================
    _, pdf_Z = eval_pchip_cdf_pdf(z_test_orig)
    
    nlog_pdf = -torch.log(pdf_Z)
    nlog_pdf.clamp_(max=200.0)  # 200 corresponds to -log(1e-87); prevents possible inf's due to precision errors.
    
    max_y_scaled = torch.max(z_test_orig, dim=1)[0]
    bias = -torch.log(max_y_scaled)

    print(f" bias.shape: {bias.shape}, nlog_pdf.shape: {nlog_pdf.shape}")
    
    all_nllh = nlog_pdf.mean(dim=1) + bias

    metrics_summary = {
        "NLLH_mean": all_nllh.mean().item(),
        "NLLH_std": all_nllh.std().item(),
        "CRPS_mean": all_crps.mean().item(),
        "CRPS_std": all_crps.std().item(),
        "Wasserstein_mean": all_w1.mean().item(),
        "Wasserstein_std": all_w1.std().item(),
        "KS_mean": all_ks.mean().item(),
        "KS_std": all_ks.std().item(),
    }
    
    instance_summary = {
        "NLLH": all_nllh.detach().cpu(), 
        "CRPS": all_crps.detach().cpu(), 
        "Wasserstein": all_w1.detach().cpu(), 
        "KS": all_ks.detach().cpu()
    }

    return metrics_summary, instance_summary

def calculate_all_distribution_metrics_qrf_logspace_kde(
    y_test_orig,
    preds, 
    *,
    device, 
    y_scaler=None,  # API compatibility
    N_grid_points,
):
    """
    Evaluates QRF natively in Z-space.
    Uses Linear Interpolation for the CDF (Wasserstein, KS, CRPS).
    Uses Gaussian KDE for the PDF, with the EXACT TabPFN NLLH bias correction logic.
    """
    y_test_orig = torch.as_tensor(y_test_orig, dtype=torch.float32, device=device)
    z_test_orig = torch.log1p(y_test_orig)
    z_preds = torch.as_tensor(preds, dtype=torch.float32, device=device)
    
    # Enforce strict monotonicity of predicted quantiles
    z_preds = torch.cummax(z_preds, dim=1)[0]

    # =========================================================
    # 1. CORE BOUNDS (Empirical Data in Z-Space)
    # =========================================================
    min_z_emp = z_test_orig.min(dim=1, keepdim=True)[0]  # shape (B, 1)
    max_z_emp = z_test_orig.max(dim=1, keepdim=True)[0]  # shape (B, 1)
    z_range = (max_z_emp - min_z_emp).clamp(min=1e-5)
    
    core_start = min_z_emp - 0.05 * z_range
    core_end = max_z_emp + 0.05 * z_range

    # =========================================================
    # 2. QRF TAIL BOUNDS & GRID INITIALIZATION
    # =========================================================
    z_model_min = z_preds[:, 0].unsqueeze(1)
    z_model_max = z_preds[:, -1].unsqueeze(1)

    global_start = torch.minimum(core_start - 0.5 * z_range, z_model_min)
    global_end = torch.maximum(core_end + 0.5 * z_range, z_model_max)

    left_pts, core_pts, right_pts = int(N_grid_points * 1/6), int(N_grid_points * 2/3), int(N_grid_points * 1/6)

    steps_left = torch.linspace(0, 1, left_pts, device=device).view(1, -1)
    z_grid_left = global_start + steps_left * (core_start - global_start)

    steps_core = torch.linspace(0, 1, core_pts + 1, device=device).view(1, -1)[:, 1:]
    z_grid_core = core_start + steps_core * (core_end - core_start)

    steps_right = torch.linspace(0, 1, right_pts + 1, device=device).view(1, -1)[:, 1:]
    z_grid_right = core_end + steps_right * (global_end - core_end)

    z_grid = torch.cat([z_grid_left, z_grid_core, z_grid_right], dim=1)  # shape (B, N_grid_points)

    # =========================================================
    # 3. CDF EVALUATION VIA LINEAR INTERPOLATION
    # =========================================================
    indicator = (z_test_orig.unsqueeze(1) <= z_grid.unsqueeze(2)).float()  
    F_emp = indicator.mean(dim=2)  # shape (B, N_grid_points)
    
    # Vectorized binary search mapping Z-grid into QRF bins
    idx = torch.searchsorted(z_preds, z_grid)  
    idx_lower = (idx - 1).clamp(min=0)
    idx_upper = idx.clamp(max=998)
    
    z_lower = torch.gather(z_preds, 1, idx_lower)
    z_upper = torch.gather(z_preds, 1, idx_upper)
    
    q_tensor = torch.arange(0.001, 1.0, 0.001, device=device)  # QRF quantiles
    q_lower = q_tensor[idx_lower]
    q_upper = q_tensor[idx_upper]
    
    # Continuous linear interpolation
    w = (z_grid - z_lower) / (z_upper - z_lower + 1e-9)
    w = w.clamp(0, 1)  
    
    F_model = q_lower + w * (q_upper - q_lower)
    
    # Clamping beyond the modeled bounds
    F_model = torch.where(z_grid < z_preds[:, 0:1], torch.zeros_like(F_model), F_model)
    F_model = torch.where(z_grid > z_preds[:, -1:], torch.ones_like(F_model), F_model)
    
    cdf_diff = F_model - F_emp
    abs_cdf_diff = torch.abs(cdf_diff)
    
    all_w1 = torch.trapezoid(abs_cdf_diff, x=z_grid, dim=1)  
    all_ks = torch.max(abs_cdf_diff, dim=1)[0]                   

    # CRPS calculation
    cvm_distance = torch.trapezoid(cdf_diff ** 2, x=z_grid, dim=1)

    z_sorted = torch.sort(z_test_orig, dim=1)[0]
    diffs = z_sorted[:, 1:] - z_sorted[:, :-1]  
    
    N = z_test_orig.shape[1]
    i = torch.arange(1, N, device=device).float()
    weights = (i / N) * (1.0 - i / N)           
    
    empirical_spread = torch.sum(weights * diffs, dim=1) 
    all_crps = cvm_distance + empirical_spread

    # =========================================================
    # 4. VECTORIZED NLLH VIA GAUSSIAN KDE (WITH Jacobian CORRECTION)
    # =========================================================
    # Bandwidth Selection: Silverman's Rule of Thumb for N=999
    # h = 1.06 * std * N^(-1/5) ≈ 0.266 * std
    std_z = torch.std(z_preds, dim=1, keepdim=True)
    h = (0.266 * std_z).clamp(min=1e-4)  # Prevent zero-division on point-mass predictions
    
    # Evaluate KDE at the observed targets (diff shape: B, O, 999)
    diff = z_test_orig.unsqueeze(-1) - z_preds.unsqueeze(1)
    h_expanded = h.unsqueeze(1) 
    
    u = diff / h_expanded
    pdf_vals = torch.exp(-0.5 * u**2) / (math.sqrt(2 * math.pi) * h_expanded)
    
    # Average across the 999 Gaussian kernels to get the PDF
    pdf_Z = pdf_vals.mean(dim=-1)  # shape (B, O)
    
    # Apply exact TabPFN NLLH correction and clamping logic
    # nlog_pdf = -torch.log(pdf_Z.clamp(min=1e-87))  # Negative log-likelihood
    nlog_pdf = -torch.log(pdf_Z)  # Negative log-likelihood
    nlog_pdf.clamp_(max=200.0)      # 200 corresponds to -log(1e-87); prevents infs
    
    # Because z_test_orig matches TabPFN's `batch_y_scaled`
    max_y_scaled = torch.max(z_test_orig, dim=1)[0]
    bias = -torch.log(max_y_scaled)
    
    all_nllh = nlog_pdf.mean(dim=1) + bias

    metrics_summary = {
        "NLLH_mean": all_nllh.mean().item(),
        "NLLH_std": all_nllh.std().item(),
        "CRPS_mean": all_crps.mean().item(),
        "CRPS_std": all_crps.std().item(),
        "Wasserstein_mean": all_w1.mean().item(),
        "Wasserstein_std": all_w1.std().item(),
        "KS_mean": all_ks.mean().item(),
        "KS_std": all_ks.std().item(),
    }
    
    instance_summary = {
        "NLLH": all_nllh.detach().cpu(), 
        "CRPS": all_crps.detach().cpu(), 
        "Wasserstein": all_w1.detach().cpu(), 
        "KS": all_ks.detach().cpu()
    }

    return metrics_summary, instance_summary

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

def calculate_all_distribution_metrics_ngboost_logspace(
    y_test_orig,
    preds, 
    *,
    device, 
    y_scaler,
    N_grid_points,
):
    """
    y_test_orig: shape (B, O) - original unscaled targets
    preds: dict - NGBoost's predicted parameters returned by test_dists.params 
           (expected keys: 's' for sigma, 'scale' for exp(mu) in max-scaled space)
    y_scaler: scaler value - the max-scaling factor used to convert from original Y space to NGBoost's max-scaled space
    device: torch device for computation
    N_grid_points: total number of points in the piecewise non-uniform grid for CDF evaluation

    returns: - metrics_summary: summary dict, instance_summary: dict of per-instance metrics
    """
    y_test_orig = torch.as_tensor(y_test_orig, dtype=torch.float32, device=device)
    y_scaler = torch.as_tensor(y_scaler, dtype=torch.float32, device=device)
    
    z_test_orig = torch.log1p(y_test_orig)

    # NGBoost test_dists.params dictionary mapping for LogNormal:
    # 's' is the shape parameter -> sigma
    # 'scale' is the scale parameter -> exp(mu)
    ngb_sigma = torch.as_tensor(preds['s'], dtype=torch.float32, device=device)
    ngb_scale = torch.as_tensor(preds['scale'], dtype=torch.float32, device=device)
    
    sigma = ngb_sigma.unsqueeze(1)
    mu = torch.log(ngb_scale).unsqueeze(1)  
    
    dist = torch.distributions.LogNormal(loc=mu, scale=sigma)

    # =========================================================
    # 1. CORE BOUNDS (Empirical Data in Z-Space)
    # =========================================================
    min_z_emp = z_test_orig.min(dim=1, keepdim=True)[0]  # shape (B, 1)
    max_z_emp = z_test_orig.max(dim=1, keepdim=True)[0]  # shape (B, 1)
    z_range = (max_z_emp - min_z_emp).clamp(min=1e-5)
    
    core_start = min_z_emp - 0.05 * z_range
    core_end = max_z_emp + 0.05 * z_range

    # =========================================================
    # 2. NGBOOST TAIL BOUNDS VIA INVERSE CDF (ICDF)
    # =========================================================
    # NGBoost modelled the max-scaled space. 
    # We query the exact 0.01% and 99.99% quantiles using the ICDF.
    p_min = torch.tensor(0.0001, device=device)
    p_max = torch.tensor(0.9999, device=device)

    y_scaled_min = dist.icdf(p_min)
    y_scaled_max = dist.icdf(p_max)

    # Convert from max-scaled space back to Original Y space
    y_orig_min = y_scaled_min / y_scaler  
    y_orig_max = y_scaled_max / y_scaler

    # Map these rigorous boundaries into our unified integration Z-space (log1p)
    z_model_min = torch.log1p(y_orig_min)
    z_model_max = torch.log1p(y_orig_max)

    global_start = torch.minimum(core_start - 0.5 * z_range, z_model_min)
    global_end = torch.maximum(core_end + 0.5 * z_range, z_model_max)

    # =========================================================
    # 3. 15K PIECEWISE NON-UNIFORM GRID
    # =========================================================
    left_pts, core_pts, right_pts = int(N_grid_points * 1/6), int(N_grid_points * 2/3), int(N_grid_points * 1/6)

    steps_left = torch.linspace(0, 1, left_pts, device=device).view(1, -1)
    z_grid_left = global_start + steps_left * (core_start - global_start)

    steps_core = torch.linspace(0, 1, core_pts + 1, device=device).view(1, -1)[:, 1:]
    z_grid_core = core_start + steps_core * (core_end - core_start)

    steps_right = torch.linspace(0, 1, right_pts + 1, device=device).view(1, -1)[:, 1:]
    z_grid_right = core_end + steps_right * (global_end - core_end)

    z_grid = torch.cat([z_grid_left, z_grid_core, z_grid_right], dim=1)  # shape (B, N_grid_points)

    # =========================================================
    # 4. CDF EVALUATION & INTEGRATION (Apples-to-Apples in Z-space)
    # =========================================================
    indicator = (z_test_orig.unsqueeze(1) <= z_grid.unsqueeze(2)).float()  
    F_emp = indicator.mean(dim=2)  # shape (B, N_grid_points)
    
    # Evaluate NGBoost CDF
    y_orig = torch.exp(z_grid) - 1.0
    y_scaled = y_orig * y_scaler  
    
    F_model = dist.cdf(y_scaled.clamp(min=0))
    
    cdf_diff = F_model - F_emp
    abs_cdf_diff = torch.abs(cdf_diff)
    
    # Integration over dz
    all_w1 = torch.trapezoid(abs_cdf_diff, x=z_grid, dim=1)  
    all_ks = torch.max(abs_cdf_diff, dim=1)[0]                   

    # CRPS calculation
    # 1. Base integral: the Cramér–von Mises distance
    cvm_distance = torch.trapezoid(cdf_diff ** 2, x=z_grid, dim=1)

    # 2. Exact Empirical Spread
    z_sorted = torch.sort(z_test_orig, dim=1)[0]
    diffs = z_sorted[:, 1:] - z_sorted[:, :-1]  # shape: (B, N-1)
    
    N = z_test_orig.shape[1]
    i = torch.arange(1, N, device=device).float()
    weights = (i / N) * (1.0 - i / N)           # shape: (N-1,)
    
    empirical_spread = torch.sum(weights * diffs, dim=1) # shape: (B,)
    
    all_crps = cvm_distance + empirical_spread

    # =========================================================
    # 5. VECTORIZED NLLH (in log-space)
    # =========================================================
    y_test_scaled = y_test_orig * y_scaler
    nlog_pdf = -dist.log_prob(y_test_scaled)  # shape (B, O)
    nlog_pdf.clamp_(max=200.0)  # 200 corresponds to -log(1e-87); prevents possible inf's due to precision errors; same threshold used for tabpfn & distnet.

    assert nlog_pdf.shape == z_test_orig.shape, f"shapes mismatched at nllh calculation: {nlog_pdf.shape} vs {z_test_orig.shape}"
    nlog_pdf += -z_test_orig  # nll correction
    bias = -torch.log(torch.max(z_test_orig, keepdim=False, dim=1)[0]) - torch.log(y_scaler)  # shape (B,)

    all_nllh = nlog_pdf.mean(dim=1) + bias

    metrics_summary = {
        "NLLH_mean": all_nllh.mean().item(),
        "NLLH_std": all_nllh.std().item(),
        "CRPS_mean": all_crps.mean().item(),
        "CRPS_std": all_crps.std().item(),
        "Wasserstein_mean": all_w1.mean().item(),
        "Wasserstein_std": all_w1.std().item(),
        "KS_mean": all_ks.mean().item(),
        "KS_std": all_ks.std().item(),
    }
    
    instance_summary = {
        "NLLH": all_nllh.detach().cpu(), 
        "CRPS": all_crps.detach().cpu(), 
        "Wasserstein": all_w1.detach().cpu(), 
        "KS": all_ks.detach().cpu()
    }

    return metrics_summary, instance_summary