import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gaussian_kde
import torch
from tabpfn_project.globals import EPS

def get_marginal_empirical_predictor(y_train):
    """
    Creates the Marginal Empirical Predictor baseline using scipy's KDE.
    
    Args:
        y_train: The target runtime values (can be 2D (N_instances, 100) or already flattened).
        
    Returns:
        cdf_object: Callable ECDF object.
        pdf_object: gaussian_kde object, which supports .pdf(x) and .logpdf(x).
    """
    # 1. Flatten the array to 1D
    y_train_flat = np.ravel(y_train)
    
    # 2. Create the CDF object
    cdf_object = ECDF(y_train_flat)
    
    # 3. Create the PDF object using scipy (defaults to Scott's Rule)
    pdf_object = gaussian_kde(y_train_flat, bw_method='scott')
    
    return cdf_object, pdf_object

def calculate_all_distribution_metrics_baseline(
    y_test_orig,
    cdf_model, 
    pdf_model,
    N_grid_points
):
    """
    y_test_orig: shape (B, O) - original unscaled target runtime values.
    cdf_model: statsmodels ECDF object (fitted on Z-space training data).
    pdf_model: scipy gaussian_kde object (fitted on Z-space training data).
    N_grid_points: total number of points in the piecewise non-uniform grid for CDF evaluation.

    returns: - metrics_summary: summary dict, instance_summary: dict of per-instance metrics
    """
    
    # 1. Cast inputs to numpy for scipy/statsmodels compatibility
    is_tensor = isinstance(y_test_orig, torch.Tensor)
    if is_tensor:
        y_test_orig_np = y_test_orig.detach().cpu().numpy()
    else:
        y_test_orig_np = np.asarray(y_test_orig)
        
    B, O = y_test_orig_np.shape
    
    # 2. Convert original Y space into Z-space (log1p scaled)
    z_test_np = np.log1p(y_test_orig_np)
    
    # =========================================================
    # 1. CORE BOUNDS (Empirical Data in Z-Space)
    # =========================================================
    min_z_emp = z_test_np.min(axis=1, keepdims=True)  # shape (B, 1)
    max_z_emp = z_test_np.max(axis=1, keepdims=True)  # shape (B, 1)
    z_range = np.maximum(max_z_emp - min_z_emp, 1e-5)
    
    core_start = min_z_emp - 0.05 * z_range
    core_end = max_z_emp + 0.05 * z_range

    # =========================================================
    # 2. BASELINE TAIL BOUNDS VIA ECDF MIN/MAX
    # =========================================================
    # statsmodels ECDF stores the sorted training points in `.x`. 
    # Index 0 is always -inf, Index 1 is the training minimum, Index -1 is the training maximum.
    z_model_min = cdf_model.x[1]
    z_model_max = cdf_model.x[-1]

    # Map these boundaries into the grid logic
    global_start = np.minimum(core_start - 0.5 * z_range, z_model_min)
    global_end = np.maximum(core_end + 0.5 * z_range, z_model_max)

    # =========================================================
    # 3. PIECEWISE NON-UNIFORM GRID
    # =========================================================
    left_pts, core_pts, right_pts = int(N_grid_points * 1/6), int(N_grid_points * 2/3), int(N_grid_points * 1/6)

    steps_left = np.linspace(0, 1, left_pts).reshape(1, -1)
    z_grid_left = global_start + steps_left * (core_start - global_start)

    steps_core = np.linspace(0, 1, core_pts + 1).reshape(1, -1)[:, 1:]
    z_grid_core = core_start + steps_core * (core_end - core_start)

    steps_right = np.linspace(0, 1, right_pts + 1).reshape(1, -1)[:, 1:]
    z_grid_right = core_end + steps_right * (global_end - core_end)

    z_grid = np.concatenate([z_grid_left, z_grid_core, z_grid_right], axis=1)  # shape (B, N_grid_points)

    # =========================================================
    # 4. CDF EVALUATION & INTEGRATION (Apples-to-Apples in Z-space)
    # =========================================================
    # Evaluate Empirical CDF
    indicator = (z_test_np[:, np.newaxis, :] <= z_grid[:, :, np.newaxis]).astype(float)
    F_emp = indicator.mean(axis=2)  # shape (B, N_grid_points)
    
    # Evaluate Baseline Model CDF (Vectorized)
    F_model = cdf_model(z_grid.flatten()).reshape(B, N_grid_points)
    
    cdf_diff = F_model - F_emp
    abs_cdf_diff = np.abs(cdf_diff)
    
    # Integration over dz
    all_w1 = np.trapezoid(abs_cdf_diff, x=z_grid, axis=1)
    all_ks = np.max(abs_cdf_diff, axis=1)

    # CRPS calculation
    # 1. Base integral: the Cramér–von Mises distance (requires grid)
    cvm_distance = np.trapezoid(cdf_diff ** 2, x=z_grid, axis=1)

    # 2. Exact Empirical Spread (Calculated directly from observations)
    z_sorted = np.sort(z_test_np, axis=1)
    diffs = z_sorted[:, 1:] - z_sorted[:, :-1]  # shape: (B, O-1)
    
    i = np.arange(1, O, dtype=float)
    weights = (i / O) * (1.0 - i / O)           # shape: (O-1,)
    
    empirical_spread = np.sum(weights * diffs, axis=1) # shape: (B,)
    all_crps = cvm_distance + empirical_spread

    # =========================================================
    # 5. VECTORIZED NLLH (in log-space)
    # =========================================================
    # Evaluate the Gaussian KDE.
    nlog_pdf = -pdf_model.logpdf(z_test_np.flatten()).reshape(B, O)
    jacobian_correction = -np.log(np.max(z_test_np, axis=1))

    assert nlog_pdf.shape[0] == jacobian_correction.shape[0] == B, "Batch size mismatch in NLLH calculation"
    
    clamp_val = -np.log(EPS)
    nlog_pdf = np.clip(nlog_pdf, a_min=None, a_max=clamp_val)
    
    all_nllh = nlog_pdf.mean(axis=1) + jacobian_correction

    # =========================================================
    # 6. SUMMARY & RETURNS
    # =========================================================
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
    
    # Convert per-instance arrays back to Torch tensors for downstream logging compatibility
    instance_summary = {
        "NLLH": torch.tensor(all_nllh, dtype=torch.float32), 
        "CRPS": torch.tensor(all_crps, dtype=torch.float32), 
        "Wasserstein": torch.tensor(all_w1, dtype=torch.float32), 
        "KS": torch.tensor(all_ks, dtype=torch.float32)
    }

    return metrics_summary, instance_summary