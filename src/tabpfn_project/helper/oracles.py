import torch
from tabpfn_project.globals import MIN_CLAMP_LLH

def calculate_all_distribution_metrics_oracle_lognormal(
    y_test_orig,
    *,
    device, 
    N_grid_points,
):
    """
    y_test_orig: shape (B, O) - original unscaled targets.
    device: torch device for computation.
    N_grid_points: total number of points in the piecewise non-uniform grid for CDF evaluation.

    returns: - metrics_summary: summary dict, instance_summary: dict of per-instance metrics
    """
    # Ensure inputs are correct dtype and on the correct device
    y_test_orig = torch.as_tensor(y_test_orig, dtype=torch.float32, device=device)
    
    # ---------------------------------------------------------
    # 0. ORACLE FIT (LogNormal modeled directly on Original Y)
    # ---------------------------------------------------------
    log_y = torch.log(y_test_orig)
    # Oracle calculates ground-truth mu and sigma per instance
    mu = log_y.mean(dim=1, keepdim=True)
    sigma = log_y.std(dim=1, correction=0, keepdim=True)
    
    # Clamp sigma to prevent division-by-zero crashes in the distribution PDF
    sigma = torch.clamp(sigma, min=1e-10)
    
    # Initialize Distribution in the original Y-space
    dist = torch.distributions.LogNormal(loc=mu, scale=sigma)

    # Convert to unified integration Z-space for distance metrics evaluation
    z_test_orig = torch.log1p(y_test_orig)

    # =========================================================
    # 1. CORE BOUNDS (Empirical Data in Z-Space)
    # =========================================================
    min_z_emp = z_test_orig.min(dim=1, keepdim=True)[0]  # shape (B, 1)
    max_z_emp = z_test_orig.max(dim=1, keepdim=True)[0]  # shape (B, 1)
    z_range = (max_z_emp - min_z_emp).clamp(min=1e-5)
    
    core_start = min_z_emp - 0.05 * z_range
    core_end = max_z_emp + 0.05 * z_range

    # =========================================================
    # 2. ORACLE TAIL BOUNDS VIA INVERSE CDF (ICDF)
    # =========================================================
    # Oracle natively models original Y-space.
    # Query exact 0.01% and 99.99% quantiles in Y-space.
    p_min = torch.tensor(0.0001, device=device)
    p_max = torch.tensor(0.9999, device=device)

    y_orig_min = dist.icdf(p_min)
    y_orig_max = dist.icdf(p_max)

    # Map these rigorous boundaries into our unified integration Z-space (log1p)
    z_model_min = torch.log1p(y_orig_min)
    z_model_max = torch.log1p(y_orig_max)

    global_start = torch.minimum(core_start - 0.5 * z_range, z_model_min)
    global_end = torch.maximum(core_end + 0.5 * z_range, z_model_max)

    # =========================================================
    # 3. 15K PIECEWISE NON-UNIFORM GRID (in Z-space)
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
    F_emp = indicator.mean(dim=2)  # shape (B, N_grid_points) - empirical CDF evaluated at each grid point
    
    # Evaluate Oracle CDF
    # Convert Z-grid back to original Y-space to query the Oracle's CDF
    y_grid = torch.expm1(z_grid)
    F_model = dist.cdf(y_grid.clamp(min=0))
    
    cdf_diff = F_model - F_emp
    abs_cdf_diff = torch.abs(cdf_diff)
    
    # Integration over dz
    all_w1 = torch.trapezoid(abs_cdf_diff, x=z_grid, dim=1)  
    all_ks = torch.max(abs_cdf_diff, dim=1)[0]                   

    # CRPS calculation (in log1p units)
    # 1. Base integral: the Cramér–von Mises distance
    cvm_distance = torch.trapezoid(cdf_diff ** 2, x=z_grid, dim=1)

    # 2. Exact Empirical Spread (Calculated from strictly sorted Z-space observations)
    z_sorted = torch.sort(z_test_orig, dim=1)[0]
    diffs = z_sorted[:, 1:] - z_sorted[:, :-1]  # shape: (B, N-1)
    
    N = z_test_orig.shape[1]
    i = torch.arange(1, N, device=device).float()
    weights = (i / N) * (1.0 - i / N)           # shape: (N-1,)
    
    empirical_spread = torch.sum(weights * diffs, dim=1) # shape: (B,)
    
    all_crps = cvm_distance + empirical_spread

    # =========================================================
    # 5. VECTORIZED NLLH (Per original Oracle formulation)
    # =========================================================
    llh = dist.log_prob(y_test_orig)
    llh.clamp_(min=MIN_CLAMP_LLH)

    assert z_test_orig.shape == llh.shape, f"Shape mismatch in llh correction: {z_test_orig.shape} vs {llh.shape}"
    llh += z_test_orig
    
    instance_nllh = -llh.mean(dim=1)
    
    # Simple max scaling bias correction used by Oracle,
    bias = -torch.log(z_test_orig.max(dim=1)[0])

    assert bias.shape == instance_nllh.shape, f"Shape mismatch in bias calculation: {bias.shape} vs {instance_nllh.shape}"
    all_nllh = instance_nllh + bias

    # =========================================================
    # 6. RETURN DICTIONARIES
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
    
    instance_summary = {
        "NLLH": all_nllh.detach().cpu(), 
        "CRPS": all_crps.detach().cpu(), 
        "Wasserstein": all_w1.detach().cpu(), 
        "KS": all_ks.detach().cpu()
    }

    return metrics_summary, instance_summary