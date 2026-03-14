import torch

def calculate_all_distribution_metrics_distnet_logspace(
    y_test_orig,
    preds, 
    *,
    device, 
    y_scaler,
    N_grid_points,
):
    """
    y_test_orig: shape (B, O) - original unscaled targets
    preds: shape (B, 2) - DistNet's predicted parameters (sigma, mu) in max-scaled space
    y_scaler: scaler value - the max-scaling factor used to convert from original Y space to DistNet's max-scaled space
    device: torch device for computation
    N_grid_points: total number of points in the piecewise non-uniform grid for CDF evaluation

    returns: - metrics_summary: summary dict, instance_summary: dict of per-instance metrics
    
    """
    y_test_orig = torch.as_tensor(y_test_orig, dtype=torch.float32, device=device)
    preds = torch.as_tensor(preds, dtype=torch.float32, device=device)
    y_scaler = torch.as_tensor(y_scaler, dtype=torch.float32, device=device)
    
    z_test_orig = torch.log1p(y_test_orig)

    sigma = preds[:, 0].unsqueeze(1)
    mu = torch.log(preds[:, 1]).unsqueeze(1)  
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
    # 2. DISTNET TAIL BOUNDS VIA INVERSE CDF (ICDF)
    # =========================================================
    # DistNet's distribution natively models the max-scaled space.
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
    # 4. CDF EVALUATION & INTEGRATION (Apples-to-Apples in Z-space (i.e., log-scaled space))
    # =========================================================
    indicator = (z_test_orig.unsqueeze(1) <= z_grid.unsqueeze(2)).float()  
    F_emp = indicator.mean(dim=2)  # shape (B, N_grid_points) - empirical CDF evaluated at each grid point
    
    # Evaluate DistNet CDF
    y_orig = torch.exp(z_grid) - 1.0
    y_scaled = y_orig * y_scaler  
    
    F_model = dist.cdf(y_scaled.clamp(min=0))
    
    cdf_diff = F_model - F_emp
    abs_cdf_diff = torch.abs(cdf_diff)
    
    # Integration over dz
    all_w1 = torch.trapezoid(abs_cdf_diff, x=z_grid, dim=1)  
    all_ks = torch.max(abs_cdf_diff, dim=1)[0]                   

    # Integration is over dz, returning CRPS in log-units (interpretation: Relative Error)
    # CRPS calculation
    # 1. Base integral: the Cramér–von Mises distance (requires grid)
    cvm_distance = torch.trapezoid(cdf_diff ** 2, x=z_grid, dim=1)

    # 2. Exact Empirical Spread (Calculated directly from observations)
    # Sort the ground-truth observations along the instance dimension
    z_sorted = torch.sort(z_test_orig, dim=1)[0]
    
    # Calculate the physical distance between consecutive sorted observations
    diffs = z_sorted[:, 1:] - z_sorted[:, :-1]  # shape: (B, N-1)
    
    # Generate the exact probability weights for each rectangle
    N = z_test_orig.shape[1]
    i = torch.arange(1, N, device=device).float()
    weights = (i / N) * (1.0 - i / N)           # shape: (N-1,)
    
    # Compute exact integral of the spread via dot product
    empirical_spread = torch.sum(weights * diffs, dim=1) # shape: (B,)
    
    all_crps = cvm_distance + empirical_spread

    # =========================================================
    # 5. VECTORIZED NLLH (in log-space)
    # =========================================================
    y_test_scaled = y_test_orig * y_scaler
    nlog_pdf = -dist.log_prob(y_test_scaled)  # shape (B, O)
    nlog_pdf.clamp_(max=200.0)  # 200 corresponds to -log(1e-87); prevents possible inf's due to precision errors; same threshold used for tabpfn

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
    
    instance_summary = {"NLLH": all_nllh.detach().cpu(), "CRPS": all_crps.detach().cpu(), "Wasserstein": all_w1.detach().cpu(), "KS": all_ks.detach().cpu()}

    return metrics_summary, instance_summary
