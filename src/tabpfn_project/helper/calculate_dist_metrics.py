import torch
from tabpfn_project.globals import MIN_CLAMP_LLH

def _create_non_uniform_grid(global_start, core_start, core_end, global_end, N_grid_points, device=None):
    """
    Generates the 1/6, 2/3, 1/6 piecewise non-uniform grid used across all metrics.
    """
    left_pts = int(N_grid_points * 1/6)
    core_pts = int(N_grid_points * 2/3)
    right_pts = int(N_grid_points * 1/6)

    steps_left = torch.linspace(0, 1, left_pts, device=device).view(1, -1)
    z_grid_left = global_start + steps_left * (core_start - global_start)

    steps_core = torch.linspace(0, 1, core_pts + 1, device=device).view(1, -1)[:, 1:]
    z_grid_core = core_start + steps_core * (core_end - core_start)

    steps_right = torch.linspace(0, 1, right_pts + 1, device=device).view(1, -1)[:, 1:]
    z_grid_right = core_end + steps_right * (global_end - core_end)

    return torch.cat([z_grid_left, z_grid_core, z_grid_right], dim=1)

def _compute_empirical_spread(z_test_orig, device):
    """
    Calculates the exact physical distance integral of the empirical spread.
    """
    z_sorted = torch.sort(z_test_orig, dim=1)[0]
    diffs = z_sorted[:, 1:] - z_sorted[:, :-1]
    
    N = z_test_orig.shape[1]
    i = torch.arange(1, N, device=device).float()
    weights = (i / N) * (1.0 - i / N)
    
    return torch.sum(weights * diffs, dim=1)

def _integrate_distribution_metrics(z_grid, F_emp, F_model, z_test_orig, device):
    """
    Performs the core integration for Wasserstein, KS, and CRPS.
    """
    cdf_diff = F_model - F_emp
    abs_cdf_diff = torch.abs(cdf_diff)
    
    # Wasserstein (W1) and KS
    w1 = torch.trapezoid(abs_cdf_diff, x=z_grid, dim=1)
    ks = torch.max(abs_cdf_diff, dim=1)[0]
    
    # CRPS = CVM distance + Empirical Spread
    cvm_distance = torch.trapezoid(cdf_diff ** 2, x=z_grid, dim=1)
    emp_spread = _compute_empirical_spread(z_test_orig, device)
    crps = cvm_distance + emp_spread
    
    return w1, ks, crps

def _format_metrics_output(all_nllh, all_crps, all_w1, all_ks):
    """
    Standardizes the return dictionary for all calculator functions.
    """
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

def calculate_all_distribution_metrics_distnet_logspace(
    y_test_orig, preds, *, device, target_scale, y_scaler, N_grid_points,
):
    assert target_scale in ["max"], "distnet supports max scaler only." 

    y_test_orig = torch.as_tensor(y_test_orig, dtype=torch.float32, device=device)
    preds = torch.as_tensor(preds, dtype=torch.float32, device=device)
    y_scaler = torch.as_tensor(y_scaler, dtype=torch.float32, device=device)
    z_test_orig = torch.log1p(y_test_orig)

    sigma = preds[:, 0].unsqueeze(1)
    mu = torch.log(preds[:, 1]).unsqueeze(1)  
    dist = torch.distributions.LogNormal(loc=mu, scale=sigma)

    # 1. Bounds
    min_z_emp = z_test_orig.min(dim=1, keepdim=True)[0]
    max_z_emp = z_test_orig.max(dim=1, keepdim=True)[0]
    z_range = (max_z_emp - min_z_emp).clamp(min=1e-5)
    core_start, core_end = min_z_emp - 0.05 * z_range, max_z_emp + 0.05 * z_range

    p_min, p_max = torch.tensor(0.0001, device=device), torch.tensor(0.9999, device=device)
    z_model_min = torch.log1p(dist.icdf(p_min) / y_scaler)
    z_model_max = torch.log1p(dist.icdf(p_max) / y_scaler)

    global_start = torch.minimum(core_start - 0.5 * z_range, z_model_min)
    global_end = torch.maximum(core_end + 0.5 * z_range, z_model_max)

    # 2. Grid and Integration
    z_grid = _create_non_uniform_grid(global_start, core_start, core_end, global_end, N_grid_points, device)
    
    indicator = (z_test_orig.unsqueeze(1) <= z_grid.unsqueeze(2)).float()  
    F_emp = indicator.mean(dim=2)
    F_model = dist.cdf((torch.expm1(z_grid) * y_scaler).clamp(min=0))
    
    all_w1, all_ks, all_crps = _integrate_distribution_metrics(z_grid, F_emp, F_model, z_test_orig, device)

    # 3. NLLH with Jacobian and Bias
    y_test_scaled = y_test_orig * y_scaler
    llh = dist.log_prob(y_test_scaled).clamp(min=MIN_CLAMP_LLH)
    llh += z_test_orig  # Jacobian correction
    bias = -torch.log(torch.max(z_test_orig, dim=1)[0]) - torch.log(y_scaler)
    all_nllh = -llh.mean(dim=1) + bias

    return _format_metrics_output(all_nllh, all_crps, all_w1, all_ks)

def calculate_all_distribution_metrics_tabpfn_logspace(
    y_test_orig, tabpfn_preds, *, device, target_scale, N_grid_points,
):
    from tabpfn_project.helper.tabpfn_helpers import cdf_tabpfn, halfnormal_with_p_weight_before, log_pdf_tabpfn
    assert target_scale in ["log", "original"], "target_scale must be 'log' or 'original'"

    y_test_orig = torch.as_tensor(y_test_orig, dtype=torch.float32, device=device)
    z_test_orig = torch.log1p(y_test_orig)

    all_crps, all_w1, all_ks, all_nllh = [], [], [], []
    instance_idx = 0
    for batch in tabpfn_preds:
        logits = batch['logits'].to(device)
        borders = batch['criterion'].borders.to(device)
        batch_size = logits.shape[0]
        batch_y_orig = y_test_orig[instance_idx : instance_idx + batch_size]
        batch_z_orig = z_test_orig[instance_idx : instance_idx + batch_size]
        
        # 1. Bounds
        min_z_emp = batch_z_orig.min(dim=1, keepdim=True)[0]
        max_z_emp = batch_z_orig.max(dim=1, keepdim=True)[0]
        z_range = (max_z_emp - min_z_emp).clamp(min=1e-5)
        core_start, core_end = min_z_emp - 0.05 * z_range, max_z_emp + 0.05 * z_range

        bucket_widths = borders[1:] - borders[:-1]
        hn_left, hn_right = halfnormal_with_p_weight_before(bucket_widths[0]), halfnormal_with_p_weight_before(bucket_widths[-1])
        p_val = torch.tensor(0.9999, device=device)
        
        if target_scale == "log":
            z_model_min, z_model_max = borders[1] - hn_left.icdf(p_val), borders[-2] + hn_right.icdf(p_val)
        else: # original
            y_model_min, y_model_max = borders[1] - hn_left.icdf(p_val), borders[-2] + hn_right.icdf(p_val)
            z_model_min, z_model_max = torch.log1p(torch.clamp(y_model_min, min=0.0)), torch.log1p(torch.clamp(y_model_max, min=0.0))

        global_start = torch.minimum(core_start - 0.5 * z_range, z_model_min.unsqueeze(0).expand(batch_size, -1))
        global_end = torch.maximum(core_end + 0.5 * z_range, z_model_max.unsqueeze(0).expand(batch_size, -1))

        # 2. Grid and Integration
        z_grid = _create_non_uniform_grid(global_start, core_start, core_end, global_end, N_grid_points, device)
        
        indicator = (batch_z_orig.unsqueeze(1) <= z_grid.unsqueeze(2)).float()
        F_emp = indicator.mean(dim=2)
        
        if target_scale == "log":
            F_tab = cdf_tabpfn(logits, z_grid, borders)
        else: # original
            F_tab_raw = cdf_tabpfn(logits, torch.expm1(z_grid), borders)
            F_tab = torch.where(z_grid < 0, torch.zeros_like(F_tab_raw), F_tab_raw)
        
        w1_b, ks_b, crps_b = _integrate_distribution_metrics(z_grid, F_emp, F_tab, batch_z_orig, device)
        all_w1.append(w1_b.detach().cpu()); all_ks.append(ks_b.detach().cpu()); all_crps.append(crps_b.detach().cpu())
        
        # 3. NLLH
        if target_scale == "log":
            llh = log_pdf_tabpfn(logits, batch_z_orig, borders).clamp(min=MIN_CLAMP_LLH)
            bias = -torch.log(torch.max(batch_z_orig, dim=1)[0])
        else: # original
            llh = log_pdf_tabpfn(logits, batch_y_orig, borders).clamp(min=MIN_CLAMP_LLH)
            llh += batch_z_orig # Jacobian
            bias = -torch.log(torch.max(batch_z_orig, dim=1)[0])

        all_nllh.append((-llh.mean(dim=1) + bias).detach().cpu())
        instance_idx += batch_size
        
    return _format_metrics_output(torch.cat(all_nllh), torch.cat(all_crps), torch.cat(all_w1), torch.cat(all_ks))

def calculate_all_distribution_metrics_logNormalDist_logspace(
    y_test_orig, lognormal_dist, *, device, N_grid_points,
):
    y_test_orig = torch.as_tensor(y_test_orig, dtype=torch.float32, device=device)
    z_test_orig = torch.log1p(y_test_orig)

    min_z_emp = z_test_orig.min(dim=1, keepdim=True)[0]
    max_z_emp = z_test_orig.max(dim=1, keepdim=True)[0]
    z_range = (max_z_emp - min_z_emp).clamp(min=1e-5)
    core_start, core_end = min_z_emp - 0.05 * z_range, max_z_emp + 0.05 * z_range

    p_min, p_max = torch.tensor(0.0001, device=device), torch.tensor(0.9999, device=device)
    z_model_min, z_model_max = torch.log1p(lognormal_dist.icdf(p_min)), torch.log1p(lognormal_dist.icdf(p_max))

    global_start = torch.minimum(core_start - 0.5 * z_range, z_model_min)
    global_end = torch.maximum(core_end + 0.5 * z_range, z_model_max)

    z_grid = _create_non_uniform_grid(global_start, core_start, core_end, global_end, N_grid_points, device)
    
    indicator = (z_test_orig.unsqueeze(1) <= z_grid.unsqueeze(2)).float()  
    F_emp = indicator.mean(dim=2)
    F_model = lognormal_dist.cdf(torch.expm1(z_grid).clamp(min=0))
    
    all_w1, all_ks, all_crps = _integrate_distribution_metrics(z_grid, F_emp, F_model, z_test_orig, device)

    llh = lognormal_dist.log_prob(y_test_orig).clamp(min=MIN_CLAMP_LLH)
    llh += z_test_orig
    bias = -torch.log(z_test_orig.max(dim=1)[0])
    all_nllh = -llh.mean(dim=1) + bias

    return _format_metrics_output(all_nllh, all_crps, all_w1, all_ks)

def calculate_all_distribution_metrics_randomForest_logspace(
    y_test_orig, preds, *, device, N_grid_points,
):
    assert preds[0].ndim == 1 and preds[1].ndim == 1, "Preds must be 1-dimensional"
    y_test_orig = torch.as_tensor(y_test_orig, dtype=torch.float32, device=device)
    means = torch.as_tensor(preds[0], dtype=torch.float32, device=device).unsqueeze(1)
    variances = torch.as_tensor(preds[1], dtype=torch.float32, device=device).unsqueeze(1)
    z_test_orig = torch.log1p(y_test_orig)

    dist = torch.distributions.Normal(loc=means, scale=torch.sqrt(variances))

    min_z_emp = z_test_orig.min(dim=1, keepdim=True)[0]
    max_z_emp = z_test_orig.max(dim=1, keepdim=True)[0]
    z_range = (max_z_emp - min_z_emp).clamp(min=1e-5)
    core_start, core_end = min_z_emp - 0.05 * z_range, max_z_emp + 0.05 * z_range

    p_min, p_max = torch.tensor(0.0001, device=device), torch.tensor(0.9999, device=device)
    z_model_min, z_model_max = dist.icdf(p_min), dist.icdf(p_max)

    global_start = torch.minimum(core_start - 0.5 * z_range, z_model_min)
    global_end = torch.maximum(core_end + 0.5 * z_range, z_model_max)

    z_grid = _create_non_uniform_grid(global_start, core_start, core_end, global_end, N_grid_points, device)
    
    indicator = (z_test_orig.unsqueeze(1) <= z_grid.unsqueeze(2)).float()  
    F_emp = indicator.mean(dim=2)
    F_model = dist.cdf(z_grid)
    
    all_w1, all_ks, all_crps = _integrate_distribution_metrics(z_grid, F_emp, F_model, z_test_orig, device)

    llh = dist.log_prob(z_test_orig).clamp(min=MIN_CLAMP_LLH)
    bias = -torch.log(torch.max(z_test_orig, dim=1)[0])
    all_nllh = -llh.mean(dim=1) + bias

    return _format_metrics_output(all_nllh, all_crps, all_w1, all_ks)
