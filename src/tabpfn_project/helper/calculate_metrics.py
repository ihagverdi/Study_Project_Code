import torch
from tabpfn_project.globals import MIN_CLAMP_LLH

def calculate_metrics_distnet(
    y_test_original, y_preds, *, device, target_scale, y_scaler, N_grid_points,
):
    assert target_scale in ["max"], "target_scale must be 'max' to ensure proper scaling of the distribution outputs for DistNet."

    y_test_original = torch.as_tensor(y_test_original, dtype=torch.float32, device=device)
    y_preds = torch.as_tensor(y_preds, dtype=torch.float32, device=device)
    y_scaler = torch.as_tensor(y_scaler, dtype=torch.float32, device=device)

    z_test_original = torch.log1p(y_test_original)

    sigma = y_preds[:, 0].unsqueeze(1)
    mu = torch.log(y_preds[:, 1]).unsqueeze(1)  
    dist = torch.distributions.LogNormal(loc=mu, scale=sigma)

    # 1. Bounds
    min_z_emp = z_test_original.min(dim=1, keepdim=True)[0]
    max_z_emp = z_test_original.max(dim=1, keepdim=True)[0]

    p_min, p_max = torch.tensor(0.0001, device=device), torch.tensor(0.9999, device=device)
    min_z_model = torch.log1p(dist.icdf(p_min) / y_scaler)
    max_z_model = torch.log1p(dist.icdf(p_max) / y_scaler)

    # 2. Grid and Integration
    global_start = torch.minimum(min_z_emp, min_z_model)
    global_end = torch.maximum(max_z_emp, max_z_model)

    steps = torch.linspace(0, 1, N_grid_points, device=device).view(1, -1)
    z_grid = global_start + steps * (global_end - global_start)
    
    indicator = (z_test_original.unsqueeze(1) <= z_grid.unsqueeze(2)).float()
    F_emp = indicator.mean(dim=2)  # Empirical CDF at grid points
    F_model = dist.cdf((torch.expm1(z_grid) * y_scaler).clamp(min=0))

    assert F_emp.shape == F_model.shape == (y_test_original.shape[0], N_grid_points), "CDF shapes must match (N_instances, N_grid_points)"
    
    all_crps, all_w1, all_ks, = _integrate_distribution_metrics(z_grid, F_emp, F_model, z_test_original, device)

    # 3. NLLH with Jacobian and Bias
    y_test_scaled = y_test_original * y_scaler
    llh = dist.log_prob(y_test_scaled).clamp(min=MIN_CLAMP_LLH)
    llh += z_test_original  # Jacobian correction
    bias = -torch.log(torch.max(z_test_original, dim=1)[0]) - torch.log(y_scaler)

    all_nllh = -llh.mean(dim=1) + bias

    return _format_metrics_output(all_nllh, all_crps, all_w1, all_ks)

def calculate_metrics_tabpfn(
    y_test_original, tabpfn_preds, *, device, target_scale, N_grid_points,
):
    assert target_scale in ["log", "original"], "target_scale must be 'log' or 'original'"

    y_test_original = torch.as_tensor(y_test_original, dtype=torch.float32, device=device)
    z_test_original = torch.log1p(y_test_original)

    all_crps, all_w1, all_ks, all_nllh = [], [], [], []
    instance_idx = 0
    for batch in tabpfn_preds:
        logits = batch['logits'].to(device)
        criterion = batch['criterion'].to(device)
        borders = criterion.borders

        batch_size = logits.shape[0]  # number of instances in the prediction batch
        batch_y_original = y_test_original[instance_idx : instance_idx + batch_size]
        batch_z_original = z_test_original[instance_idx : instance_idx + batch_size]
        
        # 1. Bounds
        min_z_emp = batch_z_original.min(dim=1, keepdim=True)[0]
        max_z_emp = batch_z_original.max(dim=1, keepdim=True)[0]

        bucket_widths = criterion.bucket_widths
        hn_left, hn_right = criterion.halfnormal_with_p_weight_before(bucket_widths[0]), criterion.halfnormal_with_p_weight_before(bucket_widths[-1])
        p_val = torch.tensor(0.9999, device=device)
        
        min_y_model, max_y_model = borders[1] - hn_left.icdf(p_val), borders[-2] + hn_right.icdf(p_val)
        if target_scale == "log":
            min_z_model, max_z_model = min_y_model, max_y_model
        elif target_scale == "original":
            min_z_model, max_z_model = torch.log1p(torch.clamp(min_y_model, min=0.0)), torch.log1p(torch.clamp(max_y_model, min=0.0))

        # 2. Grid and Integration
        global_start = torch.minimum(min_z_emp, min_z_model)
        global_end = torch.maximum(max_z_emp, max_z_model)

        steps = torch.linspace(0, 1, N_grid_points, device=device).view(1, -1)
        z_grid = global_start + steps * (global_end - global_start)

        indicator = (batch_z_original.unsqueeze(1) <= z_grid.unsqueeze(2)).float()
        F_emp = indicator.mean(dim=2)
        
        if target_scale == "log":
            F_tab = criterion.cdf(logits=logits, y=z_grid)
        elif target_scale == "original":
            F_tab_raw = criterion.cdf(logits=logits, y=torch.expm1(z_grid))
            F_tab = torch.where(z_grid < 0, torch.zeros_like(F_tab_raw), F_tab_raw)
        
        crps_b, w1_b, ks_b, = _integrate_distribution_metrics(z_grid, F_emp, F_tab, batch_z_original, device)
        all_w1.append(w1_b.detach().cpu()); all_ks.append(ks_b.detach().cpu()); all_crps.append(crps_b.detach().cpu())
        
        # 3. NLLH
        if target_scale == "log":
            llh = torch.log(criterion.pdf(logits=logits, y=batch_z_original)).clamp(min=MIN_CLAMP_LLH)
            bias = -torch.log(torch.max(batch_z_original, dim=1)[0])
        elif target_scale == "original":
            llh = torch.log(criterion.pdf(logits=logits, y=batch_y_original)).clamp(min=MIN_CLAMP_LLH)
            llh += batch_z_original # Jacobian correction
            bias = -torch.log(torch.max(batch_z_original, dim=1)[0])

        batch_nllh = -llh.mean(dim=1) + bias
        all_nllh.append(batch_nllh.detach().cpu())
        instance_idx += batch_size
        
    return _format_metrics_output(torch.cat(all_nllh), torch.cat(all_crps), torch.cat(all_w1), torch.cat(all_ks))

def calculate_metrics_lognormal(
    y_test_original, lognormal_dist, *, device, N_grid_points,
):
    y_test_original = torch.as_tensor(y_test_original, dtype=torch.float32, device=device)
    z_test_original = torch.log1p(y_test_original)

    min_z_emp = z_test_original.min(dim=1, keepdim=True)[0]
    max_z_emp = z_test_original.max(dim=1, keepdim=True)[0]

    p_min, p_max = torch.tensor(0.0001, device=device), torch.tensor(0.9999, device=device)
    min_z_model, max_z_model = torch.log1p(lognormal_dist.icdf(p_min)), torch.log1p(lognormal_dist.icdf(p_max))

    global_start = torch.minimum(min_z_emp, min_z_model)
    global_end = torch.maximum(max_z_emp, max_z_model)

    steps = torch.linspace(0, 1, N_grid_points, device=device).view(1, -1)
    z_grid = global_start + steps * (global_end - global_start)

    indicator = (z_test_original.unsqueeze(1) <= z_grid.unsqueeze(2)).float()  
    F_emp = indicator.mean(dim=2)
    F_model = lognormal_dist.cdf(torch.expm1(z_grid).clamp(min=0))
    
    all_crps, all_w1, all_ks = _integrate_distribution_metrics(z_grid, F_emp, F_model, z_test_original, device)

    llh = lognormal_dist.log_prob(y_test_original).clamp(min=MIN_CLAMP_LLH)
    llh += z_test_original  # Jacobian correction
    bias = -torch.log(z_test_original.max(dim=1)[0])
    all_nllh = -llh.mean(dim=1) + bias

    return _format_metrics_output(all_nllh, all_crps, all_w1, all_ks)

def calculate_metrics_random_forest(
    y_test_original, preds, *, device, N_grid_points,
):
    assert (preds[0].ndim == preds[1].ndim == 2) and (len(preds[0]) == len(preds[1])), "Preds must be two 2D arrays of the same length (means and variances)"
    
    y_test_original = torch.as_tensor(y_test_original, dtype=torch.float32, device=device)
    z_test_original = torch.log1p(y_test_original)

    means = torch.as_tensor(preds[0], dtype=torch.float32, device=device)
    variances = torch.as_tensor(preds[1], dtype=torch.float32, device=device)

    assert means.shape == variances.shape == (y_test_original.shape[0], 1), "Means and variances must have shape (N_instances, 1)"

    dist = torch.distributions.Normal(loc=means, scale=torch.sqrt(variances))

    min_z_emp = z_test_original.min(dim=1, keepdim=True)[0]
    max_z_emp = z_test_original.max(dim=1, keepdim=True)[0]

    p_min, p_max = torch.tensor(0.0001, device=device), torch.tensor(0.9999, device=device)
    min_z_model, max_z_model = dist.icdf(p_min), dist.icdf(p_max)

    global_start = torch.minimum(min_z_emp, min_z_model)
    global_end = torch.maximum(max_z_emp, max_z_model)
    steps = torch.linspace(0, 1, N_grid_points, device=device).view(1, -1)
    z_grid = global_start + steps * (global_end - global_start)
    
    indicator = (z_test_original.unsqueeze(1) <= z_grid.unsqueeze(2)).float()  
    F_emp = indicator.mean(dim=2)
    F_model = dist.cdf(z_grid)
    
    all_crps, all_w1, all_ks = _integrate_distribution_metrics(z_grid, F_emp, F_model, z_test_original, device)

    llh = dist.log_prob(z_test_original).clamp(min=MIN_CLAMP_LLH)
    bias = -torch.log(torch.max(z_test_original, dim=1)[0])
    all_nllh = -llh.mean(dim=1) + bias

    return _format_metrics_output(all_nllh, all_crps, all_w1, all_ks)

def _compute_empirical_spread(z_test_original, device):
    """
    Calculates the exact physical distance integral of the empirical spread.
    """
    z_sorted = torch.sort(z_test_original, dim=1)[0]
    diffs = z_sorted[:, 1:] - z_sorted[:, :-1]
    
    N = z_test_original.shape[1]
    i = torch.arange(1, N, device=device).float()
    weights = (i / N) * (1.0 - i / N)
    
    return torch.sum(weights * diffs, dim=1)

def _integrate_distribution_metrics(z_grid, F_emp, F_model, z_test_original, device):
    """
    Performs the core integration for CRPS, Wasserstein, and KS.
    """
    cdf_diff = F_model - F_emp
    abs_cdf_diff = torch.abs(cdf_diff)
    
    # Wasserstein (W1) and KS
    w1 = torch.trapezoid(abs_cdf_diff, x=z_grid, dim=1)
    ks = torch.max(abs_cdf_diff, dim=1)[0]
    
    # CRPS = L2 distance + Empirical Spread
    L2_distance = torch.trapezoid(cdf_diff ** 2, x=z_grid, dim=1)
    emp_spread = _compute_empirical_spread(z_test_original, device)
    crps = L2_distance + emp_spread
    
    return crps, w1, ks, 

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
