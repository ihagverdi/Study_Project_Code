from copy import deepcopy
import pathlib
import pickle
import platform
import pandas as pd
import warnings
import torch
import numpy as np
from scipy.stats import wilcoxon
from tabpfn_project.globals import MIN_CLAMP_LLH

def calculate_all_distribution_metrics_distnet_logspace(
    y_test_orig,
    preds, 
    *,
    device,
    target_scale, 
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
    assert target_scale in ["max"], "distnet supports max scaler only." 

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
    y_orig = torch.expm1(z_grid)
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
    llh = dist.log_prob(y_test_scaled)  # shape (B, O)
    
    assert llh.shape == y_test_scaled.shape, f"Shape mismatch: {llh.shape} vs {y_test_scaled.shape}"
    llh.clamp_(min=MIN_CLAMP_LLH)
 

    assert llh.shape == z_test_orig.shape, f"shapes mismatched at nllh calculation: {llh.shape} vs {z_test_orig.shape}"
    llh += z_test_orig  # jacobian correction

    bias = -torch.log(torch.max(z_test_orig, dim=1)[0]) - torch.log(y_scaler)  # shape (B,)

    assert bias.shape == (y_test_orig.shape[0],), f"shapes mismatched at bias calculation: {bias.shape} vs {(y_test_orig.shape[0],)}"
    assert bias.ndim == 1 and llh.ndim == 2, f"unexpected dimensions: bias {bias.ndim} vs llh {llh.ndim}"

    all_nllh = -llh.mean(dim=1) + bias

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

def calculate_all_distribution_metrics_tabpfn_logspace(
    y_test_orig,
    tabpfn_preds,
    *,
    device,
    target_scale,
    N_grid_points,
):
    """
    Compute distribution metrics for TabPFN predictive distributions in log1p space.

    Args:
        y_test_orig (array-like | torch.Tensor):
            Ground-truth targets in original Y space, shape (B, O).
        tabpfn_preds (list[dict]):
            Batched outputs from TabPFN predict(..., output_type="full").
        device (torch.device):
            Device used for tensor computation.
        target_scale (str):
            Native scale of TabPFN predictions. Must be "log" or "original".
        N_grid_points (int):
            Number of points in the piecewise non-uniform integration grid.

    Returns:
        tuple[dict[str, float], dict[str, torch.Tensor]]:
            metrics_summary: Aggregate mean/std metrics.
            instance_summary: Per-instance tensors for NLLH, CRPS, Wasserstein, and KS.
    """
    from tabpfn_project.helper.tabpfn_helpers import (
        cdf_tabpfn, 
        halfnormal_with_p_weight_before, 
        log_pdf_tabpfn
    )
    assert target_scale in ["log", "original"], "target_scale must be 'log' or 'original'"

    y_test_orig = torch.as_tensor(y_test_orig, dtype=torch.float32, device=device)

    # Ground truth in the integration space (Z-space)
    z_test_orig = torch.log1p(y_test_orig)

    all_crps, all_w1, all_ks, all_nllh = [], [], [], []
    instance_idx = 0
    for batch in tabpfn_preds:
        logits = batch['logits'].to(device)  # shape: (B, K)  
        borders = batch['criterion'].borders.to(device)  # shape: (K+1,)

        batch_size = logits.shape[0]
        batch_y_orig = y_test_orig[instance_idx : instance_idx + batch_size]
        batch_z_orig = z_test_orig[instance_idx : instance_idx + batch_size]
        
        # =========================================================
        # 1. CORE BOUNDS (Empirical Data in Z-Space / log1p-space)
        # =========================================================
        min_z_emp = batch_z_orig.min(dim=1, keepdim=True)[0]
        max_z_emp = batch_z_orig.max(dim=1, keepdim=True)[0]
        z_range = (max_z_emp - min_z_emp).clamp(min=1e-5)  # prevent zero range
        
        #  shapes: (B,1)
        core_start = min_z_emp - 0.05 * z_range
        core_end = max_z_emp + 0.05 * z_range

        # =========================================================
        # 2. TABPFN TAIL BOUNDS VIA ICDF
        # =========================================================
        bucket_widths = borders[1:] - borders[:-1]
        
        # Instantiate the HalfNormals for the tails
        hn_left = halfnormal_with_p_weight_before(bucket_widths[0])
        hn_right = halfnormal_with_p_weight_before(bucket_widths[-1])
        
        # Find exactly where the HalfNormal reaches 99.99% of its mass
        p_val = torch.tensor(0.9999, device=device)
        tail_left_ext = hn_left.icdf(p_val)
        tail_right_ext = hn_right.icdf(p_val)
        
        # Mathematical boundaries of TabPFN
        if target_scale == "log":
            # Model is natively in Z-space
            z_model_min = borders[1] - tail_left_ext 
            z_model_max = borders[-2] + tail_right_ext
        elif target_scale == "original":
            # Model is natively in Y-space (original)
            y_model_min = borders[1] - tail_left_ext
            y_model_max = borders[-2] + tail_right_ext
            
            # Map Y bounds to Z bounds.
            # We clamp at 0.0 because any negative probability mass is left-censored 
            # at y=0, meaning it mathematically collapses to z=0.
            z_model_min = torch.log1p(torch.clamp(y_model_min, min=0.0))
            z_model_max = torch.log1p(torch.clamp(y_model_max, min=0.0))

        # Global integration bounds in Z-space
        # Shapes: (B,1)
        global_start = torch.minimum(core_start - 0.5 * z_range, z_model_min.unsqueeze(0).expand(batch_size, -1))
        global_end = torch.maximum(core_end + 0.5 * z_range, z_model_max.unsqueeze(0).expand(batch_size, -1))

        # =========================================================
        # 3. PIECEWISE NON-UNIFORM GRID (Strictly in Z-Space)
        # =========================================================
        left_pts, core_pts, right_pts = int(N_grid_points * 1/6), int(N_grid_points * 2/3), int(N_grid_points * 1/6)

        steps_left = torch.linspace(0, 1, left_pts, device=device).view(1, -1)
        z_grid_left = global_start + steps_left * (core_start - global_start)

        steps_core = torch.linspace(0, 1, core_pts + 1, device=device).view(1, -1)[:, 1:]
        z_grid_core = core_start + steps_core * (core_end - core_start)

        steps_right = torch.linspace(0, 1, right_pts + 1, device=device).view(1, -1)[:, 1:]
        z_grid_right = core_end + steps_right * (global_end - core_end)

        z_grid = torch.cat([z_grid_left, z_grid_core, z_grid_right], dim=1) # shape: (B, N_grid_points)

        # =========================================================
        # 4. CDF EVALUATION & INTEGRATION (Natively in Z-Space)
        # =========================================================
        indicator = (batch_z_orig.unsqueeze(1) <= z_grid.unsqueeze(2)).float()
        F_emp = indicator.mean(dim=2)  # shape: (B, N_grid_points)
        
        if target_scale == "log":
            F_tab = cdf_tabpfn(logits, z_grid, borders)
        elif target_scale == "original":
            # Map Z-grid back to original Y-space to query the linear model
            y_grid = torch.expm1(z_grid)
            F_tab_raw = cdf_tabpfn(logits, y_grid, borders)
            
            # LEFT-CENSORING: Any probability mass assigned to z < 0 (i.e. y < 0) 
            # is physically impossible, so it is strictly censored to 0 probability.
            F_tab = torch.where(z_grid < 0, torch.zeros_like(F_tab_raw), F_tab_raw)
        
        cdf_diff = F_tab - F_emp
        abs_cdf_diff = torch.abs(cdf_diff)
        
        w1_batch = torch.trapezoid(abs_cdf_diff, x=z_grid, dim=1)
        ks_batch = torch.max(abs_cdf_diff, dim=1)[0]               

        # CRPS calculation
        # 1. Base integral: the Cramér–von Mises distance (requires grid)
        cvm_distance = torch.trapezoid(cdf_diff ** 2, x=z_grid, dim=1)
        # 2. Exact Empirical Spread (Calculated directly from observations)
        # Sort the ground-truth observations along the instance dimension
        z_sorted = torch.sort(batch_z_orig, dim=1)[0]
        
        # Calculate the physical distance between consecutive sorted observations
        diffs = z_sorted[:, 1:] - z_sorted[:, :-1]  # shape: (batch_size, N-1)
        
        # Generate the exact probability weights for each rectangle
        N = batch_z_orig.shape[1]
        i = torch.arange(1, N, device=device).float()
        weights = (i / N) * (1.0 - i / N)           # shape: (N-1,)
        
        # Compute exact integral of the spread via dot product
        empirical_spread = torch.sum(weights * diffs, dim=1) # shape: (batch_size,)
        crps_batch = cvm_distance + empirical_spread
        
        # Store batch metrics
        all_w1.append(w1_batch.detach().cpu())
        all_crps.append(crps_batch.detach().cpu())
        all_ks.append(ks_batch.detach().cpu())
        
        # =========================================================
        # 5. VECTORIZED NLLH (Evaluated in log-space)
        # =========================================================
        if target_scale == "log":
            llh = log_pdf_tabpfn(logits, batch_z_orig, borders)
            llh.clamp_(min=MIN_CLAMP_LLH)
            
            bias = -torch.log(torch.max(batch_z_orig, dim=1)[0])
            
        elif target_scale == "original":
            llh = log_pdf_tabpfn(logits, batch_y_orig, borders)
            llh.clamp_(min=MIN_CLAMP_LLH)

            jacobian = batch_z_orig
            assert llh.shape == jacobian.shape, f"Shape mismatch between llh {llh.shape} and jacobian {jacobian.shape}"
            llh += jacobian

            bias = -torch.log(torch.max(batch_z_orig, dim=1)[0])

        assert llh.shape[0] == bias.shape[0], f"Batch size mismatch between llh {llh.shape} and bias {bias.shape}"
        assert llh.ndim == 2, f"Expected llh to have shape (B, O), but got {llh.shape}"
        assert bias.ndim == 1, f"Expected bias to have shape (B,), but got {bias.shape}"
        
        batch_nllh = -llh.mean(dim=1) + bias

        all_nllh.append(batch_nllh.detach().cpu())
        
        instance_idx += batch_size
        
    all_crps = torch.cat(all_crps)
    all_w1 = torch.cat(all_w1)
    all_ks = torch.cat(all_ks)
    all_nllh = torch.cat(all_nllh)

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

    instance_summary = {"NLLH": all_nllh, "CRPS": all_crps, "Wasserstein": all_w1, "KS": all_ks}

    return metrics_summary, instance_summary

def calculate_all_distribution_metrics_logNormalDist_logspace(
    y_test_orig,
    lognormal_dist,
    *,
    device, 
    N_grid_points,
):
    """
    y_test_orig: shape (B, O) - original unscaled targets.
    lognormal_dist: lognormal distribution.
    device: torch device for computation.
    N_grid_points: total number of points in the piecewise non-uniform grid for CDF evaluation.

    returns: - metrics_summary: summary dict, instance_summary: dict of per-instance metrics
    """
    # Ensure inputs are correct dtype and on the correct device
    y_test_orig = torch.as_tensor(y_test_orig, dtype=torch.float32, device=device)
    
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

    y_orig_min = lognormal_dist.icdf(p_min)
    y_orig_max = lognormal_dist.icdf(p_max)

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
    F_model = lognormal_dist.cdf(y_grid.clamp(min=0))
    
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
    # 5. VECTORIZED NLLH (log-space)
    # =========================================================
    llh = lognormal_dist.log_prob(y_test_orig)
    llh.clamp_(min=MIN_CLAMP_LLH)

    assert z_test_orig.shape == llh.shape, f"Shape mismatch in llh correction: {z_test_orig.shape} vs {llh.shape}"
    llh += z_test_orig
    
    # max scaling bias correction
    bias = -torch.log(z_test_orig.max(dim=1)[0])

    all_nllh = -llh.mean(dim=1) + bias

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

def calculate_all_distribution_metrics_KDE_logspace(
    y_test_orig,
    y_train_flat,
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
    from statsmodels.distributions.empirical_distribution import ECDF
    from scipy.stats import gaussian_kde

    def get_marginal_empirical_predictor(y_train_flat):
        """
        Creates the Marginal Empirical Predictor baseline using scipy's KDE.
        
        Args:
            y_train_flat: The target runtime values (flattened).
            
        Returns:
            cdf_object: Callable ECDF object.
            pdf_object: gaussian_kde object, which supports .pdf(x) and .logpdf(x).
        """

        # 1. Create the CDF object
        cdf_object = ECDF(y_train_flat)
        
        # 2. Create the PDF object using scipy (defaults to Scott's Rule)
        pdf_object = gaussian_kde(y_train_flat, bw_method='scott')
        
        return cdf_object, pdf_object
    
    cdf_model, pdf_model = get_marginal_empirical_predictor(y_train_flat)
    
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
    llh = pdf_model.logpdf(z_test_np.flatten()).reshape(B, O)
    
    llh = np.clip(llh, a_min=MIN_CLAMP_LLH, a_max=None)

    jacobian_correction = -np.log(np.max(z_test_np, axis=1))

    assert llh.shape[0] == jacobian_correction.shape[0] == B, "Batch size mismatch in LLH calculation"
    
    all_nllh = -llh.mean(axis=1) + jacobian_correction

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
    
    instance_summary = {
        "NLLH": all_nllh, 
        "CRPS": all_crps, 
        "Wasserstein": all_w1, 
        "KS": all_ks
    }

    return metrics_summary, instance_summary

def calculate_all_distribution_metrics_randomForest_logspace(
    y_test_orig,
    preds, 
    *,
    device, 
    N_grid_points,
):
    """
    y_test_orig: shape (B, O) - original unscaled targets
    preds: tuple of length 2 (means, variances) - RF predicted parameters in log1p-scaled space.
           means: shape (B,)
           variances: shape (B,)
    device: torch device for computation
    N_grid_points: total number of points in the piecewise non-uniform grid for CDF evaluation

    returns: - metrics_summary: summary dict, instance_summary: dict of per-instance metrics
    """
    assert preds[0].ndim == 1 and preds[1].ndim == 1, "Preds must be 1-dimensional"

    y_test_orig = torch.as_tensor(y_test_orig, dtype=torch.float32, device=device)
    
    # Extract means and variances and reshape to (B, 1) for broadcasting across O observations
    means = torch.as_tensor(preds[0], dtype=torch.float32, device=device).unsqueeze(1)
    variances = torch.as_tensor(preds[1], dtype=torch.float32, device=device).unsqueeze(1)
    
    # Random Forest directly modeled the Z-space (log1p scaled space)
    z_test_orig = torch.log1p(y_test_orig)

    # Instantiate the predictive Gaussian distribution in Z-space
    stds = torch.sqrt(variances)
    dist = torch.distributions.Normal(loc=means, scale=stds)

    # =========================================================
    # 1. CORE BOUNDS (Empirical Data in Z-Space)
    # =========================================================
    min_z_emp = z_test_orig.min(dim=1, keepdim=True)[0]  # shape (B, 1)
    max_z_emp = z_test_orig.max(dim=1, keepdim=True)[0]  # shape (B, 1)
    z_range = (max_z_emp - min_z_emp).clamp(min=1e-5)
    
    core_start = min_z_emp - 0.05 * z_range
    core_end = max_z_emp + 0.05 * z_range

    # =========================================================
    # 2. MODEL TAIL BOUNDS VIA INVERSE CDF (ICDF)
    # =========================================================
    # RF natively models the Z-space, so we can query the quantiles directly
    p_min = torch.tensor(0.0001, device=device)
    p_max = torch.tensor(0.9999, device=device)

    z_model_min = dist.icdf(p_min)
    z_model_max = dist.icdf(p_max)

    global_start = torch.minimum(core_start - 0.5 * z_range, z_model_min)
    global_end = torch.maximum(core_end + 0.5 * z_range, z_model_max)

    # =========================================================
    # 3. 15K PIECEWISE NON-UNIFORM GRID
    # =========================================================
    left_pts = int(N_grid_points * 1/6)
    core_pts = int(N_grid_points * 2/3)
    right_pts = int(N_grid_points * 1/6)

    steps_left = torch.linspace(0, 1, left_pts, device=device).view(1, -1)
    z_grid_left = global_start + steps_left * (core_start - global_start)

    steps_core = torch.linspace(0, 1, core_pts + 1, device=device).view(1, -1)[:, 1:]
    z_grid_core = core_start + steps_core * (core_end - core_start)

    steps_right = torch.linspace(0, 1, right_pts + 1, device=device).view(1, -1)[:, 1:]
    z_grid_right = core_end + steps_right * (global_end - core_end)

    z_grid = torch.cat([z_grid_left, z_grid_core, z_grid_right], dim=1)  # shape (B, N_grid_points)

    # =========================================================
    # 4. CDF EVALUATION & INTEGRATION (in Z-space)
    # =========================================================
    # Empirical CDF
    indicator = (z_test_orig.unsqueeze(1) <= z_grid.unsqueeze(2)).float()  
    F_emp = indicator.mean(dim=2)  # shape (B, N_grid_points)
    
    # Evaluate RF CDF natively in Z-space
    F_model = dist.cdf(z_grid)
    
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
    # 5. VECTORIZED NLLH (in Z-space)
    # =========================================================
    llh = dist.log_prob(z_test_orig)  # shape (B, O)

    llh.clamp_(min=MIN_CLAMP_LLH)

    jacobian = -torch.log(torch.max(z_test_orig, dim=1)[0])
    all_nllh = -llh.mean(dim=1) + jacobian  # shape (B,)

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

class WindowsPathUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'PosixPath' and 'pathlib' in module:
            return pathlib.WindowsPath
        return super().find_class(module, name)

def find_optimal_context(results_list, scenario, target_metric, alpha=0.05, display_results=True):
    """
    Rigorously determines the optimal context size using the Principle of Parsimony 
    and a one-sided Wilcoxon signed-rank test.
    
    Parameters:
    - results_list: List of dictionaries loaded from the .pkl file
    - scenario: String, the dataset name to filter by
    - target_metric: String, e.g., "CRPS", "NLLH", "Wasserstein", "KS"
    - alpha: Float, the statistical significance threshold (default 0.05)
    
    Returns:
    - optimal_context: The selected context size (integer)
    - summary_df: A pandas DataFrame containing the thesis-ready results table
    """
    
    # ---------------------------------------------------------
    # STEP 1: Data Extraction & Instance Aggregation
    # ---------------------------------------------------------
    records = []
    for run in results_list:
        if run['scenario'] == scenario:
            # Aggregate the 200 instances by taking the mean for the target metric
            run_data = run['instance_summary'][target_metric]
            if hasattr(run_data, 'cpu'):
                run_data = run_data.detach().cpu().numpy()
            run_score = np.mean(run_data)
            records.append({
                'context_size': run['context_size'],
                'fold': run['fold'],
                'seed': run['context_seed'],
                'score': run_score
            })
            
    if not records:
        raise ValueError(f"No data found for scenario '{scenario}'. Please check the name.")
        
    df = pd.DataFrame(records)
    
    # ---------------------------------------------------------
    # STEP 2: Seed Aggregation (Handling intra-fold correlation)
    # ---------------------------------------------------------
    # Average the 5 seeds for every (context_size, fold) combination
    fold_df = df.groupby(['context_size', 'fold'])['score'].mean().reset_index()
    
    # ---------------------------------------------------------
    # STEP 3: Identify the Empirical Best (C_best)
    # ---------------------------------------------------------
    # Calculate the grand mean across the 10 independent folds
    summary_df = fold_df.groupby('context_size')['score'].agg(['mean', 'std']).reset_index()
    summary_df = summary_df.sort_values('context_size').reset_index(drop=True)
    
    # Find the context_size with the absolute lowest mean (since lower error is better)
    c_best_idx = summary_df['mean'].idxmin()
    c_best = int(summary_df.loc[c_best_idx, 'context_size'])
    
    # Extract the 10 matched fold scores for the empirical best, ensuring order by fold
    scores_best = fold_df[fold_df['context_size'] == c_best].sort_values('fold')['score'].values
    
    # ---------------------------------------------------------
    # STEP 4: Wilcoxon-Parsimony Test
    # ---------------------------------------------------------
    results_log =[]
    optimal_context = None
    
    for _, row in summary_df.iterrows():
        c_test = int(row['context_size'])
        mean_test = row['mean']
        std_test = row['std']
        
        # Extract the 10 matched fold scores for the current candidate
        scores_test = fold_df[fold_df['context_size'] == c_test].sort_values('fold')['score'].values
        
        if c_test == c_best:
            p_value = 1.0  # Comparing to itself
            is_worse = False
            status = "*** Empirical Best ***"
        else:
            # One-sided Wilcoxon test
            # H0: c_test and c_best are symmetric.
            # Ha (greater): scores_test > scores_best (i.e., c_test is strictly WORSE than c_best)
            try:
                # Suppress scipy ties warning temporarily to keep terminal clean
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stat, p_value = wilcoxon(scores_test, scores_best, alternative='greater')
            except ValueError:
                # Triggers if differences are exactly 0 across all 10 folds
                print(f"differences are exactly 0 across all 10 folds")
                p_value = 1.0 
                
            is_worse = p_value < alpha
            status = "Significantly Worse" if is_worse else "Statistical Tie"
            
        # Log the result for the thesis table
        results_log.append({
            'Context Size': c_test,
            f'Mean {target_metric}': mean_test,
            f'Std {target_metric}': std_test,
            'p-value (vs Best)': p_value,
            'Status': status
        })
        
        # Apply Parsimony Rule: 
        # Select the *smallest* context size that is <= c_best and NOT significantly worse.
        # Since we iterate in ascending order, the very first non-worse candidate is our optimal!
        if not is_worse and optimal_context is None and c_test <= c_best:
            optimal_context = c_test

    # ---------------------------------------------------------
    # STEP 5: Formatting and Thesis-Ready Output
    # ---------------------------------------------------------
    log_df = pd.DataFrame(results_log)
    # Update the status string of the chosen optimal context for display purposes
    idx_optimal = log_df.index[log_df['Context Size'] == optimal_context].tolist()[0]
    if optimal_context != c_best:
        log_df.at[idx_optimal, 'Status'] += " <-- CHOSEN (Parsimony)"
    else:
        log_df.at[idx_optimal, 'Status'] += " <-- CHOSEN"
        
    if display_results:        
        # Formatting for terminal beauty
        print(f"\n{'='*75}")
        print(f" Parsimony Analysis | Scenario: {scenario} | Metric: {target_metric}")
        print(f"{'='*75}")
        print(f"Empirical Best Context Size (C_best) : {c_best}")
        print(f"Optimal Context Size Chosen          : {optimal_context}")
        print("-" * 75)
        
        # Pandas display formatting
        formatters = {
            f'Mean {target_metric}': '{:.5f}'.format,
            f'Std {target_metric}': '{:.5f}'.format,
            'p-value (vs Best)': '{:.4f}'.format
        }
        print(log_df.to_string(index=False, formatters=formatters))
        print(f"{'='*75}\n")
    
    return optimal_context, log_df

def dict_to_cpu(d):
    result = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.detach().cpu()
        elif isinstance(v, dict):
            result[k] = dict_to_cpu(v)
        elif isinstance(v, (list, tuple)):
            processed_iterable = [
                vi.detach().cpu() if isinstance(vi, torch.Tensor) 
                else dict_to_cpu(vi) if isinstance(vi, dict) 
                else vi 
                for vi in v
            ]
            result[k] = type(v)(processed_iterable) # keep it a list or tuple
        elif hasattr(v, 'cpu') and callable(v.cpu):
            result[k] = deepcopy(v).cpu()
        else:
            result[k] = v
    return result

def load_pickle(path, access_mode='rb'):
    with open(path, access_mode) as f:
        # Use our custom Unpickler on Windows, otherwise standard pickle
        if platform.system() == 'Windows':
            results_dict = WindowsPathUnpickler(f).load()
        else:
            results_dict = pickle.load(f)
    return results_dict

def subsample_flattened_data(X_train_flat, y_train_flat, context_size, seed, subsample_method):
    """
    Subsamples the flattened dataset. Currently supports 'flatten-random' which randomly samples from the flattened training data.
    
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
    Randomly samples a subset of features from the input arrays based on the specified drop rate.
    If drop_rate >= 1.0, implements a strict marginal baseline (0 features) via a dummy column.
    
    Args:
        X_train: (n_samples, n_features)
        *arrays: Additional arrays with shape (n_samples, n_features) to subsample features from
        drop_rate: Fraction of features to drop (0.0 to 1.0)
        seed: Random seed for reproducibility.

    Returns:
        Tuple of subsampled arrays, with the same order as input (X_train, *arrays)
    """
    
    # 1. Handle the "0 Features" Marginal Baseline
    if drop_rate >= 1.0:
        dummy_X_train = X_train[:, :1] * 0.0
        processed_arrays = [arr[:, :1] * 0.0 for arr in arrays]
        
        return (dummy_X_train, *processed_arrays)

    # 2. Handle standard feature subsampling
    rng = np.random.default_rng(seed=seed)
    n_features = X_train.shape[1]
    
    # Calculate how many features to KEEP. 
    size_features = max(1, int(n_features * (1 - drop_rate)))
    
    feature_idx = rng.choice(n_features, size=size_features, replace=False)
    
    processed_arrays = [arr[:, feature_idx] for arr in arrays]

    return (X_train[:, feature_idx], *processed_arrays)

def subsample_targets_per_instance(y_train, num_samples_per_instance, seed_samples_per_instance):
    """
    Subsamples a specified number of samples per instance from the training data.
    
    Args:
        y_train: (n_instances, n_samples) - The training labels.
        num_samples_per_instance: The number of samples to subsample per instance.
        seed_samples_per_instance: The random seed for reproducibility.

    Returns:
        y_train: (n_instances, num_samples_per_instance) - The subsampled training labels.
    """
    rng = np.random.default_rng(seed=seed_samples_per_instance)
    subsample_idx = rng.choice(y_train.shape[1], size=num_samples_per_instance, replace=False)
    y_train = y_train[:, subsample_idx]
    return y_train

def load_tabpfn_preds(
    tabpfn_preds_dir,
    scenario_name,
    fold_idx,
    context_size,
    context_seed,
    seed_features,
    seed_samples_per_instance,
    feature_drop_rate,
    target_scale,
    subsample_method,
    num_samples_per_instance,
    use_cpu,
 ):
    device_tag = "cpu" if use_cpu else "gpu"
    fname = (
        f"tabpfn_{scenario_name}_{fold_idx}_{context_seed}_{seed_features}_{seed_samples_per_instance}_{feature_drop_rate}_"
        f"{context_size}_{target_scale}_{subsample_method}_{num_samples_per_instance}_{device_tag}_test_preds.pkl"
    )
    fpath = tabpfn_preds_dir / fname

    with open(fpath, "rb") as f:
        if platform.system() == "Windows":
            return WindowsPathUnpickler(f).load()
        return pickle.load(f)

def fetch_save_dict(results_dir: pathlib.Path, metadata_dir: pathlib.Path, model_name: str, search_key: str, search_value: str, scenario: str = None) -> None:
    """Build and save a nested results dictionary filtered by model and scenario."""
    experiment_results_lst = []

    for fpath in sorted(metadata_dir.glob("*.pkl")):
        with open(fpath, "rb") as f:
            if platform.system() == "Windows":
                results_dict = WindowsPathUnpickler(f).load()
            else:
                results_dict = pickle.load(f)

        if scenario is not None and results_dict.get("scenario") != scenario:
            continue
        if results_dict.get(search_key) != search_value:
            continue

        context_size = results_dict["context_size"]
        if context_size in {2**13 + 2000, 2**13 + 4000}:
            continue

        metrics_summary = None
        instance_summary = None
        best_params = None
        y_preds = None
        fit_time = None
        predict_time = None
        gpu_metrics = None
        hpo_time = None


        if model_name == "baseline":
            metrics_summary = results_dict['test_preds'][0]
            instance_summary = results_dict['test_preds'][1]

        elif model_name == "rf_baseline":
            metrics_summary = results_dict['result_metrics']['metrics_summary']
            instance_summary = results_dict['result_metrics']['instance_summary']
            best_params = results_dict['best_params']
            y_preds = results_dict['test_preds']  # [rf_means, rf_variances]
            fit_time = results_dict['result_metrics']['fit_time']
            predict_time = results_dict['result_metrics']['predict_time']
            hpo_time = results_dict['result_metrics']['hpo_time']

        elif model_name == "tabpfn":
            gpu_metrics = results_dict['result_metrics']['mem_time_stats']
            metrics_summary = results_dict['result_metrics']['metrics_summary']
            instance_summary = results_dict['result_metrics']['instance_summary']

        temp = {
            "scenario": results_dict["scenario"],
            "model": results_dict["model_name"],
            "context_size": results_dict["context_size"],
            "fold": results_dict["fold"],
            "context_seed": results_dict["seed_context"],
            "metrics_summary": metrics_summary,
            "instance_summary": instance_summary,
            "best_params": best_params,
            "y_preds": y_preds,
            "target_scale": results_dict["target_scale"],
            "fit_time": fit_time,
            "predict_time": predict_time,
            "hpo_time": hpo_time,
            "gpu_metrics": gpu_metrics,
            "use_cpu": results_dict["use_cpu"],
        }

        experiment_results_lst.append(temp)

    if scenario is None:
        scenario = "all_scenarios"
    save_file_path = results_dir / f"{model_name}_{search_key}_{search_value}_{scenario}.pkl"
    with open(save_file_path, "wb") as f:
        pickle.dump(experiment_results_lst, f)

    print(f"Saved to {save_file_path}")