import numpy as np
from scipy import stats
import torch

def calculate_llh_instance_distnet(observations, shape, scale, target_scale):
    """
    
     mean log-likelihood per instance for lognormal distribution. (FOR DISTNET)
    
    Args:
        observations: Array of max-scaled observed values for a single instance
        shape: Shape parameter of lognormal distribution
        scale: Scale parameter of lognormal distribution (e^mu)
    
    Returns:
        mean log-likelihood
    """
    if target_scale == "max":
        instance_llh = stats.distributions.lognorm.logpdf(observations, shape, loc=0, scale=scale).mean() + np.log(observations.max())
    return instance_llh

def calculate_nllh_distnet(observations, preds, target_scale="max"):
    """
    Calculate the negative log-likelihood (NLLH) for DistNet model on given instances.
    Args:
        observations: List of arrays, each containing max-scaled observations for the corresponding instance (n_instances, num_observations_per_instance)
        preds: List of tuples, each containing (shape, scale) predicted by DistNet for the corresponding instance (n_instances, 2)
    Returns:
        NLLH value
    """
    assert observations.shape[0] == preds.shape[0], "Number of instances in observations and preds must match"
    assert observations.ndim == 2 and preds.ndim == 2, "Observations and preds must be 2D arrays"
    
    llh_instances = 0.0
    n_instances = observations.shape[0]
    for obs, pred in zip(observations, preds):
        shape = pred[0]
        scale = pred[1]
        llh_instances += calculate_llh_instance_distnet(obs, shape, scale, target_scale=target_scale)
    
    nllh = -llh_instances / n_instances
    return nllh

def calculate_all_distribution_metrics_distnet(
    y_test_orig, 
    preds, 
    y_scale, 
    grid_points,
):
    """
    Vectorized calculation of CRPS, Wasserstein (W1), KS-distance, and NLLH for DistNet.
    Assumes max-scaling is the only transformation used.
    
    Args:
        y_test_orig: Tensor/Array of shape (n_instances, n_observations). ORIGINAL unscaled data.
        preds: Tensor/Array of shape (n_instances, 2). 
               preds[:, 0] = shape parameter (sigma)
               preds[:, 1] = scale parameter (exp(mu))
        y_scale: y_scale used during training
        grid_points: Resolution of the CDF integration grid.
        
    Returns:
        metrics_summary (dict): Aggregated mean/std metrics over the whole test set.
        instance_metrics (dict): Tensors of shape (n_instances,) for deep dives.
    """
    # 1. Ensure inputs are PyTorch tensors
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    y_test_orig = torch.as_tensor(y_test_orig, dtype=torch.float32, device=device)
    preds = torch.as_tensor(preds, dtype=torch.float32, device=device)
    
    # 2. Map SciPy parameters to PyTorch LogNormal parameters
    # scipy scale = exp(mu) -> mu = log(scale)
    # scipy shape = sigma
    sigma = preds[:, 0].unsqueeze(1)          # shape: (n_instances, 1)
    mu = torch.log(preds[:, 1]).unsqueeze(1)  # shape: (n_instances, 1)
    
    dist = torch.distributions.LogNormal(loc=mu, scale=sigma)

    # =========================================================
    # PART 1: CDF-BASED METRICS (CRPS, W1, KS) IN ORIGINAL SPACE
    # =========================================================
    min_y = y_test_orig.min(dim=1, keepdim=True)[0] # y_test_orig for DistNet
    max_y = y_test_orig.max(dim=1, keepdim=True)[0]
    y_range = max_y - min_y
    y_range = torch.where(y_range == 0, torch.ones_like(y_range), y_range)
    
    # Apples-to-apples: 1.5x margin on BOTH sides
    start_y = min_y - 1.5 * y_range
    end_y = max_y + 1.5 * y_range
    
    steps = torch.linspace(0, 1, grid_points, device=device).unsqueeze(0)
    y_grid = start_y + steps * (end_y - start_y)
    
    # Empirical CDF
    indicator = (y_test_orig.unsqueeze(1) <= y_grid.unsqueeze(2)).float()
    F_emp = indicator.mean(dim=2)  
    
    # Model CDF mapped from transformed space
    z_grid = y_grid * y_scale
    F_model = dist.cdf(z_grid.clamp(min=0))  # Ensure non-negativity for lognormal CDF
    
    # Numerically integrate
    cdf_diff = F_model - F_emp
    abs_cdf_diff = torch.abs(cdf_diff)
    
    all_w1 = torch.trapezoid(abs_cdf_diff, x=y_grid, dim=1)      
    all_crps = torch.trapezoid(cdf_diff ** 2, x=y_grid, dim=1)   
    all_ks = torch.max(abs_cdf_diff, dim=1)[0]                   

    # =========================================================
    # PART 2: VECTORIZED NLLH (SCALED SPACE + BIAS)
    # =========================================================
    y_scaled = y_test_orig * y_scale

    # Vectorized log PDF computation
    nlog_pdf = -dist.log_prob(y_scaled)
    nlog_pdf.clamp_(max=50.0)  # Cap to prevent numerical issues in exp/log later on, corresponds to -log(2e-22)

    # Max-scaling Bias (Jacobian adjustment)
    max_scaled = y_scaled.max(dim=1)[0]
    bias = -torch.log(max_scaled)

    # Final instance NLLH calculation (matching your: -LLH.mean() - log(obs.max()))
    all_nllh = nlog_pdf.mean(dim=1) + bias

    # =========================================================
    # AGGREGATION
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
    
    instance_metrics = {
        "NLLH": all_nllh,
        "CRPS": all_crps,
        "Wasserstein": all_w1,
        "KS": all_ks
    }
    
    return metrics_summary, instance_metrics