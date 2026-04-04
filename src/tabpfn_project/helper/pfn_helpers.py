import torch
from tabpfn_project.globals import MIN_CLAMP_LLH
from tabpfn_project.helper.utils import dict_to_cpu

def ignore_init(y: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
        ignore_loss_mask = torch.isnan(y)
        # this is just a default value, it will be ignored anyway
        y[ignore_loss_mask] =  borders[0]
        return ignore_loss_mask

def compute_scaled_log_probs(logits: torch.Tensor, bucket_widths: torch.Tensor) -> torch.Tensor:
    # this is equivalent to log(p(y)) of the density p
    bucket_log_probs = torch.log_softmax(logits, -1)
    return bucket_log_probs - torch.log(bucket_widths)

def halfnormal_with_p_weight_before(
        range_max: float,
        p: float = 0.5,
    ) -> torch.distributions.HalfNormal:
        s = range_max / torch.distributions.HalfNormal(torch.tensor(1.0)).icdf(
            torch.tensor(p),
        )
        return torch.distributions.HalfNormal(s)

def map_to_bucket_idx(y: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
        num_bars = len(borders) - 1  # number of buckets
        bucket_widths = borders[1:] - borders[:-1]
        # assert the borders are actually sorted
        assert (bucket_widths >= 0.0).all(), "borders are not sorted!"
        target_sample = torch.searchsorted(borders, y) - 1  # shape: (B, O) = shape of y
        target_sample[y == borders[0]] = 0
        target_sample[y == borders[-1]] = num_bars - 1
        return target_sample

def log_pdf_tabpfn(
        logits: torch.Tensor,
        y: torch.Tensor, 
        borders: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits: shape (B, K) where K is the number of buckets
        y: shape (B, O) where O is the number of observations per instance
        borders: shape (K+1,) where K is the number of buckets
        returns: logpdf of shape (B, O)
        """

        num_bars = len(borders) - 1
        bucket_widths = borders[1:] - borders[:-1]
        assert logits.shape[-1] == num_bars, f"mismatch between num_bars and logits_num_bars"

        y = y.clone()
        ignore_loss_mask = ignore_init(y, borders)  # shape: (B, O)
        target_sample =  map_to_bucket_idx(y, borders)  # shape: (B, O) (same as y)
        target_sample.clamp_(0,  num_bars - 1)  # bucket indices

        scaled_bucket_log_probs =  compute_scaled_log_probs(logits, bucket_widths)  # shape: (B, K)

        assert scaled_bucket_log_probs.shape[0] == target_sample.shape[0], (
            f"Shape mismatch: scaled_bucket_log_probs {scaled_bucket_log_probs.shape} vs target_sample {target_sample.shape}"
        )

        log_probs = scaled_bucket_log_probs.gather(
            -1,
            target_sample,
        )  # shape: (B, O)

        side_normals = (
             halfnormal_with_p_weight_before(bucket_widths[0]),
             halfnormal_with_p_weight_before(bucket_widths[-1]),
        )

        log_probs[target_sample == 0] += side_normals[0].log_prob(
            (borders[1] - y[target_sample == 0]).clamp(min=0.0)
        ) + torch.log(bucket_widths[0])

        log_probs[target_sample == (num_bars - 1)] += side_normals[1].log_prob(
            (y[target_sample == (num_bars - 1)] -  borders[-2]).clamp(min=0.0)) + torch.log(bucket_widths[-1])

        llh = log_probs

        if ignore_loss_mask.any():
            llh[ignore_loss_mask] = 0.0

        return llh

def cdf_tabpfn(logits: torch.Tensor, y: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    """
    logits: shape (B, K) where K is the number of buckets
    y: shape (B, O) where O is the number of observations per instance
    borders: shape (K+1,) where K is the number of buckets
    returns: cdf of shape (B, O)
    """
    num_bars = len(borders) - 1
    bucket_widths = borders[1:] - borders[:-1]

    assert num_bars == logits.shape[-1], f"{num_bars} vs {logits.shape[-1]}"

    assert logits.ndim == y.ndim and logits.shape[0] == y.shape[0], f"Batch size mismatch between logits and y: {logits.shape} vs {y.shape}"
        
    probs = torch.softmax(logits, dim=-1)  # shape: (B, K)
    
    buckets_of_ys = map_to_bucket_idx(y, borders).clamp(0, num_bars - 1)  # shape: (B, O)

    prob_so_far = torch.cumsum(probs, dim=-1) - probs  # shape: (B, K), prob_so_far[:, k] is the total probability mass of all buckets to the left of bucket k.

    # Ensure gather works across matching dimensions
    assert buckets_of_ys.ndim == prob_so_far.ndim
    prob_left_of_bucket = prob_so_far.gather(-1, buckets_of_ys)  # shape: (B, O)

    # 1. Default Assumption: Uniform Interpolation (valid for middle buckets)
    share_of_bucket_left = (
        (y - borders[buckets_of_ys]) / bucket_widths[buckets_of_ys]
    ).clamp(0.0, 1.0)  # shape: (B, O)

    # 2. Correction for the Left-most Bucket (Half-Normal extending to -infinity)
    hn_left = halfnormal_with_p_weight_before(bucket_widths[0])
    is_left_bucket = (buckets_of_ys == 0)
    
    if is_left_bucket.any():
        # The left half-normal originates at borders[1] and decays to the left.
        dist_from_right_edge = (borders[1] - y[is_left_bucket]).clamp(min=0.0)
        
        # The integral from -inf up to 'y' of a reversed half-normal 
        # is equal to 1.0 minus the standard Half-Normal CDF.
        share_of_bucket_left[is_left_bucket] = 1.0 - hn_left.cdf(dist_from_right_edge)

    # 3. Correction for the Right-most Bucket (Half-Normal extending to +infinity)
    hn_right = halfnormal_with_p_weight_before(bucket_widths[-1])
    is_right_bucket = (buckets_of_ys == num_bars - 1)
    
    if is_right_bucket.any():
        # The right half-normal originates at borders[-2] and decays to the right.
        dist_from_left_edge = (y[is_right_bucket] - borders[-2]).clamp(min=0.0)
        
        # The integral from borders[-2] up to 'y' is just the standard CDF.
        share_of_bucket_left[is_right_bucket] = hn_right.cdf(dist_from_left_edge)

    # 4. Final Aggregation
    prob_in_bucket = probs.gather(-1, buckets_of_ys) * share_of_bucket_left  # shape: (B, O)

    total_prob_ys = prob_left_of_bucket + prob_in_bucket
    return total_prob_ys.clip(0.0, 1.0)

def batch_predict_tabpfn(model, X_test, validation_batch_size):
    '''
    Batch predictor for tabpfn model.

    Parameters:
    - model: The tabPFN model with a predict method.
    - X_test: Validation input data (B, D).
    - validation_batch_size: Batch size for processing validation data.
    
    Returns:
    - tabpfn_preds (list of dicts).
    '''
    assert validation_batch_size > 0, "validation_batch_size must be a positive integer"

    n_validation_instances = X_test.shape[0]  # total number of validation instances
    validation_batch_size = min(validation_batch_size, n_validation_instances)
    tabpfn_preds = []
    for start in range(0, n_validation_instances, validation_batch_size):
        X_batch = X_test[start: start + validation_batch_size]
        with torch.no_grad():
            preds = model.predict(X_batch, output_type="full")
            tabpfn_preds.append(dict_to_cpu(preds))
            
    return tabpfn_preds

def calculate_distribution_metrics_logspace_tabpfn(
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
