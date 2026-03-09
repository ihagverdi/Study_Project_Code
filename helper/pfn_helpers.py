import torch

def ignore_init(y: torch.Tensor, borders) -> torch.Tensor:
        ignore_loss_mask = torch.isnan(y)
        # this is just a default value, it will be ignored anyway
        y[ignore_loss_mask] =  borders[0]
        return ignore_loss_mask

def compute_scaled_log_probs(logits: torch.Tensor, bucket_widths) -> torch.Tensor:
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

def map_to_bucket_idx(y: torch.Tensor, borders) -> torch.Tensor:
        num_bars = len(borders) - 1  # number of buckets
        bucket_widths = borders[1:] - borders[:-1]
        # assert the borders are actually sorted
        assert (bucket_widths >= 0.0).all(), "borders are not sorted!"
        target_sample = torch.searchsorted(borders, y) - 1
        target_sample[y == borders[0]] = 0
        target_sample[y == borders[-1]] = num_bars - 1
        return target_sample

def logpdf_tabpfn(
        logits: torch.Tensor,
        y: torch.Tensor, 
        borders,
    ) -> torch.Tensor:
        """Returns the log-pdf of tabpfn at y.

        y: B x num_observations, logits: B x num_bars.

        :param logits: Tensor of shape B x num_bars
        :param y: Tensor of shape B x num_observations
        :param borders: Tensor of shape num_bars + 1, sorted in ascending order
        :return: log(p)
        """
        num_bars = len(borders) - 1
        bucket_widths = borders[1:] - borders[:-1]
        assert logits.shape[-1] == num_bars, f"mismatch between num_bars and logits_num_bars"

        y = y.clone().view(*logits.shape[:-1])  # no trailing one dimension
        ignore_loss_mask =  ignore_init(y, borders)  # alters y (nan indices, set those in y to borders[0] value)
        target_sample =  map_to_bucket_idx(y, borders)  # shape: T x B (same as y)
        target_sample.clamp_(0,  num_bars - 1)  # bucket indices

        # range check of bucket indices
        assert (target_sample >= 0).all() and (target_sample <  num_bars).all(), (
            f"y {y} not in support set for borders (min_y, max_y) {borders}"
        )

        scaled_bucket_log_probs =  compute_scaled_log_probs(logits, bucket_widths)

        assert len(scaled_bucket_log_probs) == len(target_sample), (
            len(scaled_bucket_log_probs),
            len(target_sample),
        )

        log_probs = scaled_bucket_log_probs.gather(
            -1,
            target_sample.unsqueeze(-1),
        ).squeeze(-1)

        side_normals = (
             halfnormal_with_p_weight_before(bucket_widths[0]),
             halfnormal_with_p_weight_before(bucket_widths[-1]),
        )

        log_probs[target_sample == 0] += side_normals[0].log_prob(
            ( borders[1] - y[target_sample == 0]).clamp(min=0.0),
        ) + torch.log(bucket_widths[0])

        log_probs[target_sample ==  num_bars - 1] += side_normals[1].log_prob(
            (y[target_sample == num_bars - 1] -  borders[-2]).clamp(
                min=0.0,
            ),
        ) + torch.log(bucket_widths[-1])

        llh = log_probs

        if ignore_loss_mask.any():
            llh[ignore_loss_mask] = 0.0

        return llh

def cdf_tabpfn(logits: torch.Tensor, ys: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    num_bars = len(borders) - 1
    bucket_widths = borders[1:] - borders[:-1]

    assert num_bars == logits.shape[-1], f"{num_bars} vs {logits.shape[-1]}"

    if len(ys.shape) < len(logits.shape) and len(ys.shape) == 1:
        ys = ys.repeat((*logits.shape[:-1], 1))
    else:
        assert ys.shape[:-1] == logits.shape[:-1], (
            f"ys.shape: {ys.shape} logits.shape: {logits.shape}"
        )
        
    probs = torch.softmax(logits, dim=-1)
    
    # map_to_bucket_idx automatically assigns y < borders[0] to bucket 0 
    # and y > borders[-1] to bucket K-1. We clamp just to be safe.
    buckets_of_ys = map_to_bucket_idx(ys, borders).clamp(0, num_bars - 1)

    prob_so_far = torch.cumsum(probs, dim=-1) - probs
    # Ensure gather works across matching dimensions
    assert len(buckets_of_ys.shape) == len(prob_so_far.shape)
    prob_left_of_bucket = prob_so_far.gather(-1, buckets_of_ys)
    # else:
    #     prob_left_of_bucket = prob_so_far.gather(-1, buckets_of_ys.unsqueeze(-1)).squeeze(-1)

    # 1. Default Assumption: Uniform Interpolation (valid for middle buckets)
    share_of_bucket_left = (
        (ys - borders[buckets_of_ys]) / bucket_widths[buckets_of_ys]
    ).clamp(0.0, 1.0)

    # 2. Correction for the Left-most Bucket (Half-Normal extending to -infinity)
    hn_left = halfnormal_with_p_weight_before(bucket_widths[0])
    is_left_bucket = (buckets_of_ys == 0)
    
    if is_left_bucket.any():
        # The left half-normal originates at borders[1] and decays to the left.
        dist_from_right_edge = (borders[1] - ys[is_left_bucket]).clamp(min=0.0)
        
        # The integral from -inf up to 'ys' of a reversed half-normal 
        # is equal to 1.0 minus the standard Half-Normal CDF.
        share_of_bucket_left[is_left_bucket] = 1.0 - hn_left.cdf(dist_from_right_edge)

    # 3. Correction for the Right-most Bucket (Half-Normal extending to +infinity)
    hn_right = halfnormal_with_p_weight_before(bucket_widths[-1])
    is_right_bucket = (buckets_of_ys == num_bars - 1)
    
    if is_right_bucket.any():
        # The right half-normal originates at borders[-2] and decays to the right.
        dist_from_left_edge = (ys[is_right_bucket] - borders[-2]).clamp(min=0.0)
        
        # The integral from borders[-2] up to 'ys' is just the standard CDF.
        share_of_bucket_left[is_right_bucket] = hn_right.cdf(dist_from_left_edge)

    # 4. Final Aggregation
    prob_in_bucket = probs.gather(-1, buckets_of_ys) * share_of_bucket_left
    prob_left_of_ys = prob_left_of_bucket + prob_in_bucket

    # REMOVED: The manual truncation lines (ys <= borders[0] = 0.0)
    # The half-normals mathematically handle points outside the training borders natively.

    return prob_left_of_ys.clip(0.0, 1.0)


def calculate_all_distribution_metrics_tabpfn(
    y_test_orig, 
    tabpfn_preds, 
    target_transform_fn, 
    target_scale,
    grid_points,
    scale_args=None,
):
    """
    Vectorized calculation of CRPS, Wasserstein (W1), KS-distance, and NLLH.
    
    Args:
        y_test_orig: Array of shape (n_instances, n_observations). ORIGINAL unscaled data.
        tabpfn_preds: List of dictionaries[val_batch_i, ...] containing 'logits'
        borders: The bucket borders tensor used by TabPFN (in transformed space)
        target_transform_fn: Function mapping original targets to TabPFN's space 
        target_scale: Method used to scale targets ('log', 'max', 'none', 'z-score')
        scale_args: Additional args for scaling ([mean, std] for 'z-score')
        grid_points: Resolution of the CDF integration grid.
        
    Returns:
        metrics_summary (dict): Aggregated mean/std metrics over the whole test set.
        instance_metrics (dict): Tensors of shape (n_instances,) for deep dives.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_test_orig = torch.as_tensor(y_test_orig, dtype=torch.float32, device=device)
    
    all_crps, all_w1, all_ks, all_nllh = [], [], [],[]
    instance_idx = 0
    
    for batch in tabpfn_preds:
        logits = batch['logits'].to(device)  # shape: (batch_size, num_bars)
        borders = batch['criterion'].borders.to(device)  # shape: (num_bars+1,)

        batch_size = logits.shape[0]
        
        # Ground truth observations for this batch: shape (batch_size, n_observations)
        batch_y_orig = y_test_orig[instance_idx : instance_idx + batch_size].to(device) 
        batch_y_scaled = target_transform_fn(batch_y_orig)
        n_obs = batch_y_orig.shape[1]
        
        # =========================================================
        # PART 1: CDF-BASED METRICS (CRPS, W1, KS) IN ORIGINAL SPACE
        # =========================================================
        min_y = batch_y_orig.min(dim=1, keepdim=True)[0]
        max_y = batch_y_orig.max(dim=1, keepdim=True)[0]
        y_range = max_y - min_y
        y_range = torch.where(y_range == 0, torch.ones_like(y_range), y_range)
        
        # Apples-to-apples: 1.5x margin on BOTH sides
        start_y = min_y - 1.5 * y_range
        end_y = max_y + 1.5 * y_range
        
        steps = torch.linspace(0, 1, grid_points, device=device).unsqueeze(0)
        y_grid = start_y + steps * (end_y - start_y)
        
        # Empirical CDF
        indicator = (batch_y_orig.unsqueeze(1) <= y_grid.unsqueeze(2)).float()
        F_emp = indicator.mean(dim=2)  
        
        # TabPFN CDF
        if target_scale == "log":
            y_grid_safe = y_grid.clamp(min=-1 + 1e-10) # ensure we don't go below log1p domain
        else:
            y_grid_safe = y_grid # max, z-score, and none can handle negative y_grids normally
            
        z_grid = target_transform_fn(y_grid_safe)
        F_tab = cdf_tabpfn(logits, z_grid, borders)
        
        # Numerically integrate
        cdf_diff = F_tab - F_emp
        abs_cdf_diff = torch.abs(cdf_diff)
        
        w1_batch = torch.trapezoid(abs_cdf_diff, x=y_grid, dim=1)
        crps_batch = torch.trapezoid(cdf_diff ** 2, x=y_grid, dim=1) 
        ks_batch = torch.max(abs_cdf_diff, dim=1)[0]               
        
        all_w1.append(w1_batch)
        all_crps.append(crps_batch)
        all_ks.append(ks_batch)
        
        # =========================================================
        # PART 2: VECTORIZED NLLH (SCALED SPACE + JACOBIAN + BIAS)
        # =========================================================
        
        # Expand logits to match y: (batch_size, n_observations, num_bars)
        logits_expanded = logits.unsqueeze(1).expand(-1, n_obs, -1)
        
        # Fully vectorized NLLH computation. Output shape: (batch_size, n_observations)
        nlog_pdf = -logpdf_tabpfn(logits=logits_expanded, y=batch_y_scaled, borders=borders)
        
        # Cap infinite values
        nlog_pdf.clamp_(max=50.0)  # corresponds to prob of 2e-22, which is already tiny and prevents numerical issues in exp/log later on.

        # Apply Jacobian adjustments & compute bias vectorized over the batch
        if target_scale == "log":
            nlog_pdf += batch_y_scaled  # Jacobian for log
            # Max over dim=1 (observations) -> shape: (batch_size,)
            max_exp_minus_1 = torch.max(torch.exp(batch_y_scaled) - 1.0, dim=1)[0]
            bias = -torch.log(max_exp_minus_1)
            
        elif target_scale in ["max", "none"]:
            max_scaled = torch.max(batch_y_scaled, dim=1)[0]
            bias = -torch.log(max_scaled)
            
        elif target_scale == "z-score":
            mean_t = torch.as_tensor(scale_args[0], device=device, dtype=batch_y_scaled.dtype)
            std_t = torch.as_tensor(scale_args[1], device=device, dtype=batch_y_scaled.dtype)
            scaler = std_t * batch_y_scaled + mean_t
            max_scaler = torch.max(scaler, dim=1)[0]
            bias = torch.log(std_t) - torch.log(max_scaler)
            
        else:
            raise ValueError("target_scale must be 'none', 'max', 'log' or 'z-score'")

        # Mean across observations for each instance -> shape: (batch_size,)
        nlog_pdf_val = nlog_pdf.mean(dim=1)
        
        # Final instance NLLH -> shape: (batch_size,)
        batch_nllh = nlog_pdf_val + bias
        all_nllh.append(batch_nllh)
        
        instance_idx += batch_size
        
    # Aggregate results
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
        "KS_std": all_ks.std().item()
    }
    
    instance_metrics = {
        "NLLH": all_nllh,
        "CRPS": all_crps,
        "Wasserstein": all_w1,
        "KS": all_ks
    }
    
    return metrics_summary, instance_metrics

