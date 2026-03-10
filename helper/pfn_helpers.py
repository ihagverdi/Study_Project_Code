import torch
import time

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

        y: B, logits: B x num_bars.

        :param logits: Tensor of shape B x num_bars
        :param y: Tensor of shape B
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


def calculate_all_distribution_metrics_tabpfn_logspace(
    y_test_orig, 
    tabpfn_preds, 
    target_scale,
    grid_points,
):
    assert target_scale == "log", "Tabpfn support only 'log' scaler atm."

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_test_orig = torch.as_tensor(y_test_orig, dtype=torch.float32, device=device)
    
    # Ground truth in the integration space
    z_test_orig = torch.log1p(y_test_orig)
    
    all_crps, all_w1, all_ks, all_nllh = [], [], [], []
    instance_idx = 0
    
    for batch in tabpfn_preds:
        logits = batch['logits'].to(device)  
        borders = batch['criterion'].borders.to(device)  

        batch_size = logits.shape[0]
        batch_y_orig = y_test_orig[instance_idx : instance_idx + batch_size]
        batch_z_orig = z_test_orig[instance_idx : instance_idx + batch_size]
        n_obs = batch_y_orig.shape[1]
        
        # =========================================================
        # 1. CORE BOUNDS (Empirical Data in Z-Space (i.e., log-space))
        # =========================================================
        min_z_emp = batch_z_orig.min(dim=1, keepdim=True)[0]
        max_z_emp = batch_z_orig.max(dim=1, keepdim=True)[0]
        z_range = (max_z_emp - min_z_emp).clamp(min=1e-5)
        
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
        
        # Mathematical boundaries of TabPFN in Z-space
        z_model_min = borders[1] - tail_left_ext
        z_model_max = borders[-2] + tail_right_ext

        # Global integration bounds
        global_start = torch.minimum(core_start - 0.5 * z_range, z_model_min.unsqueeze(0).expand(batch_size, -1))
        global_end = torch.maximum(core_end + 0.5 * z_range, z_model_max.unsqueeze(0).expand(batch_size, -1))

        # =========================================================
        # 3. PIECEWISE NON-UNIFORM GRID
        # =========================================================
        left_pts, core_pts, right_pts = int(grid_points * 1/6), int(grid_points * 2/3), int(grid_points * 1/6)

        steps_left = torch.linspace(0, 1, left_pts, device=device).view(1, -1)
        z_grid_left = global_start + steps_left * (core_start - global_start)

        steps_core = torch.linspace(0, 1, core_pts + 1, device=device).view(1, -1)[:, 1:]
        z_grid_core = core_start + steps_core * (core_end - core_start)

        steps_right = torch.linspace(0, 1, right_pts + 1, device=device).view(1, -1)[:, 1:]
        z_grid_right = core_end + steps_right * (global_end - core_end)

        z_grid = torch.cat([z_grid_left, z_grid_core, z_grid_right], dim=1) # shape: (batch_size, grid_points)

        # =========================================================
        # 4. CDF EVALUATION & INTEGRATION (Natively in Log-Space)
        # =========================================================
        indicator = (batch_z_orig.unsqueeze(1) <= z_grid.unsqueeze(2)).float()
        F_emp = indicator.mean(dim=2)
        
        # TabPFN is already modeled in Z-space, no un-scaling required!
        F_tab = cdf_tabpfn(logits, z_grid, borders)
        
        cdf_diff = F_tab - F_emp
        abs_cdf_diff = torch.abs(cdf_diff)
        
        # Integration is over dz, returning CRPS in log-units (interpretation: Relative Error)
        w1_batch = torch.trapezoid(abs_cdf_diff, x=z_grid, dim=1)
        crps_batch = torch.trapezoid(cdf_diff ** 2, x=z_grid, dim=1) 
        ks_batch = torch.max(abs_cdf_diff, dim=1)[0]               
        
        all_w1.append(w1_batch)
        all_crps.append(crps_batch)
        all_ks.append(ks_batch)
        
        # =========================================================
        # 5. VECTORIZED NLLH (Evaluated in log-space)
        # =========================================================
        target_transform_fn = lambda y: torch.log1p(y)  # TODO: add support for other scalers, move fn to the input section.
        batch_y_scaled = target_transform_fn(batch_y_orig)
        logits_expanded = logits.unsqueeze(1).expand(-1, n_obs, -1)
        nlog_pdf = -logpdf_tabpfn(logits=logits_expanded, y=batch_y_scaled, borders=borders)
        nlog_pdf.clamp_(max=200.0)  # 200 corresponds to -log(1e-87); prevents possible inf's due to precision errors.
        
        if target_scale == "log":  # (nllh in log-space)
            max_y_scaled = torch.max(batch_y_scaled, dim=1)[0]
            bias = -torch.log(max_y_scaled)

        # if target_scale == "log":  # old version (nllh in original runtime space)
        #     nlog_pdf += batch_y_scaled  
        #     max_exp_minus_1 = torch.max(torch.exp(batch_y_scaled) - 1.0, dim=1)[0]
        #     bias = -torch.log(max_exp_minus_1)

        batch_nllh = nlog_pdf.mean(dim=1) + bias
        all_nllh.append(batch_nllh)
        
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

def batch_predict_tabpfn(model, X_test, validation_batch_size):
    '''
    Batch predictor for tabpfn model.

    Parameters:
    - model: The tabPFN model with a predict method.
    - X_test: Validation input data (n_instances, n_features).
    - validation_batch_size: Batch size for processing validation data.
    
    Returns:
    - tabpfn_preds, tabpfn_predict_time.
    '''
    assert validation_batch_size > 0, "validation_batch_size must be a positive integer"

    n_validation_instances = X_test.shape[0]  # total number of validation instances
    validation_batch_size = min(validation_batch_size, n_validation_instances)
    tabpfn_predict_time = 0.0
    tabpfn_preds = []
    for start in range(0, n_validation_instances, validation_batch_size):
        X_batch = X_test[start: start + validation_batch_size]
        with torch.no_grad():
            tabpfn_predict_time_start = time.time()
            preds = model.predict(X_batch, output_type="full")
            tabpfn_predict_time += (time.time() - tabpfn_predict_time_start)
            tabpfn_preds.append(preds)
            
    return tabpfn_preds, tabpfn_predict_time