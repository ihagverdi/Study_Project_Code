import torch
import numpy as np

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
    from tabpfn_project.helper.utils import dict_to_cpu

    n_validation_instances = X_test.shape[0]  # total number of validation instances
    validation_batch_size = min(validation_batch_size, n_validation_instances)
    tabpfn_preds = []
    for start in range(0, n_validation_instances, validation_batch_size):
        X_batch = X_test[start: start + validation_batch_size]
        with torch.no_grad():
            preds = model.predict(X_batch, output_type="full")
            tabpfn_preds.append(dict_to_cpu(preds))
            
    return tabpfn_preds

def oracle_predict_tabpfn(model, y_test_scaled):
    from tabpfn_project.helper.utils import dict_to_cpu

    tabpfn_preds = []
    for prob_inst in y_test_scaled:
        X_temp = np.zeros(shape=(prob_inst.shape[0], 1))
        model.fit(X_temp, prob_inst)

        with torch.no_grad():
            preds = model.predict(X_temp[0:1], output_type="full")
            tabpfn_preds.append(dict_to_cpu(preds))
    
    return tabpfn_preds

