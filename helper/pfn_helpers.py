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

def logPDF_pfn(
        logits: torch.Tensor,
        y: torch.Tensor, 
        borders,
    ) -> torch.Tensor:
        """Returns the log-pdf of tabpfn at y.

        y: B, logits: B x num_bars.

        :param logits: Tensor of shape B x num_bars
        :param y: Tensor of shape B
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
