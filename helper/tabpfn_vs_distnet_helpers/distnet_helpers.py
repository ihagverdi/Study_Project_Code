import numpy as np
from scipy import stats


def calculate_llh_instance_distnet(observations, shape, scale, target_scale):
    """
    
     mean log-likelihood per instance for lognormal distribution. (FOR DISTNET)
    
    Args:
        observations: Array of max-scaled observed values for a single instance
        shape: Shape parameter of lognormal distribution
        scale: Scale parameter of lognormal distribution
    
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
