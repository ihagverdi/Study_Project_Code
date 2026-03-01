import time
import numpy as np
import torch
from helper.pfn_helpers import logPDF_pfn

def predict_and_calculate_nllh_tabpfn(model, X_test, y_test, validation_batch_size, target_scale, args=None):
    '''
    Calculate the negative log-likelihood (NLLH) for TabPFN.

    Parameters:
    - model: The tabPFN model with a predict method.
    - X_test: Validation input data (n_instances, n_features).
    - y_test: Scaled validation target data (n_instances, num_samples_per_instance).
    - validation_batch_size: Batch size for processing validation data.
    - target_scale: Scaling method for targets ('none', 'max', 'log', or 'z-score').
    - args: Additional arguments for scaling ([mean, std] for 'z-score').
    
    Returns:
    - nllh, tabpfn_preds, tabpfn_predict_time.
    '''
    assert validation_batch_size > 0, "validation_batch_size must be a positive integer"
    assert X_test.ndim == 2 and y_test.ndim == 2, "X_test and y_test must be 2D arrays"
    assert X_test.shape[0] == y_test.shape[0], "X_test and y_test must have the same number of instances"

    nllh = 0.0
    n_validation_instances = X_test.shape[0]  # total number of validation instances
    validation_batch_size = min(validation_batch_size, n_validation_instances)
    tabpfn_predict_time = 0.0

    tabpfn_preds = []
    for start in range(0, n_validation_instances, validation_batch_size):
        X_batch = X_test[start: start + validation_batch_size]
        y_batch = y_test[start: start + validation_batch_size]

        with torch.no_grad():
            tabpfn_predict_time_start = time.time()
            preds = model.predict(X_batch, output_type="full")
            tabpfn_predict_time += (time.time() - tabpfn_predict_time_start)

            tabpfn_preds.append(preds)
            logits = preds["logits"]
            borders = preds["criterion"].borders
            # foreach instance loop.
            for instance_idx, obss in enumerate(y_batch):
                logits_i = logits[instance_idx:instance_idx+1]  # keep batch dimension
                obs_t = torch.as_tensor(obss, device=logits_i.device, dtype=logits_i.dtype)
                assert obs_t.ndim == 1, "Each target instance must be a 1D array"
                logits_rep = torch.repeat_interleave(logits_i, repeats=obs_t.shape[0], dim=0)

                nlog_pdf = -logPDF_pfn(logits=logits_rep, y=obs_t, borders=borders)  # -log(p) of shape (n,)

                # Cap infinite values to a maximum penalty (50 nats ≈ prob of 2e-22)
                max_nll = 50.0
                nlog_pdf.clamp_(max=max_nll)

                if target_scale == "log":
                    assert nlog_pdf.shape == obs_t.shape, "nlog_pdf and obs_t must have the same shape for log scaling"
                    nlog_pdf += obs_t  # jacobian adjustment
                    bias = -torch.log(torch.max(torch.exp(obs_t) - 1))  # prevent easy instances dominating the nllh score

                elif target_scale == "max" or target_scale == "none":
                    bias = -torch.log(torch.max(obs_t))

                elif target_scale == "z-score":
                    assert args is not None, "std and mean values must be provided for z-score scaling"
                    assert len(args) == 2, "args must contain (mean, std) for z-score scaling"
                    mean, std = args[0], args[1]
                    
                    # Ensure mean and std are on the same device as obs_t
                    mean_t = torch.as_tensor(mean, device=obs_t.device, dtype=obs_t.dtype)
                    std_t = torch.as_tensor(std, device=obs_t.device, dtype=obs_t.dtype)
                    
                    scaler = std_t * obs_t + mean_t
                    bias = torch.log(std_t) - torch.log(torch.max(scaler))
                    

                else:
                    raise ValueError("target_scale must be either 'none', 'max', 'log' or 'z-score'")

                nlog_pdf_val = nlog_pdf.mean().item()
                bias_val = bias.item()

                if not np.isfinite(bias_val):
                    print(f"⚠️  Instance {start + instance_idx}: bias={bias_val}, max(obs)={torch.max(obs_t).item():.6e}, min(obs)={torch.min(obs_t).item():.6e}")
                
                if not np.isfinite(nlog_pdf_val):
                    print(f"⚠️  Instance {start + instance_idx}: nlog_pdf={nlog_pdf_val}")

                instance_nllh = nlog_pdf_val + bias_val
                nllh += instance_nllh

    
    nllh /= n_validation_instances
    return nllh, tabpfn_preds, tabpfn_predict_time