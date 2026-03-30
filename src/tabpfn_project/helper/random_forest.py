import numpy as np
import torch
from tabpfn_project.globals import MAX_CLAMP_VAL_NLLH

class TreeNode:
    """A node in the Custom Regression Tree."""
    # Using __slots__ prevents the creation of a __dict__ for every node, 
    # saving massive amounts of memory and lookup time in large trees.
    __slots__ =['is_leaf', 'mean', 'var', 'split_feature', 'split_value', 'left', 'right']
    
    def __init__(self):
        self.is_leaf = False
        self.mean = None
        self.var = None
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None

class HutterRegressionTree:
    """
    A single unpruned regression tree with randomized split locations 
    and empirical variance tracking.
    """
    def __init__(self, n_min=5, perc=0.5, var_min=0.01):
        self.n_min = n_min
        self.perc = perc
        self.var_min = var_min
        self.root = None
        self.p = None
        self.v = None

    def fit(self, X, y):
        self.p = X.shape[1]
        self.v = max(1, int(np.floor(self.perc * self.p)))
        
        # Ensure contiguous arrays for fast memory access
        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(y)
        
        # Pass an index array rather than copying the dataset recursively
        idx = np.arange(X.shape[0])
        self.root = self._build_tree(X, y, idx)

    def _build_tree(self, X, y, idx):
        node = TreeNode()
        
        y_node = y[idx]
        n_samples = len(idx)
        
        # 1. Stopping Criterion
        if n_samples <= self.n_min or np.all(y_node == y_node[0]):
            self._make_leaf(node, y_node)
            return node

        # Variable Subsampling
        features = np.random.choice(self.p, self.v, replace=False)

        best_sse = float('inf')
        best_feat = None
        best_xk = None
        best_xl = None

        # Precompute total sum and squared sum for the node to avoid repeated calculations
        sum_total = np.sum(y_node)
        sq_sum_total = np.sum(np.square(y_node))

        for j in features:
            X_j = X[idx, j]
            
            # Sort indices to easily test all possible splits
            sort_idx = np.argsort(X_j)
            X_j_sorted = X_j[sort_idx]
            
            # Find boundaries where adjacent feature values differ
            valid_mask = X_j_sorted[:-1] != X_j_sorted[1:]
            
            if not np.any(valid_mask):
                continue
                
            y_sorted = y_node[sort_idx]

            # Vectorized Sum-of-Squared-Errors (SSE) calculation using running sums
            sum_left_full = np.cumsum(y_sorted)[:-1]
            sq_sum_left_full = np.cumsum(np.square(y_sorted))[:-1]
            count_left_full = np.arange(1, n_samples)
            
            # Slice arrays to compute SSE ONLY for valid split locations (massive speedup)
            sum_left = sum_left_full[valid_mask]
            sq_sum_left = sq_sum_left_full[valid_mask]
            count_left = count_left_full[valid_mask]
            
            sum_right = sum_total - sum_left
            sq_sum_right = sq_sum_total - sq_sum_left
            count_right = n_samples - count_left

            # Numerically stable variance * N (which is SSE)
            var_left = np.maximum(0.0, sq_sum_left - (np.square(sum_left) / count_left))
            var_right = np.maximum(0.0, sq_sum_right - (np.square(sum_right) / count_right))
            valid_sse = var_left + var_right

            # Fast index extraction using numpy built-ins 
            min_idx_in_valid = np.argmin(valid_sse)
            min_sse = valid_sse[min_idx_in_valid]

            if min_sse < best_sse:
                best_sse = min_sse
                best_feat = j
                # Map back to original index
                best_i = np.nonzero(valid_mask)[0][min_idx_in_valid]
                best_xk = X_j_sorted[best_i]
                best_xl = X_j_sorted[best_i+1]

        # Edge Case: If all randomly selected features are constant
        if best_feat is None:
            self._make_leaf(node, y_node)
            return node

        # Randomized Split Locations
        node.split_feature = best_feat
        node.split_value = np.random.uniform(best_xk, best_xl)

        # Split data and recurse (using index masks to avoid deepcopies)
        left_mask = X[idx, best_feat] <= node.split_value
        
        node.left = self._build_tree(X, y, idx[left_mask])
        node.right = self._build_tree(X, y, idx[~left_mask])

        return node

    def _make_leaf(self, node, y):
        """Converts a node to a leaf, computing mean and variance bounds."""
        node.is_leaf = True
        node.mean = np.mean(y)
        # Variance Bounding
        node.var = max(np.var(y), self.var_min)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        means = np.empty(X.shape[0])
        vars_ = np.empty(X.shape[0])
        
        # Iterative batched traversal avoids high overhead Python recursion limits
        queue = [(self.root, np.arange(X.shape[0]))]
        while queue:
            node, idx = queue.pop()
            if node.is_leaf:
                means[idx] = node.mean
                vars_[idx] = node.var
            else:
                left_mask = X[idx, node.split_feature] <= node.split_value
                
                left_idx = idx[left_mask]
                if len(left_idx) > 0:
                    queue.append((node.left, left_idx))
                    
                right_idx = idx[~left_mask]
                if len(right_idx) > 0:
                    queue.append((node.right, right_idx))
                    
        return means, vars_

    def _traverse(self, x, node):
        # Kept strictly for backward compatibility if you call it individually
        if node.is_leaf:
            return node.mean, node.var
        if x[node.split_feature] <= node.split_value:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)

class HutterRandomForest:
    """
    Random Forest implementing the exact modifications from Hutter et al. 2014.
    """
    def __init__(self, n_trees=10, min_samples_split=5, ratio_features=0.5, var_min=0.01):
        self.B = n_trees
        self.n_min = min_samples_split
        self.perc = ratio_features
        self.var_min = var_min
        self.trees =[]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        self.trees =[]
        for _ in range(self.B):
            tree = HutterRegressionTree(n_min=self.n_min, perc=self.perc, var_min=self.var_min)
            tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        
        # Pre-allocate output arrays instead of dynamically calling list.append()
        all_means = np.empty((self.B, n_samples))
        all_vars = np.empty((self.B, n_samples))
        
        for i, tree in enumerate(self.trees):
            m, v = tree.predict(X)
            all_means[i] = m
            all_vars[i] = v

        # Mixture Model Aggregation (Fully Vectorized)
        final_mean = np.mean(all_means, axis=0)
        
        avg_of_vars = np.mean(all_vars, axis=0)
        avg_of_squared_means = np.mean(np.square(all_means), axis=0)
        
        final_var = avg_of_vars + avg_of_squared_means - np.square(final_mean)
        final_var = np.maximum(final_var, self.var_min)

        return final_mean, final_var


def calculate_all_distribution_metrics_rf_baseline(
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
    nlog_pdf = -dist.log_prob(z_test_orig)  # shape (B, O)

    nlog_pdf.clamp_(max=MAX_CLAMP_VAL_NLLH)
    jacobian = -torch.log(torch.max(z_test_orig, dim=1)[0])
    all_nllh = nlog_pdf.mean(dim=1) + jacobian  # shape (B,)

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