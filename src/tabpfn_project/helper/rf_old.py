import numpy as np

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
