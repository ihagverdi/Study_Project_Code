import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

class RuntimePredictionRandomForest(RandomForestRegressor):
    """
    Random Forest model for Algorithm Runtime Prediction as described by 
    Frank Hutter et al. (2014) in "Algorithm Runtime Prediction".
    
    Predicts both the empirical mean and predictive variance (uncertainty).
    """
    
    def __init__(
        self,
        n_estimators=10,        # B = 10 throughout the experiments
        max_features=0.5,       # perc = 0.5
        min_samples_split=6,    # "grown until each node contains no more than n_min data points" (n_min = 5)
                                # setting this to 6 ensures nodes with <= 5 points are NOT split.
        bootstrap=False,        # "using the full training set for each tree"
        var_min=0.01,           # sigma^2_{min} = 0.01
        criterion="squared_error", # Required to ensure tree impurity equals variance
        random_state=0,
        **kwargs                # Expose remainder of scikit-learn's kwargs for full compatibility
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            criterion=criterion,
            random_state=random_state,
            **kwargs
        )
        self.var_min = var_min

    def fit(self, X, y, sample_weight=None):
        """
        Fit the random forest model. Handles user-defined shape constraints.
        X shape: (B, D)
        y shape: (B, 1)
        """
        # Scikit-learn expects 1D arrays for single-target regression (B,). 
        # We handle the user's expected (B, 1) shape silently here.
        if y.ndim == 2 and y.shape[1] == 1:
            y = np.ravel(y)
            
        if self.criterion != "squared_error":
            warnings.warn("Variance calculation relies on the 'squared_error' "
                          "criterion to represent empirical variance in leaves.")
            
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        """
        Predict mean and predictive variance for X based on the Law of Total Variance.
        
        X shape: (B, D)
        Returns: Tuple of arrays (means, variances), both of shape (B, 1)
        """
        check_is_fitted(self)
        
        n_samples = X.shape[0]
        B = self.n_estimators
        
        # Arrays to hold the individual tree predictions and leaf variances
        mu_b_all = np.zeros((B, n_samples))
        sigma2_b_all = np.zeros((B, n_samples))
        
        for i, estimator in enumerate(self.estimators_):
            # 1. Get the leaf node index each sample falls into
            leaves = estimator.apply(X)
            
            # 2. Extract the mean prediction (mu_b) for that leaf
            # tree_.value holds the output values of the tree nodes. Shape: (n_nodes, 1, 1)
            mu_b = estimator.tree_.value[leaves, 0, 0]
            mu_b_all[i, :] = mu_b
            
            # 3. Extract the empirical variance (sigma^2_b) of that leaf
            # Under the 'squared_error' criterion, the calculated impurity is exactly the variance
            sigma2_b = estimator.tree_.impurity[leaves]
            
            # Apply the floor threshold sigma^2_{min}
            sigma2_b = np.maximum(sigma2_b, self.var_min)
            sigma2_b_all[i, :] = sigma2_b
            
        # --- Apply the Law of Total Variance (Eq. 5 in paper) ---
        
        # Mean across trees: (1 / B) * sum(mu_b)
        mu = np.mean(mu_b_all, axis=0)
        
        # Component 1: Average variance of the trees: (1 / B) * sum(sigma2_b)
        mean_of_variances = np.mean(sigma2_b_all, axis=0)
        
        # Component 2: Variance across the means of the trees
        # (1 / B) * sum(mu_b^2) - mu^2  <--> np.var(..., ddof=0)
        variance_of_means = np.var(mu_b_all, axis=0, ddof=0)
        
        # Total Variance
        sigma2 = mean_of_variances + variance_of_means
        
        # Return as (B, 1) shapes to maintain rigorous correspondence with input shapes
        return mu.reshape(-1, 1), sigma2.reshape(-1, 1)
