import numpy as np


def remove_timeouts(runningtimes, cutoff, features=None, sat_ls=None):
    """
    Remove all instances with more than one value >= cutoff
    """

    if features is None:
        features = [0] * runningtimes.shape[0]
    if sat_ls is None:
        sat_ls = [0] * runningtimes.shape[0]

    new_rt = list()
    new_ft = list()
    new_sl = list()
    assert runningtimes.shape[0] == len(features) == len(sat_ls)
    for instance, feature, sat in zip(runningtimes, features, sat_ls):
        if not np.any(instance >= cutoff):
            new_ft.append(feature)
            new_rt.append(instance)
            new_sl.append(sat)
    print("Discarding %d (%d) instances because not stated TIMEOUTS" %
          (len(features) - len(new_ft), len(features)))
    return np.array(new_rt), np.array(new_ft), new_sl


def remove_instances_with_status(runningtimes, features, sat_ls=None,
                                 status="CRASHED"):
    if sat_ls is None:
        print("Could not remove %s instances" % status)

    new_rt = list()
    new_ft = list()
    new_sl = list()
    assert runningtimes.shape[0] == len(features) == len(sat_ls)
    for f, r, s in zip(features, runningtimes, sat_ls):
        if not status in s:
            new_rt.append(r)
            new_sl.append(s)
            new_ft.append(f)
    print("Discarding %d (%d) instances because of %s" %
          (len(features) - len(new_ft), len(features), status))
    return np.array(new_rt), np.array(new_ft), new_sl


def remove_constant_instances(runningtimes, features, sat_ls=None):
    if sat_ls is None:
        sat_ls = [0] * runningtimes.shape[0]

    new_rt = list()
    new_ft = list()
    new_sl = list()
    assert runningtimes.shape[0] == len(features) == len(sat_ls)
    for f, r, s in zip(features, runningtimes, sat_ls):
        if np.std(f) > 0:
            new_rt.append(r)
            new_sl.append(s)
            new_ft.append(f)
    print("Discarding %d (%d) instances because of constant features" %
          (len(features) - len(new_ft), len(features)))
    return np.array(new_rt), np.array(new_ft), new_sl


def feature_imputation(features, impute_val=-512, impute_with="median"):
    cntr = 0
    if impute_with == "median":
        for col in range(features.shape[1]):
            cntr += features[:, col].tolist().count(impute_val)
            med = np.median(features[:, col])
            features[:, col] = [med if i == impute_val else i for i in features[:, col]]
    print("Imputed %d values with %s" % (cntr, impute_with))
    return features


def remove_zeros(runningtimes, features=None, sat_ls=None):
    """
    Remove all instances with more than one value == 0
    """

    if features is None:
        features = [0] * runningtimes.shape[0]
    if sat_ls is None:
        sat_ls = [0] * runningtimes.shape[0]

    new_rt = list()
    new_ft = list()
    new_sl = list()
    assert runningtimes.shape[0] == len(features) == len(sat_ls)
    for instance, feature, sat in zip(runningtimes, features, sat_ls):
        if not np.any(instance <= 0):
            new_ft.append(feature)
            new_rt.append(instance)
            new_sl.append(sat)
    print("Discarding %d (%d) instances because of ZEROS" % (len(features) - len(new_ft), len(features)))
    return np.array(new_rt), np.array(new_ft), new_sl


def det_constant_features(X):
    """
    Return a list with constant features
    :param X:
    :return:
    """
    max_ = X.max(axis=0)
    min_ = X.min(axis=0)
    diff = max_ - min_

    det_idx = np.where(diff <= 10e-10)
    print(f"Discarding {det_idx[0].shape[0]} features (out of {X.shape[1]})")
    return det_idx


def det_transformation(X):
    """
    Return min max scaling
    """
    min_ = np.min(X, axis=0)
    max_ = np.max(X, axis=0) - min_
    return min_, max_


def delete_constant_features(tra_X, *arrays):
    # Remove constant features
    del_idx = det_constant_features(tra_X)
    tra_X = np.delete(tra_X, del_idx, axis=1)
    processed_arrays = [np.delete(arr, del_idx, axis=1) for arr in arrays]
    return tra_X, *processed_arrays

def preprocess_features(tra_X, *arrays, scal="meanstd"):
    """
    Preprocesses training data and applies the same transformation to any number of 
    additional arrays (validation, test, etc).
    :returns: tuple of processed arrays, with training data first
    """

    if scal == "minmax":
        # Calculate scaling parameters from training data
        min_, max_ = det_transformation(tra_X)
        
        # Apply to training data
        tra_X = (tra_X - min_) / max_
        
        # Apply to other arrays
        processed_arrays = [(arr - min_) / max_ for arr in processed_arrays]
        
    elif scal == "meanstd":
        # Calculate scaling parameters from training data
        mean_ = tra_X.mean(axis=0)
        std_ = tra_X.std(axis=0)
        
        # Safety: prevent division by zero if a feature has 0 variance 
        # (should be handled by delete_constant_features, but good practice to double check)
        std_[std_ == 0] = 1.0
        
        # Apply to training data
        tra_X = (tra_X - mean_) / std_
        
        # Apply to other arrays
        processed_arrays = [(arr - mean_) / std_ for arr in processed_arrays]

    # Return training data + unpacked processed arrays
    return (tra_X, *processed_arrays)