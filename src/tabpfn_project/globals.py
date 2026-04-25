''' DISNET global variables and constants '''
'''------------------------------------------------------'''
from tabpfn_project.helper.data_source_release import get_sc_dict
from tabpfn_project.paths import DISTNET_DATA_DIR

RANDOM_STATE=0  # random state for data splitting, model initialization.
N_FOLDS = 10  # number of folds for cross-validation in distnet data.

N_GRID_POINTS = 15000
MAX_HPO_TRIALS = 1000000
MAX_HPO_WCT = 3600
DISTNET_SCENARIOS = list(get_sc_dict(DISTNET_DATA_DIR).keys())

DISTNET_N_EPOCHS = 1000
DISTNET_BATCH_SIZE = 16
DISTNET_WCT = 3540
DISTNET_ES_PATIENCE = 50


MODELS = ["distnet", "tabpfn", "bayesian_distnet", "random_forest", "lognormal"]
TARGET_SCALES = ["log", "max", "original"]

DISTNET_CONTEXT_SIZES = [2**i for i in range(5, 18)]  # context sizes to evaluate on, from 32 to 131072.
DISTNET_DROP_RATES = [0.25, 0.50, 0.75, 0.90, 0.95, 1.0]  # drop rates to evaluate on.
DISTNET_CONTEXT_SEEDS = [j*100 for j in range(1,6)]
DISTNET_DROP_SEEDS = [k*1000 for k in range(1,6)]

LLH_EPSILON = 1e-10

'''------------------------------------------------------'''
TABPFN_VAL_BATCH_SIZE = 1000

