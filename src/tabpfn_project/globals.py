''' DISNET global variables and constants '''
'''------------------------------------------------------'''
from tabpfn_project.helper.data_source_release import get_sc_dict
from tabpfn_project.paths import DISTNET_DATA_DIR

RANDOM_STATE=0  # random state for data splitting, model initialization.
N_FOLDS = 10
N_GRID_POINTS = 15000

DISTNET_SCENARIOS = get_sc_dict(DISTNET_DATA_DIR).keys()

MODELS = ["distnet", "tabpfn", "random_forest", "dist_lognormal"]
TARGET_SCALES = ["log", "max", "original"]
SUBSAMPLE_METHOD_CHOICES = ["flatten-random"]

DISTNET_CONTEXT_SIZES = [2**i for i in range(5, 18)]  # context sizes to evaluate on, from 32 to 131072.
DISTNET_DROP_RATES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
DISTNET_CONTEXT_SEEDS = [j*100 for j in range(1,6)]
DISTNET_DROP_SEEDS = [k*1000 for k in range(1,6)]

MIN_CLAMP_LLH = -200.0  # ~= np.log(1e-87), to prevent numerical issues in log-space. 


'''------------------------------------------------------'''

