''' DISNET global variables and constants '''
'''------------------------------------------------------'''
RANDOM_STATE=0  # random state for data splitting, model initialization.
N_FOLDS = 10
N_GRID_POINTS = 15000

DISTNET_SCENARIOS = [
    "clasp_factoring",
    "saps-CVVAR",
    "spear_qcp",
    "yalsat_qcp",
    "spear_swgcp",
    "yalsat_swgcp",
    "lpg-zeno",
]

MODELS = ["tabpfn", "distnet", "ngboost", "qrf"]

TARGET_SCALES = ["log", "z-score", "max"]

DISTNET_CONTEXT_SIZES = [2**i for i in range(5, 18)]  # context sizes to evaluate on, from 32 to 131072.
DISTNET_DROP_RATES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
DISTNET_CONTEXT_SEEDS = [j*100 for j in range(1,6)]
DISTNET_DROP_SEEDS = [k*1000 for k in range(1,6)]

MAX_CLAMP_VAL_NLLH = 200.0  # corresponds to a minimum likelihood of exp(-200) ~ 1e-87, which is very small and should not cause numerical issues.


'''------------------------------------------------------'''

