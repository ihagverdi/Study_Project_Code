# src/tabpfn_project/paths.py
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# Now define all your standard project directories relative to ROOT_DIR
DATA_DIR = ROOT_DIR / "data"
DISTNET_DATA_DIR = DATA_DIR / "distnet_data"

NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

CONTEXT_SIZES_DIR = ROOT_DIR / "experiments" / "data" / "experiment_context_sizes"

RESULTS_DIR = ROOT_DIR / "results"

# Create the directories if they don't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

