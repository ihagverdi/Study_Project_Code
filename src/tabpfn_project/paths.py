# src/tabpfn_project/paths.py
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# Now define all your standard project directories relative to ROOT_DIR
DISTNET_DATA_DIR = ROOT_DIR / "data" / "distnet_data"

NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

EXPERIMENTS_DATA_DIR = ROOT_DIR / "experiments_data"

RESULTS_DIR = ROOT_DIR / "results"

# Create the directories if they don't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

