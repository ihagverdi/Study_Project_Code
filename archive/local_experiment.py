import subprocess
import itertools
import time
from datetime import datetime
from tabpfn_project.paths import ROOT_DIR
from tabpfn_project.globals import N_FOLDS, DISTNET_SCENARIOS, DISTNET_CONTEXT_SEEDS

def main():

    CONTEXT_SIZES = [2**i for i in range(5, 13)]  # from 32 to 4096
    
    experiment_configs = list(itertools.product(
        DISTNET_SCENARIOS, 
        CONTEXT_SIZES, 
        list(range(N_FOLDS)),
        DISTNET_CONTEXT_SEEDS
    ))
    
    total_runs = len(experiment_configs)
    print(f"Starting experiment sweep: {total_runs} total runs.\n")

    # 3. Loop through configs and run them as isolated subprocesses
    for i, (scenario, context_size, fold, context_seed) in enumerate(experiment_configs, start=1):
        print(f"{'='*50}")
        print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"🧪 Run {i}/{total_runs} - Scenario: {scenario}, Context Size: {context_size}, Fold: {fold}, Context Seed: {context_seed}")
        print(f"{'='*50}")
        
        # Build the command exactly as you would type it in the Windows terminal
        # "uv run python -m module.path --arg value"
        cmd = [
            "uv", "run", "python", "-m", "tabpfn_project.scripts.main",
            "--scenario", str(scenario),
            "--model", "tabpfn",
            "--context_size", str(context_size),
            "--fold", str(fold),
            "--target_scale", "log",
            "--seed_context", str(context_seed),
            "--subsample_method", "flatten-random",
            "--save_dir", "local_experiment_results_tabpfn_context_sweep_32_to_4096",
        ]
        
        start_time = time.time()
        
        try:
            # 4. Execute the command
            # cwd=ROOT_DIR ensures it always runs from the project root.
            # check=True makes it raise an error if the script crashes.
            subprocess.run(cmd, cwd=ROOT_DIR, check=True)
            
        except subprocess.CalledProcessError as e:
            # If one experiment OOMs or crashes, catch it and continue the rest!
            print(f"\n[ERROR] Experiment {i} failed with exit code {e.returncode}.")
            print("Moving to the next experiment...\n")
            continue
        except KeyboardInterrupt:
            print("\n[INFO] Sweep aborted by user (Ctrl+C).")
            break
            
        elapsed = time.time() - start_time
        print(f"\n✅[SUCCESS] Run {i} completed in {elapsed:.2f} seconds.\n")
    
    print(f"🏁All experiments completed. Total runs attempted: {total_runs}.")

if __name__ == "__main__":
    main()