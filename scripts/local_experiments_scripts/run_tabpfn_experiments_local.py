import subprocess
import sys

# --- CONFIGURATION ---
SCENARIOS = [
    "clasp_factoring",
    "saps-CVVAR",
    "spear_qcp",
    "yalsat_qcp",
    "spear_swgcp",
    "yalsat_swgcp",
    "lpg-zeno",
]

CONTEXT_SIZES = [2048, 4096, 8192]
SEEDS = [100, 200, 300]
FOLDS = range(10)
MODEL = "tabpfn"  # Changed to variable so save_dir matches automatically

# --- EXECUTION LOOP ---
total_jobs = len(SCENARIOS) * len(CONTEXT_SIZES) * len(SEEDS) * len(FOLDS)
print(f"Starting execution of {total_jobs} jobs locally...")

current_job = 0
for seed in SEEDS:
    for scenario in SCENARIOS:
        for context in CONTEXT_SIZES:  
            for fold in FOLDS:
                current_job += 1
                
                # Dynamic output directory based on model and scenario
                save_dir = f"{MODEL}/seeds/{seed}/{scenario}"
                
                print(f"🧪[{current_job}/{total_jobs}] Running: {scenario} | Context: {context} | Fold: {fold} | Seed: {seed}")

                # Construct the command as a list of strings (safer than shell=True)
                command = [
                    sys.executable, "-m", "source.experiment_1",
                    "--model", MODEL,
                    "--scenario", scenario,
                    "--fold", str(fold),
                    "--seed", str(seed),
                    "--context_size", str(context),
                    "--num_samples_per_instance", "100",
                    "--save_dir", save_dir,
                    "--target_scale", "log",
                    "--subsample_method", "flatten-random",
                    
                ]

                # Run the command
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error running job {current_job}: {e}")
                    # Optional: break or continue depending on preference
                except KeyboardInterrupt:
                    print("\nStopping execution...")
                    sys.exit(0)

print("✅All local jobs finished.")