import itertools
import os
import json
import subprocess
import time
from config.base_configs import CONFIG as BASE_CONFIG
from config.hyperparameter_grid import HYPERPARAM_GRID

def generate_experiments():
    keys, values = zip(*HYPERPARAM_GRID.items())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def run_experiment(exp_id, params):
    config = BASE_CONFIG.copy()
    config.update(params)
    config["exp_name"] = f"run_{exp_id:03d}"

    exp_dir = os.path.join("experiments", config["exp_name"])
    os.makedirs(exp_dir, exist_ok=True)

    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n Running experiment {exp_id}: {config['exp_name']}")
    print(f"Config saved to {config_path}")

    # Log output
    with open(os.path.join(exp_dir, "train_log.txt"), "w") as log_file:
        subprocess.run(
            ["python", "train_generator_only.py", "--config_path", config_path],
            stdout=log_file, stderr=subprocess.STDOUT
        )

    print(f" Finished experiment {exp_id} | Results saved to {exp_dir}\n")
    time.sleep(2)

if __name__ == "__main__":
    experiments = list(generate_experiments())
    print(f"Total experiments to run: {len(experiments)}")

    for i, params in enumerate(experiments, start=1):
        try:
            run_experiment(i, params)
        except Exception as e:
            print(f" Experiment {i} failed: {e}")
            continue
