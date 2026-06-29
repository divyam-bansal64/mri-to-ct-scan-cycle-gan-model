# experiment_runner_confirm.py

import os
import json
import subprocess
import time
from config.base_configs import CONFIG as BASE_CONFIG
from config.hyperparameter_grid_confirm import HYPERPARAM_GRID


def run_experiment(params):
    """Run a single experiment based on provided parameters."""
    config = BASE_CONFIG.copy()
    config.update(params)

    exp_dir = os.path.join("experiments", config["exp_name"])
    os.makedirs(exp_dir, exist_ok=True)

    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\nRunning experiment: {config['exp_name']}")
    print(f"Saving config to: {config_path}")

    log_path = os.path.join(exp_dir, "train_log.txt")
    with open(log_path, "w") as log_file:
        subprocess.run(
            ["python", "train_full_model_confirm.py", "--config_path", config_path],
            stdout=log_file,
            stderr=subprocess.STDOUT
        )

    print(f"Finished experiment: {config['exp_name']}\n")
    time.sleep(2)


def main():
    print("============================================")
    print("CycleGAN Confirmatory Experiment Runner")
    print("============================================")

    for params in HYPERPARAM_GRID:
        run_experiment(params)

    print("All confirmatory runs completed successfully.")


if __name__ == "__main__":
    main()
