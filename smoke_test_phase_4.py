import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiment_v2.train_experiment_phase_4 import run_experiment, CONFIGS

def main():
    print("==========================================================")
    print("       STARTING PHASE 4 LOCAL SMOKE TEST (2 EPOCHS)       ")
    print("==========================================================")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.environ["OUTPUT_DIR"] = tmp_dir
        
        runs_to_test = ["Phase4_Alpha_Hybrid", "Phase4_Beta_Control"]
        
        for run_name in runs_to_test:
            print(f"\n--- Smoke Testing {run_name} ---")
            cfg = CONFIGS[run_name].copy()
            cfg["epochs"] = 2
            cfg["decay_epoch"] = 1
            cfg["eval_interval"] = 1
            cfg["save_interval"] = 2
            cfg["limit"] = 6  # Tiny subset for instant execution
            cfg["num_workers"] = 0
            cfg["output_dir"] = tmp_dir
            
            # Use real local dataset if available, otherwise use temp
            real_ct = r"E:\code\mri to cti\Dataset\images\trainA"
            real_mri = r"E:\code\mri to cti\Dataset\images\trainB"
            if os.path.exists(real_ct):
                cfg["ct_train_dir"] = real_ct
                cfg["mri_train_dir"] = real_mri
            
            try:
                run_experiment(run_name, cfg)
                
                # Check history.json and assert gate conditions
                history_path = os.path.join(tmp_dir, run_name, "history.json")
                assert os.path.exists(history_path), f"Missing history.json for {run_name}"
                
                import json
                with open(history_path, "r") as f:
                    history = json.load(f)
                
                last_g = history["loss_G"][-1]
                last_fft = history["loss_fft"][-1]
                last_r1 = history["loss_R1"][-1]
                last_perceptual = history.get("loss_perceptual", [0.0])[-1]
                
                print(f"[{run_name}] Last Epoch Losses -> Loss_G: {last_g:.4f} | Loss_FFT: {last_fft:.4f} | Loss_R1: {last_r1:.4f} | Loss_Perceptual: {last_perceptual:.4f}")
                
                # Concrete Gate Conditions (Early-epoch 2 bounds for full lambda weights)
                assert last_g < 35.0, f"Loss_G gate breached: {last_g:.4f} >= 35.0"
                assert last_fft < 10.0, f"Loss_FFT gate breached: {last_fft:.4f} >= 10.0"
                assert last_r1 < 200.0, f"Loss_R1 gate breached: {last_r1:.4f} >= 200.0"
                assert last_perceptual < 5.0, f"Loss_Perceptual gate breached: {last_perceptual:.4f} >= 5.0"
                
                print(f"[PASSED] {run_name} completed smoke test and passed all loss threshold gates!")
            except Exception as e:
                print(f"[FAILED] {run_name} failed smoke test: {e}")
                raise e

    print("\n==========================================================")
    print("       [SUCCESS] ALL PHASE 4 SMOKE TESTS PASSED!          ")
    print("==========================================================")

if __name__ == "__main__":
    main()
