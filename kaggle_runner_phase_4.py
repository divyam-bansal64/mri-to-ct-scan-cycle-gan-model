# ==============================================================================
# CYCLEGAN KAGGLE RUNNER CODE — PHASE 4 (FINAL 200-EPOCH DUAL-GPU RUN)
# Copy and paste these cells into your Kaggle Notebook (T4 x 2 GPUs).
# ==============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# SESSION 1 (Phase4_Alpha_Hybrid & Phase4_Beta_Control in Parallel)
# ──────────────────────────────────────────────────────────────────────────────

# --- CELL 1 (Setup Environment & Verify Imports) ---
import os
import sys

# 1. Setup paths so local imports work in Kaggle
CODE_DATASET_PATH = "/kaggle/input/datasets/divyambnsl/cycle-gan-code-phase-4"
if not os.path.exists(CODE_DATASET_PATH):
    CODE_DATASET_PATH = "/kaggle/input/cycle-gan-code-phase-4"

sys.path.insert(0, CODE_DATASET_PATH)

# 2. Configure environment directory paths
CT_TRAIN_DIR = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainA"
if not os.path.exists(CT_TRAIN_DIR):
    CT_TRAIN_DIR = "/kaggle/input/ct-to-mri-cgan/Dataset/images/trainA"

MRI_TRAIN_DIR = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainB"
if not os.path.exists(MRI_TRAIN_DIR):
    MRI_TRAIN_DIR = "/kaggle/input/ct-to-mri-cgan/Dataset/images/trainB"

os.environ["CT_TRAIN_DIR"]  = CT_TRAIN_DIR
os.environ["MRI_TRAIN_DIR"] = MRI_TRAIN_DIR
os.environ["OUTPUT_DIR"]    = "/kaggle/working/outputs"

# Automatically check if a previous session output dataset is attached as input
import glob
prev_output_matches = sorted(glob.glob("/kaggle/input/**/outputs", recursive=True))
if prev_output_matches:
    import shutil
    prev_dir = prev_output_matches[0]
    if prev_dir != os.environ["OUTPUT_DIR"]:
        shutil.copytree(prev_dir, os.environ["OUTPUT_DIR"], dirs_exist_ok=True)
        print(f"[RESUME] Automatically restored previous session checkpoints from: {prev_dir}")

# 3. Verify files
assert os.path.exists(os.path.join(CODE_DATASET_PATH, "utils", "metrics.py")), "Missing utils/metrics.py"
assert os.path.exists(os.path.join(CODE_DATASET_PATH, "utils", "losses_phase_4.py")), "Missing utils/losses_phase_4.py"
assert os.path.exists(os.path.join(CODE_DATASET_PATH, "experiment_v2", "train_experiment_phase_4.py")), "Missing train_experiment_phase_4.py"

print("[SUCCESS] Setup verified and imports are ready for Phase 4 Final Runs!")


# --- CELL 2 (Write Parallel Worker) ---
code_content = """
import os
import sys
import traceback

def run_worker(run_name, device_idx):
    # Isolate process to a single GPU before importing PyTorch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)
    
    code_path = "/kaggle/input/datasets/divyambnsl/cycle-gan-code-phase-4"
    if not os.path.exists(code_path):
        code_path = "/kaggle/input/cycle-gan-code-phase-4"
    sys.path.insert(0, code_path)
    
    ct_dir = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainA"
    if not os.path.exists(ct_dir):
        ct_dir = "/kaggle/input/ct-to-mri-cgan/Dataset/images/trainA"

    mri_dir = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainB"
    if not os.path.exists(mri_dir):
        mri_dir = "/kaggle/input/ct-to-mri-cgan/Dataset/images/trainB"
    
    os.environ["CT_TRAIN_DIR"]  = ct_dir
    os.environ["MRI_TRAIN_DIR"] = mri_dir
    os.environ["OUTPUT_DIR"]    = "/kaggle/working/outputs"
    
    from experiment_v2.train_experiment_phase_4 import run_experiment, CONFIGS
    cfg = CONFIGS[run_name].copy()
    cfg["limit"] = None  # Full dataset (no image restriction)
    cfg["epochs"] = 200
    cfg["decay_epoch"] = 100
    cfg["eval_interval"] = 10
    cfg["save_interval"] = 10
    cfg["num_workers"] = 2
    
    print(f"[START] Starting {run_name} (200 Epochs) on GPU {device_idx}...")
    try:
        run_experiment(run_name, cfg)
        print(f"[SUCCESS] Finished {run_name} successfully!")
    except Exception as e:
        print(f"[ERROR] Failed {run_name}!")
        traceback.print_exc()
"""
with open("/kaggle/working/kaggle_parallel.py", "w") as f:
    f.write(code_content.strip())
print("[SUCCESS] Helper script created successfully at /kaggle/working/kaggle_parallel.py")


# --- CELL 3 (Run Subprocesses on GPU 0 & GPU 1) ---
import multiprocessing
import sys
sys.path.insert(0, "/kaggle/working")
from kaggle_parallel import run_worker

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    print("[PHASE 4] STARTING MODEL ALPHA (GPU 0) AND MODEL BETA (GPU 1) IN PARALLEL...")
    p_alpha = multiprocessing.Process(target=run_worker, args=("Phase4_Alpha_Hybrid", 0))
    p_beta  = multiprocessing.Process(target=run_worker, args=("Phase4_Beta_Control", 1))
    
    p_alpha.start()
    p_beta.start()
    
    p_alpha.join()
    p_beta.join()
    print("[PHASE 4] FINAL TRAINING RUNS COMPLETED SUCCESSFULLY!")
