# ==============================================================================
# CYCLEGAN KAGGLE RUNNER CODES — PHASE 3 (7 RUNS IN PARALLEL)
# Copy and paste these cells into your Kaggle Notebooks.
# ==============================================================================

# ==============================================================================
# SECTION 1 — ACCOUNT 1 (demonbnsl) — 4 VGG RUNS (5 Hours Wall-Clock)
# ==============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# SESSION 1 (Runs 6v2 & 9 in Parallel)
# ──────────────────────────────────────────────────────────────────────────────

# --- CELL 1 (Setup Environment) ---
import os
import sys

# 1. Setup paths so local imports work in Kaggle
CODE_DATASET_PATH = "/kaggle/input/datasets/demonbnsl/cycle-gan-code-phase-3"
sys.path.insert(0, CODE_DATASET_PATH)

# 2. Configure environment directory paths
os.environ["CT_TRAIN_DIR"]  = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainA"
os.environ["MRI_TRAIN_DIR"] = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainB"
os.environ["OUTPUT_DIR"]    = "/kaggle/working/outputs"

# 3. Verify files
assert os.path.exists(os.path.join(CODE_DATASET_PATH, "utils", "metrics.py")), "Missing utils/metrics.py"
assert os.path.exists(os.path.join(CODE_DATASET_PATH, "utils", "losses_phase_3.py")), "Missing utils/losses_phase_3.py"
assert os.path.exists(os.path.join(CODE_DATASET_PATH, "experiment_v2", "train_experiment_phase_3.py")), "Missing train_experiment_phase_3.py"

print("[SUCCESS] Setup verified and imports are ready for Session 1!")


# --- CELL 2 (Write Parallel Worker) ---
code_content = """
import os
import sys
import traceback

def run_worker(run_name, device_idx):
    # Isolate process to a single GPU before importing PyTorch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)
    sys.path.insert(0, "/kaggle/input/datasets/demonbnsl/cycle-gan-code-phase-3")
    
    os.environ["CT_TRAIN_DIR"]  = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainA"
    os.environ["MRI_TRAIN_DIR"] = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainB"
    os.environ["OUTPUT_DIR"]    = "/kaggle/working/outputs"
    
    from experiment_v2.train_experiment_phase_3 import run_experiment, CONFIGS
    cfg = CONFIGS[run_name].copy()
    cfg["limit"] = 500
    cfg["epochs"] = 50
    cfg["eval_interval"] = 10
    cfg["save_interval"] = 10
    cfg["num_workers"] = 2
    
    print(f"[START] Starting {run_name} on GPU {device_idx}...")
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


# --- CELL 3 (Run Session 1 Subprocesses) ---
import multiprocessing
import sys
sys.path.insert(0, "/kaggle/working")
from kaggle_parallel import run_worker

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    print("[SESSION 1] STARTING RUNS 6v2 AND 9 IN PARALLEL...")
    p6 = multiprocessing.Process(target=run_worker, args=("Run6v2_H_full_deep_VGG", 0))
    p9 = multiprocessing.Process(target=run_worker, args=("Run9_H_full_VGG_FFT_combo", 1))
    
    p6.start()
    p9.start()
    
    p6.join()
    p9.join()
    print("[SESSION 1] COMPLETED SUCCESSFULLY!")


# ──────────────────────────────────────────────────────────────────────────────
# SESSION 2 (Runs 10 & 11 in Parallel)
# ──────────────────────────────────────────────────────────────────────────────

# --- CELL 1 (Setup Environment) ---
import os
import sys

# 1. Setup paths so local imports work in Kaggle
CODE_DATASET_PATH = "/kaggle/input/datasets/demonbnsl/cyclegan-experiment-code"
sys.path.insert(0, CODE_DATASET_PATH)

# 2. Configure environment directory paths
os.environ["CT_TRAIN_DIR"]  = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainA"
os.environ["MRI_TRAIN_DIR"] = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainB"
os.environ["OUTPUT_DIR"]    = "/kaggle/working/outputs"

print("[SUCCESS] Setup verified and imports are ready for Session 2!")


# --- CELL 2 (Write Parallel Worker) ---
# (Same Cell 2 content as Session 1 above, execute to generate /kaggle/working/kaggle_parallel.py)


# --- CELL 3 (Run Session 2 Subprocesses) ---
import multiprocessing
import sys
sys.path.insert(0, "/kaggle/working")
from kaggle_parallel import run_worker

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    print("[SESSION 2] STARTING RUNS 10 AND 11 IN PARALLEL...")
    p10 = multiprocessing.Process(target=run_worker, args=("Run10_H_full_idt_VGG", 0))
    p11 = multiprocessing.Process(target=run_worker, args=("Run11_H_full_idt_VGG_FFT_combo", 1))
    
    p10.start()
    p11.start()
    
    p10.join()
    p11.join()
    print("[SESSION 2] COMPLETED SUCCESSFULLY!")


# ==============================================================================
# SECTION 2 — ACCOUNT 2 (divyambnsl) — 3 REDO RUNS (5 Hours Wall-Clock)
# ==============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# SESSION 3 (Runs 4v2 & 3v2 in Parallel)
# ──────────────────────────────────────────────────────────────────────────────

# --- CELL 1 (Setup Environment) ---
import os
import sys

# 1. Setup paths so local imports work in Kaggle
CODE_DATASET_PATH = "/kaggle/input/datasets/divyambnsl/cycle-gan-code-phase-3"
sys.path.insert(0, CODE_DATASET_PATH)

# 2. Configure environment directory paths
os.environ["CT_TRAIN_DIR"]  = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainA"
os.environ["MRI_TRAIN_DIR"] = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainB"
os.environ["OUTPUT_DIR"]    = "/kaggle/working/outputs"

# 3. Verify files
assert os.path.exists(os.path.join(CODE_DATASET_PATH, "utils", "metrics.py")), "Missing utils/metrics.py"
assert os.path.exists(os.path.join(CODE_DATASET_PATH, "utils", "losses_phase_3.py")), "Missing utils/losses_phase_3.py"
assert os.path.exists(os.path.join(CODE_DATASET_PATH, "experiment_v2", "train_experiment_phase_3.py")), "Missing train_experiment_phase_3.py"

print("[SUCCESS] Setup verified and imports are ready for Session 3!")


# --- CELL 2 (Write Parallel Worker) ---
code_content = """
import os
import sys
import traceback

def run_worker(run_name, device_idx):
    # Isolate process to a single GPU before importing PyTorch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)
    sys.path.insert(0, "/kaggle/input/datasets/divyambnsl/cycle-gan-code-phase-3")
    
    os.environ["CT_TRAIN_DIR"]  = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainA"
    os.environ["MRI_TRAIN_DIR"] = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainB"
    os.environ["OUTPUT_DIR"]    = "/kaggle/working/outputs"
    
    from experiment_v2.train_experiment_phase_3 import run_experiment, CONFIGS
    cfg = CONFIGS[run_name].copy()
    cfg["limit"] = 500
    cfg["epochs"] = 50
    cfg["eval_interval"] = 10
    cfg["save_interval"] = 10
    cfg["num_workers"] = 2
    
    print(f"[START] Starting {run_name} on GPU {device_idx}...")
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


# --- CELL 3 (Run Session 3 Subprocesses) ---
import multiprocessing
import sys
sys.path.insert(0, "/kaggle/working")
from kaggle_parallel import run_worker

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    print("[SESSION 3] STARTING RUNS 4v2 AND 3v2 IN PARALLEL...")
    p4 = multiprocessing.Process(target=run_worker, args=("Run4v2_H_full_normalized_FFT", 0))
    p3 = multiprocessing.Process(target=run_worker, args=("Run3v2_H_full_corrected_R1", 1))
    
    p4.start()
    p3.start()
    
    p4.join()
    p3.join()
    print("[SESSION 3] COMPLETED SUCCESSFULLY!")


# ──────────────────────────────────────────────────────────────────────────────
# SESSION 4 (Run 8v2 on GPU 0)
# ──────────────────────────────────────────────────────────────────────────────

# --- CELL 1 (Setup Environment) ---
import os
import sys

# 1. Setup paths so local imports work in Kaggle
CODE_DATASET_PATH = "/kaggle/input/datasets/divyambnsl/cycle-gan-code-phase-3"
sys.path.insert(0, CODE_DATASET_PATH)

# 2. Configure environment directory paths
os.environ["CT_TRAIN_DIR"]  = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainA"
os.environ["MRI_TRAIN_DIR"] = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainB"
os.environ["OUTPUT_DIR"]    = "/kaggle/working/outputs"

print("[SUCCESS] Setup verified and imports are ready for Session 4!")


# --- CELL 2 (Write Parallel Worker) ---
# (Same Cell 2 content as Session 3 above, execute to generate /kaggle/working/kaggle_parallel.py)


# --- CELL 3 (Run Session 4 Subprocess) ---
import multiprocessing
import sys
sys.path.insert(0, "/kaggle/working")
from kaggle_parallel import run_worker

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    print("[SESSION 4] STARTING RUN 8v2...")
    p8 = multiprocessing.Process(target=run_worker, args=("Run8v2_H_full_R1_FFT", 0))
    
    p8.start()
    p8.join()
    print("[SESSION 4] COMPLETED SUCCESSFULLY!")
