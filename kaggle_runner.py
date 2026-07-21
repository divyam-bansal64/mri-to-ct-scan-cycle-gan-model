# ==============================================================================
# CYCLEGAN KAGGLE RUNNER CODES (4 RUNS PER ACCOUNT SEQUENCE)
# Copy and paste these blocks into separate cells in your Kaggle Notebook.
# ==============================================================================

# ==============================================================================
# SECTION 1 — ACCOUNT 1 (Runs 0 & 5, followed by Runs 6 & 4 in parallel)
# ==============================================================================

# ------------------------------------------------------------------------------
# COPY AND PASTE THIS ENTIRE BLOCK INTO CELL 1:
# ------------------------------------------------------------------------------
import os
import sys

# 1. Setup paths so local imports work in Kaggle
CODE_DATASET_PATH = "/kaggle/input/datasets/demonbnsl/cyclegan-experiment-code"
sys.path.insert(0, CODE_DATASET_PATH)

# 2. Configure environment directory paths
os.environ["CT_TRAIN_DIR"]  = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainA"
os.environ["MRI_TRAIN_DIR"] = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainB"
os.environ["OUTPUT_DIR"]    = "/kaggle/working/outputs"

# 3. Verify code dataset structure is correct
assert os.path.exists(os.path.join(CODE_DATASET_PATH, "utils", "metrics.py")), "Missing utils/metrics.py"
assert os.path.exists(os.path.join(CODE_DATASET_PATH, "config", "ref_stats_A.pth")), "Missing config/ref_stats_A.pth"
assert os.path.exists(os.path.join(CODE_DATASET_PATH, "experiment_v2", "train_experiment.py")), "Missing experiment_v2/train_experiment.py"

print("[SUCCESS] Setup verified and imports are ready!")


# ------------------------------------------------------------------------------
# COPY AND PASTE THIS ENTIRE BLOCK INTO CELL 2:
# ------------------------------------------------------------------------------
code_content = """
import os
import sys
import traceback

def run_worker(run_name, device_idx):
    # Isolate this process to a single GPU before importing PyTorch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)
    
    sys.path.insert(0, "/kaggle/input/datasets/demonbnsl/cyclegan-experiment-code")
    
    os.environ["CT_TRAIN_DIR"]  = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainA"
    os.environ["MRI_TRAIN_DIR"] = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainB"
    os.environ["OUTPUT_DIR"]    = "/kaggle/working/outputs"
    
    from experiment_v2.train_experiment import run_experiment, CONFIGS
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


# ------------------------------------------------------------------------------
# COPY AND PASTE THIS ENTIRE BLOCK INTO CELL 3:
# ------------------------------------------------------------------------------
import multiprocessing
import sys
import time

# Ensure working directory is in import path
sys.path.insert(0, "/kaggle/working")
from kaggle_parallel import run_worker

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    print("[PHASE 1] STARTING PHASE 1...")
    p0 = multiprocessing.Process(target=run_worker, args=("Run0_H_full_baseline", 0))
    p5 = multiprocessing.Process(target=run_worker, args=("Run5_G_low_idt_dice", 1))
    
    p0.start()
    p5.start()
    
    p0.join()
    p5.join()
    print("[PHASE 1] COMPLETED!")
    
    print("[PHASE 2] STARTING PHASE 2...")
    p6 = multiprocessing.Process(target=run_worker, args=("Run6_H_full_perceptual_vgg", 0))
    p4 = multiprocessing.Process(target=run_worker, args=("Run4_F_asym_idt_FFT_resizeconv", 1))
    
    p6.start()
    p4.start()
    
    p6.join()
    p4.join()
    print("[PHASE 2] COMPLETED!")
    print("[SUCCESS] All 4 runs for Account 1 are finished!")


# ==============================================================================
# SECTION 2 — ACCOUNT 2 (Runs 2 & 3, followed by Runs 1 & 8 in parallel)
# ==============================================================================

# ------------------------------------------------------------------------------
# COPY AND PASTE THIS ENTIRE BLOCK INTO CELL 1:
# ------------------------------------------------------------------------------
import os
import sys

# 1. Setup paths so local imports work in Kaggle
CODE_DATASET_PATH = "/kaggle/input/datasets/divyambnsl/cyclegan-experiments-run"
sys.path.insert(0, CODE_DATASET_PATH)

# 2. Configure environment directory paths
os.environ["CT_TRAIN_DIR"]  = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainA"
os.environ["MRI_TRAIN_DIR"] = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainB"
os.environ["OUTPUT_DIR"]    = "/kaggle/working/outputs"

# 3. Verify code dataset structure is correct
assert os.path.exists(os.path.join(CODE_DATASET_PATH, "utils", "metrics.py")), "Missing utils/metrics.py"
assert os.path.exists(os.path.join(CODE_DATASET_PATH, "config", "ref_stats_A.pth")), "Missing config/ref_stats_A.pth"
assert os.path.exists(os.path.join(CODE_DATASET_PATH, "experiment_v2", "train_experiment.py")), "Missing experiment_v2/train_experiment.py"

print("[SUCCESS] Setup verified and imports are ready!")


# ------------------------------------------------------------------------------
# COPY AND PASTE THIS ENTIRE BLOCK INTO CELL 2:
# ------------------------------------------------------------------------------
code_content = """
import os
import sys
import traceback

def run_worker(run_name, device_idx):
    # Isolate this process to a single GPU before importing PyTorch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)
    
    sys.path.insert(0, "/kaggle/input/datasets/divyambnsl/cyclegan-experiments-run")
    
    os.environ["CT_TRAIN_DIR"]  = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainA"
    os.environ["MRI_TRAIN_DIR"] = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainB"
    os.environ["OUTPUT_DIR"]    = "/kaggle/working/outputs"
    
    from experiment_v2.train_experiment import run_experiment, CONFIGS
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


# ------------------------------------------------------------------------------
# COPY AND PASTE THIS ENTIRE BLOCK INTO CELL 3:
# ------------------------------------------------------------------------------
import multiprocessing
import sys
import time

# Ensure working directory is in import path
sys.path.insert(0, "/kaggle/working")
from kaggle_parallel import run_worker

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    print("[PHASE 1] STARTING PHASE 1...")
    p2 = multiprocessing.Process(target=run_worker, args=("Run2_H_full_resizeconv_bare", 0))
    p3 = multiprocessing.Process(target=run_worker, args=("Run3_H_full_resizeconv_R1", 1))
    
    p2.start()
    p3.start()
    
    p2.join()
    p3.join()
    print("[PHASE 1] COMPLETED!")
    
    print("[PHASE 2] STARTING PHASE 2...")
    p1 = multiprocessing.Process(target=run_worker, args=("Run1_A_baseline_dice", 0))
    p8 = multiprocessing.Process(target=run_worker, args=("Run8_B_capacity_R1_FFT", 1))
    
    p1.start()
    p8.start()
    
    p1.join()
    p8.join()
    print("[PHASE 2] COMPLETED!")
    print("[SUCCESS] All 4 runs for Account 2 are finished!")


# ==============================================================================
# SECTION 3 — SINGLE RUN (Run 7: F_asym_idt_scheduled_TTUR)
# ==============================================================================

# --- OPTION A: RUN ON ACCOUNT 1 (demonbnsl) ---

# COPY AND PASTE THIS BLOCK INTO CELL 1:
import os
import sys

# 1. Setup paths so local imports work in Kaggle
CODE_DATASET_PATH = "/kaggle/input/datasets/demonbnsl/cyclegan-experiment-code"
sys.path.insert(0, CODE_DATASET_PATH)

# 2. Configure environment directory paths
os.environ["CT_TRAIN_DIR"]  = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainA"
os.environ["MRI_TRAIN_DIR"] = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainB"
os.environ["OUTPUT_DIR"]    = "/kaggle/working/outputs"

# 3. Verify files
assert os.path.exists(os.path.join(CODE_DATASET_PATH, "utils", "metrics.py")), "Missing utils/metrics.py"
assert os.path.exists(os.path.join(CODE_DATASET_PATH, "config", "ref_stats_A.pth")), "Missing config/ref_stats_A.pth"

print("[SUCCESS] Setup verified and imports are ready!")


# COPY AND PASTE THIS BLOCK INTO CELL 2:
import traceback
from experiment_v2.train_experiment import run_experiment, CONFIGS

run_name = "Run7_F_asym_idt_scheduled_TTUR"
cfg = CONFIGS[run_name].copy()

# Full budget settings
cfg["limit"] = 500
cfg["epochs"] = 50
cfg["eval_interval"] = 10
cfg["save_interval"] = 10
cfg["num_workers"] = 2

print(f"\n==========================================")
print(f"[START] Starting Run: {run_name}")
print(f"==========================================")

try:
    history, best_ssim = run_experiment(run_name, cfg)
    print(f"[SUCCESS] Completed Run: {run_name} (Best SSIM: {best_ssim:.4f})")
except Exception as e:
    print(f"[ERROR] Failed Run: {run_name}")
    traceback.print_exc()


# --- OPTION B: RUN ON ACCOUNT 2 (divyambnsl) ---

# COPY AND PASTE THIS BLOCK INTO CELL 1:
import os
import sys

# 1. Setup paths so local imports work in Kaggle
CODE_DATASET_PATH = "/kaggle/input/datasets/divyambnsl/cyclegan-experiments-run"
sys.path.insert(0, CODE_DATASET_PATH)

# 2. Configure environment directory paths
os.environ["CT_TRAIN_DIR"]  = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainA"
os.environ["MRI_TRAIN_DIR"] = "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainB"
os.environ["OUTPUT_DIR"]    = "/kaggle/working/outputs"

# 3. Verify files
assert os.path.exists(os.path.join(CODE_DATASET_PATH, "utils", "metrics.py")), "Missing utils/metrics.py"
assert os.path.exists(os.path.join(CODE_DATASET_PATH, "config", "ref_stats_A.pth")), "Missing config/ref_stats_A.pth"

print("[SUCCESS] Setup verified and imports are ready!")


# COPY AND PASTE THIS BLOCK INTO CELL 2:
import traceback
from experiment_v2.train_experiment import run_experiment, CONFIGS

run_name = "Run7_F_asym_idt_scheduled_TTUR"
cfg = CONFIGS[run_name].copy()

# Full budget settings
cfg["limit"] = 500
cfg["epochs"] = 50
cfg["eval_interval"] = 10
cfg["save_interval"] = 10
cfg["num_workers"] = 2

print(f"\n==========================================")
print(f"[START] Starting Run: {run_name}")
print(f"==========================================")

try:
    history, best_ssim = run_experiment(run_name, cfg)
    print(f"[SUCCESS] Completed Run: {run_name} (Best SSIM: {best_ssim:.4f})")
except Exception as e:
    print(f"[ERROR] Failed Run: {run_name}")
    traceback.print_exc()

