import os
import sys
import torch
import numpy as np

# Resolve imports when executing directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiment_v2.train_experiment import run_experiment, CONFIGS

def run_smoke_test():
    print("=" * 60)
    # Correct capitalization
    print("CYCLEGAN 5-EPOCH SMOKE TEST MATRIX RUNNER")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Overrides for rapid execution
    smoke_overrides = {
        "epochs": 5,
        "limit": 50,          # 50 images per domain instead of 500
        "save_interval": 99,  # do not waste time writing checkpoints
        "eval_interval": 1,   # execute validation metrics loop every epoch
    }
    
    # 9 Configurations list
    configs_to_test = [
        "Run0_H_full_baseline",
        "Run1_A_baseline_dice",
        "Run2_H_full_resizeconv_bare",
        "Run3_H_full_resizeconv_R1",
        "Run4_F_asym_idt_FFT_resizeconv",
        "Run5_G_low_idt_dice",
        "Run6_H_full_perceptual_vgg",
        "Run7_F_asym_idt_scheduled_TTUR",
        "Run8_B_capacity_R1_FFT"
    ]
    
    test_results = {}
    
    for name in configs_to_test:
        print("\n" + "-"*50)
        print(f"🚀 Starting Smoke Test for: {name}")
        print("-"*50)
        
        # Merge overrides with base config
        cfg = CONFIGS[name].copy()
        cfg.update(smoke_overrides)
        
        passed = True
        failure_reasons = []
        
        try:
            # Run the training loop for 5 epochs
            history, best_ssim = run_experiment(name, cfg)
            
            # Check 1: NaN detection
            loss_G = history.get("loss_G", [])
            loss_D = history.get("loss_D", [])
            
            if any(np.isnan(loss_G)) or any(np.isnan(loss_D)):
                passed = False
                failure_reasons.append("NaN detected in loss values")
                
            # Check 2: Divergence check (loss_G should not explode > 100)
            if len(loss_G) > 0 and loss_G[-1] > 100.0:
                passed = False
                failure_reasons.append(f"Loss G exploded/diverged: {loss_G[-1]:.2f} > 100")
                
            # Check 3: D/G ratio check (fails the run if collapsed)
            if len(loss_G) > 0 and len(loss_D) > 0:
                ratio = loss_D[-1] / max(loss_G[-1], 1e-8)
                if ratio < 0.01 or ratio > 100.0:
                    passed = False
                    failure_reasons.append(f"D/G loss ratio collapsed: {ratio:.4f} (expected 0.01 to 100)")
                    
        except RuntimeError as e:
            passed = False
            if "out of memory" in str(e).lower():
                failure_reasons.append("CUDA Out Of Memory (OOM) error")
            else:
                failure_reasons.append(f"Runtime error: {e}")
        except Exception as e:
            passed = False
            failure_reasons.append(f"Unexpected exception: {e}")
            
        test_results[name] = {
            "passed": passed,
            "reasons": failure_reasons
        }
        
    # Final Visual Summary Report
    print("\n" + "="*70)
    print("5-EPOCH SMOKE TEST MATRIX RESULTS")
    print("="*70)
    print(f"{'Configuration Name':35s} | {'Status':8s} | {'Notes / Failures'}")
    print("-"*70)
    for name, res in test_results.items():
        status_str = "✅ PASS" if res["passed"] else "❌ FAIL"
        reasons_str = "; ".join(res["reasons"]) if res["reasons"] else "None (training stable)"
        print(f"{name:35s} | {status_str:8s} | {reasons_str}")
    print("="*70)

if __name__ == "__main__":
    run_smoke_test()
