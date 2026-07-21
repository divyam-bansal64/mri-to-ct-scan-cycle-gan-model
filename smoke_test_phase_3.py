import os
import sys
import torch
import numpy as np

# Resolve imports when executing directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiment_v2.train_experiment_phase_3 import run_experiment, CONFIGS

def run_smoke_test():
    print("=" * 60)
    print("CYCLEGAN PHASE 3 — 5-EPOCH SMOKE TEST MATRIX RUNNER")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Overrides for rapid smoke test execution
    smoke_overrides = {
        "epochs": 5,
        "limit": 50,          # 50 images per domain instead of 500
        "save_interval": 99,  # do not waste time writing checkpoints
        "eval_interval": 1,   # execute validation metrics loop every epoch
    }
    
    configs_to_test = [
        "Run3v2_H_full_corrected_R1",
        "Run4v2_H_full_normalized_FFT",
        "Run6v2_H_full_deep_VGG",
        "Run8v2_H_full_R1_FFT",
        "Run9_H_full_VGG_FFT_combo",
        "Run10_H_full_idt_VGG",
        "Run11_H_full_idt_VGG_FFT_combo"
    ]
    
    test_results = {}
    
    for name in configs_to_test:
        print("\n" + "-"*50)
        print(f"Starting Smoke Test for: {name}")
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
            loss_fft = history.get("loss_fft", [])
            loss_R1 = history.get("loss_R1", [])
            
            if any(np.isnan(loss_G)) or any(np.isnan(loss_D)):
                passed = False
                failure_reasons.append("NaN detected in loss values")
                
            # Check 2: G loss bounds check (loss_G should not exceed 5.0 under normalized losses)
            if len(loss_G) > 0 and loss_G[-1] > 5.0:
                passed = False
                failure_reasons.append(f"G-loss limit exceeded: {loss_G[-1]:.2f} > 5.0 (expected normalized range)")
                
            # Check 3: FFT loss bounds check (should not exceed 1.0)
            if len(loss_fft) > 0 and loss_fft[-1] > 1.0:
                passed = False
                failure_reasons.append(f"FFT-loss limit exceeded: {loss_fft[-1]:.2f} > 1.0 (normalization bug check)")
                
            # Check 4: R1 loss bounds check (should not exceed 5.0)
            if len(loss_R1) > 0 and loss_R1[-1] > 5.0:
                passed = False
                failure_reasons.append(f"R1-loss limit exceeded: {loss_R1[-1]:.2f} > 5.0 (scaling bug check)")
                
            # Check 5: D/G ratio check (fails the run if collapsed)
            if len(loss_G) > 0 and len(loss_D) > 0:
                ratio = loss_D[-1] / max(loss_G[-1], 1e-8)
                if ratio < 0.01 or ratio > 100.0:
                    passed = False
                    failure_reasons.append(f"D/G loss ratio collapsed: {ratio:.4f} (expected 0.01 to 100)")
                    
            # Check 6: FFT_B eval artifact warning (checks if high-frequency ratio > 0.02)
            fft_ratio_B = history.get("val_fft_ratio_fake_B", [])
            if len(fft_ratio_B) > 0 and fft_ratio_B[-1] > 0.02:
                failure_reasons.append(f"WARNING: MRI high-freq ratio is high: {fft_ratio_B[-1]:.4f} > 0.02 (artifact warning)")
                    
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
    print("PHASE 3 — 5-EPOCH SMOKE TEST MATRIX RESULTS")
    print("="*70)
    print(f"{'Configuration Name':30s} | {'Status':8s} | {'Notes / Failures'}")
    print("-"*70)
    for name, res in test_results.items():
        status_str = "PASS" if res["passed"] else "FAIL"
        reasons_str = "; ".join(res["reasons"]) if res["reasons"] else "None (training stable)"
        print(f"{name:30s} | {status_str:8s} | {reasons_str}")
    print("="*70)

if __name__ == "__main__":
    run_smoke_test()

# End of smoke test runner

