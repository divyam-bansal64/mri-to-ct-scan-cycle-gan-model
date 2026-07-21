import os
import json
from tabulate import import_failed  # Fallback if tabulate is not installed

def generate_report():
    outputs_dir = r"E:\code\mri to cti\experiment_v2\outputs"
    report_path = os.path.join(outputs_dir, "experiment_summary_report.txt")
    
    if not os.path.exists(outputs_dir):
        print(f"Error: Outputs directory not found at {outputs_dir}")
        return
        
    configs = sorted([d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))])
    
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("MRI <-> CT CYCLEGAN: LOCAL EXPERIMENT SUMMARY REPORT")
    report_lines.append("=" * 100)
    report_lines.append("\n")
    
    for config in configs:
        history_path = os.path.join(outputs_dir, config, "history.json")
        if not os.path.exists(history_path):
            continue
            
        report_lines.append("-" * 100)
        report_lines.append(f"CONFIGURATION: {config}")
        report_lines.append("-" * 100)
        
        try:
            with open(history_path, "r") as f:
                h = json.load(f)
        except Exception as e:
            report_lines.append(f"  Error loading history.json: {e}\n")
            continue
            
        loss_G = h.get("loss_G", [])
        loss_D = h.get("loss_D", [])
        
        ssim_A = h.get("val_rec_ssim_A", [])
        ssim_B = h.get("val_rec_ssim_B", [])
        mae_A = h.get("val_rec_mae_A", [])
        mae_B = h.get("val_rec_mae_B", [])
        idt_A = h.get("val_idt_mae_A", [])
        idt_B = h.get("val_idt_mae_B", [])
        
        # Calculate validation epochs (every 10 epochs)
        val_step = 10
        total_epochs = len(loss_G)
        
        header = f"{'Epoch':6s} | {'Loss_G':9s} | {'Loss_D':9s} | {'Rec SSIM A':10s} | {'Rec SSIM B':10s} | {'Rec MAE A':9s} | {'Rec MAE B':9s} | {'Idt MAE A':9s} | {'Idt MAE B':9s}"
        report_lines.append(header)
        report_lines.append("-" * len(header))
        
        for i in range(total_epochs):
            epoch_num = i + 1
            lg = f"{loss_G[i]:.4f}" if i < len(loss_G) else "N/A"
            ld = f"{loss_D[i]:.4f}" if i < len(loss_D) else "N/A"
            
            # Since validation metrics are evaluated only every 10 epochs
            if epoch_num % val_step == 0:
                val_idx = (epoch_num // val_step) - 1
                sa = f"{ssim_A[val_idx]:.4f}" if val_idx < len(ssim_A) else "N/A"
                sb = f"{ssim_B[val_idx]:.4f}" if val_idx < len(ssim_B) else "N/A"
                ma = f"{mae_A[val_idx]:.4f}" if val_idx < len(mae_A) else "N/A"
                mb = f"{mae_B[val_idx]:.4f}" if val_idx < len(mae_B) else "N/A"
                ia = f"{idt_A[val_idx]:.4f}" if val_idx < len(idt_A) else "N/A"
                ib = f"{idt_B[val_idx]:.4f}" if val_idx < len(idt_B) else "N/A"
                
                row = f"{epoch_num:6d} | {lg:9s} | {ld:9s} | {sa:10s} | {sb:10s} | {ma:9s} | {mb:9s} | {ia:9s} | {ib:9s}"
                report_lines.append(row)
            elif epoch_num == total_epochs:
                # Always print the final epoch's losses even if not a validation step
                row = f"{epoch_num:6d} | {lg:9s} | {ld:9s} | {'-':10s} | {'-':10s} | {'-':9s} | {'-':9s} | {'-':9s} | {'-':9s}"
                report_lines.append(row)
                
        report_lines.append("\n")
        
    report_content = "\n".join(report_lines)
    
    # Save the report
    try:
        with open(report_path, "w") as f:
            f.write(report_content)
        print(f"Report successfully compiled and saved to:\n{report_path}\n")
    except Exception as e:
        print(f"Error saving report: {e}")
        
    # Print the report to stdout
    print(report_content)

if __name__ == "__main__":
    generate_report()
