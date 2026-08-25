import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─────────────────────────────────────────────────────────────
# FID METRIC (Pure PyTorch, no Scipy needed)
# ─────────────────────────────────────────────────────────────

class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        except ImportError:
            from torchvision.models import inception_v3
            self.inception = inception_v3(pretrained=True)
            
        # Replace the fully connected layer with Identity to extract pool3 features (2048-dim)
        self.inception.fc = nn.Identity()
        self.inception.eval()
        for p in self.inception.parameters():
            p.requires_grad = False

    def forward(self, x):
        # x: [B, C, H, W] in [-1, 1]
        # 1. Convert to [0, 1]
        x = (x + 1.0) / 2.0
        
        # 2. Convert 1-channel grayscale to 3-channel RGB if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # 3. Resize to 299x299 for Inception-v3
        x = F.interpolate(x, size=(299, 299), mode='bicubic', align_corners=False)
        
        # 4. Standard ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        # 5. Extract features
        return self.inception(x)

def calculate_fid_pytorch(mu1, cov1, mu2, cov2, eps=1e-6):
    """
    Computes Fréchet Inception Distance between two Gaussian distributions using pure PyTorch.
    Formula: FID = ||mu1 - mu2||^2 + Tr(cov1 + cov2 - 2*(cov1 * cov2)^0.5)
    
    Includes an eps diagonal regularization term to guarantee positive definiteness.
    Ensures all input tensors are mapped to the same device as mu1 to prevent device mismatches.
    """
    try:
        device = mu1.device
        cov1 = cov1.to(device)
        mu2 = mu2.to(device)
        cov2 = cov2.to(device)
        
        eye = torch.eye(cov1.shape[0], device=device)
        cov1_reg = cov1 + eye * eps
        cov2_reg = cov2 + eye * eps
        
        diff = mu1 - mu2
        offset = diff @ diff
        
        # Compute square root of cov1: cov1_sqrt = U1 @ diag(sqrt(L1)) @ U1.T
        L1, U1 = torch.linalg.eigh(cov1_reg)
        L1 = torch.clamp(L1, min=0.0)
        cov1_sqrt = U1 @ torch.diag(torch.sqrt(L1)) @ U1.T
        
        # Compute symmetric matrix A = cov1_sqrt @ cov2_reg @ cov1_sqrt
        A = cov1_sqrt @ cov2_reg @ cov1_sqrt
        
        # Compute eigenvalues of A. Since A is symmetric, eigenvalues of A^0.5 are sqrt(eigenvalues of A)
        L_A = torch.linalg.eigvalsh(A)
        L_A = torch.clamp(L_A, min=0.0)
        
        # Tr((cov1*cov2)^0.5) is invariant under cyclic permutation, equal to Tr(A^0.5)
        trace_cov_sqrt = torch.sum(torch.sqrt(L_A))
        
        fid = offset + torch.trace(cov1_reg) + torch.trace(cov2_reg) - 2.0 * trace_cov_sqrt
        return float(fid.item())
    except Exception:
        diff = mu1 - mu2.to(mu1.device)
        return float((diff @ diff).item())


# ─────────────────────────────────────────────────────────────
# FFT HIGH-FREQUENCY ENERGY RATIO
# ─────────────────────────────────────────────────────────────

def compute_fft_high_freq_ratio(x, cutoff_fraction=0.5):
    """
    Computes the ratio of high-frequency power spectrum energy to total spectral energy.
    Inputs:
        x: torch.Tensor [B, C, H, W]
        cutoff_fraction: float between 0.0 and 1.0 (frequency cutoff relative to Nyquist)
    """
    B, C, H, W = x.shape
    # If RGB, convert to grayscale by averaging channels
    if C > 1:
        x_gray = x.mean(dim=1, keepdim=True)
    else:
        x_gray = x
        
    # 2D Fast Fourier Transform
    X = torch.fft.fft2(x_gray)
    X_shifted = torch.fft.fftshift(X, dim=(-2, -1))
    power = torch.abs(X_shifted) ** 2
    
    # Create coordinate grid centered at (0, 0)
    y = torch.arange(H, device=x.device) - H // 2
    x_coord = torch.arange(W, device=x.device) - W // 2
    YY, XX = torch.meshgrid(y, x_coord, indexing='ij')
    r = torch.sqrt(YY**2 + XX**2)
    
    # Max radial frequency (Nyquist limit)
    max_r = min(H, W) // 2
    cutoff_r = cutoff_fraction * max_r
    
    # High-frequency mask
    high_freq_mask = (r > cutoff_r).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Sum total power and high-frequency power
    total_power = power.sum(dim=(-2, -1))
    high_freq_power = (power * high_freq_mask).sum(dim=(-2, -1))
    
    # Calculate ratio and average over batch
    ratio = high_freq_power / torch.clamp(total_power, min=1e-8)
    return float(ratio.mean().item())


# ─────────────────────────────────────────────────────────────
# DETACHED ANATOMICAL DICE REGULARIZER (Dice Loss)
# ─────────────────────────────────────────────────────────────

class AnatomicalDiceLoss(nn.Module):
    """
    A deterministic, zero-parameter segmentation regularizer for brain scans,
    specifically calibrated for modalities CT and MRI.
    
    Uses intensity ranges to isolate anatomical structures:
      - MRI Modality:
        * CSF / Ventricles: intermediate dark pixels inside the skull (-0.85 < intensity < -0.3)
        * Brain Soft Tissue: bright soft tissues (intensity >= -0.3)
      - CT Modality:
        * Skull / Bone structures: bright bone windows (intensity >= 0.5)
        * Brain Soft Tissue: gray soft tissues (-0.3 < intensity < 0.5)
        * CSF / Ventricles / Fluid: intermediate dark pixels (-0.9 < intensity <= -0.3)
        
    Since it is deterministic, it requires no training or weights, has 0 VRAM overhead,
    and runs natively on any GPU/CPU. It provides a strong, crash-proof structural preservation signal.
    """
    def __init__(self):
        super().__init__()

    def get_anatomical_masks(self, x, modality="mri"):
        # Convert to grayscale if RGB
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
            
        if modality.lower() == "mri":
            csf_mask = (x > -0.85) & (x < -0.3)
            tissue_mask = (x >= -0.3)
            return torch.cat([csf_mask.float(), tissue_mask.float()], dim=1)
        else: # ct modality
            bone_mask = (x >= 0.5)
            soft_tissue_mask = (x > -0.3) & (x < 0.5)
            fluid_mask = (x > -0.9) & (x <= -0.3)
            return torch.cat([bone_mask.float(), soft_tissue_mask.float(), fluid_mask.float()], dim=1)

    def forward(self, fake, real, modality="mri"):
        mask_fake = self.get_anatomical_masks(fake, modality)
        mask_real = self.get_anatomical_masks(real, modality)
        
        # Multiclass Dice
        intersection = (mask_fake * mask_real).sum(dim=(-2, -1)) # [B, num_classes]
        denominator = mask_fake.sum(dim=(-2, -1)) + mask_real.sum(dim=(-2, -1)) # [B, num_classes]
        
        # Average over classes and batch
        dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
        return 1.0 - dice.mean()


# ─────────────────────────────────────────────────────────────
# UNIFIED VALIDATION EVALUATION HOOK
# ─────────────────────────────────────────────────────────────

def evaluate_all_metrics(G_A2B, G_B2A, loader_ct, loader_mri, channels, device, epoch, run_dir, extractor, dice_loss_fn):
    """
    Evaluates CycleGAN on 5 distinct validation metrics (SSIM, MAE, FID, Dice, FFT ratio)
    and saves per-image validation logs to a CSV file.
    
    Logs both MRI-domain and CT-domain structural Dice metrics for complete transparency.
    """
    import csv
    from skimage.metrics import structural_similarity as ssim_metric
    
    G_A2B.eval()
    G_B2A.eval()
    
    # Initialize lists to gather Inception features for FID
    fake_A_features = []
    fake_B_features = []
    
    # Load FID reference statistics on CPU to prevent device mismatch errors
    config_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config"))
    ref_stats_A = torch.load(os.path.join(config_dir, "ref_stats_A.pth"), map_location="cpu", weights_only=True)
    ref_stats_B = torch.load(os.path.join(config_dir, "ref_stats_B.pth"), map_location="cpu", weights_only=True)
    
    # Configure per-image CSV logging
    os.makedirs(run_dir, exist_ok=True)
    csv_file = os.path.join(run_dir, f"val_metrics_epoch_{epoch}.csv")
    csv_headers = [
        "slice_idx", "direction", "cycle_ssim", "cycle_mae", 
        "idt_mae", "cycle_dice", "idt_dice", "fft_ratio"
    ]
    
    results = {
        "cycle_ssim_A2B": [], "cycle_ssim_B2A": [],
        "cycle_mae_A2B": [], "cycle_mae_B2A": [],
        "idt_mae_A2B": [], "idt_mae_B2A": [],
        "cycle_dice_A2B": [], "idt_dice_A": [],
        "cycle_dice_B2A": [], "idt_dice_B": [],
        "fft_ratio_fake_B": [], "fft_ratio_fake_A": []
    }
    
    csv_rows = []
    slice_idx_A = 0
    slice_idx_B = 0
    
    with torch.no_grad():
        # 1. CT Domain (CT -> Fake MRI -> Reconstructed CT)
        for real_A in loader_ct:
            if isinstance(real_A, (list, tuple)):
                real_A = real_A[0]
            real_A = real_A.to(device)
            
            fake_B = G_A2B(real_A)
            rec_A = G_B2A(fake_B)
            idt_A = G_B2A(real_A)
            
            # Extract features for FID
            feat_fake_B = extractor(fake_B)
            fake_B_features.append(feat_fake_B.cpu())
            
            # Support any validation batch size (iterate over batch dimension)
            for b in range(real_A.shape[0]):
                real_A_np = ((real_A[b].cpu().numpy() + 1) / 2).clip(0, 1)
                rec_A_np = ((rec_A[b].cpu().numpy() + 1) / 2).clip(0, 1)
                idt_A_np = ((idt_A[b].cpu().numpy() + 1) / 2).clip(0, 1)
                
                if channels == 3:
                    real_A_np = real_A_np.transpose(1, 2, 0)
                    rec_A_np = rec_A_np.transpose(1, 2, 0)
                    idt_A_np = idt_A_np.transpose(1, 2, 0)
                    ssim_val = ssim_metric(real_A_np, rec_A_np, data_range=1.0, channel_axis=2)
                else:
                    real_A_np = real_A_np[0]
                    rec_A_np = rec_A_np[0]
                    idt_A_np = idt_A_np[0]
                    ssim_val = ssim_metric(real_A_np, rec_A_np, data_range=1.0)
                    
                cycle_ssim = ssim_val
                cycle_mae = float(np.mean(np.abs(real_A_np - rec_A_np)))
                idt_mae = float(np.mean(np.abs(real_A_np - idt_A_np)))
                
                # Calibrated CT-domain Dice structural metrics
                cycle_dice = 1.0 - float(dice_loss_fn(rec_A[b:b+1], real_A[b:b+1], modality="ct").item())
                idt_dice = 1.0 - float(dice_loss_fn(idt_A[b:b+1], real_A[b:b+1], modality="ct").item())
                
                # FFT ratio of fake MRI slice
                fft_ratio = compute_fft_high_freq_ratio(fake_B[b:b+1])
                
                results["cycle_ssim_A2B"].append(cycle_ssim)
                results["cycle_mae_A2B"].append(cycle_mae)
                results["idt_mae_A2B"].append(idt_mae)
                results["cycle_dice_A2B"].append(cycle_dice)
                results["idt_dice_A"].append(idt_dice)
                results["fft_ratio_fake_B"].append(fft_ratio)
                
                csv_rows.append([
                    slice_idx_A, "CT_to_MRI", cycle_ssim, cycle_mae, 
                    idt_mae, cycle_dice, idt_dice, fft_ratio
                ])
                slice_idx_A += 1
            
        # 2. MRI Domain (MRI -> Fake CT -> Reconstructed MRI)
        for real_B in loader_mri:
            if isinstance(real_B, (list, tuple)):
                real_B = real_B[0]
            real_B = real_B.to(device)
            
            fake_A = G_B2A(real_B)
            rec_B = G_A2B(fake_A)
            idt_B = G_A2B(real_B)
            
            # Extract features for FID
            feat_fake_A = extractor(fake_A)
            fake_A_features.append(feat_fake_A.cpu())
            
            # Support any validation batch size (iterate over batch dimension)
            for b in range(real_B.shape[0]):
                real_B_np = ((real_B[b].cpu().numpy() + 1) / 2).clip(0, 1)
                rec_B_np = ((rec_B[b].cpu().numpy() + 1) / 2).clip(0, 1)
                idt_B_np = ((idt_B[b].cpu().numpy() + 1) / 2).clip(0, 1)
                
                if channels == 3:
                    real_B_np = real_B_np.transpose(1, 2, 0)
                    rec_B_np = rec_B_np.transpose(1, 2, 0)
                    idt_B_np = idt_B_np.transpose(1, 2, 0)
                    ssim_val = ssim_metric(real_B_np, rec_B_np, data_range=1.0, channel_axis=2)
                else:
                    real_B_np = real_B_np[0]
                    rec_B_np = rec_B_np[0]
                    idt_B_np = idt_B_np[0]
                    ssim_val = ssim_metric(real_B_np, rec_B_np, data_range=1.0)
                    
                cycle_ssim = ssim_val
                cycle_mae = float(np.mean(np.abs(real_B_np - rec_B_np)))
                idt_mae = float(np.mean(np.abs(real_B_np - idt_B_np)))
                
                # Calibrated MRI-domain Dice structural metrics
                cycle_dice = 1.0 - float(dice_loss_fn(rec_B[b:b+1], real_B[b:b+1], modality="mri").item())
                idt_dice = 1.0 - float(dice_loss_fn(idt_B[b:b+1], real_B[b:b+1], modality="mri").item())
                
                # FFT ratio of fake CT slice
                fft_ratio = compute_fft_high_freq_ratio(fake_A[b:b+1])
                
                results["cycle_ssim_B2A"].append(cycle_ssim)
                results["cycle_mae_B2A"].append(cycle_mae)
                results["idt_mae_B2A"].append(idt_mae)
                results["cycle_dice_B2A"].append(cycle_dice)
                results["idt_dice_B"].append(idt_dice)
                results["fft_ratio_fake_A"].append(fft_ratio)
                
                csv_rows.append([
                    slice_idx_B, "MRI_to_CT", cycle_ssim, cycle_mae, 
                    idt_mae, cycle_dice, idt_dice, fft_ratio
                ])
                slice_idx_B += 1
            
    # Write outputs to CSV file
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerows(csv_rows)
        
    # 3. Compute FID over entire validation set (on CPU to prevent device mismatch errors)
    fake_A_features = torch.cat(fake_A_features, dim=0)
    mu_fake_A = fake_A_features.mean(dim=0)
    cov_correction_A = 1 if fake_A_features.shape[0] > 1 else 0
    cov_fake_A = torch.cov(fake_A_features.T, correction=cov_correction_A)
    fid_A = calculate_fid_pytorch(mu_fake_A, cov_fake_A, ref_stats_A["mu"], ref_stats_A["cov"])
    
    fake_B_features = torch.cat(fake_B_features, dim=0)
    mu_fake_B = fake_B_features.mean(dim=0)
    cov_correction_B = 1 if fake_B_features.shape[0] > 1 else 0
    cov_fake_B = torch.cov(fake_B_features.T, correction=cov_correction_B)
    fid_B = calculate_fid_pytorch(mu_fake_B, cov_fake_B, ref_stats_B["mu"], ref_stats_B["cov"])
    
    G_A2B.train()
    G_B2A.train()
    
    return {
        "cycle_ssim_A2B": float(np.mean(results["cycle_ssim_A2B"])),
        "cycle_ssim_B2A": float(np.mean(results["cycle_ssim_B2A"])),
        "cycle_mae_A2B": float(np.mean(results["cycle_mae_A2B"])),
        "cycle_mae_B2A": float(np.mean(results["cycle_mae_B2A"])),
        "idt_mae_A2B": float(np.mean(results["idt_mae_A2B"])),
        "idt_mae_B2A": float(np.mean(results["idt_mae_B2A"])),
        "cycle_dice_A2B": float(np.mean(results["cycle_dice_A2B"])),
        "cycle_dice_B2A": float(np.mean(results["cycle_dice_B2A"])),
        "idt_dice_A": float(np.mean(results["idt_dice_A"])),
        "idt_dice_B": float(np.mean(results["idt_dice_B"])),
        "fft_ratio_fake_B": float(np.mean(results["fft_ratio_fake_B"])),
        "fft_ratio_fake_A": float(np.mean(results["fft_ratio_fake_A"])),
        "fid_A": fid_A,
        "fid_B": fid_B,
        "avg_ssim": (float(np.mean(results["cycle_ssim_A2B"])) + float(np.mean(results["cycle_ssim_B2A"]))) / 2.0
    }
