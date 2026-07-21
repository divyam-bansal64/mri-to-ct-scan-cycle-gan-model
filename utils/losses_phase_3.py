import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# NORMALIZED FFT DISTORTION LOSS (High-Pass Power Ratio)
# ─────────────────────────────────────────────────────────────

def compute_fft_loss_v2(fake, real, cutoff_fraction=0.5):
    """
    Computes L1 distance between normalized high-frequency power ratios
    of fake and real images. Bounded in [0, 0.05], avoiding G-loss inflation.
    One-sided: only penalizes excess high-frequency artifacts in fake.
    """
    # If RGB, convert to grayscale
    if fake.shape[1] > 1:
        fake_gray = fake.mean(dim=1, keepdim=True)
        real_gray = real.mean(dim=1, keepdim=True)
    else:
        fake_gray = fake
        real_gray = real
        
    H, W = fake_gray.shape[-2:]
    
    # 2D Fast Fourier Transform
    X_fake = torch.fft.fft2(fake_gray)
    X_fake_shifted = torch.fft.fftshift(X_fake, dim=(-2, -1))
    
    X_real = torch.fft.fft2(real_gray)
    X_real_shifted = torch.fft.fftshift(X_real, dim=(-2, -1))
    
    power_fake = torch.abs(X_fake_shifted) ** 2
    power_real = torch.abs(X_real_shifted) ** 2
    
    # Create coordinate grid centered at (0, 0)
    y = torch.arange(H, device=fake.device) - H // 2
    x_coord = torch.arange(W, device=fake.device) - W // 2
    YY, XX = torch.meshgrid(y, x_coord, indexing='ij')
    r = torch.sqrt(YY**2 + XX**2)
    
    # Compute high-frequency mask
    max_r = min(H, W) // 2
    cutoff_r = cutoff_fraction * max_r
    high_freq_mask = (r > cutoff_r).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Calculate ratio of high frequency power to total power
    total_fake = power_fake.sum(dim=(-2, -1))
    high_fake = (power_fake * high_freq_mask).sum(dim=(-2, -1))
    fake_ratio = high_fake / torch.clamp(total_fake, min=1e-8)
    
    total_real = power_real.sum(dim=(-2, -1))
    high_real = (power_real * high_freq_mask).sum(dim=(-2, -1))
    real_ratio = high_real / torch.clamp(total_real, min=1e-8)
    
    # One-sided penalty: only penalize if fake has MORE high frequency than real
    excess = F.relu(fake_ratio - real_ratio)
    return excess.mean()


# ─────────────────────────────────────────────────────────────
# DEEP VGG SEMANTIC PERCEPTUAL LOSS
# ─────────────────────────────────────────────────────────────

class VGGPerceptualLossV2(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        except ImportError:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)
            
        # Use deep semantic layers: relu4_2 (index 23) and relu3_2 (index 16)
        # Prevents high-frequency texture-matching artifacts caused by shallow layers
        self.features_deep = nn.Sequential(*list(vgg.features.children())[:23])  # relu4_2
        self.features_mid = nn.Sequential(*list(vgg.features.children())[:16])   # relu3_2
        
        for model in [self.features_deep, self.features_mid]:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            
    def forward(self, fake, real):
        # Convert grayscale to 3-channel
        if fake.shape[1] == 1:
            fake = fake.repeat(1, 3, 1, 1)
        if real.shape[1] == 1:
            real = real.repeat(1, 3, 1, 1)
            
        # Normalize range [-1, 1] to ImageNet expectations
        mean = torch.tensor([0.485, 0.456, 0.406], device=fake.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=fake.device).view(1, 3, 1, 1)
        
        fake_norm = ((fake + 1.0) / 2.0 - mean) / std
        real_norm = ((real + 1.0) / 2.0 - mean) / std
        
        # Calculate loss across scale hierarchy (weight deep semantic structure more)
        feat_fake_deep = self.features_deep(fake_norm)
        feat_real_deep = self.features_deep(real_norm)
        feat_fake_mid = self.features_mid(fake_norm)
        feat_real_mid = self.features_mid(real_norm)
        
        loss_deep = F.l1_loss(feat_fake_deep, feat_real_deep)
        loss_mid = F.l1_loss(feat_fake_mid, feat_real_mid)
        
        return 0.7 * loss_deep + 0.3 * loss_mid
