import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# FFT DISTORTION LOSS (High-Pass Filtered)
# ─────────────────────────────────────────────────────────────

def compute_fft_loss(fake, real, cutoff_fraction=0.5):
    """
    Computes L1 distance between the 2D Fast Fourier Transform magnitude spectra
    specifically targeting high-frequency bands (cutoff_fraction to Nyquist).
    Aligns training directly with the high-frequency validation metric.
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
    
    # Create coordinate grid centered at (0, 0)
    y = torch.arange(H, device=fake.device) - H // 2
    x_coord = torch.arange(W, device=fake.device) - W // 2
    YY, XX = torch.meshgrid(y, x_coord, indexing='ij')
    r = torch.sqrt(YY**2 + XX**2)
    
    # Compute high-frequency mask matching metrics.py
    max_r = min(H, W) // 2
    cutoff_r = cutoff_fraction * max_r
    high_freq_mask = (r > cutoff_r).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Minimize L1 distance ONLY on the high-frequency components
    fake_high = torch.abs(X_fake_shifted) * high_freq_mask
    real_high = torch.abs(X_real_shifted) * high_freq_mask
    
    return F.l1_loss(fake_high, real_high)


# ─────────────────────────────────────────────────────────────
# VGG PERCEPTUAL LOSS
# ─────────────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        except ImportError:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)
            
        # Extract features up to relu2_2 (index 9 in vgg.features)
        self.features = nn.Sequential(*list(vgg.features.children())[:9])
        self.features.eval()
        for p in self.features.parameters():
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
        
        # Calculate L1 loss between feature maps
        feat_fake = self.features(fake_norm)
        feat_real = self.features(real_norm)
        return F.l1_loss(feat_fake, feat_real)
