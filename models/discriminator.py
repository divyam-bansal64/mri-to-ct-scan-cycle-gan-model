import torch
import torch.nn as nn
from typing import Tuple

# ============================================
# PatchGAN Discriminator (N-layer)
# Reference: CycleGAN / Pix2Pix style
# ============================================
class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator that outputs a patch map (not a single scalar).
    - input_nc: input channels (3 for RGB)
    - ndf: base number of filters (64 typical)
    - n_layers: number of downsampling conv layers (4 gives ~30x30 output for 256 input)
    - use_spect: whether to apply spectral normalization to conv layers (optional)
    """
    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 4, use_spect: bool = False):
        super(NLayerDiscriminator, self).__init__()
        norm_layer = nn.InstanceNorm2d

        # helper to optionally apply spectral normalization
        def conv(ni, no, ks=4, stride=2, pad=1, use_spect=use_spect):
            c = nn.Conv2d(ni, no, kernel_size=ks, stride=stride, padding=pad, bias=True)
            return nn.utils.spectral_norm(c) if use_spect else c

        sequence = []
        # First layer: no normalization, LeakyReLU
        sequence += [
            conv(input_nc, ndf, ks=4, stride=2, pad=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        # Middle layers
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                conv(ndf * nf_mult_prev, ndf * nf_mult, ks=4, stride=2, pad=1),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        # Penultimate layer: stride 1 to keep receptive field appropriate
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            conv(ndf * nf_mult_prev, ndf * nf_mult, ks=4, stride=1, pad=1),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Final conv: single-channel output (patch map)
        sequence += [
            conv(ndf * nf_mult, 1, ks=4, stride=1, pad=1)
            # No activation here if you use BCEWithLogitsLoss
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)  # shape: (N, 1, H_patch, W_patch)


# ============================================
# Factory to create both discriminators D_A and D_B
# ============================================
def define_discriminators(input_nc: int = 3, ndf: int = 64, n_layers: int = 4,
                          use_spect: bool = False, device: str = "cuda") -> Tuple[nn.Module, nn.Module]:
    """
    Returns:
      D_A: discriminator for domain A (CT)
      D_B: discriminator for domain B (MRI)
    """
    D_A = NLayerDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=n_layers, use_spect=use_spect).to(device)
    D_B = NLayerDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=n_layers, use_spect=use_spect).to(device)
    return D_A, D_B


# ============================================
# Quick test
# ============================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D_A, D_B = define_discriminators(device=device)

    # sanity inputs: batch size 1, 3 channels, 256x256
    real_ct = torch.randn(1, 3, 256, 256).to(device)
    real_mri = torch.randn(1, 3, 256, 256).to(device)

    out_A = D_A(real_ct)
    out_B = D_B(real_mri)

    print("D_A output shape:", out_A.shape)  # e.g., (1,1,30,30) depending on n_layers
    print("D_B output shape:", out_B.shape)
