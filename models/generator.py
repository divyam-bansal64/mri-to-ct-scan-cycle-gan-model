import torch
import torch.nn as nn


# ======================================================
# 🧱 Residual Block (Core of ResNet Generator)
# ======================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)  # Skip connection


# ======================================================
# 🌀 ResNet Generator (for CT↔MRI)
# ======================================================
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=6):
        """
        input_nc: number of input channels (3 for RGB)
        output_nc: number of output channels (3 for RGB)
        ngf: base number of filters (64 standard)
        n_blocks: number of residual blocks (6 for 256x256)
        """
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()

        # Initial Convolution Layer (7x7 kernel)
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        # Downsampling layers (reduce spatial size, increase depth)
        n_downsampling = 2
        in_channels = ngf
        for i in range(n_downsampling):
            out_channels = in_channels * 2
            model += [
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels

        # Residual blocks
        for i in range(n_blocks):
            model += [ResidualBlock(in_channels)]

        # Upsampling layers (increase spatial size, reduce depth)
        for i in range(n_downsampling):
            out_channels = in_channels // 2
            model += [
                nn.ConvTranspose2d(in_channels, out_channels,
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ======================================================
# 🔄 Define both Generators (A2B and B2A)
# ======================================================
def define_generators(input_nc=3, output_nc=3, ngf=64, n_blocks=6, device='cuda'):
    """
    Creates and returns two generators:
      - G_A2B: CT → MRI
      - G_B2A: MRI → CT
    """
    G_A2B = ResnetGenerator(input_nc, output_nc, ngf, n_blocks).to(device)
    G_B2A = ResnetGenerator(output_nc, input_nc, ngf, n_blocks).to(device)
    return G_A2B, G_B2A


# ======================================================
# 🧪 Test Run
# ======================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    G_A2B, G_B2A = define_generators(device=device)

    # Simulated dataset samples
    real_CT = torch.randn(1, 3, 256, 256).to(device)   # domain A
    real_MRI = torch.randn(1, 3, 256, 256).to(device)  # domain B

    fake_MRI = G_A2B(real_CT)
    fake_CT = G_B2A(real_MRI)

    print("CT → MRI output:", fake_MRI.shape)
    print("MRI → CT output:", fake_CT.shape)
