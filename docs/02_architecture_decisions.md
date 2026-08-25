# Architecture Decisions

Every architectural choice here was tested experimentally. This document covers what was chosen,
what was tried and rejected, and why.

---

## Generator: ResNet-9

The generator follows the standard CycleGAN architecture — an encoder-bottleneck-decoder with
residual blocks in the middle:

```
Input (1×256×256)
    ↓
ReflectionPad(3) → Conv2d(1→64, 7×7) → InstanceNorm → ReLU
    ↓
Conv2d(3×3, stride=2) × 2    [downsampling: 256→128→64 spatial, 64→128→256 channels]
    ↓
ResidualBlock × 9            [64×64 spatial, 256 channels]
    ↓
ConvTranspose2d × 2          [upsampling: 64→128→256 spatial, 256→128→64 channels]
    ↓
ReflectionPad(3) → Conv2d(64→1, 7×7) → Tanh
    ↓
Output (1×256×256), range [-1, 1]
```

**Why 9 residual blocks:** The CycleGAN paper recommends 9 blocks for 256×256 input. Phase 1 confirmed
this — 4 blocks (Run 8) dropped SSIM from 0.894 to 0.869. The bottleneck at 64×64 with 256 channels
gives each residual block enough receptive field to capture anatomical structures like ventricles
and sulci without losing spatial precision.

**Why InstanceNorm instead of BatchNorm:** With batch size 1 (standard for CycleGAN), BatchNorm
computes statistics from a single sample, which is effectively useless and noisy. InstanceNorm
normalizes per-image, per-channel and is invariant to batch size. CT and MRI also have completely
different global intensity distributions — BatchNorm's running statistics would bleed between domains.

**Why Tanh output:** Output range [-1, 1] matches the input normalization (Normalize(0.5, 0.5)
maps [0,1] → [-1,1]). Tanh keeps output bounded, which prevents gradient explosion and is standard
practice in image-to-image GANs.

---

## Upsampling: ConvTranspose2d vs ResizeConv (Bilinear)

This was tested directly. Results from Phase 2:

| Method | SSIM | Notes |
| :--- | :---: | :--- |
| ConvTranspose2d | 0.894 | Learned upsampling, produces sharp edges |
| ResizeConv (bilinear + Conv2d) | 0.842 | 5.8% SSIM drop from this change alone |

Bilinear interpolation is a spatial low-pass filter — it averages neighboring pixels before passing
to the convolution. The convolution cannot recover high-frequency detail that was already destroyed
by averaging. The result is inherently blurrier output regardless of training duration.

ConvTranspose2d (fractional-strided convolution) learns the upsampling kernel, so the network can
produce sharp edges. The checkerboard artifact risk is mitigated by using kernel_size=3 (odd) with
stride=2 and output_padding=1.

---

## Discriminator: PatchGAN

A PatchGAN discriminator outputs a spatial map (30×30 for 256×256 input) rather than a single
real/fake scalar. Each cell in the output corresponds to a 70×70 receptive field in the input.

```
Input (1×256×256)
    ↓
Conv2d(1→64, 4×4, stride=2) → LeakyReLU(0.2)
Conv2d(64→128, 4×4, stride=2) → InstanceNorm → LeakyReLU(0.2)
Conv2d(128→256, 4×4, stride=2) → InstanceNorm → LeakyReLU(0.2)
Conv2d(256→512, 4×4, stride=1) → InstanceNorm → LeakyReLU(0.2)
Conv2d(512→1, 4×4, stride=1)
    ↓
Output: (1×30×30) — one real/fake score per 70×70 patch
```

This is better than a global discriminator for medical images because brain anatomy has repetitive
local structure (cortical folds, bone boundaries). Patch-level discrimination forces the generator
to produce realistic local textures, not just a globally plausible image.

**Why MSE loss (LSGAN) instead of BCE:** When the discriminator is confident, BCE gradients vanish
(log(1) → 0). MSE provides non-zero gradients even for well-classified samples, which keeps the
generator receiving useful feedback throughout training.

---

## Training Design

**Replay buffer (size 50):** The discriminator trains on a mix of current and historical fake images
(50% probability each). Without this, the discriminator overfits to the generator's latest output
distribution and oscillates as the generator changes.

**Learning rate schedule:** Constant at 0.0002 for the first 100 epochs, then linear decay to 0
over the next 100. This gives the model time to learn the general mapping before fine-tuning.

**Equal learning rates — no TTUR:** TTUR (Two Time-scale Update Rule) sets the discriminator LR
4x higher than the generator. Tested in Phase 2, it failed every time:

| Run | Config | What happened |
| :--- | :--- | :--- |
| Run 3 | TTUR + R1 | Violent discriminator loss oscillation |
| Run 7 | TTUR + identity decay | Generator loss reversed direction after epoch 33 |
| Run 8 | TTUR + R1 + FFT | Combined instability from both |

TTUR was designed for StyleGAN, which doesn't have cycle or identity losses. In CycleGAN, the
generator already has a harder optimization problem (three separate loss components vs one for the
discriminator). Giving the discriminator extra speed makes the imbalance worse.

**Grayscale (1 channel):** CT and MRI are inherently grayscale. Using 3-channel RGB triples memory
for zero information gain and requires the network to learn that R=G=B. Phase 1 confirmed grayscale
matches or beats 3-channel quality.

---

## Augmentation

```python
transforms.RandomHorizontalFlip(p=0.5)                  # brain is bilaterally symmetric
transforms.RandomRotation(degrees=5)                      # heads aren't always perfectly aligned
transforms.RandomAffine(degrees=0, scale=(0.9, 1.1))     # different patients, different head sizes
```

Not used and why:
- **Vertical flip** — brain anatomy is not vertically symmetric (skull base ≠ vertex)
- **Color jitter** — meaningless on grayscale images
- **Elastic deformation** — brain has a rigid skull; elastic transforms produce anatomically
  impossible geometry

Validation transforms never include random augmentation. Applying randomness to validation would
make metrics non-reproducible across runs.
