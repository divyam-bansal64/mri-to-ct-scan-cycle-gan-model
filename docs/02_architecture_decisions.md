# Architecture Decisions: Generator, Discriminator & Training Design

> Design rationale for every architectural choice made during the project, including what worked, what failed, and why.

---

## 1. Generator Architecture: ResNet-9

### What We Used

A **ResNet-based generator** with 9 residual blocks, following the CycleGAN paper (Zhu et al., 2017). The architecture follows an **encoder-transformer-decoder** pattern:

```
Input (1×256×256)
    ↓
[Encoder] ReflectionPad(3) → Conv2d(1→64, 7×7) → InstanceNorm → ReLU
    ↓
[Downsample ×2] Conv2d(3×3, stride=2) → InstanceNorm → ReLU
    64 → 128 → 256 channels, spatial: 256 → 128 → 64
    ↓
[Residual Blocks ×9] Conv2d(3×3) → InstanceNorm → ReLU → Conv2d(3×3) → InstanceNorm + Skip
    256 channels, spatial: 64×64
    ↓
[Upsample ×2] ConvTranspose2d(3×3, stride=2) → InstanceNorm → ReLU
    256 → 128 → 64 channels, spatial: 64 → 128 → 256
    ↓
[Output] ReflectionPad(3) → Conv2d(64→1, 7×7) → Tanh
    ↓
Output (1×256×256), range [-1, 1]
```

### Why 9 Residual Blocks

The CycleGAN paper recommends 9 blocks for 256×256 images and 6 blocks for 128×128. Our Phase 1 experiments confirmed that 9 blocks outperformed reduced configurations:

- **9 blocks**: SSIM = 0.894 (Phase 2, Run 0)
- **4 blocks** (Run 8, Phase 2): SSIM = 0.869 — insufficient capacity to learn the mapping

The bottleneck operates at 64×64 spatial resolution with 256 channels, giving each residual block a receptive field large enough to capture anatomical structures (ventricles, sulci) while maintaining spatial precision.

### Why InstanceNorm (Not BatchNorm)

**InstanceNorm2d** normalizes each image independently (per-channel, per-sample). **BatchNorm2d** normalizes across the batch.

In unpaired translation, each batch contains a **single image** (batch_size=1 is standard for CycleGAN). BatchNorm with batch_size=1 computes statistics from a single sample, producing noisy normalization that destabilizes training. InstanceNorm is invariant to batch size.

Additionally, CT and MRI have fundamentally different intensity distributions. BatchNorm's running statistics would blend these distributions if both domains were processed by shared layers.

### Why ConvTranspose2d (Not ResizeConv)

We tested two upsampling strategies:

| Method | Mechanism | SSIM (Phase 2) | Issue |
| :--- | :--- | :---: | :--- |
| **ConvTranspose2d** | Learned fractional-strided convolution | **0.894** | Minor checkerboard risk |
| **ResizeConv** (bilinear + Conv2d) | Bilinear interpolation → convolution | 0.842 | **Low-pass filter** — blurs output |

Bilinear interpolation averages neighboring pixels, acting as a spatial low-pass filter. The subsequent convolution cannot recover the high-frequency detail that was already destroyed. This produced a **5.8% SSIM drop** from a single architectural change.

**ConvTranspose2d** learns the upsampling kernel, allowing it to produce sharp edges. The theoretical checkerboard artifact risk is mitigated by using kernel_size=3 (odd) with stride=2 and output_padding=1.

### Why Tanh Output Activation

The generator outputs values in [-1, 1] via `Tanh`. This matches the input normalization `Normalize((0.5,), (0.5,))` which maps [0,1] → [-1,1]. Using Tanh (vs. no activation or Sigmoid) ensures:
- Output is bounded — prevents gradient explosion from unbounded predictions
- Symmetric range matches the L1 loss landscape
- Standard practice in image-to-image GANs

---

## 2. Discriminator Architecture: PatchGAN

### What We Used

A **PatchGAN discriminator** (Isola et al., 2017) that outputs a spatial map of real/fake predictions rather than a single scalar:

```
Input (1×256×256)
    ↓
Conv2d(1→64, 4×4, stride=2) → LeakyReLU(0.2)          # No normalization
    ↓
Conv2d(64→128, 4×4, stride=2) → InstanceNorm → LeakyReLU(0.2)
    ↓
Conv2d(128→256, 4×4, stride=2) → InstanceNorm → LeakyReLU(0.2)
    ↓
Conv2d(256→512, 4×4, stride=1) → InstanceNorm → LeakyReLU(0.2)
    ↓
Conv2d(512→1, 4×4, stride=1)    # No activation (used with MSE loss)
    ↓
Output: (1×30×30) patch map
```

### Why PatchGAN (Not Global Discriminator)

Each cell in the 30×30 output map corresponds to a **70×70 receptive field** in the input image. The discriminator classifies whether each 70×70 patch looks real or fake, rather than making a single judgment about the entire image.

Benefits:
- **Captures local texture**: Medical images have repetitive local structure (brain folds, bone boundaries). Patch-level discrimination forces the generator to produce realistic local textures.
- **Fewer parameters**: A 30×30 PatchGAN has far fewer parameters than a full-image discriminator, reducing overfitting risk.
- **Spatial gradients**: Each spatial location provides an independent gradient signal to the generator, producing richer training feedback than a single scalar.

### Why MSE Loss (Not BCE)

We used **MSE (Least Squares GAN)** loss instead of the original GAN's binary cross-entropy:

```python
def gan_loss(pred, target_is_real):
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)
```

LSGAN (Mao et al., 2017) produces more stable training than BCE because:
- BCE saturates when the discriminator is confident (gradient → 0)
- MSE provides non-zero gradients even for well-classified samples
- Shown to produce higher quality images in practice

---

## 3. Training Design Decisions

### Replay Buffer (Size 50)

A **replay buffer** stores the 50 most recently generated fake images. When training the discriminator, instead of always using the current generator's output, we sample from this buffer with 50% probability.

```python
class ReplayBuffer:
    def push_and_pop(self, data):
        if random.uniform(0, 1) > 0.5:
            # Use historical fake instead of current fake
            return self.data[random.choice]
```

This prevents the discriminator from overfitting to the generator's current output distribution and reduces training oscillation.

### Learning Rate Schedule

**Linear decay** starting at the midpoint of training:

```python
# Constant LR for first 100 epochs, then linearly decay to 0 over next 100
lr_G = lr_D = 0.0002
decay_epoch = 100  # Start decay at epoch 100 (out of 200)
```

This gives the model 100 epochs to learn the general mapping, then gradually fine-tunes by reducing the step size. The linear (not cosine) schedule was chosen for simplicity and is standard in CycleGAN.

### Equal Learning Rates (No TTUR)

**TTUR** (Two Time-scale Update Rule) uses a faster learning rate for the discriminator (e.g., lr_D = 4×lr_G). We tested this in Phase 2 and it **failed in 3/3 runs**:

| Run | TTUR Config | Outcome |
| :--- | :--- | :--- |
| Run 3 | lr_G=0.0001, lr_D=0.0004 + R1 | Violent D-loss oscillation |
| Run 7 | lr_G=0.0001, lr_D=0.0004 + identity decay | Loss reversal after epoch 33 |
| Run 8 | lr_G=0.0001, lr_D=0.0004 + R1 + FFT | Combined instability |

TTUR works in StyleGAN (single-image generation, no cycle loss). In CycleGAN, the generator already has a harder optimization landscape (adversarial + cycle + identity losses). Giving the discriminator extra speed advantage makes the imbalance worse.

**Lesson**: Equal learning rates (lr_G = lr_D = 0.0002) for CycleGAN. Always.

### Grayscale (1-Channel, Not RGB)

Both CT and MRI brain images are inherently grayscale modalities. Using 3-channel RGB:
- Triples memory usage for no information gain
- Introduces cross-channel gradient interactions that can cause color artifacts
- Requires the model to learn that R=G=B (wasted capacity)

Phase 1 confirmed that 1-channel grayscale with appropriate augmentation matches or exceeds 3-channel training quality.

---

## 4. Augmentation Strategy

```python
transforms.RandomHorizontalFlip(p=0.5)           # Brain symmetry
transforms.RandomRotation(degrees=5)               # Slight head tilt
transforms.RandomAffine(degrees=0, scale=(0.9, 1.1))  # Minor zoom variation
```

### Why These Specific Augmentations

- **Horizontal flip**: Brain anatomy is approximately bilaterally symmetric. Flipping doubles effective dataset size.
- **±5° rotation**: Patients' heads are rarely perfectly aligned in the scanner. Small rotations simulate realistic positioning variation.
- **0.9–1.1× scale**: Different patients have different head sizes. Minor zoom simulates this variation without distorting anatomy.
- **No vertical flip**: Brain anatomy is not vertically symmetric (skull base ≠ vertex).
- **No color jitter**: Grayscale images — color augmentation is meaningless.
- **No elastic deformation**: Brain anatomy has rigid bone boundaries — elastic transforms would produce anatomically impossible images.

### Critical Implementation Detail: Separate Transforms for Train/Val

```python
transform_train = Compose([Resize, Flip, Rotate, Scale, ToTensor, Normalize])
transform_val   = Compose([Resize, ToTensor, Normalize])  # No augmentation
```

Validation images must use deterministic transforms to produce reproducible metrics. Applying random augmentation to validation data would make metrics noisy and unreliable.
