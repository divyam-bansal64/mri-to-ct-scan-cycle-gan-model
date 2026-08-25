# Errors to Remember — CycleGAN 9-Run Loss Function Analysis

> This document catalogs every loss function implementation error discovered during the 9-run hyperparameter search (July 2026). Keep as a reference for future GAN training.

---

## Error 1: FFT Loss Not Normalized

**Runs affected**: 4, 8  
**Severity**: 🔴 Critical — made both runs unusable

### What we did

```python
# losses.py — compute_fft_loss()
fake_high = torch.abs(X_fake_shifted) * high_freq_mask   # Raw FFT magnitudes
real_high = torch.abs(X_real_shifted) * high_freq_mask
return F.l1_loss(fake_high, real_high)                    # L1 on raw values
```

Raw 2D FFT magnitudes on 256×256 images are in the range of **thousands to tens of thousands**. The L1 distance between fake and real spectra was outputting values around **5-12 per image**.

### What our eval metric did

```python
# metrics.py — compute_fft_high_freq_ratio()
power = torch.abs(X_shifted) ** 2
ratio = high_freq_power / total_power    # Normalized to [0, 1], outputs ~0.003
```

### The mismatch

| | Training Loss | Eval Metric |
|---|---|---|
| **Magnitude** | Raw FFT amplitudes (O(1000s)) | Power ratio (O(0.001)) |
| **Output range** | ~5-12 per call | ~0.001-0.05 |
| **What it measures** | "Match the real image's exact frequency fingerprint" | "Does the fake have too much high-freq energy?" |

### What happened

- `λ_fft = 1.0` added **~8-16** to Loss_G
- All other losses combined = **~2-3**
- FFT dominated 80% of the gradient signal
- Generator spent its capacity matching Fourier spectra instead of making good images
- Run 4: D/G ratio crashed to 0.012 (discriminator collapse)
- Run 8: Loss_G stayed at 9-18 for all 50 epochs

### The fix

```python
def compute_fft_loss_v2(fake, real, cutoff_fraction=0.5):
    # Compute power ratio (same as eval metric)
    fake_ratio = high_freq_power_fake / total_power_fake   # [0, 1]
    real_ratio = high_freq_power_real / total_power_real   # [0, 1]
    
    # One-sided: only penalize EXCESS high-freq (artifacts)
    excess = F.relu(fake_ratio - real_ratio)
    return excess.mean()   # Output: ~0.001-0.01
```

### Evidence

| Run | Epoch 1 Loss_G | Standard components (est.) | FFT contribution (est.) |
|-----|:-:|:-:|:-:|
| Run 0 (no FFT) | 2.79 | 2.79 | 0.00 |
| Run 4 (FFT) | **15.59** | ~2.8 | **~12.8** |
| Run 8 (FFT + R1) | **18.88** | ~2.8 | **~16.1** |

---

## Error 2: R1 Penalty Scaled 32× Too Strong

**Runs affected**: 3, 8  
**Severity**: 🔴 Critical — caused violent D-loss oscillation

### What we did

```python
# train_experiment.py line 727
loss_R1 = 0.5 * (r1_penalty_A + r1_penalty_B) * cfg["lambda_R1"] * 16.0
#                                                 ^^^^^^^^^^^^     ^^^^
#                                                   = 10.0        × 16
#                                         Effective weight = 10 × 16 × 0.5 = 80.0
```

### What StyleGAN2 actually does

StyleGAN2 (the source of this formula) uses `γ/2 × d_reg_interval` where `γ = 10`. But their code **pre-divides** the base weight, so the effective strength is ~2.5, not 80.

Our implementation skipped the pre-division, giving us **32× the intended strength**.

### What happened

The discriminator was hit with massive gradient penalties every 16 steps, causing:

**Run 3 — D-loss oscillation (first 30 epochs):**
```
Epoch  1: Loss_D = 0.91
Epoch  2: Loss_D = 6.18  ← R1 overcorrection
Epoch  3: Loss_D = 1.13  ← D recovers
Epoch  4: Loss_D = 6.32  ← R1 overcorrection again
Epoch 15: Loss_D = 7.74  ← Largest spike
Epoch 20: Dice_Idt_B = 0.067  ← Output was anatomical gibberish
```

**Run 8 — Same oscillation + FFT compounding:**
```
Epoch  6: Loss_D = 4.27
Epoch 13: Loss_D = 5.51
Epoch 22: Loss_D = 6.97  ← Largest spike
```

### The fix

```python
# Either: correct the scaling
loss_R1 = 0.5 * r1_penalty * (cfg["lambda_R1"] / 2.0)  # Effective weight = 2.5

# Or: simplify — apply every step with lower weight
if cfg.get("lambda_R1", 0.0) > 0.0:   # Every step, not every 16
    loss_R1 = 0.5 * r1_penalty * 1.0   # λ = 1.0, no interval multiplication
```

### Compounding factor: TTUR

All R1 runs also used TTUR (D_lr = 4× G_lr). Both R1 and TTUR strengthen the discriminator. Using both together created a double-amplification effect — the discriminator became too strong too fast, oscillated violently, and damaged the generator's learned features permanently.

**Rule**: Never combine R1 with TTUR. Pick one.

---

## Error 3: VGG Perceptual Loss at Wrong Layer Depth

**Runs affected**: 6  
**Severity**: 🟠 High — caused late-training artifact growth

### What we did

```python
# losses.py line 65
self.features = nn.Sequential(*list(vgg.features.children())[:9])  # relu2_2
```

`relu2_2` is a **shallow** VGG layer. It captures:
- Edges, small textures, fine patterns
- NOT shapes, objects, or semantics

### What happened

The VGG loss at relu2_2 told the generator: *"Match the exact edge patterns of the real image."* The generator responded by sharpening edges aggressively, which created **Gibbs-ringing artifacts** (high-frequency oscillations around sharp boundaries like the skull edge).

**FFT_B trajectory in Run 6:**
```
Epoch 10: FFT_B = 0.003  ← Normal
Epoch 30: FFT_B = 0.004  ← Normal  
Epoch 40: FFT_B = 0.010  ← Starting to rise
Epoch 50: FFT_B = 0.038  ← 10× worse! Artifact explosion
```

The shallow VGG was literally **causing** the checkerboard/ringing artifacts the FFT metric was detecting.

### The fix

```python
# Use deeper layers that capture shapes/semantics, not edges/textures
self.features_deep = nn.Sequential(*list(vgg.features.children())[:23])  # relu4_2
self.features_mid  = nn.Sequential(*list(vgg.features.children())[:16])  # relu3_2

# Weight deeper features more
loss = 0.7 * F.l1_loss(deep_fake, deep_real) + 0.3 * F.l1_loss(mid_fake, mid_real)
```

| VGG Layer | What it sees | Artifact risk |
|-----------|-------------|:-:|
| relu1_2 (idx 4) | Pixel-level noise | 🔴 Very high |
| **relu2_2 (idx 9)** ← we used this | Edge textures | 🟠 High |
| relu3_2 (idx 16) | Local patterns/shapes | 🟡 Low |
| relu4_2 (idx 23) | Object-level semantics | ✅ Safe |

---

## Error 4: VGG and Dice Applied to Wrong Images

**Runs affected**: 1, 5, 6 (Dice: 1, 5; VGG: 6)  
**Severity**: 🟡 Moderate — made losses redundant rather than harmful

### What we did

```python
# Dice applied to cycle-reconstructed images
loss_dice_A = eval_dice_loss(rec_A, real_A, modality="ct")   # rec_A = G_B2A(G_A2B(real_A))
loss_dice_B = eval_dice_loss(rec_B, real_B, modality="mri")  # rec_B = G_A2B(G_B2A(real_B))

# VGG applied to cycle-reconstructed images
loss_perceptual = eval_perceptual_loss(rec_B, real_B) + eval_perceptual_loss(rec_A, real_A)
```

### Why this is wrong

The **cycle-consistency L1 loss** already optimizes `rec_A ≈ real_A`. Adding Dice and VGG on the same pair (`rec_A vs real_A`) is just a weaker, more expensive version of what L1 already does.

The losses should be applied to **translated images** (`fake_A, fake_B`) — the actual generator output we care about:

```python
# Correct target: translated images
loss_dice_A = eval_dice_loss(fake_A, real_A, modality="ct")   # fake_CT vs real_CT
loss_perceptual = eval_perceptual_loss(fake_B, real_B)        # fake_MRI vs real_MRI
```

### Evidence of redundancy

Run 0 (no Dice) vs Run 1 (with Dice) achieved nearly identical Dice eval scores:
- Run 0: Idt_Dice = 0.882 / 0.810
- Run 1: Idt_Dice = 0.880 / 0.819

The Dice training loss added zero value because it was doing the same job as L1.

---

## Error 5: Dice Loss Uses Hard Thresholds (Zero Gradient)

**Runs affected**: 1, 5  
**Severity**: 🟡 Moderate — gradient signal was mostly noise

### What we did

```python
# metrics.py — AnatomicalDiceLoss.get_anatomical_masks()
bone_mask = (x >= 0.5)               # Boolean: True/False
soft_tissue_mask = (x > -0.3) & (x < 0.5)  # Boolean: True/False
```

### Why this is wrong

`(x >= 0.5)` is a **step function** — its derivative is zero everywhere (except at exactly 0.5, where it's undefined). When PyTorch backpropagates through this, the generator receives **zero useful gradient** from the Dice loss.

The Dice formula itself (`2 × intersection / denominator`) does produce some gradient through the soft multiplication `mask_fake * mask_real`, but since the masks themselves are binary, the gradient is extremely noisy and essentially random with respect to small intensity changes.

### The fix

```python
# Soft sigmoid approximation — smooth gradient everywhere
def soft_threshold(x, threshold, temperature=10.0):
    return torch.sigmoid(temperature * (x - threshold))

bone_mask = soft_threshold(x, 0.5, temperature=10.0)    # Smooth 0→1 transition
```

---

## Error 6: Identity Weight Decayed Too Low

**Runs affected**: 7  
**Severity**: 🟠 High — caused late-epoch performance reversal

### What we did

```python
# Linear decay from epoch 15 to 50
# lambda_idt_B: 2.5 → 0.5
lambda_idt_B_current = start_weight - fraction * (start_weight - end_weight)
```

### What happened

The identity loss acts as a safety net: *"If you're given a CT, your output should still look like a CT."* Decaying it to 0.5 meant the generator lost this anchor.

**Loss_G reversed direction after epoch 33:**
```
Epoch 29: Loss_G = 2.22  ← Still improving
Epoch 33: Loss_G = 1.95  ← Best point
Epoch 38: Loss_G = 2.00  ← Getting WORSE
Epoch 41: Loss_G = 2.15  ← Clearly worse
Epoch 50: Loss_G = 2.00  ← Stuck higher
```

**SSIM_A also decreased**: 0.884 (epoch 30) → 0.874 (epoch 50). The model got worse, not better.

### The fix

Don't decay identity at all. Run 0 proved that constant λ_idt = 2.5 works best.

If you must decay: use a **high floor** (minimum 1.5) with cosine schedule:
```python
floor = 1.5  # Never go below this
lambda_idt_B = floor + (2.5 - floor) * 0.5 * (1 + cos(π × fraction))
```

---

## Error 7: ResizeConv Uses Bilinear (Too Smooth)

**Runs affected**: 2, 3, 4  
**Severity**: 🟡 Moderate — inherently limits output sharpness

### What we did

Replaced `ConvTranspose2d` (standard CycleGAN upsampling) with:
```python
nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
nn.Conv2d(ngf, ngf, kernel_size=3, padding=1)
```

### Why this hurts

Bilinear interpolation is a **low-pass filter** — it smooths the image by averaging neighboring pixels. It physically cannot create sharp edges or fine detail. The subsequent Conv2d can only work with what bilinear gives it.

**Run 0 (ConvTranspose) vs Run 2 (ResizeConv):**
- SSIM: 0.894 vs 0.842 → **5.8% drop** just from this change

### Ironic interaction with FFT loss (Run 4)

Run 4 combined ResizeConv (which blurs) with FFT loss (which demands sharp high-frequency content). The generator was being told to produce sharp details using a tool that fundamentally cannot. This contradiction contributed to Run 4's collapse.

### The fix

If avoiding ConvTranspose checkerboard, use **nearest-neighbor** (less smoothing) or **PixelShuffle** (learns upsampling):
```python
# Better alternative
nn.Upsample(scale_factor=2, mode='nearest'),  # Not bilinear
nn.Conv2d(ngf, ngf, 3, padding=1)
```

---

## Error 8: TTUR Combined with Other D-Strengtheners

**Runs affected**: 3, 7, 8  
**Severity**: 🟠 High — amplified instability in every run that used it

### What we did

TTUR = Two Time-scale Update Rule: discriminator learns 4× faster than generator.
```python
lr_G = 0.0001
lr_D = 0.0004  # 4× faster
```

### Why it failed

Every single TTUR run had problems:
- **Run 3** (TTUR + R1): Violent D-loss oscillation, 30 epochs of chaos
- **Run 7** (TTUR + identity decay): D outpaced G as identity decayed, causing reversal  
- **Run 8** (TTUR + R1 + FFT): D-loss oscillation + FFT domination

Meanwhile, every **equal LR** run (0, 1, 2, 5, 6) trained smoothly.

TTUR works in StyleGAN where training is different (progressive growing, no cycle loss). In CycleGAN, the generator already has a harder job (cycle + identity + adversarial), so giving the discriminator even more advantage is counterproductive.

### The fix

Use equal learning rates. Don't use TTUR for CycleGAN.

---

## Quick Reference Table

| Error | Runs Hit | Type | Impact |
|-------|:--------:|:----:|--------|
| FFT not normalized | 4, 8 | Implementation bug | Loss_G inflated 5× |
| R1 scaled 32× too strong | 3, 8 | Implementation bug | D-loss oscillation |
| VGG wrong layer (relu2_2) | 6 | Implementation bug | Late-epoch artifact growth |
| Dice/VGG on wrong images | 1, 5, 6 | Design error | Made losses redundant |
| Dice hard thresholds | 1, 5 | Design error | Zero gradient signal |
| Identity decayed to 0.5 | 7 | Design error | Late-epoch reversal |
| Bilinear ResizeConv | 2, 3, 4 | Wrong tool choice | Output too smooth |
| TTUR + other D-boosters | 3, 7, 8 | Bad combination | Amplified instability |

---

## Code Implementation Errors (Found & Fixed in Phase 3 Review)

### Error 25: VGG Perceptual Loss Targeting Reconstructed Images (Redundant with L1)
- **Severity**: 🔴 Critical
- **Symptom**: Run 6v2 and Run 9 VGG losses were redundant with the standard L1 cycle loss, failing to test the desired anti-artifact perceptual hypothesis.
- **Cause**: The code applied VGG loss to `rec_A`/`rec_B` (cycle-reconstructions). Because cycle-reconstruction is already optimized by the cycle L1 loss, VGG was redundant.
- **Fix**: Cycle-reconstructed images are the only mathematically valid paired targets for spatial feature losses (like VGG) in unpaired translation tasks, since translated images `fake_A`/`fake_B` are unpaired and would cause anatomical warping if L1 VGG loss were applied. The correct fix is to keep VGG on reconstructed/identity images, but ensure we use **deep semantic layers** (`relu4_2` + `relu3_2`) to prevent edge artifacts.

### Error 26: FFT Loss Target Mismatch (Applied to Reconstructed Instead of Translated Images)
- **Severity**: 🟠 High
- **Symptom**: Training FFT loss did not target the pathway where high-frequency ringing/checkerboard artifacts actually occurred.
- **Cause**: The training FFT loss was computed on cycle-reconstructed images (`rec_A`/`rec_B`). Because the artifacts emerge in the translated generated scans (`fake_A`/`fake_B`), the loss failed to act as a guardrail.
- **Fix**: Because FFT magnitude alignment matches global spectral statistics and is spatial-agnostic, it does NOT require paired scans. We must apply the normalized FFT loss directly to the translated generated scans `fake_A` and `fake_B` relative to `real_A` and `real_B`.

### Error 27: Smoke Test Validation Blindness
- **Severity**: 🟠 High
- **Symptom**: Sub-component losses (like broken FFT scaling or R1 penalty regressions) could pass the local smoke test undetected.
- **Cause**: `smoke_test_phase_3.py` only checked total `loss_G` and did not check `loss_fft`, `loss_R1`, or `loss_perceptual` individually.
- **Fix**: Add separate history tracking arrays in `train_experiment_phase_3.py` for each auxiliary loss term, return them, and write explicit threshold gates in `smoke_test_phase_3.py` to assert correct scaling bounds.

### Error 28: Incomplete Configuration Alignment
- **Severity**: 🟡 Medium
- **Symptom**: Validation split did not match the plan's 80/20 specification, and early stopping patience was dead code.
- **Cause**: `val_split` was set to 0.1 (90/10 split) instead of 0.2, and `"fid_gated_stop": True` was omitted from the Run 6v2 and Run 9 configs lists.
- **Fix**: Align the configuration dicts and base config parameters directly with the implementation plan specifications.

### Error 29: Evaluation Function Keyword Signature Mismatch
- **Severity**: 🔴 Critical — caused evaluation loop crash during smoke test
- **Symptom**: `TypeError: evaluate_all_metrics() got an unexpected keyword argument 'loader_A'`
- **Cause**: `evaluate_all_metrics()` in `utils/metrics.py` expects positional arguments `(G_A2B, G_B2A, loader_ct, loader_mri, channels, device, epoch, run_dir, extractor, dice_loss_fn)`. The call site in `train_experiment_phase_4.py` used incorrect keyword names (`loader_A`, `loader_B`, `dice_metric_fn`, `output_dir`).
- **Fix**: Instantiated `eval_extractor = InceptionFeatureExtractor()` and `eval_dice_loss = AnatomicalDiceLoss()` in `train_experiment_phase_4.py`, updated `evaluate_all_metrics` arguments to match signature, and updated dictionary keys (`cycle_ssim_A2B`, `cycle_ssim_B2A`, `cycle_mae_A2B`, `cycle_mae_B2A`, `idt_mae_A2B`, `idt_mae_B2A`, `cycle_dice_A2B`, `cycle_dice_B2A`).

### Error 30: FID Covariance Ill-Conditioning on Small Subsets
- **Severity**: 🟠 High — caused LinAlgError in local smoke test with small validation sets
- **Symptom**: `torch._C._LinAlgError: linalg.eigh: The algorithm failed to converge because the input matrix is ill-conditioned` and `UserWarning: cov(): degrees of freedom is <= 0`
- **Cause**: When running local smoke tests with small sample counts (e.g. 1 validation image), `torch.cov` has zero degrees of freedom ($N - 1 = 0$), producing a zero/singular covariance matrix that breaks `torch.linalg.eigh`.
- **Fix**: Passed `correction=0` to `torch.cov` when $N \le 1$ in `utils/metrics.py` and wrapped `calculate_fid_pytorch()` eigenvalue decomposition in a `try-except` block with L2 norm mean difference fallback.

---

*Document updated: July 21, 2026. Authoritative reference catalog.*

