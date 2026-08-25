# Loss Function Analysis: Dynamics, Bugs & Corrected Implementations

> This document analyzes every loss component used across 4 phases, including 30 implementation 
> errors discovered during training. Each bug includes the original code, the impact on training, 
> and the corrected implementation.

---

## 1. CycleGAN Loss Landscape Overview

The total generator loss is a weighted sum of multiple objectives:

$$\mathcal{L}_G = \underbrace{\mathcal{L}_{\text{GAN}}}_{\text{Adversarial}} + \underbrace{\lambda_{\text{cyc}} \cdot \mathcal{L}_{\text{cycle}}}_{\text{Reconstruction}} + \underbrace{\lambda_{\text{idt}} \cdot \mathcal{L}_{\text{identity}}}_{\text{Preservation}} + \underbrace{\lambda_{\text{FFT}} \cdot \mathcal{L}_{\text{FFT}}}_{\text{Frequency}} + \underbrace{\lambda_{\text{VGG}} \cdot \mathcal{L}_{\text{perceptual}}}_{\text{Texture}}$$

Each component pulls the generator in a different direction:

| Loss | What It Wants | Weight Used | Effect When Dominant |
| :--- | :--- | :---: | :--- |
| **Adversarial** (GAN) | Fool the discriminator → realistic outputs | 1.0 | Sharp but potentially unstable |
| **Cycle L1** | $G_{B2A}(G_{A2B}(x)) \approx x$ | **10.0** | Conservative, blurry outputs |
| **Identity L1** | $G_{B2A}(A) \approx A$ (don't change same-domain) | **5.0** | Forces input copying behavior |
| **FFT** | Match high-frequency spectral energy | 5.0–10.0 | Suppresses artifacts (if normalized) |
| **VGG Perceptual** | Match deep CNN feature representations | 0.3–1.0 | Encourages texture richness |
| **R1 Gradient Penalty** | Regularize discriminator gradients | 1.0 | Stabilizes D at ~0.5 equilibrium |

### The Weight Imbalance Problem

```
Total "reproduce input" weight:  λ_cycle (10.0) + λ_identity (5.0) = 15.0
Total "translate to target" weight: λ_GAN = 1.0

Ratio: 15:1 in favor of input preservation
```

This 15:1 ratio means the generator receives 15× stronger gradient signals to **reproduce the input** than to **transform it into the target domain**. The rational optimization strategy is: apply a minimal global intensity shift (just enough to reduce the adversarial loss slightly) while preserving the input structure perfectly (to minimize the dominant cycle+identity losses).

---

## 2. Individual Loss Deep-Dives

### 2.1 Adversarial Loss (LSGAN)

```python
def gan_loss(pred, target_is_real):
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)
```

**How it works**: The generator wants `D(fake) = 1.0` (fool discriminator). The discriminator wants `D(real) = 1.0` and `D(fake) = 0.0`. MSE (Least Squares) is used instead of BCE for gradient stability.

**Healthy equilibrium**: When `loss_D ≈ 0.5`, the discriminator is unsure about real vs fake — the game is balanced. When `loss_D < 0.4`, the discriminator is winning (vanishing generator gradients). When `loss_D > 0.6`, the generator is winning (discriminator needs to catch up).

### 2.2 Cycle Consistency Loss

```python
rec_A = G_B2A(G_A2B(real_A))  # CT → MRI → CT (should ≈ original CT)
rec_B = G_A2B(G_B2A(real_B))  # MRI → CT → MRI (should ≈ original MRI)

loss_cycle_A = L1Loss(rec_A, real_A) * lambda_cycle  # λ = 10.0
loss_cycle_B = L1Loss(rec_B, real_B) * lambda_cycle
```

**Intent**: Ensures the translation is *reversible* — if you translate CT→MRI→CT, you should get back the original CT.

**Problem**: This is the primary cause of the input-copying behavior. With λ=10.0, the easiest way to guarantee perfect round-trip reconstruction is to **never change the image in the first place** (or hide the original in imperceptible noise).

### 2.3 Identity Loss

```python
idt_A = G_B2A(real_A)  # CT generator given a CT → should output the same CT
idt_B = G_A2B(real_B)  # MRI generator given an MRI → should output the same MRI

loss_idt_A = L1Loss(idt_A, real_A) * lambda_identity  # λ = 5.0
loss_idt_B = L1Loss(idt_B, real_B) * lambda_identity
```

**Intent**: Prevents the generator from making unnecessary changes. If G_B2A already receives a CT, it shouldn't distort it.

**Problem**: This literally trains the generator to be an **identity function** (output = input) with weight 5.0. Combined with cycle loss (10.0), the model receives a total weight of 15.0 for "don't change the image."

### 2.4 FFT Frequency Loss

**Buggy version (Phase 1-2)** — used raw magnitudes:
```python
# ❌ Raw FFT magnitudes: output range O(1000)
fake_high = torch.abs(X_fake_shifted) * high_freq_mask  
real_high = torch.abs(X_real_shifted) * high_freq_mask
return F.l1_loss(fake_high, real_high)  # Returns ~5-12 per image
```

**Corrected version (Phase 3-4)** — normalized power ratio:
```python
# ✅ Power ratio: output range O(0.001)
fake_ratio = high_freq_power_fake / total_power_fake
real_ratio = high_freq_power_real / total_power_real
excess = F.relu(fake_ratio - real_ratio)  # One-sided: only penalize artifacts
return excess.mean()  # Returns ~0.001-0.01
```

**Impact of the bug**: With λ_FFT=1.0, the raw FFT added ~8-16 to Loss_G. Total standard losses were ~2-3. FFT dominated **80% of the gradient signal**, causing the generator to spend all its capacity matching Fourier spectra instead of producing realistic images, triggering severe discriminator collapse (Phase 2 Run 4):

![Phase 2 Run 4 FFT Collapse](assets/phase2_run4_fft_collapse_epoch_050.png)
*Phase 2 Run 4: Severe structural disruption caused by raw FFT magnitude inflation (Loss_G exploded to ~18.0).*

---

### 2.5 VGG Perceptual Loss

**Buggy version (Phase 1-2)** — shallow layer:
```python
# ❌ relu2_2 (index 9): captures edges/textures, causes Gibbs ringing
self.features = nn.Sequential(*list(vgg.features.children())[:9])
```

**Corrected version (Phase 3-4)** — deep semantic layers:
```python
# ✅ relu4_2 (index 23) + relu3_2 (index 16): captures shapes/semantics
self.features_deep = nn.Sequential(*list(vgg.features.children())[:23])
self.features_mid  = nn.Sequential(*list(vgg.features.children())[:16])
loss = 0.7 * L1(deep_fake, deep_real) + 0.3 * L1(mid_fake, mid_real)
```

**Impact of the bug**: Shallow VGG layers (`relu2_2`) forced the network to match pixel-level edge patterns, causing high-frequency Gibbs ringing oscillations around skull boundaries (Phase 2 Run 6):

![Phase 2 Run 6 VGG Ringing](assets/phase2_run6_vgg_ringing_epoch_050.png)
*Phase 2 Run 6: Gibbs-ringing high-frequency boundary artifacts caused by shallow relu2_2 perceptual supervision.*

**Additional bug in Phase 4**: VGG was applied to **identity passes** (idt_A, idt_B) instead of **translated outputs** (fake_A, fake_B), meaning the translated images received zero perceptual guidance.

### 2.6 R1 Gradient Penalty

```python
# Compute gradient of D's output w.r.t. real input
grad_real = torch.autograd.grad(D(real).sum(), real, create_graph=True)[0]
r1_penalty = grad_real.pow(2).sum(dim=(1,2,3)).mean()
loss_R1 = 0.5 * r1_penalty * lambda_R1
```

**What it does**: Penalizes the discriminator for having large gradients on real images. This prevents the discriminator from becoming too sharp/confident, keeping the GAN game balanced.

**Phase 2 bug**: The penalty was scaled 32× too strong (effective weight = 80 instead of ~2.5), causing violent loss_D oscillation. Phase 3+ fixed this to λ_R1=1.0 applied every step.

**Phase 3-4 result**: R1 was the single most impactful improvement. Runs with R1 maintained loss_D ≈ 0.48-0.50 (perfect equilibrium) while runs without R1 saw loss_D drift down to 0.39-0.40 (discriminator overpowering).

---

## 3. Complete Error Catalog Summary

For full details with code, see [errors_to_remember.md](../errors_to_remember.md).

| # | Error | Runs Hit | Severity | Category |
| :---: | :--- | :--- | :---: | :--- |
| 1 | FFT loss uses raw magnitudes (O(1000)) | 4, 8 | 🔴 Critical | Scale mismatch |
| 2 | R1 penalty scaled 32× too strong | 3, 8 | 🔴 Critical | Implementation bug |
| 3 | VGG at relu2_2 (shallow, edge-sensitive) | 6 | 🟠 High | Wrong layer |
| 4 | Dice/VGG applied to reconstructed (not translated) images | 1, 5, 6 | 🟡 Moderate | Wrong target |
| 5 | Dice uses hard boolean thresholds (zero gradient) | 1, 5 | 🟡 Moderate | Non-differentiable |
| 6 | Identity weight decayed to 0.5 (too low) | 7 | 🟠 High | Design error |
| 7 | ResizeConv bilinear interpolation blurs output | 2, 3, 4 | 🟡 Moderate | Wrong tool |
| 8 | TTUR (4× D learning rate) in CycleGAN | 3, 7, 8 | 🟠 High | Bad combination |
| 25 | VGG perceptual loss redundant with cycle L1 | Phase 3 | 🔴 Critical | Wrong target |
| 26 | FFT loss on reconstructed instead of translated images | Phase 3 | 🟠 High | Wrong target |
| 27 | Smoke test didn't validate sub-component losses | Phase 3 | 🟠 High | Testing gap |
| 28 | Val split 0.1 instead of planned 0.2 | Phase 3 | 🟡 Medium | Config mismatch |
| 29 | evaluate_all_metrics() keyword argument mismatch | Phase 4 | 🔴 Critical | API mismatch |
| 30 | FID covariance ill-conditioning on small val sets | Phase 4 | 🟠 High | Numerical stability |

---

## 4. Key Insight: Loss Component Interaction Effects

Losses don't just add — they interact. Several compounding failures were discovered:

### R1 + TTUR Double Amplification
Both R1 and TTUR strengthen the discriminator. Using both simultaneously created violent oscillation because the discriminator alternated between being over-penalized (R1 step) and over-trained (high LR).

### ResizeConv + FFT Loss Contradiction
ResizeConv (bilinear) is a low-pass filter that removes high frequencies. FFT loss demands the generator produce realistic high-frequency content. The generator was told to produce sharp details using a tool that fundamentally cannot — this contributed to Run 4's complete collapse.

### Cycle L1 + Identity L1 + VGG on Reconstruction = Triple Redundancy
All three losses optimize for `output ≈ input` when applied to the same path. They reinforce the input-copying behavior with combined weight ~16.0 vs adversarial weight 1.0.
