# Experiment Matrix: All Configurations & Results

> Complete record of every experiment run across 4 phases (20+ configurations, 50–200 epochs each).

---

## Phase 1: Architecture Search (1 Configuration, 50 Epochs)

### Winner: H_full

| Parameter | Value |
| :--- | :--- |
| Generator blocks | 9 (ResNet) |
| Upsampling | ConvTranspose2d |
| Channels | 1 (Grayscale) |
| λ_cycle | 5.0 |
| λ_identity | 2.5 / 2.5 |
| Augmentation | Flip + ±5° rotation + 0.9–1.1× scale |
| **SSIM at epoch 50** | **0.829** |

Key findings:
- ConvTranspose2d > ResizeConv (always)
- Augmentation is critical for generalization
- Grayscale is viable (avoids RGB artifacts)

---

## Phase 2: 9-Run Hyperparameter Search (50 Epochs Each)

| Run | Configuration | SSIM_A ↑ | SSIM_B ↑ | FID_A ↓ | FID_B ↓ | Stability |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | H_full baseline | **0.894** | 0.750 | 172.5 | **128.0** | ✅ Best |
| **6** | H_full + VGG (relu2_2) | 0.890 | **0.778** | 178.5 | 153.0 | ✅ Good |
| **3** | H_full + ResizeConv + R1(×32) + TTUR | 0.882 | 0.769 | 269.2 | 247.7 | ⚠️ Volatile |
| **8** | 4-block Gen + R1(×32) + FFT(raw) + TTUR | 0.869 | 0.771 | 282.2 | 243.8 | ⚠️ Oscillating |
| **1** | Baseline + Dice loss | 0.868 | 0.731 | 207.0 | 150.4 | ✅ Good |
| **5** | Low identity + Dice loss | 0.867 | 0.751 | 170.2 | 119.3 | ✅ Good |
| **7** | Asymmetric identity + TTUR | 0.874 | 0.725 | 195.5 | 177.8 | ⚠️ Reversed |
| **2** | H_full + ResizeConv (no extras) | 0.842 | 0.694 | 158.4 | 121.7 | ✅ Good |
| **4** | Asymmetric + FFT(raw) + ResizeConv | 0.806 | 0.585 | 212.6 | 177.3 | ❌ D-collapse |

### Phase 2 Key Findings
- **Run 0 (pure baseline) won** — risk-adjusted best performer
- **8 critical bugs discovered** (see [03_loss_function_analysis.md](03_loss_function_analysis.md))
- TTUR failed in 3/3 runs
- ResizeConv caused 5.8% SSIM drop
- Dice loss added zero value over L1 cycle loss

---

## Phase 3: Corrected Loss Implementations (7 Runs, 50 Epochs Each)

All runs use H_full as base. TTUR dropped. Identity constant. ConvTranspose only.

| Run | New Loss | λ Values | SSIM_A ↑ | SSIM_B ↑ | FID_A ↓ | FID_B ↓ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **3v2** | Corrected R1 | λ_R1=1.0 | **0.968** | **0.897** | 268.7 | 258.0 |
| **8v2** | R1 + Normalized FFT | λ_R1=1.0, λ_FFT=10.0 | **0.967** | **0.896** | 254.8 | 249.1 |
| **10** | Identity VGG (relu4_2) | λ_VGG=1.0 | 0.903 | 0.751 | **149.2** | **119.0** |
| **9** | Cycle VGG + FFT | λ_VGG=1.0, λ_FFT=10.0 | 0.901 | 0.754 | **147.4** | **119.8** |
| **4v2** | Normalized FFT only | λ_FFT=10.0 | 0.896 | 0.733 | 156.2 | 117.2 |
| **6v2** | Deep VGG (cycle mode) | λ_VGG=1.0 | 0.878 | 0.720 | 159.0 | 123.1 |
| **11** | Identity VGG + FFT | λ_VGG=1.0, λ_FFT=10.0 | 0.893 | 0.724 | 153.3 | 128.9 |

### Phase 3 Key Findings

**R1 gradient penalty was transformative**: Runs 3v2 and 8v2 achieved SSIM 0.967 (vs 0.894 in Phase 2) — a massive jump. The discriminator maintained perfect equilibrium (loss_D ≈ 0.49) throughout training.

**The SSIM vs FID tradeoff**: R1 runs achieved the highest SSIM but the highest FID. VGG runs achieved the lowest FID but lower SSIM. This is the L1 oversmoothing paradox — pixel-perfect reconstruction (high SSIM) produces blurry outputs that don't match the real image distribution (high FID).

**Discriminator health**: Without R1, all discriminators showed loss_D dropping from 0.49 to 0.39-0.40 by epoch 50 — indicating the discriminator overpowering the generator.

---

## Phase 4: Full 200-Epoch Training (2 Runs)

### Configuration Comparison

| Parameter | Alpha Hybrid | Beta Control |
| :--- | :--- | :--- |
| R1 Penalty | λ_R1 = 1.0 | λ_R1 = 1.0 |
| FFT Loss | λ_FFT = 5.0 (reduced) | λ_FFT = 10.0 |
| VGG Perceptual | λ_VGG = 0.5 (identity mode) | None |
| Cycle L1 | λ_cycle = 10.0 | λ_cycle = 10.0 |
| Identity L1 | λ_idt = 5.0 | λ_idt = 5.0 |

### Final Metrics at Epoch 200

| Metric | Alpha Hybrid | Beta Control | Winner |
| :--- | :---: | :---: | :--- |
| **SSIM_A** (CT→MRI structure) | **0.9959** | 0.9958 | Tied |
| **SSIM_B** (MRI→CT structure) | 0.9554 | **0.9650** | Beta |
| **FID_A** (MRI realism) | **183.71** | 185.74 | Alpha |
| **FID_B** (CT realism) | 229.09 | **218.31** | Beta |
| **MAE_A** | **0.0019** | 0.0021 | Alpha |
| **MAE_B** | 0.0142 | **0.0127** | Beta |
| **Dice_idt_MRI** | 0.9315 | **0.9438** | Beta |
| **FFT_CT** | 0.0028 | **0.0023** | Beta |
| **FFT_MRI** | 0.0013 | **0.0010** | Beta |
| Loss_G (final) | 0.8819 | **0.8708** | Beta |
| Loss_D (final) | 0.4733 | 0.4683 | Both stable |

### Phase 4 Metric Progression (Beta Control)

| Epoch | SSIM_A | SSIM_B | FID_A | FID_B | Loss_G | Loss_D |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 10 | 0.9356 | 0.7528 | 222.1 | 238.2 | 2.849 | 1.143 |
| 50 | 0.9747 | 0.9219 | 219.8 | 223.9 | 1.331 | 0.517 |
| 100 | 0.9818 | 0.9407 | 213.2 | 224.2 | 1.078 | 0.485 |
| 150 | 0.9930 | 0.9552 | 197.9 | 220.5 | 0.951 | 0.476 |
| 200 | 0.9958 | 0.9650 | 185.7 | 218.3 | 0.871 | 0.468 |

### Phase 4 Outcome: Metrics Improve, Images Do Not

Despite steadily improving numerical metrics (SSIM approaching 1.0, FID slowly decreasing), visual inspection reveals that both models converge to **input replication** rather than cross-modality translation. The high SSIM scores measure how well the model copies the input, not how well it translates.

This is the central research finding of this project — see [05_failure_mode_taxonomy.md](05_failure_mode_taxonomy.md).
