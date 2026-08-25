# Experiment Results

All runs used 256×256 grayscale brain images, Adam optimizer, and CycleGAN base architecture.
Runs of 50 epochs each for phases 1–3, 200 epochs for phase 4.

---

## Phase 1: Architecture Baseline (1 run, 50 epochs)

Single run to establish the base architecture. Winner: **H_full** (9 ResNet blocks, ConvTranspose2d):

| Parameter | Value |
| :--- | :--- |
| Generator blocks | 9 (ResNet) |
| Upsampling | ConvTranspose2d |
| Channels | 1 (Grayscale) |
| λ_cycle | 5.0 |
| λ_identity | 2.5 / 2.5 |
| Augmentation | Flip + ±5° rotation + 0.9–1.1× scale |
| SSIM at epoch 50 | 0.829 |

Key findings: ConvTranspose2d beats ResizeConv, grayscale training is viable, augmentation is critical.

---

## Phase 2: 9-Run Hyperparameter Search (50 epochs each)

| Run | Configuration | SSIM_A | SSIM_B | FID_A | FID_B | Stability |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | H_full baseline | 0.894 | 0.750 | 172.5 | 128.0 | Stable — best overall |
| 6 | H_full + VGG (relu2_2) | 0.890 | 0.778 | 178.5 | 153.0 | Stable |
| 3 | H_full + ResizeConv + R1(×32) + TTUR | 0.882 | 0.769 | 269.2 | 247.7 | Volatile |
| 8 | 4-block Gen + R1(×32) + FFT(raw) + TTUR | 0.869 | 0.771 | 282.2 | 243.8 | Oscillating |
| 1 | Baseline + Dice loss | 0.868 | 0.731 | 207.0 | 150.4 | Stable |
| 5 | Low identity + Dice loss | 0.867 | 0.751 | 170.2 | 119.3 | Stable |
| 7 | Asymmetric identity + TTUR | 0.874 | 0.725 | 195.5 | 177.8 | Reversed direction |
| 2 | H_full + ResizeConv (no extras) | 0.842 | 0.694 | 158.4 | 121.7 | Stable |
| 4 | Asymmetric + FFT(raw) + ResizeConv | 0.806 | 0.585 | 212.6 | 177.3 | D-collapse |

The plain baseline (Run 0) won on risk-adjusted performance. Every configuration that added
complexity either introduced instability or produced no improvement. TTUR failed in all 3 runs
it appeared in. ResizeConv caused a consistent 5.8% SSIM penalty. The FFT bug (raw magnitudes)
caused discriminator collapse in Run 4. This phase found 8 critical implementation bugs.

---

## Phase 3: Corrected Losses (7 runs, 50 epochs each)

All runs use H_full base with TTUR dropped, identity constant, ConvTranspose only.

| Run | Added loss | λ values | SSIM_A | SSIM_B | FID_A | FID_B |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 3v2 | Corrected R1 | λ_R1=1.0 | 0.968 | 0.897 | 268.7 | 258.0 |
| 8v2 | R1 + normalized FFT | λ_R1=1.0, λ_FFT=10.0 | 0.967 | 0.896 | 254.8 | 249.1 |
| 10 | Identity VGG (relu4_2) | λ_VGG=1.0 | 0.903 | 0.751 | 149.2 | 119.0 |
| 9 | Cycle VGG + FFT | λ_VGG=1.0, λ_FFT=10.0 | 0.901 | 0.754 | 147.4 | 119.8 |
| 4v2 | Normalized FFT only | λ_FFT=10.0 | 0.896 | 0.733 | 156.2 | 117.2 |
| 6v2 | Deep VGG (cycle mode) | λ_VGG=1.0 | 0.878 | 0.720 | 159.0 | 123.1 |
| 11 | Identity VGG + FFT | λ_VGG=1.0, λ_FFT=10.0 | 0.893 | 0.724 | 153.3 | 128.9 |

R1 was the single most impactful change — runs 3v2 and 8v2 jumped from 0.894 to 0.968 SSIM,
and the discriminator held steady at loss_D ≈ 0.49 throughout. Without R1, discriminators
drifted to 0.39–0.40 by epoch 50, indicating the discriminator was overpowering the generator.

The tradeoff: R1 runs had the highest SSIM but also highest FID. VGG runs had lower SSIM but
much better FID. This is the L1 oversmoothing effect — better pixel-level reconstruction produces
blurrier outputs that don't match the real image distribution. See
[06_metrics_literacy.md](06_metrics_literacy.md).

---

## Phase 4: Full 200-Epoch Training (2 runs)

### Configurations

| Parameter | Alpha Hybrid | Beta Control |
| :--- | :--- | :--- |
| R1 | λ_R1 = 1.0 | λ_R1 = 1.0 |
| FFT | λ_FFT = 5.0 (reduced) | λ_FFT = 10.0 |
| VGG | λ_VGG = 0.5 (identity mode) | None |
| Cycle | λ_cycle = 10.0 | λ_cycle = 10.0 |
| Identity | λ_idt = 5.0 | λ_idt = 5.0 |

### Final Metrics (Epoch 200)

| Metric | Alpha | Beta |
| :--- | :---: | :---: |
| SSIM_A (CT→MRI structural similarity) | 0.9959 | 0.9958 |
| SSIM_B (MRI→CT structural similarity) | 0.9554 | 0.9650 |
| FID_A (MRI realism) | 183.71 | 185.74 |
| FID_B (CT realism) | 229.09 | 218.31 |
| MAE_A | 0.0019 | 0.0021 |
| MAE_B | 0.0142 | 0.0127 |
| Loss_G (final) | 0.8819 | 0.8708 |
| Loss_D (final) | 0.4733 | 0.4683 |

### Epoch-by-Epoch Progression (Beta Control)

| Epoch | SSIM_A | SSIM_B | FID_A | FID_B | Loss_G | Loss_D |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 10 | 0.9356 | 0.7528 | 222.1 | 238.2 | 2.849 | 1.143 |
| 50 | 0.9747 | 0.9219 | 219.8 | 223.9 | 1.331 | 0.517 |
| 100 | 0.9818 | 0.9407 | 213.2 | 224.2 | 1.078 | 0.485 |
| 150 | 0.9930 | 0.9552 | 197.9 | 220.5 | 0.951 | 0.476 |
| 200 | 0.9958 | 0.9650 | 185.7 | 218.3 | 0.871 | 0.468 |

SSIM climbs steadily toward 1.0. FID improves slightly but stays above 180. Visual inspection
of the output images shows both models converging to input replication — the SSIM improvement
measures how well the model copies the input, not how well it translates. See
[05_failure_mode_taxonomy.md](05_failure_mode_taxonomy.md) for the visual analysis.
