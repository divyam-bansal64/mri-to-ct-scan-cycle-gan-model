# RECOMMENDATIONS AND EXPERIMENTS — Phase 3 Update

> This document supplements `RECOMMENDATIONS_AND_EXPERIMENTS.pdf` (Phase 1) with Phase 2 results, errors discovered, and the Phase 3 plan. The PDF remains valid for Phase 1 history; this document covers everything after.

---

## Phase 1 Recap (from PDF)

**Winner**: H_full configuration (SSIM 0.8290 at 50 epochs)
- 9 ResNet blocks, ConvTranspose, Grayscale 1ch
- λ_cycle=5.0, λ_identity=2.5/2.5
- Flip + ±5° rotation + scale 0.9-1.1 augmentation

**Key findings**: ConvTranspose > ResizeConv, augmentation critical, grayscale viable with augmentation.

**Errors fixed (PDF Errors 9-16)**: Dataset transform sharing, unpaired VGG, unfiltered FFT, missing early-stop, spectral norm init, 3ch validation mismatch, Jupyter multiprocessing, CUDA compatibility.

---

## Phase 2: 9-Run Experiment Results

### Final Metrics at Epoch 50

| Run | Config | SSIM_A | SSIM_B | FID_A | FID_B | Dice_cycle_A | FFT_avg | Stability |
|-----|--------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **0** | H_full baseline | **0.894** | 0.750 | 172.5 | **128.0** | 0.824 | 0.005 | ✅ Best |
| **6** | H_full + VGG | 0.890 | **0.778** | 178.5 | 153.0 | **0.848** | **0.021** ⚠️ | ✅ Good |
| **3** | H_full + ResizeConv + R1 | 0.882 | 0.769 | 269.2 | 247.7 | 0.832 | 0.001 | ⚠️ Volatile |
| **8** | B_capacity + R1 + FFT | 0.869 | 0.771 | 282.2 | 243.8 | 0.795 | 0.003 | ⚠️ Oscillating |
| **1** | A_baseline + Dice | 0.868 | 0.731 | 207.0 | 150.4 | 0.798 | 0.004 | ✅ Good |
| **5** | G_low_idt + Dice | 0.867 | 0.751 | 170.2 | 119.3 | 0.807 | 0.011 | ✅ Good |
| **7** | F_asym_idt + sched + TTUR | 0.874 | 0.725 | 195.5 | 177.8 | 0.812 | 0.004 | ⚠️ Reversed |
| **2** | H_full + ResizeConv bare | 0.842 | 0.694 | 158.4 | 121.7 | 0.795 | 0.004 | ✅ Good |
| **4** | F_asym_idt + FFT + ResizeConv | 0.806 | 0.585 | 212.6 | 177.3 | 0.764 | 0.004 | ❌ D-collapse |

### Winner: Run 0 (H_full baseline)

Best risk-adjusted result: highest SSIM_A, strong FID, excellent training stability, zero artifacts.

---

## Phase 2: Errors Discovered (Errors 17-24)

> Full details in [errors_to_remember.md](file:///e:/code/mri%20to%20cti/errors_to_remember.md)

| Error # | Component | Runs Hit | Severity | Description |
|:-:|----------|:-:|:-:|-------------|
| 17 | FFT training loss | 4, 8 | 🔴 Critical | Raw FFT magnitudes (O(1000s)) not normalized to ratio (O(0.001)). Inflated Loss_G by +8-16. |
| 18 | R1 scaling | 3, 8 | 🔴 Critical | Effective weight = 80 (should be ~2.5). Incorrect StyleGAN2 lazy reg adaptation: `λ=10 × 16 = 160 × 0.5 = 80`. |
| 19 | VGG layer depth | 6 | 🟠 High | relu2_2 (shallow texture layer) caused exponential FFT_B artifact growth: 0.003→0.038. Should use relu4_2 (semantic). |
| 20 | VGG/Dice target | 1, 5, 6 | 🟡 Moderate | Applied to cycle-reconstructed images (redundant with cycle L1). Should target translated images. |
| 21 | Dice gradients | 1, 5 | 🟡 Moderate | Hard boolean thresholds `(x >= 0.5)` have zero gradient. Generator receives no useful signal. |
| 22 | Identity decay floor | 7 | 🟠 High | Linear 2.5→0.5 removed safety net. Loss_G reversed 1.93→2.15 after epoch 33. |
| 23 | TTUR + CycleGAN | 3, 7, 8 | 🟠 High | 4× D learning rate failed in 3/3 runs. CycleGAN generators already disadvantaged by multi-loss optimization. |
| 24 | ResizeConv smoothing | 2, 3, 4 | 🟡 Moderate | Bilinear interpolation is a low-pass filter. SSIM dropped 0.894→0.842 from this swap alone. |

---

## Phase 3: Corrected v2 Experiment Plan

### Design Principles

1. **All runs use H_full as base** — the proven winner from Phase 2
2. **Equal learning rates only** — TTUR dropped permanently
3. **No identity decay** — constant λ_idt=2.5 proven optimal
4. **No ResizeConv** — ConvTranspose confirmed superior
5. **No Dice training loss** — redundant with cycle L1
6. **Grayscale only** — avoids RGB cross-channel artifacts

### The 7-Run v2 Matrix

| Run | New Loss | λ Value | What It Tests |
|-----|----------|---------|---------------|
| **0** | None (existing) | — | Control baseline |
| **3v2** | Corrected R1 (every step, no TTUR) | λ_R1=1.0 | Does gentle R1 improve stability? |
| **4v2** | Normalized FFT (power ratio, one-sided) | λ_fft=5.0 | Does normalized FFT suppress artifacts? |
| **6v2** | Deep VGG on cycle (Option B) | λ_perceptual=0.3 | Does cycle VGG keep Run 6's quality? |
| **8v2** | Corrected R1 + Normalized FFT | λ_R1=1.0, λ_fft=5.0 | Do R1 + FFT work together? |
| **9** | Deep VGG + Normalized FFT (Cycle) | λ_perceptual=0.3, λ_fft=5.0 | Combo of Option B VGG + FFT |
| **10** | Deep VGG on identity (Option A) | λ_perceptual=0.3 | Does identity VGG keep visual quality? |
| **11** | Deep VGG + Normalized FFT (Identity) | λ_perceptual=0.3, λ_fft=5.0 | Combo of Option A VGG + FFT |

### What Was Permanently Dropped

| Component | Reason |
|-----------|--------|
| Dice loss | Redundant — Run 0 matched Dice scores without it |
| TTUR | 3/3 failures — wrong for CycleGAN |
| Identity decay | Caused late-epoch reversal |
| Asymmetric identity | Hurt anatomy preservation |
| ResizeConv | Low-pass filter, loses detail |
| 4-block generator | Capacity-limited |
| RGB | Cross-channel gradient leakage |

### Decision Flow After v2

```
Run 6v2 beats Run 0 on SSIM + no FFT spike?
├── YES → Does Run 9 beat Run 6v2?
│         ├── YES → Run 9 → 200-epoch
│         └── NO  → Run 6v2 → 200-epoch
└── NO → Run 0 → 200-epoch (baseline wins)
```

### Full Plan

See [implementation_plan.md](file:///e:/code/mri%20to%20cti/implementation_plan.md) (v7) for complete execution sequence, corrected loss code, smoke test criteria, and timeline.

---

*Last updated: July 18, 2026 — Phase 3 plan approved.*
