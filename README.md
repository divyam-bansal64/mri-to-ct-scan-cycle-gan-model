# MRI ↔ CT CycleGAN: A Research Case Study in Medical Image Translation

> ⚠️ **This is a research and engineering case study, not a medical tool.**  
> The models trained in this project do not produce medically accurate translations.  
> This repository documents the technical journey, failure modes, and engineering  
> lessons learned from attempting unpaired CT↔MRI translation with CycleGAN.

---

## What This Project Offers

| Category | What You'll Find |
| :--- | :--- |
| 📊 **Systematic Experiment Design** | 4 phases, 20+ configurations, up to 200-epoch training runs |
| 🔍 **Failure Mode Taxonomy** | Why CycleGAN fails at cross-modality medical image translation — with visual evidence |
| 🐛 **30 Implementation Bugs Cataloged** | Loss function bugs with code, impact analysis, and corrected implementations |
| 📐 **Metrics Deep-Dive** | Why SSIM ≈ 0.99 can be dangerously misleading in image translation |
| 🧠 **Domain Physics** | CT vs MRI imaging fundamentals and the information asymmetry problem |
| 🛠️ **Image Processing Patterns** | Medical normalization, HU windowing, augmentation strategies, FFT analysis |

---

## Key Finding: The Model Replicates Instead of Translating

After 200 epochs of training across multiple configurations, visual inspection reveals that the CycleGAN generators learned to **apply global intensity shifts to the input image** rather than performing genuine cross-modality translation:

![Phase 4 Epoch 200 Sample Grid](docs/assets/phase4_epoch_200.png)
*Panel 1: Real CT | Panel 2: Fake MRI (retains CT skull) | Panel 3: Real MRI | Panel 4: Fake CT (hallucinated bright blobs)*

* **CT → "MRI" output (Panel 2)**: Retains the bright skull rim (real MRI shows skull as dark), same internal contrast — effectively a brightness-adjusted copy of the input CT.
* **MRI → "CT" output (Panel 4)**: Early epochs produce a recolored MRI. Later epochs develop bright artifact blobs with no anatomical basis (partial mode collapse).

**Why high SSIM is misleading here**: Cycle consistency loss (λ=10.0) + identity loss (λ=5.0) = 15× reconstruction weight vs 1× adversarial weight. The model chose the path of least resistance — hide input information steganographically and reproduce it on the return trip.

See [docs/05_failure_mode_taxonomy.md](docs/05_failure_mode_taxonomy.md) for full visual progression and analysis.

---

## Project Structure & Knowledge Generalizability

Each file is tagged with its **scope of applicability** — much of this repo is useful far beyond this specific project:

> 🌍 = Any GAN / deep learning project &nbsp;│&nbsp; 🔄 = Image-to-image translation &nbsp;│&nbsp; 🏥 = Medical imaging AI &nbsp;│&nbsp; 📌 = This project only

```
├── docs/
│   ├── 01_problem_formulation.md       🏥  CT vs MRI physics & information asymmetry
│   ├── 02_architecture_decisions.md    🔄  Generator, discriminator, upsampling, normalization
│   ├── 03_loss_function_analysis.md    🌍  Loss dynamics, weight balancing, scale normalization
│   ├── 04_experiment_matrix.md         📌  All experiment configs and results tables
│   ├── 05_failure_mode_taxonomy.md     🔄  Input copying, steganography, L1 oversmoothing
│   ├── 06_metrics_literacy.md          🌍  SSIM trap, FID interpretation, metric bundles
│   ├── 07_image_preprocessing.md       🏥  HU windowing, CLAHE, medical augmentation
│   └── 08_lessons_and_future_work.md   🔄  Engineering rules, CUT/Diffusion alternatives
│
├── models/
│   ├── generator.py                    🔄  ResNet-9 Generator (reusable for any CycleGAN)
│   └── discriminator.py               🔄  PatchGAN Discriminator (reusable)
│
├── utils/
│   ├── losses.py                       🌍  VGG perceptual loss, FFT loss (buggy originals)
│   ├── losses_phase_4.py              🌍  Corrected: normalized FFT, deep VGG (relu4_2)
│   ├── metrics.py                      🌍  SSIM, FID, Dice, FFT ratio implementations
│   └── dataset.py                      🏥  Unpaired medical image dataset loader
│
├── experiment_v2/
│   └── train_experiment_phase_4.py     📌  Phase 4 training script (200-epoch)
│
├── errors_to_remember.md              🌍  30 implementation bugs with fixes
└── README.md                           📌  Project overview
```

**~70% of documentation is generalizable** beyond this project — loss analysis, failure modes, metrics guides, and architecture decisions apply to anyone working with GANs or image translation.

---

## Experiment Phases Overview

| Phase | Runs | Epochs | Key Objective | Key Outcome |
| :---: | :---: | :---: | :--- | :--- |
| **1** | 1 | 50 | Find best base architecture | H_full (9 ResNet blocks, ConvTranspose) won |
| **2** | 9 | 50 | Hyperparameter search (losses, LR, upsampling) | 8 critical bugs discovered; Run 0 baseline best |
| **3** | 7 | 50 | Corrected loss implementations (R1, FFT, VGG) | R1 penalty proven essential for D stability |
| **4** | 2 | 200 | Full training with best configurations | Model replicates input; doesn't translate |

---

## Technical Highlights

### Loss Function Bug Discovery

| Bug | Severity | Impact |
| :--- | :---: | :--- |
| FFT loss using raw magnitudes (O(1000)) instead of power ratios (O(0.001)) | 🔴 Critical | Inflated generator loss by 5-16×, caused discriminator collapse |
| R1 gradient penalty scaled 32× too strong | 🔴 Critical | Violent discriminator loss oscillation |
| VGG perceptual loss at relu2_2 (shallow) instead of relu4_2 (semantic) | 🟠 High | Gibbs-ringing artifacts, 10× FFT spike |
| TTUR (4× discriminator LR) combined with CycleGAN | 🟠 High | Failed in 3/3 runs; wrong for multi-loss generators |

See [docs/03_loss_function_analysis.md](docs/03_loss_function_analysis.md) and [errors_to_remember.md](errors_to_remember.md) for complete catalog.

### Why CycleGAN Cannot Solve This Problem

1. **Information asymmetry**: CT soft tissue is a ~uniform gray blob; MRI reveals rich tissue contrast. CT→MRI requires hallucinating information that doesn't exist in the source.
2. **Cycle consistency incentivizes copying**: With λ_cycle=10.0, the easiest way to guarantee perfect reconstruction is to hide the input in imperceptible noise patterns.
3. **L1 loss rewards blurriness**: The generator outputs the pixel-wise mean of all valid targets, achieving high SSIM but zero perceptual quality.

See [docs/01_problem_formulation.md](docs/01_problem_formulation.md) for the physics explanation.

---

## Requirements

```
torch>=1.12
torchvision>=0.13
Pillow
numpy
scikit-image
```

---

## Dataset

This project uses the [CT-to-MRI cGAN dataset](https://www.kaggle.com/datasets/darren2020/ct-to-mri-cgan) from Kaggle:
- **Domain A (CT)**: Axial brain CT slices
- **Domain B (MRI)**: Axial brain MRI slices
- Unpaired — no patient-level correspondence between CT and MRI images

---

## License

This project is released for educational and research purposes.

---

## Acknowledgments

- CycleGAN architecture based on [Zhu et al., 2017](https://arxiv.org/abs/1703.10593)
- Dataset: [Darren2020/ct-to-mri-cgan](https://www.kaggle.com/datasets/darren2020/ct-to-mri-cgan) on Kaggle
- Training infrastructure: Kaggle T4×2 GPU notebooks
