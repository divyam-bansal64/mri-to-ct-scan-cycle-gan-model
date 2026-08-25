# MRI to CT — CycleGAN Research & Failure Analysis

> **Note:** This is not a working medical image translator. The model doesn't perform accurate
> cross-modality conversion. What this repository does instead is document, in detail, what
> went wrong, why it went wrong, and what I learned from it — which turned out to be considerably
> more useful than a model that "works."

---

## Background

This started as an attempt to train a CycleGAN to convert between CT and MRI brain scans without
paired training data. After running 4 phases of experiments (~20+ configurations, up to 200 training
epochs each on Kaggle T4 GPUs), the honest conclusion is that the model learned to copy the input
image with a brightness shift rather than performing any actual modality translation.

The reasons for this are partly architectural (CycleGAN's cycle-consistency loss actively incentivizes
input preservation), partly physics-based (CT and MRI carry fundamentally different amounts of
information — the conversion isn't symmetric), and partly implementation (30 bugs found and fixed
across the course of training, including some that inflated the generator loss by 5-16x without
obvious error messages).

Those three things — the architecture analysis, the physics, and the bugs — are what the docs
in this repo try to capture properly.

---

## Key Finding

After 200 epochs, the generators produce this:

![Phase 4 Epoch 200 Sample Grid](docs/assets/phase4_epoch_200.png)

*Left to right: Real CT — "Fake MRI" — Real MRI — "Fake CT"*

The "Fake MRI" (panel 2) still has the bright skull rim that only appears in CT. The "Fake CT"
(panel 4) develops large bright blob artifacts in the brain parenchyma with no anatomical basis.
Neither generator is translating — one is copying, the other is hallucinating.

The SSIM score at epoch 200 was 0.9959, which sounds impressive. It means the fake MRI is
structurally 99.6% identical to the source CT — exactly the problem.

Details and visual progression from epoch 10 to 200 are in
[docs/05_failure_mode_taxonomy.md](docs/05_failure_mode_taxonomy.md).

---

## Repository Structure

The docs cover material at different levels of specificity. Files useful beyond this specific
project are noted:

```
docs/
  01_problem_formulation.md       [medical imaging]    CT vs MRI physics, information asymmetry
  02_architecture_decisions.md    [image translation]  Generator, discriminator, upsampling choices
  03_loss_function_analysis.md    [general GAN]        Loss dynamics, weight imbalance, scale bugs
  04_experiment_matrix.md         [this project]       All run configs and result tables
  05_failure_mode_taxonomy.md     [image translation]  Input copying, steganography, mode collapse
  06_metrics_literacy.md          [general GAN]        Why SSIM is misleading in translation tasks
  07_image_preprocessing.md       [medical imaging]    HU windowing, normalization, augmentation
  08_lessons_and_future_work.md   [image translation]  What to do differently, CUT vs CycleGAN

models/
  generator.py                    [image translation]  ResNet-9 Generator
  discriminator.py                [image translation]  PatchGAN Discriminator

utils/
  losses.py                       [general GAN]        VGG and FFT losses (original versions)
  losses_phase_4.py               [general GAN]        Corrected FFT (normalized) and deep VGG
  metrics.py                      [general GAN]        SSIM, FID, Dice, FFT ratio
  dataset.py                      [medical imaging]    Unpaired image dataset loader

experiment_v2/
  train_experiment_phase_4.py     [this project]       200-epoch training script

errors_to_remember.md             [general GAN]        30 bugs documented with before/after code
```

The general GAN and image translation docs are written to be self-contained — you don't need
to know anything about this specific project to find them useful.

---

## Experiment Summary

| Phase | Configs | Epochs | What I was testing | What happened |
| :---: | :---: | :---: | :--- | :--- |
| 1 | 1 | 50 | Architecture baseline (ResNet blocks, upsampling methods) | 9-block + ConvTranspose was the clear winner |
| 2 | 9 | 50 | Hyperparameter search across loss combinations | Found 8 bugs; plain baseline outperformed everything |
| 3 | 7 | 50 | Corrected loss implementations (R1, normalized FFT, deep VGG) | R1 gradient penalty was the single most impactful fix |
| 4 | 2 | 200 | Full-length training with best configs | Both models converge to input copying |

---

## The Bugs Worth Knowing About

These four caused the most damage and are worth understanding if you're working on any GAN:

| Bug | Impact |
| :--- | :--- |
| FFT loss computed on raw magnitudes (O(1000)) instead of power ratios (O(0.001)) | Generator loss inflated 5-16x, completely dominated training |
| R1 gradient penalty scaled 32x too strong | Discriminator loss oscillated violently — never stabilized |
| VGG perceptual loss extracted from relu2_2 instead of relu4_2 | Gibbs ringing artifacts around skull boundary — FFT ratio spiked 10x |
| TTUR (4x discriminator learning rate) applied to CycleGAN | Failed in every run it was used — wrong assumption for multi-loss generators |

Full catalog with code: [errors_to_remember.md](errors_to_remember.md)
Full analysis: [docs/03_loss_function_analysis.md](docs/03_loss_function_analysis.md)

---

## Why CycleGAN Specifically Struggles Here

Three reasons, in rough order of importance:

**The physics is asymmetric.** CT measures X-ray attenuation — brain soft tissue all looks
roughly the same (~40 HU). MRI measures T1/T2 relaxation, which distinguishes gray matter,
white matter, CSF, and more. CT→MRI requires generating tissue contrast that doesn't exist
in the source signal. MRI→CT is a compression problem; CT→MRI is a generative problem.

**Cycle consistency actively discourages translation.** The identity and cycle losses together
weighted 15x heavier than the adversarial loss. The optimal strategy under that loss landscape
is to preserve the input — which is what both models did.

**L1 loss rewards blurry averages.** When the correct output is uncertain, L1 minimization
produces the mean of all plausible outputs. That mean is blurry and structurally similar to
the input, so SSIM goes up even as the output quality goes down.

See [docs/01_problem_formulation.md](docs/01_problem_formulation.md) for the full physics writeup.

---

## Running It

```bash
pip install -r requirements.txt
```

Dataset: [CT-to-MRI cGAN on Kaggle](https://www.kaggle.com/datasets/darren2020/ct-to-mri-cgan)
(axial brain slices, unpaired CT and MRI domains)

```bash
python experiment_v2/train_experiment_phase_4.py Phase4_Beta_Control
```

---

## References

- Zhu et al. — [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) (ICCV 2017)
- Chu et al. — [CycleGAN, a Master of Steganography](https://arxiv.org/abs/1712.02950) (2017)
- Park et al. — [Contrastive Learning for Unpaired Image-to-Image Translation](https://arxiv.org/abs/2007.15651) (ECCV 2020)
