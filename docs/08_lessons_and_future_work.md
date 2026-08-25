# Lessons Learned & Future Work

> What we would do differently knowing what we know now, and modern alternatives to CycleGAN 
> for cross-modality medical image translation.

---

## 1. Core Takeaways

### 1.1 The Physics Matters More Than the Architecture

The most important factor in medical image translation is not the model architecture, loss function, or training schedule — it's the **information content of the source modality**.

CT→MRI requires generating information that doesn't exist in the source. No amount of architectural improvement can overcome this fundamental limitation with CycleGAN's unpaired framework. The model cannot "learn" what gray/white matter looks like from CT alone — it needs paired supervision or a strong generative prior.

### 1.2 Loss Weight Ratios Determine Model Behavior

The 15:1 ratio of reconstruction-to-adversarial loss weights predetermined the outcome. The model was mathematically incentivized to copy the input. This isn't a bug — it's the rational optimal strategy given the loss landscape.

**Guideline**: In any GAN training, explicitly calculate the total weight contribution of each "behavioral category":
- "Preserve input" losses (cycle L1, identity L1)
- "Change to target" losses (adversarial)
- "Quality/regularization" losses (VGG, FFT, R1)

If "preserve" >> "change", the model will preserve.

### 1.3 Metrics Can Actively Mislead

SSIM = 0.9959 sounds like a success. It's actually evidence of failure in a translation task. Always:
- Compare metrics across different failure modes (input copying has distinct metric signatures)
- Use multiple metrics that capture different aspects (SSIM + FID at minimum)
- Visually inspect outputs — if they look like the input, metrics are measuring the wrong thing

### 1.4 Bugs Compound Non-Linearly

Individual bugs (wrong FFT scale, wrong R1 weight, wrong VGG layer) might be survivable alone. But bugs interact — R1 + TTUR created a double amplification effect. ResizeConv + FFT created an impossible optimization. Always test components in isolation before combining.

### 1.5 Negative Results Are Valuable

This project did not produce a useful medical image translator. But it produced:
- A taxonomy of CycleGAN failure modes
- 30 documented and corrected implementation errors
- A deep understanding of loss function dynamics
- Practical knowledge of medical image handling

Negative results are underrepresented in ML publications and repositories. Documenting why something doesn't work saves others months of effort.

---

## 2. What We Would Do Differently

### 2.1 Start with a Feasibility Check

Before training any models, we should have:
1. Computed the **mutual information** between CT and MRI brain images — this would have quantified the information asymmetry
2. Established a **human baseline** — shown a radiologist a CT and asked them to sketch the corresponding MRI. If a human expert can't do it, a model can't either
3. Focused on the **MRI→CT direction** (pseudo-CT generation) where the physics supports the task

### 2.2 Reduce Reconstruction Loss Dominance

If using CycleGAN, start with lower reconstruction weights:
```python
lambda_cycle = 2.0    # Instead of 10.0
lambda_identity = 0.5  # Instead of 5.0
# Now: "preserve" = 2.5 vs "change" = 1.0 → ratio of 2.5:1 instead of 15:1
```

The trade-off: Lower reconstruction weight allows more domain shift but risks spatial distortion. This is the fundamental tension in unpaired translation.

### 2.3 Use Translation-Aware Evaluation

Instead of measuring SSIM between source and reconstructed images (which rewards input copying), we should measure:
- **Cross-domain SSIM**: Compare generated MRI to actual MRI images of similar anatomy
- **Domain classifier accuracy**: Train a simple CNN to distinguish real CT from real MRI; feed it the generated images to see if they classify as the target domain
- **Clinical feature scores**: Have domain experts rate anatomical plausibility (we did this — the doctor confirmed non-viable outputs)

---

## 3. Modern Alternatives to CycleGAN

### 3.1 Contrastive Unpaired Translation (CUT)

**Paper**: Park et al., "Contrastive Learning for Unpaired Image-to-Image Translation" (ECCV 2020)

**Key idea**: Replace cycle consistency with **PatchNCE** (Patch-wise Noise Contrastive Estimation). Instead of requiring the round-trip $G_{B2A}(G_{A2B}(x)) \approx x$, CUT requires that local patches in the generated image share **maximum mutual information** with corresponding patches in the input.

**Why it helps**:
- **No cycle path** → eliminates the steganographic hiding incentive
- **One generator only** → no need for inverse mapping
- **Local correspondence** → preserves anatomy without requiring pixel-perfect reconstruction
- Shown to produce sharper, more realistic outputs than CycleGAN

### 3.2 Latent Diffusion Models with ControlNet

**Papers**: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022); Zhang & Agrawala, "Adding Conditional Control to Text-to-Image Diffusion Models" (ICCV 2023)

**Key idea**: Use a pre-trained diffusion model as a generative prior. Condition the generation process on the source image (CT) via ControlNet or T2I-Adapter, which provides spatial structure guidance without restricting the generative capacity.

**Why it helps for CT→MRI**:
- Diffusion models learn strong generative priors from large datasets
- ControlNet constrains spatial structure to match the CT anatomy
- The diffusion process naturally generates rich textures and tissue contrast
- Handles the "imagination" problem — the diffusion model has learned what MRI tissue contrast looks like from training data

### 3.3 Paired Registration + Pix2Pix

If any co-registered CT-MRI datasets are available (same patient, same session, spatially aligned):

1. Use **deformable registration** (ANTs SyN, Elastix) to align CT and MRI volumes
2. Train a **Pix2Pix** (paired) model with direct pixel supervision
3. Add perceptual loss on the translated outputs (not reconstructed)

**Why it's better**: Paired supervision eliminates the need for cycle consistency entirely. The model receives direct feedback on translation quality.

**Limitation**: Requires co-registered multi-modal datasets, which are expensive to acquire.

### 3.4 Domain-Adapted Foundation Models

Emerging approach: Fine-tune medical image foundation models (e.g., MedSAM, BiomedCLIP) for cross-modality synthesis. These models already encode medical image understanding and can be adapted for specific translation tasks with much less data.

---

## 4. Summary of Engineering Rules

From 4 phases and 30 bugs, these rules emerged:

| Rule | Source |
| :--- | :--- |
| Always normalize custom losses to O(0.01–1.0) before adding to total loss | Error #1 (FFT) |
| Never combine R1 with TTUR in CycleGAN | Error #2, #8 |
| Use deep VGG layers (relu4_2+) for perceptual loss, never shallow (relu2_2) | Error #3 |
| Apply auxiliary losses (VGG, FFT, Dice) to translated outputs, not reconstructed | Errors #4, #25, #26 |
| Use ConvTranspose2d over ResizeConv (bilinear is a low-pass filter) | Error #7 |
| Keep identity loss constant — don't decay it | Error #6 |
| Use equal learning rates (lr_G = lr_D) for CycleGAN | Error #8 |
| Smoke tests must validate individual loss components, not just total loss | Error #27 |
| Use soft thresholds (sigmoid) instead of hard boolean for differentiable masks | Error #5 |
| Validate with ≥ 20% holdout for reliable FID computation | Errors #28, #30 |
