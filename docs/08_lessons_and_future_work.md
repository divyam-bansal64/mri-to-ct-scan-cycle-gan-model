# Lessons and Future Directions

---

## What Actually Mattered

Looking back across 4 phases and 200 epochs of training, a few things turned out to be more
important than I expected going in.

**The physics mattered more than the architecture.** The CT-to-MRI direction is a generative problem
by nature — the model is being asked to synthesize tissue contrast that doesn't exist in the CT
attenuation signal. No architecture fixes that. The right question to ask before training was:
"what information does the source modality carry, and is it sufficient to determine the target?"
For CT→MRI, the answer is no. For MRI→CT, it mostly is.

**Loss weight ratios determine behavior more than loss choice.** With cycle loss at λ=10.0 and
identity loss at λ=5.0, the model received 15x stronger signal to preserve the input than to
transform it. The model optimized exactly what we asked it to optimize — that just turned out
to be the wrong thing. Before combining multiple losses, it's worth explicitly calculating the
total gradient weight in each behavioral category: how much is pushing the model to preserve vs
how much is pushing it to change.

**Metrics can confirm a failure while appearing to show success.** SSIM of 0.9959 passed every
automated check. It took visual inspection and then FID cross-reference to understand that this
score was measuring how well the model copied the input, not how well it translated.

**Bugs compound.** Individual problems were sometimes survivable — the wrong FFT scale on its own,
or R1 alone. But R1 and TTUR together created double discriminator amplification. ResizeConv and
FFT loss together created an impossible optimization (bilinear removes the high frequencies that
FFT loss demands). Testing components in isolation before combining them would have saved several
wasted runs.

---

## What I Would Do Differently

**Start with a feasibility sanity check.** Before training anything, compute the mutual information
between CT and MRI brain images quantitatively. Alternatively, ask a domain expert: "looking at
this CT, can you predict what the MRI would show?" If a radiologist says no, the model won't
manage it either. I consulted a doctor after training 200 epochs — the conversation should have
happened first.

**Focus on MRI→CT, not CT→MRI.** MRI→CT (pseudo-CT synthesis) is actively used in clinical
settings for MRI-only radiotherapy planning. The physics supports it. CT→MRI synthesis remains
largely unsolved precisely because of the information asymmetry discussed in doc 01.

**Reduce reconstruction loss dominance before anything else.** If revisiting CycleGAN for this
kind of problem, starting point would be:

```python
lambda_cycle = 2.0    # instead of 10.0
lambda_identity = 0.5  # instead of 5.0
# Ratio: ~2.5:1 preserve vs change, instead of 15:1
```

The tradeoff is that lower reconstruction weight allows more domain shift but risks spatial
distortion. That tension is unavoidable — it's the core difficulty of unpaired translation.

**Measure translation, not reconstruction.** SSIM between source and output tells you how much
the model preserved the input. That's not what you want to know. More useful: train a simple
domain classifier (CT vs MRI) and check whether the generated images get classified as the target
domain. Or compare generated outputs to real target-domain images, not back to the source.

---

## Modern Alternatives

**CUT (Contrastive Unpaired Translation)** — Park et al., ECCV 2020

Replaces cycle consistency with PatchNCE (patch-wise contrastive estimation). Instead of requiring
round-trip reconstruction, CUT requires that local patches in the generated image share maximum
mutual information with the corresponding patches in the input. One generator, no cycle path, no
incentive to hide information steganographically. Produces sharper outputs than CycleGAN in most
comparisons. This would be the first thing to try for any new unpaired translation project.

**Latent diffusion with ControlNet** — Rombach et al., CVPR 2022; Zhang & Agrawala, ICCV 2023

Use a pre-trained diffusion model as a generative prior conditioned on the source image via
ControlNet. The diffusion model has already learned what MRI tissue contrast looks like from
training data — ControlNet just constrains spatial structure to match the CT anatomy. This is
the approach most likely to handle the CT→MRI direction, because it brings an external generative
prior rather than trying to learn it from the translation data alone.

**Pix2Pix with registered pairs**

If co-registered CT-MRI data exists (same patient, same session, spatially aligned), paired
supervision eliminates the need for cycle consistency entirely. Deformable registration with ANTs
or Elastix can align volumes from different modalities. The model gets direct pixel-level feedback
on translation quality, which is a fundamentally easier optimization problem.

**Medical foundation models**

MedSAM, BiomedCLIP, and similar models encode medical image understanding from large multi-modal
training sets. Fine-tuning one for cross-modality synthesis requires far less task-specific data
and starts from a much stronger prior.

---

## Engineering Rules From This Project

These came directly from bugs and failed runs, not from theory. See
[errors_to_remember.md](../errors_to_remember.md) for the full context on each.

| Rule | Came from |
| :--- | :--- |
| Normalize custom losses to O(0.01–1.0) before adding to the total | Error #1 — FFT was O(1000) |
| Don't combine R1 with TTUR | Errors #2, #8 — double discriminator amplification |
| VGG perceptual loss: use relu4_2 or deeper, not relu2_2 | Error #3 — ringing artifacts |
| Apply auxiliary losses to translated outputs, not cycle-reconstructed | Errors #4, #25, #26 |
| ConvTranspose2d over bilinear ResizeConv | Error #7 — bilinear is a low-pass filter |
| Don't decay identity loss below ~1.5 | Error #6 — generator lost its anchor |
| Equal learning rates for CycleGAN | Error #8 — TTUR failed 3/3 times |
| Smoke tests need to check sub-component losses, not just total loss | Error #27 |
| Use sigmoid soft thresholds, not boolean, for differentiable masks | Error #5 |
| Keep validation set at ≥ 20% for stable FID computation | Errors #28, #30 |
