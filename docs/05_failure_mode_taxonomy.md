# What the Model Is Actually Doing

The training outputs three types of failure, each visible at a different stage. This document
explains what to look for in the sample grids, what the images show across training, and why
the model behaves this way given the loss landscape it was optimized under.

---

## Reading the Sample Grids

Each saved image is a 4-panel grid:

```python
grid = make_grid(
    torch.cat([sample_ct, fake_mri, sample_mri, fake_ct], dim=0),
    nrow=4, normalize=True
)
```

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│   Panel 1    │   Panel 2    │   Panel 3    │   Panel 4    │
│   Real CT    │  "Fake MRI"  │   Real MRI   │  "Fake CT"   │
│   (input)    │  G_A2B(CT)   │   (input)    │  G_B2A(MRI)  │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

Compare Panel 1 vs Panel 2 for CT→MRI quality. Compare Panel 3 vs Panel 4 for MRI→CT quality.
If translation is working, Panel 2 should look like Panel 3 (real MRI), and Panel 4 should look
like Panel 1 (real CT). If they look like the same-panel input, the model is copying.

---

## Visual Progression

#### Epoch 10 — Early training, global intensity shift
![Phase 4 Epoch 10 Grid](assets/phase4_epoch_010.png)
*Panel 2 is already structurally identical to Panel 1. Panel 4 is a recolored MRI.*

#### Epoch 100 — Input copying stabilized
![Phase 4 Epoch 100 Grid](assets/phase4_epoch_100.png)
*Panel 2 still retains the CT skull brightness. Panel 4 starts developing unmotivated bright patches.*

#### Epoch 200 (Alpha Hybrid) — Copying and mode collapse
![Phase 4 Epoch 200 Alpha Grid](assets/phase4_epoch_200.png)
*Panel 2: CT skull still visible. Panel 4: large bright blob artifacts with no anatomical correspondence.*

#### Epoch 200 (Beta Control) — Artifact confirmation
![Phase 4 Epoch 200 Beta Grid](assets/phase4_beta_epoch_200.png)
*Panel 4 shows even more pronounced cauliflower-like bright regions.*

---

## Failure 1: Input Copying (CT→MRI Direction)

The CT→MRI generator applies a global intensity shift and nothing else. Panel 2 always looks like
a brightness-adjusted version of Panel 1.

Why: Identity loss trains the generator to be an identity function:

```python
idt_A = G_B2A(real_A)
loss_idt_A = L1Loss(idt_A, real_A) * 5.0  # output must equal input
```

The network can't distinguish whether its input is a "real CT being passed through the identity
case" or a "translated MRI being passed through the cycle case" — it shares parameters for both.
The safe strategy is minimal modification in all cases.

Evidence: SSIM between Real CT and Fake MRI = 0.9959 at epoch 200. A real CT and a real MRI of the
same brain region have SSIM roughly 0.3–0.5. 0.9959 means the fake MRI is structurally
indistinguishable from the source CT — the intended transformation did not happen.

---

## Failure 2: Steganographic Hiding (Cycle Consistency Trap)

Cycle loss requires: `G_B2A(G_A2B(CT)) ≈ CT` with weight λ=10.0.

The model discovered it can satisfy this without translating: encode the original CT's full pixel
information into imperceptible high-frequency noise in the "Fake MRI," then decode it on the return
trip.

```
Real CT ──[G_A2B]──► "Fake MRI"  (visually: CT + brightness shift)
                         │
                         │  hidden: pixel-level noise encoding of original CT
                         │
                         ▼
                ──[G_B2A]──► Reconstructed CT ≈ Real CT  ✓ (cycle loss satisfied)
```

The "translation" doesn't need to look like an MRI — it just needs to contain enough hidden signal
for the return generator to reconstruct the original CT. Cycle SSIM ≈ 0.996 while the fake images
are visually CT-like confirms this is happening.

This behavior was documented by Chu et al. (2017) in "CycleGAN, a Master of Steganography."

---

## Failure 3: Blob Artifacts (MRI→CT Direction, Late Epochs)

By epoch 150–200, the MRI→CT generator produces large bright blob regions in the brain parenchyma
with no anatomical basis. These are not bone — they appear in the wrong location and wrong shape.

What happened: The discriminator's simplest signal for "real CT" is "bright high-attenuation regions
exist" (corresponding to skull bone). The generator learned this shortcut: generate bright regions
anywhere. It never learned that bone is a thin peripheral shell, not a diffuse internal pattern.

This gets worse as training continues because the learning rate decay (starting at epoch 100)
reduces the step size, locking the generator into this local minimum rather than escaping it.

---

## Why All of This Happened

| Failure | Weight pushing it | Observable result |
| :--- | :---: | :--- |
| Identity shortcut | λ_identity = 5.0 | Fake MRI looks like Real CT |
| Steganographic hiding | λ_cycle = 10.0 | Perfect cycle SSIM without visible translation |
| L1 oversmoothing | Combined 15.0 | SSIM near 1.0, images look plastic |
| Blob hallucination | Discriminator shortcut | Bright blobs in Fake CT |

Total gradient pressure to preserve the input: **15.0×**
Total gradient pressure to change modality: **1.0×**

The model is behaving rationally. It optimized exactly what the loss function asked it to.
The loss function asked for the wrong thing.
