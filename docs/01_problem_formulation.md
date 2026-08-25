# CT vs MRI Physics & Why Translation Is Hard

The most important thing to understand before looking at any model results is that CT and MRI
don't just look different — they measure fundamentally different physical properties of tissue.
That distinction is what makes CT-to-MRI translation so difficult and MRI-to-CT more tractable.

---

## How CT Works

CT measures **X-ray attenuation** — how much radiation tissue absorbs as an X-ray beam passes
through it. The result is quantified in Hounsfield Units (HU), a standardized scale:

| Tissue | HU Value | Appearance on CT |
| :--- | :---: | :--- |
| Air | -1000 | Black |
| Fat | -100 to -50 | Dark gray |
| Water | 0 | Gray |
| Soft tissue (brain, muscle) | +20 to +60 | Gray |
| Brain gray matter | +37 to +45 | Gray |
| Brain white matter | +20 to +30 | Gray |
| Bone (cancellous) | +300 to +500 | Light |
| Bone (cortical/skull) | +800 to +1900 | Bright white |

The critical thing to notice: **gray matter (+40 HU) and white matter (+25 HU) differ by only
15 HU** out of a 2000+ HU range. On a standard brain CT, the entire parenchyma looks like a
roughly uniform gray mass. Bone and air stand out clearly; soft tissue does not.

CT is good for: bone fractures, hemorrhage, calcifications, skull anatomy.  
CT is poor for: soft tissue differentiation, white matter lesions, ventricular detail.

---

## How MRI Works

MRI measures **T1 and T2 relaxation** of hydrogen protons in water and fat molecules. A strong
magnetic field (1.5T or 3T) aligns proton spins; RF pulses disturb that alignment, and the
recovery signal is measured.

The key difference from CT: there is no single fixed intensity scale. By varying the pulse
sequence parameters, the same scanner produces radically different images of the same anatomy:

| Sequence | CSF | Gray Matter | White Matter | Clinical use |
| :--- | :---: | :---: | :---: | :--- |
| T1-weighted | Dark | Gray | Light | Anatomy, post-contrast enhancement |
| T2-weighted | Bright | Gray | Dark | Edema, inflammation, lesions |
| FLAIR | Dark (suppressed) | Gray | Dark | Periventricular lesions, MS plaques |
| DWI | Variable | Variable | Variable | Acute stroke |

Gray and white matter have dramatically different T1/T2 relaxation times, so MRI produces
high-contrast boundaries between tissue types that are invisible on CT.

---

## The Core Problem: Information Asymmetry

This is why CT-to-MRI is genuinely hard and MRI-to-CT is more feasible:

**CT → MRI** means going from low-entropy to high-entropy. The CT shows brain parenchyma as
a near-uniform ~40 HU blob. The model must somehow synthesize gray/white matter boundaries,
CSF brightness, and tissue contrast that was never present in the source signal.
It also needs to decide *which MRI sequence* to produce — T1 and T2 weightings look completely
different, and CT carries no information about which is appropriate.
This is fundamentally a generative problem, not a translation problem.

**MRI → CT** means going from high-entropy to low-entropy. MRI already shows clear boundaries
between skull, brain, ventricles, and air. The task is to map these regions to their approximate
HU values — a compression rather than a generation. This is why pseudo-CT synthesis from MRI is
an active clinical research area (used in MRI-only radiotherapy planning).

| Direction | Type | Difficulty |
| :--- | :--- | :--- |
| CT → MRI | Generative — must create information that doesn't exist in source | Very hard |
| MRI → CT | Compressive — maps rich signal to a simpler scale | Feasible |

---

## Why CycleGAN Makes This Worse

CycleGAN treats both directions with the same architecture and loss weights, ignoring the
asymmetry entirely. The cycle consistency constraint (`G_B2A(G_A2B(CT)) ≈ CT` with λ=10.0)
means the CT→MRI generator literally cannot afford to lose the spatial information in the input —
it needs to preserve it for the round-trip to succeed. This pushes the generator toward copying
the input rather than transforming it.

The physics of CT→MRI says "generate new information". The cycle loss says "preserve all
existing information". These are directly contradictory requirements.

---

## References

- Johnstone et al. "Systematic Review of Synthetic CT Generation Methodologies for MRI-Only
  Radiation Therapy." *International Journal of Radiation Oncology* (2018).
- Han, X. "MR-based synthetic CT generation using a deep convolutional neural network."
  *Medical Physics* (2017).
- Zhu et al. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks."
  *ICCV* (2017).
