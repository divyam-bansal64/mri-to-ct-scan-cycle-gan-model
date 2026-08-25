# Problem Formulation: CT vs MRI Physics & Information Asymmetry

> This document explains *why* cross-modality medical image translation between CT and MRI 
> is fundamentally challenging, grounded in imaging physics rather than model architecture.

---

## 1. How CT Imaging Works

**Computed Tomography (CT)** measures the **X-ray attenuation coefficient** of tissue. An X-ray beam passes through the body, and detectors measure how much radiation was absorbed along each path. Mathematical reconstruction (filtered back-projection or iterative methods) produces a volumetric image.

### Hounsfield Units (HU)

CT images are quantified in **Hounsfield Units**, a standardized linear scale:

| Tissue | HU Value | Appearance |
| :--- | :---: | :--- |
| Air | -1000 | Black |
| Fat | -100 to -50 | Dark gray |
| Water | 0 | Gray |
| Soft tissue (muscle, organs) | +20 to +60 | Gray |
| **Brain gray matter** | **+37 to +45** | **Gray** |
| **Brain white matter** | **+20 to +30** | **Gray** |
| Bone (cancellous) | +300 to +500 | Light |
| Bone (cortical/skull) | +800 to +1900 | Bright white |

### Key Property: Low Soft-Tissue Contrast

Brain gray matter (~40 HU) and white matter (~25 HU) differ by only **~15 HU** out of a 2000+ HU dynamic range. In a standard CT image, the brain parenchyma appears as a **nearly uniform gray mass** — the intricate folding patterns of gyri and sulci, the ventricle boundaries, and white matter tracts are barely distinguishable.

**CT excels at**: Bone fractures, calcifications, hemorrhage (acute blood = 50-70 HU), and air/fluid boundaries.

**CT struggles with**: Differentiating normal brain tissue types, detecting subtle white matter lesions, visualizing nerve tracts.

---

## 2. How MRI Imaging Works

**Magnetic Resonance Imaging (MRI)** exploits the **quantum spin properties of hydrogen protons** (¹H) in water and fat molecules. A strong external magnetic field (1.5T or 3T) aligns proton spins. Radiofrequency (RF) pulses perturb this alignment, and the recovery signal is measured.

### Relaxation Parameters

| Parameter | What It Measures | Effect on Image |
| :--- | :--- | :--- |
| **T1 (spin-lattice relaxation)** | How quickly protons realign with the magnetic field | Controls tissue brightness in T1-weighted images |
| **T2 (spin-spin relaxation)** | How quickly the transverse magnetization decays | Controls tissue brightness in T2-weighted images |
| **Proton density** | Concentration of hydrogen atoms | Contributes to overall signal intensity |

### Multiple Contrast Weightings

Unlike CT (one physical quantity → one image), MRI can produce **many different images** of the same anatomy by varying pulse sequence parameters:

| Sequence | CSF | Gray Matter | White Matter | Fat | Clinical Use |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **T1-weighted** | Dark | Gray | Light | Bright | Anatomy, post-gadolinium enhancement |
| **T2-weighted** | Bright | Gray | Dark | Gray | Edema, inflammation, lesions |
| **FLAIR** | Dark (suppressed) | Gray | Dark | Gray | Periventricular lesions, MS plaques |
| **DWI** | Variable | Variable | Variable | Variable | Acute stroke (restricted diffusion) |

### Key Property: Rich Soft-Tissue Contrast

Brain gray matter and white matter have dramatically different T1 and T2 relaxation times, producing **high-contrast boundaries** between tissue types. Ventricles, sulci, basal ganglia, and white matter tracts are all clearly delineated.

---

## 3. The Information Asymmetry Problem

### CT → MRI: A One-to-Many Ill-Posed Problem

```
CT Scan (Single Measurement)
    │
    │  X-ray attenuation: Brain ≈ 40 HU (uniform gray)
    │
    ▼  Which MRI contrast should the model produce?
    │
    ├── T1-weighted? (WM bright, GM gray, CSF dark)
    ├── T2-weighted? (WM dark, GM gray, CSF bright)  
    ├── FLAIR?       (CSF suppressed, lesions bright)
    └── DWI?         (Diffusion-weighted)
```

**The fundamental issue**: CT measures electron density. Brain soft tissue (gray matter, white matter, CSF) all have similar electron densities (~40 HU ± 20). But these same tissues have **dramatically different** T1/T2 relaxation times in MRI.

A model attempting CT→MRI must **synthesize tissue contrast information that does not exist in the CT attenuation data**. This is not a translation problem — it's a **generative hallucination** problem. The model must decide:
- Where are the gray/white matter boundaries? (CT can't tell you)
- How bright should the CSF be? (Depends on T1 vs T2 weighting — CT doesn't know)
- Are there any white matter lesions? (Invisible on CT, visible on FLAIR MRI)

### MRI → CT: A Better-Posed Problem

```
MRI Scan (Rich Tissue Information)
    │
    │  T1/T2 relaxation: Detailed tissue boundaries
    │  Skull/bone: Dark (fast T2 decay, low signal)
    │  Soft tissue: Rich contrast (GM, WM, CSF distinguished)
    │
    ▼  Map to single CT contrast
    │
    └── CT: Bone = bright, Soft tissue ≈ uniform gray, Air = black
```

MRI provides **more information than CT needs**. The skull boundary is visible in MRI (dark region). Soft tissue geometry is well-defined. The model needs to map these structures to their approximate Hounsfield Unit values — a relatively deterministic, many-to-one compression.

This is why **pseudo-CT generation from MRI** is an active clinical research area (used in MRI-only radiotherapy planning), while CT→MRI synthesis remains largely unsolved.

### Summary: Information Flow Direction

| Direction | Information Flow | Problem Type | Difficulty |
| :--- | :--- | :--- | :---: |
| **CT → MRI** | Low-entropy → High-entropy | **Generative** (must create information) | 🔴 Very Hard |
| **MRI → CT** | High-entropy → Low-entropy | **Compressive** (must select/average information) | 🟢 Feasible |

---

## 4. Implications for CycleGAN Training

CycleGAN treats both directions symmetrically — same architecture, same loss weights, same training dynamics. But the physics is asymmetric:

1. **G_A2B (CT→MRI)** is asked to generate rich tissue contrast from a nearly uniform input. Without the physical information to guide it, the generator defaults to the **easiest solution**: apply a global intensity transformation and preserve the input structure. This explains the "input copying" behavior observed in our experiments.

2. **G_B2A (MRI→CT)** has a feasible but still challenging task. It must learn to brighten skull regions and flatten soft-tissue contrast. Our models partially learned this mapping but developed bright artifact blobs rather than learning true attenuation physics.

3. **Cycle consistency makes the asymmetry worse**: Because $G_{B2A}(G_{A2B}(CT)) \approx CT$ must hold with high weight (λ=10.0), the CT→MRI generator cannot afford to lose any spatial information from the input CT. This forces it into the steganographic hiding strategy rather than genuine translation.

---

## References

- Johnstone, E., et al. "Systematic Review of Synthetic Computed Tomography Generation Methodologies for Use in Magnetic Resonance Imaging–Only Radiation Therapy." *International Journal of Radiation Oncology* (2018).
- Han, X. "MR-based synthetic CT generation using a deep convolutional neural network method." *Medical Physics* (2017).
- Zhu, J.-Y., et al. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." *ICCV* (2017).
