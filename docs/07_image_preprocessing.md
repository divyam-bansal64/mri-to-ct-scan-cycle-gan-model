# Image Preprocessing for Medical Images

> Best practices for handling medical imaging data (CT and MRI) in deep learning pipelines, 
> derived from practical experience during this project.

---

## 1. Understanding Medical Image Formats

### DICOM (Real Clinical Data)

Clinical CT and MRI data is stored in **DICOM** (.dcm) format, which includes:
- **Pixel data**: Raw scanner output (often 12-bit or 16-bit integers)
- **Metadata**: Patient info, scan parameters, spatial calibration
- **Rescale parameters**: `RescaleSlope` and `RescaleIntercept` for converting raw values to Hounsfield Units

```python
import pydicom

ds = pydicom.dcmread("scan.dcm")
raw_pixels = ds.pixel_array                        # uint16 array
hu_values = raw_pixels * ds.RescaleSlope + ds.RescaleIntercept  # Convert to HU
```

### PNG/JPEG (Research Datasets)

Research datasets (like the one used in this project) often provide pre-processed images as PNG or JPEG:
- Already converted from DICOM
- Intensity range mapped to [0, 255] (8-bit)
- **Original HU/intensity scale is lost** — this limits what preprocessing we can do

---

## 2. CT Windowing

### Why Windowing Matters

CT images span a huge dynamic range: air (-1000 HU) to cortical bone (+1900 HU). Displaying this entire range on an 8-bit display compresses soft tissue differences into a tiny gray band.

**Windowing** selects a clinically relevant intensity range and maps it to the full display range:

```python
def apply_window(image_hu, center, width):
    """Apply CT window to Hounsfield Unit image."""
    lower = center - width / 2
    upper = center + width / 2
    windowed = np.clip(image_hu, lower, upper)
    windowed = (windowed - lower) / (upper - lower)  # Normalize to [0, 1]
    return windowed
```

### Common Brain CT Windows

| Window | Center (HU) | Width (HU) | Visualizes |
| :--- | :---: | :---: | :--- |
| **Brain** | +40 | 80 | Soft tissue (gray matter, white matter, edema) |
| **Bone** | +400 | 1800 | Skull, facial bones, calcifications |
| **Subdural** | +75 | 215 | Subdural hematomas, near-bone pathology |
| **Stroke** | +40 | 40 | Subtle early ischemic changes |

### Impact on Model Training

Without windowing (using raw full-range normalization):
```
Air (-1000) ─────────── Brain (40) ──── Bone (1000+)
    │                      │                │
    ▼                      ▼                ▼
  0.000                  0.520            1.000
```

Brain soft tissue occupies only ~2% of the normalized range (0.48 to 0.52). The model sees brain parenchyma as a nearly uniform gray band, making it impossible to learn tissue differentiation.

With brain windowing (center=40, width=80):
```
Lower (0 HU) ──── Gray matter (37-45 HU) ──── White matter (20-30 HU) ──── Upper (80 HU)
      │                    │                           │                        │
      ▼                    ▼                           ▼                        ▼
    0.000                0.462-0.562                0.250-0.375               1.000
```

Now gray/white matter differences span ~20% of the normalized range — learnable by the model.

**Lesson learned**: The dataset used in this project provided pre-normalized PNG images without the ability to apply custom windowing. For future work with DICOM data, windowing should be the **first preprocessing step**.

---

## 3. MRI Intensity Normalization

### The Problem: MRI Has No Standardized Intensity Scale

Unlike CT (where water = 0 HU globally), MRI intensities are **arbitrary**:
- Different scanners produce different intensity ranges
- Different coils produce spatial intensity bias
- The same tissue can have different values across patients and sessions

### Normalization Approaches

#### Simple Min-Max Normalization
```python
# Map to [0, 1]
image = (image - image.min()) / (image.max() - image.min())
```
**Problem**: Sensitive to outliers. One hot pixel can compress the entire useful range.

#### Percentile Clipping + Normalization
```python
# Clip extreme values, then normalize
p1, p99 = np.percentile(image, [1, 99])
image = np.clip(image, p1, p99)
image = (image - p1) / (p99 - p1)
```
**Better**: Robust to outliers. Standard practice in medical image AI.

#### Z-Score Normalization (Per-Patient)
```python
# Normalize to zero mean, unit variance
brain_mask = image > threshold  # Only normalize within brain
mean = image[brain_mask].mean()
std = image[brain_mask].std()
image = (image - mean) / std
```
**Best for training**: Makes intensity distributions comparable across patients. Requires brain masking to avoid including background pixels.

### What We Used

```python
transforms.Normalize((0.5,), (0.5,))  # Maps [0,1] → [-1,1]
```

This is the simplest approach and is standard for CycleGAN. The images were already pre-processed PNGs in [0, 255], converted to [0, 1] by `ToTensor()`, then mapped to [-1, 1].

**Limitation**: This doesn't account for inter-patient intensity variation or the different dynamic ranges of CT vs MRI.

---

## 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)

### What It Does

CLAHE enhances local contrast by:
1. Dividing the image into small tiles (e.g., 8×8)
2. Computing a histogram for each tile
3. Clipping the histogram at a threshold (preventing over-amplification of noise)
4. Redistributing the clipped pixels uniformly
5. Using bilinear interpolation between tiles to avoid border artifacts

```python
import cv2

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(image_uint8)
```

### Why It Helps Medical Images

Medical images often have low global contrast but important **local** contrast differences. CLAHE boosts these local differences without creating global intensity distortions:

- **Brain CT**: Makes subtle gray/white matter boundaries visible
- **MRI**: Enhances contrast in regions with similar signal intensity
- **Post-processing**: Can be applied to generated images to make subtle translations visible

### When to Apply

- **Before training**: If the input images have poor contrast (apply to both domains consistently)
- **For visualization only**: Apply to generated samples for display purposes without affecting training
- **Not during validation**: Augmentation should not affect evaluation metrics

---

## 5. Data Augmentation for Medical Images

### Safe Augmentations

| Augmentation | Rationale | Parameters Used |
| :--- | :--- | :--- |
| **Horizontal flip** | Brain is approximately bilaterally symmetric | p=0.5 |
| **Small rotation** | Patients' heads are rarely perfectly aligned | ±5° |
| **Small scale** | Head size varies across patients | 0.9–1.1× |

### Dangerous Augmentations (Avoided)

| Augmentation | Why It's Dangerous |
| :--- | :--- |
| **Vertical flip** | Brain anatomy is NOT vertically symmetric |
| **Large rotation (> 15°)** | Produces unrealistic head positions |
| **Elastic deformation** | Brain has rigid skull — elastic transforms create impossible anatomy |
| **Color jitter** | Grayscale modality — meaningless |
| **Random erasing/cutout** | Removes diagnostically relevant regions |
| **Mixup/CutMix** | Blends different patients — anatomically nonsensical |

### Critical: Separate Train/Val Transforms

```python
# ✅ Correct: augmentation only on training data
transform_train = Compose([Resize(256), Flip, Rotate, Scale, ToTensor, Normalize])
transform_val   = Compose([Resize(256), ToTensor, Normalize])

# ❌ Wrong: same augmented transform for validation
# This makes validation metrics noisy and non-reproducible
```

---

## 6. Dataset Splitting for Unpaired Translation

### The Unpaired Challenge

In CycleGAN, there is no correspondence between CT and MRI images — they come from different patients. The "train/val split" must be done **independently** for each domain:

```python
# Deterministic 80/20 split with fixed seed
gen = torch.Generator().manual_seed(42)
ct_train, ct_val = random_split(ct_dataset, [0.8, 0.2], generator=gen)
mri_train, mri_val = random_split(mri_dataset, [0.8, 0.2], generator=gen)
```

### Important Considerations

1. **No data leakage**: If multiple slices come from the same patient, all slices from one patient must be in either train or val (not split across both)
2. **Balanced splits**: Ensure both train and val contain representative slice positions (not all basal ganglia slices in train and all vertex slices in val)
3. **Reproducible**: Use a fixed random seed so the same split is used across all experiment runs
