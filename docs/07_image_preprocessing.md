# Image Preprocessing Notes

Practical notes on handling CT and MRI data in a deep learning pipeline, based on what
was done in this project and what should have been done differently.

---

## Medical Image Formats

**Clinical data uses DICOM (.dcm)**, which stores raw 12-bit or 16-bit scanner output alongside
metadata about scan parameters, spatial calibration, and rescale coefficients:

```python
import pydicom

ds = pydicom.dcmread("scan.dcm")
raw_pixels = ds.pixel_array                                         # uint16 raw values
hu_values = raw_pixels * ds.RescaleSlope + ds.RescaleIntercept     # convert to Hounsfield Units
```

**Research datasets typically provide PNGs** already converted from DICOM. This dataset (Kaggle)
provided 8-bit PNGs with the original HU scale lost. That's a significant limitation — it means
you can't apply CT windowing, which would have meaningfully improved what the model could learn.

---

## CT Windowing

CT spans a huge dynamic range — air is -1000 HU, cortical bone is +1900 HU. Without windowing,
brain soft tissue occupies roughly 2% of the normalized intensity range (about 0.48 to 0.52).
Everything looks uniform gray and the model has essentially no signal to learn tissue differences from.

**Windowing** clips to a clinically relevant range and stretches it to fill the display:

```python
def apply_window(image_hu, center, width):
    lower = center - width / 2
    upper = center + width / 2
    windowed = np.clip(image_hu, lower, upper)
    return (windowed - lower) / (upper - lower)
```

Common brain windows:

| Window | Center (HU) | Width (HU) | Shows |
| :--- | :---: | :---: | :--- |
| Brain | +40 | 80 | Soft tissue — gray/white matter, edema |
| Bone | +400 | 1800 | Skull, calcifications |
| Subdural | +75 | 215 | Near-bone pathology, hematomas |
| Stroke | +40 | 40 | Subtle early ischemic changes |

With brain windowing, gray/white matter contrast spans ~20% of the normalized range rather than
2%, which is actually learnable. For any future work using DICOM data directly, windowing should
happen before anything else.

---

## MRI Normalization

CT has a universal intensity scale (water = 0 HU, air = -1000 HU). MRI does not — intensities are
arbitrary and vary between scanners, coils, and even scan sessions for the same patient. This needs
to be handled explicitly.

Three common approaches:

**Simple min-max:** Scales to [0, 1] based on image extremes. Fast, but a single outlier pixel
crushes the useful contrast into a narrow band.

```python
image = (image - image.min()) / (image.max() - image.min())
```

**Percentile clipping + normalization:** More robust. Clip extremes first, then scale.
Standard practice in medical image AI.

```python
p1, p99 = np.percentile(image, [1, 99])
image = np.clip(image, p1, p99)
image = (image - p1) / (p99 - p1)
```

**Z-score within brain mask:** Normalizes to zero mean, unit variance, computed only inside
the brain (excludes background). Makes intensity distributions comparable across patients.

```python
brain_mask = image > threshold
mean = image[brain_mask].mean()
std  = image[brain_mask].std()
image = (image - mean) / std
```

What was used in this project:

```python
transforms.Normalize((0.5,), (0.5,))  # maps [0,1] → [-1,1]
```

Standard for CycleGAN, simple, but doesn't account for inter-patient variation or the different
dynamic ranges of CT vs MRI.

---

## CLAHE

CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances local contrast without
amplifying noise. It divides the image into tiles (e.g. 8×8), equalizes each tile's histogram
independently, then blends tiles with bilinear interpolation to avoid boundary artifacts.

```python
import cv2
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(image_uint8)
```

Useful for making subtle gray/white matter boundaries visible in CT, or boosting local contrast in
MRI regions with similar signal intensity. Can be applied to generated samples for display purposes.

Don't apply CLAHE to validation images — it's a random-state operation that makes metrics
non-reproducible.

---

## Augmentation

What was used:

```python
transforms.RandomHorizontalFlip(p=0.5)              # brain is bilaterally symmetric
transforms.RandomRotation(degrees=5)                  # slight head tilt
transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)) # head size variation
```

What was avoided and why:

| Augmentation | Why avoided |
| :--- | :--- |
| Vertical flip | Brain is not vertically symmetric |
| Large rotation (> 15°) | Produces anatomically unrealistic head positions |
| Elastic deformation | Brain has a rigid skull — elastic warps create impossible geometry |
| Color jitter | Grayscale images — meaningless |
| Random erasing / cutout | Removes diagnostically relevant regions |
| Mixup / CutMix | Blends anatomy from different patients — nonsensical |

Validation transforms must be deterministic — no randomness. Applying augmentation to validation
makes metrics noisy and non-reproducible across runs.

---

## Dataset Splitting for Unpaired Translation

In CycleGAN, CT and MRI come from different patients with no correspondence. The split is
done independently for each domain:

```python
gen = torch.Generator().manual_seed(42)  # fixed seed for reproducibility
ct_train,  ct_val  = random_split(ct_dataset,  [0.8, 0.2], generator=gen)
mri_train, mri_val = random_split(mri_dataset, [0.8, 0.2], generator=gen)
```

If multiple slices come from the same patient, all slices from that patient need to stay in the
same split — mixing them between train and val is data leakage. Also check that slice positions
are representative in both splits (you don't want all basal ganglia slices in train and all
vertex slices in validation).
