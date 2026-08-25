# Metrics Literacy: Why High SSIM ≠ Good Translation

> Understanding what evaluation metrics actually measure, their failure modes, and how 
> to interpret them critically in image-to-image translation tasks.

---

## 1. The Fundamental Problem: Metrics Measure Similarity, Not Translation

All standard image quality metrics compare **two images** and output a scalar "similarity" score. But in image translation, we're evaluating a **transformation** — we want to know:

1. Did the output change modality? (CT → MRI)
2. Did it preserve anatomy?
3. Does it look realistic?

No single metric captures all three. Worse, metrics can be **actively misleading** when the model learns shortcuts.

---

## 2. SSIM (Structural Similarity Index)

### What It Measures

SSIM compares two images across three components:

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

| Component | What It Captures |
| :--- | :--- |
| **Luminance** ($\mu$) | Mean pixel intensity similarity |
| **Contrast** ($\sigma$) | Standard deviation similarity |
| **Structure** ($\sigma_{xy}$) | Correlation between pixel patterns |

### Why SSIM Rewards Input Copying

If a model outputs a slightly adjusted copy of the input:
- **Luminance**: Nearly identical means → high score
- **Contrast**: Nearly identical variance → high score
- **Structure**: Identical spatial patterns → near-perfect score

**In our experiments**: SSIM_A = 0.9959 at epoch 200. This means the "Fake MRI" and the Real CT have 99.6% structural similarity. For a model that's supposed to *change* the image from one modality to another, this score is evidence of **failure**, not success.

### The "SSIM Sweet Spot" for Translation

| SSIM Range | Interpretation for Image Translation |
| :--- | :--- |
| **0.95 – 1.00** | 🔴 **Suspicious**: Model is likely copying the input |
| **0.80 – 0.95** | 🟡 Model preserves structure with some transformation |
| **0.65 – 0.80** | 🟢 Meaningful domain shift while preserving anatomy |
| **< 0.65** | 🔴 Excessive distortion — anatomy may be corrupted |

For cross-modality medical translation, SSIM ~0.75–0.85 between source and translated output is expected if genuine translation occurs, because the intensity distributions of CT and MRI are fundamentally different.

### SSIM Rewards Blurriness

L1 loss minimizes pixel error. The safest strategy under uncertainty is outputting the **mean prediction** — a smooth, blurry average. This average image has:
- High SSIM (close to the input and to the mean of all targets)
- High perceptual distance from any real image (blurry ≠ realistic)

A completely blurred (Gaussian-smoothed) version of an image typically achieves SSIM > 0.90 against the original. SSIM does not penalize loss of fine detail as aggressively as human perception does.

---

## 3. MAE (Mean Absolute Error)

### What It Measures

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |x_i - y_i|$$

The average pixel-wise absolute difference. Lower is better.

### Limitations

- **Sensitive to global intensity shifts**: If one image is uniformly brighter by 0.1, MAE increases proportionally regardless of structural quality.
- **Not perceptually meaningful**: A 1-pixel shift of a sharp edge has very low MAE but is visually noticeable. A uniform blurring has moderate MAE but is very noticeable.
- **Same problem as SSIM**: Rewards the mean prediction. A blurry average image has lower MAE than any specific sharp prediction.

### In Our Experiments

MAE_A = 0.0019 at epoch 200 — near-zero error. This confirms the model is reproducing the input nearly exactly, not translating.

---

## 4. FID (Fréchet Inception Distance)

### What It Measures

FID compares the **distribution of generated images** to the **distribution of real images** in Inception-v3 feature space:

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$$

Where $(\mu_r, \Sigma_r)$ are the mean and covariance of real image features, and $(\mu_g, \Sigma_g)$ are for generated images.

### Why FID Is More Honest

FID doesn't compare individual image pairs — it compares **distributions**. A model that copies inputs will have a generated distribution that looks like the source domain (CT), not the target domain (MRI). FID will penalize this.

| FID Range | Interpretation |
| :--- | :--- |
| **< 50** | Excellent — generated images are nearly indistinguishable from real |
| **50 – 150** | Good — realistic with noticeable differences |
| **150 – 250** | Moderate — visible quality gaps |
| **> 250** | Poor — significant distributional difference |

### In Our Experiments

FID_A = 183.71 (Alpha, epoch 200). Despite SSIM = 0.9959, the FID shows that the generated "MRI" images **do not look like real MRIs**. They look like intensity-shifted CTs, which is exactly what they are.

### FID Limitations

- **Sample size sensitivity**: FID variance increases dramatically with small sample sizes (< 50 images). Our validation set was limited.
- **Inception bias**: Inception-v3 was trained on ImageNet (natural images). Its features may not capture medical image-specific quality differences.
- **Mean-only comparison**: FID compares distribution means and covariances but not higher-order statistics. Two distributions can have identical mean/covariance but look very different.

---

## 5. Dice Score (Anatomical Overlap)

### What It Measures

$$\text{Dice} = \frac{2 |A \cap B|}{|A| + |B|}$$

Measures the overlap between segmented anatomical regions (bone, soft tissue, ventricles) in the generated and real images. Requires binarization/thresholding.

### In Our Context

We used intensity-based thresholding to create approximate anatomical masks:
- **Bone mask**: pixels > 0.5 (normalized intensity)
- **Soft tissue mask**: pixels between -0.3 and 0.5

### Limitations

- **Threshold sensitivity**: Hard thresholds (x > 0.5) create binary masks with zero gradient — the generator cannot learn from Dice if used as a training loss (Error #5 in our catalog)
- **Proxy for real segmentation**: Intensity thresholding is a crude approximation. Real anatomical segmentation requires trained segmentation models.
- **Domain-dependent**: A Dice score of 0.95 between a "Fake MRI" and the source CT means the thresholded regions match — which is expected if the model is copying the input.

---

## 6. FFT Power Ratio

### What It Measures

The ratio of high-frequency spectral power to total spectral power in the generated image:

$$\text{FFT Ratio} = \frac{\sum_{r > r_{\text{cutoff}}} |F(x)|^2}{\sum_{\text{all}} |F(x)|^2}$$

Where $F(x)$ is the 2D Fourier transform and $r_{\text{cutoff}}$ is half the maximum frequency.

### Why It Matters for GANs

GAN generators (especially those using ConvTranspose2d or bilinear upsampling) can produce **checkerboard artifacts** — regular high-frequency patterns visible in the Fourier domain. The FFT ratio detects these:

| FFT Ratio | Interpretation |
| :--- | :--- |
| **< 0.005** | ✅ Clean — no detectable artifacts |
| **0.005 – 0.015** | ⚠️ Mild high-frequency excess |
| **> 0.015** | 🔴 Artifact alert — visible ringing/checkerboard |

### In Our Experiments

FFT ratios stayed low (0.001–0.003) throughout training. This confirms no checkerboard artifacts — but also reflects that the model isn't generating new high-frequency content (it's copying the input, which has natural frequency distribution).

---

## 7. The Lesson: Use Metrics as a Bundle, Not Individually

| Metric | What It Catches | What It Misses |
| :--- | :--- | :--- |
| **SSIM** | Structural distortion | Input copying, blurriness |
| **MAE** | Large pixel errors | Perceptual quality |
| **FID** | Distribution mismatch | Individual image quality |
| **Dice** | Anatomical region shifts | Texture/contrast changes |
| **FFT** | Frequency artifacts | Spatial artifacts, copying |

### Red Flag Combinations

| Pattern | Diagnosis |
| :--- | :--- |
| High SSIM + High FID | **Input copying** — model preserves structure but doesn't change domain |
| High SSIM + Low MAE + High FID | **Oversmoothing** — blurry copies that are pixel-accurate but not realistic |
| Low SSIM + Low FID | **Actual translation** (but verify anatomy preservation separately) |
| Increasing SSIM + Increasing FID | **Progressive oversmoothing** — model getting blurrier over epochs |
