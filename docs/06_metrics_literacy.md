# Metrics: What They Actually Measure and Where They Lie

Standard image quality metrics compare two images and return a similarity score. The problem in
image *translation* is that we're not asking "are these two images similar?" — we're asking "did
the model actually change the modality?" Those are different questions, and standard metrics answer
the first one while we need the second.

This caused significant confusion during this project. A SSIM of 0.9959 at epoch 200 sounds like
the model is working well. It actually means the model output is 99.6% structurally identical to
the input — which is exactly the failure we were trying to avoid.

---

## SSIM (Structural Similarity Index)

SSIM compares two images across luminance (mean intensity), contrast (variance), and structural
correlation:

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

**Why it rewards input copying:** If the model outputs a brightness-shifted copy of the input,
all three components stay high — the means are similar, the variances are similar, and the spatial
patterns are identical. SSIM cannot distinguish between "the model preserved anatomy while changing
modality" and "the model just copied the input."

**Why it rewards blurry outputs:** L1 loss minimizes pixel error by outputting the mean of all
plausible predictions. That mean is blurry, but it's close to the input in SSIM terms. A
Gaussian-blurred image typically achieves SSIM > 0.90 against the original — SSIM does not
penalize loss of fine detail as aggressively as human perception does.

For cross-modality translation, a SSIM of 0.95–1.00 between source and output should raise
suspicion rather than confidence. If the model genuinely translated CT to MRI, the output should
look different — the skull brightness changes, tissue contrast changes. A score near 1.0 means
the output looks like the input, which means translation didn't happen.

A realistic expectation for successful CT↔MRI translation would be SSIM roughly in the 0.70–0.85
range — enough structural preservation to maintain anatomy, enough change to indicate a genuine
domain shift.

---

## MAE (Mean Absolute Error)

$$\text{MAE} = \frac{1}{N} \sum_{i} |x_i - y_i|$$

MAE_A = 0.0019 at epoch 200. Near-zero pixel error is the expected result when the model is
copying the input. MAE shares SSIM's problem — it measures how close two images are, not whether
the right transformation occurred.

It also doesn't capture perceptual quality: a uniform brightness shift produces low MAE, a sharp
edge displaced by 1 pixel produces moderate MAE, but both can be completely unacceptable visually.

---

## FID (Fréchet Inception Distance)

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$$

FID is more honest for translation tasks because it compares *distributions* rather than image pairs.
A model that copies CT inputs will produce a generated distribution that looks like CT, not MRI —
and FID will catch this even if SSIM looks good.

FID_A = 183.71 despite SSIM = 0.9959. The generated "MRI" images don't match the real MRI
distribution because they are, structurally, CTs.

Rough interpretation for reference:

| FID | What it suggests |
| :---: | :--- |
| < 50 | Generated distribution is close to real |
| 50 – 150 | Noticeable quality gap |
| 150 – 250 | Significant distributional mismatch |
| > 250 | Generated images look like a different domain entirely |

Limitations worth knowing: FID variance is high with small validation sets (< 50 images). The
Inception-v3 features it uses were trained on ImageNet natural images, so they may not capture
the right quality features for medical images specifically.

---

## Dice Score

$$\text{Dice} = \frac{2 |A \cap B|}{|A| + |B|}$$

Dice measures overlap between segmented anatomical regions. We used intensity thresholding to
approximate masks (pixels > 0.5 for bone, -0.3 to 0.5 for soft tissue). This has two problems:

First, hard thresholds (x > 0.5) have zero gradient — the generator can't learn from a Dice loss
built on binary masks. This was Error #5 in the catalog.

Second, a high Dice score between a fake MRI and the source CT just means the thresholded regions
match — which is guaranteed if the model is copying the input. Like SSIM, high Dice in a
translation context can be evidence of failure rather than success.

---

## FFT Power Ratio

$$\text{FFT Ratio} = \frac{\sum_{r > r_{\text{cutoff}}} |F(x)|^2}{\sum_{\text{all}} |F(x)|^2}$$

Measures the fraction of image spectral energy in high frequencies. Useful for detecting GAN
artifacts: ConvTranspose2d and bilinear upsampling can produce regular high-frequency patterns
(checkerboard or ringing) that are visible in the Fourier domain.

FFT ratios stayed low throughout training (0.001–0.003), which confirmed there were no checkerboard
artifacts. It also reflects that the model wasn't generating new high-frequency content — it was
preserving the input's natural frequency distribution.

---

## The Main Lesson

No single metric tells you whether translation succeeded. The combination that caught the failure
here was:

- **SSIM** very high → model output is structurally close to the input (suspicious)
- **FID** elevated → generated distribution doesn't match the target domain (confirms failure)
- **Visual inspection** → fake MRI has CT skull brightness, confirming input copying

Patterns to watch for:

| SSIM | FID | Likely diagnosis |
| :--- | :--- | :--- |
| Very high + rising | Elevated | Input copying |
| Moderate | Low | Actual translation (verify anatomy separately) |
| High | Low | Oversmoothing — blurry but realistic distribution |
| High | Rising | Progressive oversmoothing as training continues |
