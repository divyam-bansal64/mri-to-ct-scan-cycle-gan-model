from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

OUTPUT_PATH = r"E:\code\mri to cti\MODEL_DOCUMENTATION.pdf"

# ─────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────

styles = getSampleStyleSheet()

def make_style(name, parent="Normal", fontSize=10, leading=14, spaceBefore=4,
               spaceAfter=4, textColor=colors.black, bold=False, italic=False,
               alignment=TA_JUSTIFY, leftIndent=0):
    return ParagraphStyle(
        name=name,
        parent=styles[parent],
        fontSize=fontSize,
        leading=leading,
        spaceBefore=spaceBefore,
        spaceAfter=spaceAfter,
        textColor=textColor,
        fontName=("Helvetica-BoldOblique" if bold and italic else
                  "Helvetica-Bold" if bold else
                  "Helvetica-Oblique" if italic else "Helvetica"),
        alignment=alignment,
        leftIndent=leftIndent,
    )

title_style     = make_style("DocTitle",    fontSize=22, leading=28, bold=True,
                              textColor=colors.HexColor("#1a1a2e"), alignment=TA_CENTER,
                              spaceBefore=10, spaceAfter=6)
subtitle_style  = make_style("DocSubtitle", fontSize=12, leading=16, italic=True,
                              textColor=colors.HexColor("#444444"), alignment=TA_CENTER,
                              spaceAfter=20)
h1_style        = make_style("H1", fontSize=16, leading=20, bold=True,
                              textColor=colors.HexColor("#1a1a2e"), spaceBefore=18, spaceAfter=6)
h2_style        = make_style("H2", fontSize=13, leading=17, bold=True,
                              textColor=colors.HexColor("#2d4a7a"), spaceBefore=14, spaceAfter=4)
h3_style        = make_style("H3", fontSize=11, leading=15, bold=True,
                              textColor=colors.HexColor("#3a5a8a"), spaceBefore=10, spaceAfter=3)
body_style      = make_style("Body", fontSize=10, leading=15, spaceAfter=6)
code_style      = ParagraphStyle(
    "Code", parent=styles["Normal"], fontSize=8.5, leading=13, spaceAfter=4,
    textColor=colors.HexColor("#1a1a1a"), leftIndent=20, fontName="Courier",
    alignment=TA_LEFT,
)
bullet_style    = make_style("Bullet", fontSize=10, leading=14, leftIndent=20,
                              spaceAfter=3)
note_style      = make_style("Note", fontSize=9.5, leading=13, italic=True,
                              textColor=colors.HexColor("#555555"), leftIndent=20, spaceAfter=4)
error_style     = make_style("Error", fontSize=10, leading=14, bold=True,
                              textColor=colors.HexColor("#c0392b"), spaceAfter=2)
fix_style       = make_style("Fix", fontSize=10, leading=14, bold=True,
                              textColor=colors.HexColor("#27ae60"), spaceAfter=2)

def P(text, style=body_style):
    return Paragraph(text, style)

def H1(text): return P(text, h1_style)
def H2(text): return P(text, h2_style)
def H3(text): return P(text, h3_style)
def Body(text): return P(text, body_style)
def Code(text): return P(text.replace(" ", "&nbsp;").replace("\n", "<br/>"), code_style)
def Bullet(text): return P(f"&bull;&nbsp;&nbsp;{text}", bullet_style)
def Note(text): return P(f"<i>Note: {text}</i>", note_style)
def SP(n=6): return Spacer(1, n)
def HR(): return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"),
                             spaceAfter=6, spaceBefore=6)

def make_table(data, col_widths=None, header=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d4a7a")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f4f7fb"), colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ALIGN",      (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]
    t.setStyle(TableStyle(style))
    return t


# ─────────────────────────────────────────────────────────────
# DOCUMENT CONTENT
# ─────────────────────────────────────────────────────────────

content = []

# ── COVER ────────────────────────────────────────────────────
content += [
    SP(60),
    P("MRI ↔ CT CycleGAN", title_style),
    P("Complete Model Documentation", subtitle_style),
    HR(),
    SP(10),
    P("Full-scale bidirectional brain scan translation using unpaired CycleGAN.<br/>"
      "Trained on Kaggle T4 GPU — 150 epochs — 256×256px — ~5000 images.",
      make_style("CoverBody", fontSize=11, leading=16, alignment=TA_CENTER,
                 textColor=colors.HexColor("#444444"))),
    SP(10),
    make_table([
        ["Parameter", "Value"],
        ["Architecture", "CycleGAN (ResnetGenerator + NLayerDiscriminator)"],
        ["Image Size", "256 × 256 px"],
        ["Training Epochs", "150"],
        ["Dataset", "darren2020/ct-to-mri-cgan (Kaggle)"],
        ["Total Images", "~4974 (2486 CT + 2488 MRI)"],
        ["Best Checkpoint", "Epoch 110 (Val SSIM: 0.3771)"],
        ["CT→MRI Test SSIM", "0.264 ± 0.120"],
        ["MRI→CT Test SSIM", "0.486 ± 0.075"],
    ], col_widths=[7*cm, 10*cm]),
    PageBreak(),
]

# ── SECTION 1 — PROJECT OVERVIEW ─────────────────────────────
content += [
    H1("1. Project Overview"),
    HR(),
    Body("This project implements a CycleGAN to perform bidirectional image-to-image translation "
         "between MRI (Magnetic Resonance Imaging) and CT (Computed Tomography) brain scans. "
         "Both translation directions are trained simultaneously within a single model:"),
    SP(4),
    Bullet("CT → MRI: Convert CT scan appearance to MRI-style soft tissue contrast"),
    Bullet("MRI → CT: Convert MRI scan appearance to CT-style bone/density contrast"),
    SP(6),
    Body("CycleGAN was specifically chosen because the dataset is <b>unpaired</b> — CT and MRI images "
         "are not aligned by patient identity, scan session, or anatomical slice position. "
         "A supervised paired model such as pix2pix cannot be applied here because it requires "
         "corresponding image pairs at pixel level."),
    SP(6),
    Body("CycleGAN solves the unpaired problem by training two generators and two discriminators "
         "simultaneously, enforced by a cycle-consistency constraint: translating an image to the "
         "other domain and then back should reconstruct the original image faithfully. This provides "
         "a self-supervised training signal without needing paired data."),
    SP(6),
    H2("1.1 Project Pipeline"),
    Body("The project followed a structured four-phase methodology:"),
    SP(4),
    make_table([
        ["Phase", "Description", "Scale"],
        ["Phase 1 — Encoder Search", "Tested encoder hyperparameters, identified best 7 configs", "Small"],
        ["Phase 2 — Grid Search", "28 full configurations × 10 epochs × 50 images", "Small"],
        ["Phase 3 — Confirmation", "Top 4 configs × 30 epochs × 50 images", "Small"],
        ["Phase 4 — Full Training", "Winning config × 150 epochs × full dataset × 256px", "Full"],
    ], col_widths=[5*cm, 9*cm, 3*cm]),
    SP(8),
    H2("1.2 Why CycleGAN Over Other Architectures"),
    Body("Several architectures were considered for this task:"),
    SP(4),
    make_table([
        ["Architecture", "Paired?", "Suitable Here?", "Reason"],
        ["pix2pix", "Yes", "No", "Requires pixel-aligned pairs"],
        ["CycleGAN", "No", "Yes", "Works with unpaired domains"],
        ["UNIT", "No", "Yes", "Heavier, shared latent space"],
        ["MUNIT", "No", "Yes", "Multimodal, more complex"],
        ["StarGAN", "No", "Partial", "Better for 3+ domains"],
        ["Diffusion Models", "No", "Possible", "High compute, overkill for this scale"],
    ], col_widths=[4*cm, 2.5*cm, 3*cm, 7.5*cm]),
    SP(6),
    Body("CycleGAN is the standard choice for unpaired medical image translation and has extensive "
         "literature validation on MRI/CT datasets specifically. The cycle-consistency loss provides "
         "strong geometric preservation guarantees which are critical for medical imaging — "
         "a translated image must preserve anatomical structure."),
    PageBreak(),
]

# ── SECTION 2 — DATASET ──────────────────────────────────────
content += [
    H1("2. Dataset"),
    HR(),
    H2("2.1 Source and Access"),
    Body("The dataset used is <b>darren2020/ct-to-mri-cgan</b> hosted on Kaggle. "
         "It contains brain scan images in both CT and MRI modalities, pre-split into train and test sets."),
    SP(4),
    Bullet("<b>Kaggle path:</b> /kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/"),
    Bullet("<b>Local path:</b> E:\\code\\mri to cti\\Dataset\\images\\"),
    SP(6),
    H2("2.2 Folder Structure"),
    Code("Dataset/images/\n"
         "├── trainA/   — CT training images    (1742 images, ct*.png)\n"
         "├── trainB/   — MRI training images   (1744 images, mri*.png)\n"
         "├── testA/    — CT test images         (744 images)\n"
         "└── testB/    — MRI test images        (744 images)"),
    SP(6),
    H2("2.3 Data Split"),
    Body("The dataset already provides a train/test split via the trainA/trainB and testA/testB folders. "
         "An additional validation split was carved out at runtime from the training data:"),
    SP(4),
    make_table([
        ["Split", "CT Images", "MRI Images", "Purpose"],
        ["Train", "1568", "1570", "Model weight updates each epoch"],
        ["Validation", "174", "174", "Val SSIM for best checkpoint selection"],
        ["Test", "744", "744", "Final evaluation only — never seen during training"],
    ], col_widths=[3.5*cm, 3.5*cm, 3.5*cm, 7*cm]),
    SP(6),
    Note("CT has 1742 and MRI has 1744 training images — a 2 image difference. The zip() in the "
         "training loop stops at the shorter domain (1568 CT batches per epoch), so 2 MRI images "
         "are skipped per epoch. Effect on training is negligible."),
    SP(6),
    H2("2.4 Image Preprocessing"),
    Body("All images go through the following preprocessing pipeline before entering the network:"),
    SP(4),
    make_table([
        ["Step", "Operation", "Why"],
        ["Resize", "256×256 with BICUBIC interpolation",
         "Standardize input size; BICUBIC preserves edge sharpness better than bilinear"],
        ["RandomHorizontalFlip", "50% chance flip (experiment_v2 only)",
         "Data augmentation; doubles effective dataset size"],
        ["ToTensor", "PIL [0,255] → float [0,1]", "PyTorch tensor format"],
        ["Normalize", "mean=0.5, std=0.5 → maps [0,1] to [-1,1]",
         "Matches generator Tanh output range"],
    ], col_widths=[3.5*cm, 5.5*cm, 8*cm]),
    SP(6),
    Body("Images are loaded as RGB (3-channel) even though brain scans are grayscale. "
         "PIL's .convert('RGB') duplicates the single grayscale channel across R, G, B. "
         "This allows the same model architecture to be used regardless of source modality."),
    PageBreak(),
]

# ── SECTION 3 — HYPERPARAMETER SEARCH ────────────────────────
content += [
    H1("3. Hyperparameter Search"),
    HR(),
    H2("3.1 Phase 2 — 28-Run Grid Search"),
    Body("After identifying the best 7 encoder configurations in Phase 1, a full grid search was run "
         "combining encoder parameters with decoder/discriminator parameters. Each of the 28 runs "
         "was trained for 10 epochs on 50 images per domain at 128px resolution."),
    SP(4),
    Body("Each run produced a train_log.txt and config.json. A summary CSV was generated by computing "
         "statistics over the <b>last 20 log entries</b> of each run (not all entries — confirmed by "
         "cross-referencing CSV values against log files)."),
    SP(4),
    Body("<b>Scoring metric:</b>"),
    Code("combined_score = lossG_mean_last20 + lossD_mean_last20\n"
         "               + G_stability + D_stability"),
    Body("Lower combined_score = better. This rewards low average loss AND stable training "
         "(low variance between log entries)."),
    SP(6),
    H2("3.2 Phase 3 — 30-Epoch Confirmation"),
    Body("Top 4 configs from Phase 2 were run for 30 epochs on 50 images. Visual inspection of "
         "generated images at epoch 30 was the primary selection criterion:"),
    SP(4),
    make_table([
        ["Config", "Rank", "Outcome at Epoch 30"],
        ["confirm_run_012", "1st", "Winner — clean images, stable losses, no artifacts"],
        ["confirm_run_018", "2nd", "Grid artifacts on fakeB (n_blocks=6 caused issues)"],
        ["confirm_run_002", "3rd", "Severe checkerboard + noise artifacts"],
        ["confirm_run_004", "5th", "Severe checkerboard + noise artifacts"],
    ], col_widths=[4*cm, 2.5*cm, 10.5*cm]),
    SP(6),
    H2("3.3 Winning Configuration — confirm_run_012"),
    make_table([
        ["Hyperparameter", "Value", "Description"],
        ["lr_G", "0.0001", "Generator learning rate"],
        ["lr_D", "0.0001", "Discriminator learning rate"],
        ["lambda_cycle", "5.0", "Cycle-consistency loss weight"],
        ["lambda_identity", "2.5", "Identity loss weight"],
        ["n_blocks", "4", "Residual blocks in generator"],
        ["n_layers_D", "3", "PatchGAN discriminator depth"],
        ["use_spect", "True", "Spectral normalization on discriminator"],
        ["ngf / ndf", "64", "Generator / discriminator base feature channels"],
        ["batch_size", "1", "Images per training step"],
        ["image_size", "128 → 256", "Search at 128px, full training at 256px"],
    ], col_widths=[4.5*cm, 3*cm, 9.5*cm]),
    PageBreak(),
]

# ── SECTION 4 — ARCHITECTURE ─────────────────────────────────
content += [
    H1("4. Model Architecture"),
    HR(),
    H2("4.1 CycleGAN Overview — Four Networks"),
    Body("CycleGAN trains four networks simultaneously:"),
    SP(4),
    make_table([
        ["Network", "Role", "Input", "Output"],
        ["G_A2B", "Generator CT→MRI", "CT image (3×256×256)", "Fake MRI (3×256×256)"],
        ["G_B2A", "Generator MRI→CT", "MRI image (3×256×256)", "Fake CT (3×256×256)"],
        ["D_A", "Discriminator CT", "CT or Fake CT (3×256×256)", "Patch map (1×30×30)"],
        ["D_B", "Discriminator MRI", "MRI or Fake MRI (3×256×256)", "Patch map (1×30×30)"],
    ], col_widths=[2.5*cm, 4*cm, 5*cm, 5.5*cm]),
    SP(8),
    H2("4.2 Generator — ResnetGenerator"),
    Body("The generator follows the Encoder → Residual Blocks → Decoder architecture "
         "from the original CycleGAN paper (Johnson et al. 2016). It processes the full image "
         "at multiple scales and must output a complete 256×256 image."),
    SP(4),
    H3("Architecture Layers"),
    Code("Input: 3 × 256 × 256\n\n"
         "[Encoder]\n"
         "ReflectionPad2d(3)\n"
         "Conv2d(3 → 64,  kernel=7, stride=1)  + InstanceNorm + ReLU\n"
         "Conv2d(64 → 128, kernel=3, stride=2)  + InstanceNorm + ReLU\n"
         "Conv2d(128 → 256, kernel=3, stride=2) + InstanceNorm + ReLU\n\n"
         "[Bottleneck — 4 Residual Blocks @ 256 channels]\n"
         "ResBlock × 4\n\n"
         "[Decoder]\n"
         "ConvTranspose2d(256 → 128, stride=2, output_padding=1) + InstanceNorm + ReLU\n"
         "ConvTranspose2d(128 → 64,  stride=2, output_padding=1) + InstanceNorm + ReLU\n"
         "ReflectionPad2d(3)\n"
         "Conv2d(64 → 3, kernel=7) + Tanh\n\n"
         "Output: 3 × 256 × 256  (range [-1, 1])"),
    SP(6),
    H3("Residual Block Detail"),
    Code("def forward(x):\n"
         "    residual = x\n"
         "    x = ReflectionPad2d(1) → Conv2d(256,256,3) → InstanceNorm → ReLU\n"
         "    x = ReflectionPad2d(1) → Conv2d(256,256,3) → InstanceNorm\n"
         "    return x + residual   # skip connection"),
    SP(6),
    H3("Design Choices Explained"),
    make_table([
        ["Choice", "Alternative", "Why This Choice"],
        ["ReflectionPad2d", "Zero padding",
         "Avoids border artifacts — mirrors edge pixels instead of inserting zeros"],
        ["InstanceNorm", "BatchNorm",
         "BatchNorm with batch_size=1 is equivalent to no normalization. "
         "InstanceNorm normalizes per-image, works correctly at batch_size=1"],
        ["ConvTranspose2d", "Bilinear upsample + Conv",
         "Learnable upsampling; combined with spectral norm on discriminator "
         "prevents checkerboard artifacts"],
        ["Tanh output", "Sigmoid",
         "Output range [-1,1] matches normalized input range — consistent "
         "with the [-1,1] normalization applied to real images"],
        ["n_blocks=4", "9 (paper default)",
         "Chosen from hyperparameter search at 128px. Paper recommends 9 for "
         "256px — this is a known limitation of the current model"],
    ], col_widths=[4*cm, 4*cm, 9*cm]),
    SP(4),
    Body("<b>Total parameters: 5,477,379</b>"),
    SP(8),
    H2("4.3 Discriminator — NLayerDiscriminator (PatchGAN)"),
    Body("The discriminator does not classify the full image as real or fake. Instead it outputs "
         "a spatial patch map where each value represents the realness score of a local image region. "
         "With n_layers_D=3 at 256px input, the output is 30×30 — each patch value covers "
         "approximately 70×70 pixels of the input."),
    SP(4),
    Body("This patch-based approach has several advantages over a full-image discriminator: "
         "it operates on local texture and structure rather than global composition, "
         "is faster to compute, and produces sharper high-frequency details in generated images."),
    SP(4),
    H3("Architecture Layers"),
    Code("Input: 3 × 256 × 256\n\n"
         "Conv2d(3 → 64,   stride=2) + SpectralNorm + LeakyReLU(0.2)    [no InstanceNorm on first layer]\n"
         "Conv2d(64 → 128,  stride=2) + SpectralNorm + InstanceNorm + LeakyReLU(0.2)\n"
         "Conv2d(128 → 256, stride=2) + SpectralNorm + InstanceNorm + LeakyReLU(0.2)\n"
         "Conv2d(256 → 512, stride=1) + SpectralNorm + InstanceNorm + LeakyReLU(0.2)\n"
         "Conv2d(512 → 1,   kernel=4, padding=1) + SpectralNorm\n\n"
         "Output: 1 × 30 × 30  (patch realness scores)"),
    SP(6),
    H3("Spectral Normalization"),
    Body("use_spect=True applies spectral normalization to every convolutional layer in the discriminator "
         "(including the final layer). Spectral norm constrains the Lipschitz constant of the discriminator "
         "by normalizing weights by their largest singular value. This:"),
    SP(4),
    Bullet("Prevents the discriminator from becoming too powerful relative to the generator"),
    Bullet("Stabilizes GAN training — less oscillation in loss values"),
    Bullet("Was a key finding from the hyperparameter search: all top configs had use_spect=True"),
    SP(4),
    Body("<b>Total parameters: 2,764,737</b>"),
    SP(6),
    H3("Why Generator Has More Parameters Than Discriminator"),
    Body("The generator must reconstruct a complete 256×256 image pixel by pixel — it has an encoder, "
         "4 residual blocks, and a full decoder. The discriminator only needs to output a 30×30 "
         "grid of scores — a much simpler task requiring far less capacity. The asymmetry in task "
         "complexity directly causes the 5.4M vs 2.7M parameter difference."),
    SP(8),
    H2("4.4 Weight Initialization"),
    Body("All convolutional layers are initialized with a normal distribution (mean=0.0, std=0.02). "
         "InstanceNorm layers are initialized with normal(mean=1.0, std=0.02) for weights and 0 for bias. "
         "This is the standard CycleGAN initialization. Small random weights at std=0.02 (smaller than "
         "the PyTorch default of 0.01 * kaiming) help prevent symmetry-breaking issues where all "
         "neurons learn the same features at the start of training."),
    PageBreak(),
]

# ── SECTION 5 — TRAINING CONFIG ──────────────────────────────
content += [
    H1("5. Training Configuration"),
    HR(),
    H2("5.1 Loss Functions"),
    H3("Adversarial Loss — LSGAN (Least Squares GAN)"),
    Body("Standard GAN loss uses Binary Cross-Entropy (BCE). LSGAN replaces it with Mean Squared Error (MSE):"),
    Code("# Generator wants discriminator to output 1 (thinks it is real)\n"
         "loss_GAN = MSE(D(G(x)), ones_like(D(G(x))))\n\n"
         "# Discriminator: real images → 1, fake images → 0\n"
         "loss_D = 0.5 * (MSE(D(real), 1) + MSE(D(fake), 0))"),
    SP(4),
    Body("LSGAN is more stable than BCE-GAN because it penalizes samples that are on the correct "
         "side of the decision boundary but far from it — BCE saturates and produces near-zero "
         "gradients for confident predictions, while MSE maintains gradient signal throughout training."),
    SP(6),
    H3("Cycle-Consistency Loss"),
    Body("The core constraint that enables unpaired training:"),
    Code("loss_cycle = L1(G_B2A(G_A2B(real_CT)),  real_CT)  * lambda_cycle\n"
         "           + L1(G_A2B(G_B2A(real_MRI)), real_MRI) * lambda_cycle\n\n"
         "lambda_cycle = 5.0"),
    Body("Translating real_CT → fake_MRI → reconstructed_CT should recover the original CT image. "
         "This forces the generators to learn a meaningful mapping rather than ignoring the input. "
         "lambda_cycle=5.0 was the winning value from the search — strong enough to enforce geometry "
         "without overwhelming the adversarial signal."),
    SP(6),
    H3("Identity Loss"),
    Code("loss_identity = L1(G_B2A(real_CT),  real_CT)  * lambda_identity\n"
         "              + L1(G_A2B(real_MRI), real_MRI) * lambda_identity\n\n"
         "lambda_identity = 2.5  (= lambda_cycle / 2, standard ratio)"),
    Body("When a generator receives an image already in its target domain, it should return "
         "that image unchanged. This preserves overall color and structure consistency. "
         "The standard ratio is lambda_identity = 0.5 * lambda_cycle."),
    Note("For CT↔MRI translation, the two modalities look very different. "
         "Identity loss may be too constraining for CT→MRI, preventing the generator from "
         "making the large textural changes needed. This is being investigated in experiment_v2 "
         "with lambda_identity=0.5."),
    SP(6),
    H3("Total Generator Loss"),
    Code("loss_G = loss_GAN_A2B + loss_GAN_B2A\n"
         "       + loss_cycle_A + loss_cycle_B\n"
         "       + loss_idt_A   + loss_idt_B"),
    SP(8),
    H2("5.2 Optimizers"),
    Body("Both generators share one Adam optimizer; both discriminators share one Adam optimizer. "
         "One optimizer.step() updates both generators simultaneously, one updates both discriminators."),
    SP(4),
    make_table([
        ["Optimizer", "Networks", "lr", "beta1", "beta2"],
        ["optimizer_G", "G_A2B + G_B2A", "0.0001", "0.5", "0.999"],
        ["optimizer_D", "D_A + D_B", "0.0001", "0.5", "0.999"],
    ], col_widths=[3.5*cm, 4.5*cm, 3*cm, 2.5*cm, 2.5*cm]),
    SP(4),
    Body("beta1=0.5 (vs PyTorch default 0.9) is standard for GAN training. Lower beta1 means "
         "the optimizer forgets momentum faster, which helps with the non-stationary loss landscape "
         "of adversarial training where the target keeps moving."),
    SP(6),
    H2("5.3 Learning Rate Schedule — Linear Decay"),
    Body("Constant LR for first half, linear decay to zero for second half:"),
    Code("def lr_lambda(epoch):\n"
         "    e = epoch + 1  # LambdaLR is 0-indexed internally\n"
         "    if e < decay_epoch:    # epochs 1-75: constant\n"
         "        return 1.0\n"
         "    # epochs 75-150: linear decay to 0\n"
         "    return max(0.0, 1.0 - (e - decay_epoch) / (total_epochs - decay_epoch))"),
    SP(4),
    Body("The LR at epoch 80 (where session 2 crashed) was approximately 0.0000933. "
         "The LR reaches exactly 0.000000 at epoch 150 (confirmed in training logs)."),
    SP(6),
    H2("5.4 Image Replay Buffer"),
    Body("The replay buffer stores the last 50 generated images per domain. During discriminator "
         "training, each fake image has a 50% probability of being replaced by a randomly selected "
         "stored image from the buffer:"),
    Code("def push_and_pop(images):\n"
         "    for img in images:\n"
         "        if buffer not full:\n"
         "            add to buffer, return current\n"
         "        else if random() > 0.5:\n"
         "            swap random stored image with current\n"
         "            return stored image  # discriminator sees older fake\n"
         "        else:\n"
         "            return current fake"),
    Body("This exposes the discriminator to a diverse history of generator outputs rather than "
         "only the most recent batch, preventing the discriminator from overfitting to the "
         "current generator state and stabilizing the adversarial dynamic."),
    PageBreak(),
]

# ── SECTION 6 — CODE WALKTHROUGH ─────────────────────────────
content += [
    H1("6. Code Walkthrough — kaggle_notebook.py"),
    HR(),
    Body("The full training is implemented as a single Python script organized into 12 sections. "
         "Each section is described below with its purpose, key decisions, and any non-obvious behavior."),
    SP(6),
    H2("Section 1 — Setup and Imports"),
    Body("Imports all required libraries and sets random seeds across Python, NumPy, and PyTorch "
         "for full reproducibility. SEED=42 is applied to all random number generators. "
         "Device detection runs at startup — if CUDA is unavailable the script falls back to CPU "
         "but will be extremely slow at 256px."),
    SP(4),
    Bullet("torch.cuda.manual_seed_all(SEED) — seeds all CUDA devices, not just the current one"),
    Bullet("skimage.metrics: structural_similarity and peak_signal_noise_ratio used for evaluation only"),
    SP(6),
    H2("Section 2 — Dataset Path Explorer"),
    Body("Walks the Kaggle input directory up to 4 levels deep and prints the folder tree. "
         "This section exists purely for verification before any training code runs. "
         "Running this cell first is how the correct dataset paths were discovered — "
         "the Kaggle input path includes the dataset owner username (darren2020) which is "
         "not immediately obvious from the dataset page URL."),
    SP(6),
    H2("Section 3 — Configuration"),
    Body("All hyperparameters and file paths live in a single CONFIG dictionary. "
         "Nothing is hardcoded elsewhere in the script — all downstream sections read from CONFIG. "
         "This makes the notebook reproducible: changing one CONFIG entry propagates everywhere."),
    SP(4),
    Body("Three output directories are created at the start of this section:"),
    Bullet("/kaggle/working/outputs/ — root output directory"),
    Bullet("/kaggle/working/outputs/checkpoints/ — model weight files (.pth)"),
    Bullet("/kaggle/working/outputs/samples/ — sample grid images per epoch"),
    SP(6),
    H2("Section 4 — Dataset"),
    H3("MRICTDataset class"),
    Body("Loads all image files from a flat directory (no subdirectories expected). "
         "Files are sorted alphabetically for deterministic ordering across runs. "
         "Images are converted to RGB on load using PIL's .convert('RGB') — this handles "
         "grayscale source files by duplicating the single channel across R, G, B, making "
         "the dataset compatible with the 3-channel model architecture."),
    SP(4),
    Body("Raises RuntimeError immediately on empty directory rather than silently running "
         "on zero data — a fast-fail approach that surfaces path issues early."),
    SP(4),
    H3("split_dataset function"),
    Body("PyTorch's random_split is seeded with a fixed Generator to ensure the same "
         "90/10 train/validation split every time the notebook runs. This is critical for "
         "multi-session training — if the split were random, different sessions would "
         "have different validation sets, making SSIM comparisons meaningless."),
    SP(4),
    H3("DataLoaders"),
    Body("Training loaders use shuffle=True (random order each epoch, good for SGD) and "
         "pin_memory=True (locks tensor memory pages for faster CPU→GPU transfer). "
         "Validation and test loaders use shuffle=False — consistent ordering is required "
         "so that zip(loader_src, loader_tgt) matches the same positional index each time "
         "for SSIM computation."),
    SP(6),
    H2("Section 5 — Model Architecture"),
    Body("Defines all four network classes and the init_weights function. "
         "All models are instantiated, moved to device (.to(device)), "
         "and weight-initialized in this section. Parameter counts are printed "
         "to confirm the architecture is correct before training begins."),
    SP(6),
    H2("Section 6 — Losses, Optimizers, Schedulers"),
    H3("gan_loss function"),
    Body("Creates a target tensor of ones (real) or zeros (fake) matching the discriminator "
         "output shape using torch.ones_like / torch.zeros_like. This automatically handles "
         "whatever spatial dimensions the discriminator produces without hardcoding shape. "
         "MSE between prediction and target implements LSGAN loss."),
    SP(4),
    H3("lr_lambda and the 0-index offset"),
    Body("PyTorch's LambdaLR scheduler internally tracks an epoch counter starting at 0 "
         "(not 1). The lambda function receives this 0-indexed value. Since our training "
         "loop uses 1-indexed epochs (range(1, 151)), comparisons like "
         "'epoch < decay_epoch' would be off by one without correction. "
         "The fix: e = epoch + 1 at the start of lr_lambda maps "
         "the 0-indexed internal counter to 1-indexed training epochs correctly."),
    SP(6),
    H2("Section 7 — Image Replay Buffer"),
    Body("Implements the history buffer from the original GAN training paper (Shrivastava et al., 2017). "
         "The buffer uses a plain Python list rather than collections.deque because the random swap "
         "operation requires integer index access (buffer[idx] = img) — deque supports this syntax "
         "but at O(n) cost and with unexpected behavior for assignment. See Error 3 in Section 8."),
    SP(6),
    H2("Section 8 — Validation SSIM"),
    Body("compute_val_ssim iterates over validation loader pairs using zip() for positional matching. "
         "Critical details:"),
    SP(4),
    Bullet("G.eval() is called before inference to disable training-mode behaviors "
           "(dropout if present, BatchNorm running stats). G.train() restores training mode after."),
    Bullet("Images are denormalized from [-1,1] to [0,1] before SSIM computation: "
           "(tensor + 1) / 2"),
    Bullet("channel_axis=2 in skimage's ssim because numpy arrays after transpose are H×W×C"),
    Bullet("data_range=1.0 because images are in [0,1] range after denormalization"),
    SP(6),
    H2("Section 9 — Training Loop"),
    H3("Per-batch training order"),
    Body("Within each batch, generators are trained first, then discriminators:"),
    Code("1. optimizer_G.zero_grad()\n"
         "   compute identity losses: G_B2A(real_CT)≈real_CT, G_A2B(real_MRI)≈real_MRI\n"
         "   compute GAN losses:      D_B(G_A2B(real_CT))→1, D_A(G_B2A(real_MRI))→1\n"
         "   compute cycle losses:    G_B2A(fake_MRI)≈real_CT, G_A2B(fake_CT)≈real_MRI\n"
         "   total loss_G.backward()\n"
         "   optimizer_G.step()\n\n"
         "2. optimizer_D.zero_grad()\n"
         "   D_A: MSE(D_A(real_CT),1) + MSE(D_A(buffer_fake_CT),0)\n"
         "   D_B: MSE(D_B(real_MRI),1) + MSE(D_B(buffer_fake_MRI),0)\n"
         "   loss_D.backward()\n"
         "   optimizer_D.step()"),
    SP(4),
    Body("<b>Important:</b> fake_A and fake_B computed during generator training are reused for "
         "cycle loss — they are not recomputed. This is efficient and correct. The discriminator "
         "receives these tensors via .detach() which removes them from the computation graph, "
         "preventing gradients from flowing back through the generator during discriminator training."),
    SP(4),
    H3("Resume logic cascade"),
    Body("On startup, the training loop checks three locations in priority order:"),
    Code("1. save_resume_path  = /kaggle/working/outputs/checkpoints/resume_state.pth\n"
         "   (current session's checkpoint — highest priority, most recent)\n\n"
         "2. full_resume_path  = /kaggle/input/datasets/.../resume_state.pth\n"
         "   (uploaded from previous session — full state: weights + optimizer + scheduler)\n\n"
         "3. partial_resume_dir = /kaggle/input/.../cycle-gan-resume-checkpoint-80/\n"
         "   (individual weight files when resume_state.pth was unavailable)"),
    SP(4),
    Body("Partial resume (path 3) loads model weights only — optimizer states are restarted fresh "
         "and the scheduler is fast-forwarded by calling step() 79 times before training begins. "
         "History is pre-filled with NaN for the missing epochs to keep the x-axis correct in plots."),
    SP(6),
    H2("Section 10 — Loss Curves"),
    Body("Plots three subplots: (1) generator and discriminator loss over all training epochs, "
         "(2) validation SSIM for both directions over validation epochs, "
         "(3) CT→MRI SSIM with a vertical line marking the LR decay start epoch. "
         "val_epochs is derived from actual history length rather than hardcoded, "
         "to handle partial sessions where history has fewer entries than expected."),
    SP(6),
    H2("Section 11 — Test Set Evaluation"),
    Body("Loads the best checkpoint weights (G_A2B_best.pth and G_B2A_best.pth), "
         "runs inference over the full 744-image test set, and computes MAE, SSIM, "
         "and PSNR per image. Reports mean ± standard deviation for each metric. "
         "Results are saved to test_metrics.json for reproducible reference. "
         "map_location=device is specified on torch.load to handle cases where the checkpoint "
         "was saved on a GPU but the evaluation runs on a different device."),
    SP(6),
    H2("Section 12 — Visual Results"),
    Body("show_results generates a 3-row × 6-column figure: "
         "row 1 = input images, row 2 = generated images, row 3 = real target images. "
         "Run for both CT→MRI and MRI→CT directions on the test set. "
         "Saved as PNG files in the outputs directory. "
         "Provides a qualitative sanity check complementing the quantitative metrics."),
    PageBreak(),
]

# ── SECTION 7 — TRAINING PROCESS ─────────────────────────────
content += [
    H1("7. Training Process and Sessions"),
    HR(),
    Body("Full training required 3 sessions across multiple days due to Kaggle's 9-hour session limit. "
         "Each session is documented below including issues encountered."),
    SP(6),
    H2("7.1 Session 1 — Epochs 1 to 70"),
    make_table([
        ["Item", "Detail"],
        ["Kaggle Version", "Version 1"],
        ["GPU", "T4 (16GB VRAM)"],
        ["Duration", "~9 hours (session timeout)"],
        ["Epochs Completed", "1–70 (72 seen in logs but last checkpoint at 70)"],
        ["Outcome", "Clean run — natural timeout"],
        ["Last Checkpoint", "resume_state.pth at epoch 70"],
        ["Best SSIM", "0.3577 at epoch unknown within session"],
    ], col_widths=[4.5*cm, 12.5*cm]),
    SP(4),
    Body("Epoch timing: epoch 1 completed at 620 seconds (~10.3 minutes). "
         "All subsequent epochs consistent at ~10 min/epoch. "
         "This gave the estimate of ~35-40 epochs per 9-hour session, ~4 sessions total."),
    SP(6),
    H2("7.2 Session 2 — Epochs 71 to 80 (Crashed)"),
    make_table([
        ["Item", "Detail"],
        ["Kaggle Version", "Version 2"],
        ["GPU", "T4"],
        ["Resumed From", "Epoch 70 via uploaded resume_state.pth"],
        ["Crash Point", "Epoch 80 — during resume_state.pth save"],
        ["Error", "RuntimeError: open file failed — Read-only file system"],
        ["Files Saved Before Crash",
         "G_A2B_epoch80.pth, G_B2A_epoch80.pth, G_A2B_best.pth, G_B2A_best.pth, D_A_best.pth, D_B_best.pth"],
        ["Best SSIM", "0.3602 at epoch 80"],
    ], col_widths=[4.5*cm, 12.5*cm]),
    SP(4),
    Body("The crash occurred because the resume path variable was overwritten with the "
         "read-only Kaggle input path when falling back to the uploaded checkpoint. "
         "See Error 2 in Section 8 for full details and fix."),
    SP(6),
    H2("7.3 Session 3 — Aborted (No GPU)"),
    make_table([
        ["Item", "Detail"],
        ["Kaggle Version", "Version 3"],
        ["GPU", "None — CPU only"],
        ["Duration", "~25 minutes (aborted manually)"],
        ["Outcome", "Detected Device: cpu in logs, stopped immediately"],
        ["Cause", "GPU accelerator not re-enabled when creating new session"],
    ], col_widths=[4.5*cm, 12.5*cm]),
    SP(6),
    H2("7.4 Session 3 (Retry) — Epochs 81 to 150"),
    make_table([
        ["Item", "Detail"],
        ["Kaggle Version", "Version 4"],
        ["GPU", "T4"],
        ["Resume Type", "Partial — individual weight files from epoch 80"],
        ["Files Loaded", "G_A2B_best.pth, G_B2A_best.pth, D_A_best.pth, D_B_best.pth"],
        ["Optimizer State", "Restarted fresh — Adam momentum lost"],
        ["Scheduler", "Fast-forwarded 79 steps to match epoch 80 LR position"],
        ["LR at Start", "0.000093 (correct for epoch 81)"],
        ["Duration", "~11 hours"],
        ["Outcome", "Completed successfully — epoch 150 reached"],
        ["Best SSIM", "0.3771 at epoch 110"],
    ], col_widths=[4.5*cm, 12.5*cm]),
    SP(6),
    H2("7.5 Session Timeline Summary"),
    make_table([
        ["Session", "Version", "Epochs", "Outcome"],
        ["1", "V1", "1 → 70", "Clean run — natural timeout at 9 hours"],
        ["2", "V2", "71 → 80", "Crashed saving checkpoint — read-only path bug"],
        ["3 attempt 1", "V3", "—", "Aborted — CPU only, no GPU"],
        ["3 attempt 2", "V4", "81 → 150", "Completed — training finished at epoch 150"],
    ], col_widths=[3.5*cm, 2.5*cm, 4*cm, 7*cm]),
    PageBreak(),
]

# ── SECTION 8 — ERRORS AND FIXES ─────────────────────────────
content += [
    H1("8. Errors and Fixes"),
    HR(),
    Body("Eight bugs or mistakes were identified and fixed during the course of development and training. "
         "Each is documented with root cause analysis and the exact fix applied."),
    SP(8),

    H2("Error 1 — Wrong Dataset Path"),
    P("What went wrong:", error_style),
    Body("The initial dataset path assumed /kaggle/input/ct-to-mri-cgan/ with subdirectories "
         "train/CT and train/MRI. The actual Kaggle input path was: "
         "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainA"),
    SP(4),
    Body("<b>Why it happened:</b> Kaggle dataset input paths include the dataset owner's username "
         "(darren2020) which does not appear in the dataset URL. The internal folder structure "
         "inside the dataset (Dataset/images/trainA vs expected dataset/image/train/CT) "
         "was also non-standard and not visible without running the notebook."),
    SP(4),
    Body("Additionally, the folder tree depth was limited to 2 levels (if depth > 2: continue), "
         "which hid trainA/trainB/testA/testB folders that exist at depth 3. "
         "The tree showed 'images/' folder but appeared to have 0 files inside it."),
    SP(4),
    P("Fix:", fix_style),
    Body("User ran the Kaggle default starter cell (os.walk('/kaggle/input')) which prints all file paths. "
         "Actual paths were read from the output and updated in CONFIG. "
         "Folder tree depth limit was changed from > 2 to > 4 to show all subfolders."),
    SP(8),

    H2("Error 2 — resume_state.pth Saved to Read-Only Input Path (Session 2 Crash)"),
    P("What went wrong:", error_style),
    Body("Session 2 crashed at epoch 80 with:"),
    Code("RuntimeError: [enforce fail at inline_container.cc:743]\n"
         "open file failed with strerror: Read-only file system"),
    Body("The code tried to write resume_state.pth to /kaggle/input/... which is read-only."),
    SP(4),
    Body("<b>Why it happened:</b> A single variable resume_path was used for both loading "
         "and saving the checkpoint. When the code fell back to the uploaded checkpoint "
         "(at /kaggle/input/...), it overwrote resume_path with that path. "
         "Later, torch.save(..., resume_path) tried to write to the same location — "
         "which is in the read-only input filesystem."),
    SP(4),
    P("Fix:", fix_style),
    Body("Split into two separate variables:"),
    Code("save_resume_path = '/kaggle/working/outputs/checkpoints/resume_state.pth'\n"
         "load_resume_path = None  # determined at runtime\n\n"
         "# Load from working dir first, then input dir\n"
         "if exists(save_resume_path):    load_resume_path = save_resume_path\n"
         "elif exists(full_resume_path):  load_resume_path = full_resume_path\n\n"
         "# Save always goes to working dir\n"
         "torch.save(state, save_resume_path)"),
    SP(8),

    H2("Error 3 — deque Used With Integer Indexing in ReplayBuffer"),
    P("What went wrong:", error_style),
    Body("The ReplayBuffer was implemented using collections.deque with maxlen=50, "
         "then accessed with integer indexing: tmp = self.buffer[idx].clone() and "
         "self.buffer[idx] = img. Python's deque supports [] access but it is O(n) "
         "and assignment to a deque by index produces incorrect behavior when the "
         "buffer is full (the maxlen auto-drop behavior interacts badly with index-based mutation)."),
    SP(4),
    Body("<b>Why it happened:</b> deque was chosen for its maxlen feature which automatically "
         "drops old elements. But the random-swap logic requires direct integer index access "
         "and assignment, which is the intended use case for a list, not a deque."),
    SP(4),
    P("Fix:", fix_style),
    Body("Replaced deque with a plain Python list. Manual size management: only append when "
         "len(buffer) < max_size, randomly replace an existing element when full. "
         "The deque import was also removed as it became unused."),
    SP(8),

    H2("Error 4 — val_epochs Length Mismatch in Matplotlib Plots"),
    P("What went wrong:", error_style),
    Body("Section 10 used a hardcoded x-axis: "
         "val_epochs = list(range(10, CONFIG['epochs']+1, 10)) — always 15 entries for 150 epochs. "
         "After partial resume from epoch 80, history['val_ssim_A2B'] only had entries for epochs "
         "90, 100, 110, 120, 130, 140, 150 — 7 entries. "
         "Plotting 15 x-values against 7 y-values would crash matplotlib with a length mismatch."),
    SP(4),
    Body("<b>Why it happened:</b> The x-axis was hardcoded based on total training epochs "
         "rather than derived from actual history. This worked for the first complete session "
         "but broke for partial sessions where history starts mid-training."),
    SP(4),
    P("Fix:", fix_style),
    Code("# Before:\n"
         "val_epochs = list(range(10, CONFIG['epochs'] + 1, 10))\n\n"
         "# After:\n"
         "val_epochs = list(range(10, 10 * len(history['val_ssim_A2B']) + 1, 10))"),
    SP(8),

    H2("Error 5 — LR Scheduler Warning in Partial Resume Advancement Loop"),
    P("What went wrong:", error_style),
    Body("PyTorch 1.1+ warns that scheduler.step() must be called after optimizer.step(). "
         "The advancement loop called scheduler.step() 79 times before any optimizer.step() "
         "had occurred, generating repeated warning messages in the logs."),
    SP(4),
    Body("<b>Why it happened:</b> The advancement loop is not doing real training — "
         "it is only fast-forwarding the scheduler's internal epoch counter to the correct position "
         "for resuming. No optimizer step exists or is needed. "
         "PyTorch's warning is technically correct for normal training but does not apply here."),
    SP(4),
    P("Fix:", fix_style),
    Code("import warnings\n"
         "with warnings.catch_warnings():\n"
         "    warnings.simplefilter('ignore')\n"
         "    for _ in range(partial_resume_epoch - 1):\n"
         "        scheduler_G.step()\n"
         "        scheduler_D.step()"),
    SP(8),

    H2("Error 6 — Scheduler Advanced One Extra Step (Off-by-One)"),
    P("What went wrong:", error_style),
    Body("The partial resume code advanced the scheduler 80 times before training epoch 81. "
         "The training loop also calls scheduler.step() at the end of every epoch. "
         "So after epoch 81 completed, the scheduler had been stepped 81 times total — "
         "one epoch ahead of the actual training epoch. This persisted throughout session 3, "
         "meaning the LR decay curve was shifted one epoch earlier than intended."),
    SP(4),
    Body("<b>Why it happened:</b> The advancement count matched partial_resume_epoch (80) "
         "without accounting for the training loop's own scheduler.step() call at epoch 81 end. "
         "The total after epoch 81 = 80 pre-advance + 1 loop step = 81, but epoch 81 position "
         "should correspond to 80 total steps."),
    SP(4),
    P("Fix:", fix_style),
    Code("# Before:\n"
         "for _ in range(partial_resume_epoch):      # 80 steps\n\n"
         "# After:\n"
         "for _ in range(partial_resume_epoch - 1):  # 79 steps\n"
         "# Loop adds step 80 at end of epoch 81 — total = 80 (correct)"),
    SP(8),

    H2("Error 7 — Session Started Without GPU"),
    P("What went wrong:", error_style),
    Body("Version 3 ran with Device: cpu instead of Device: cuda. "
         "Training at 256px on CPU would take approximately 100+ minutes per epoch "
         "versus ~10 minutes on T4 GPU — making 70 remaining epochs take ~120 hours on CPU."),
    SP(4),
    Body("<b>Why it happened:</b> When creating a new Kaggle version (Save & Run All), "
         "the GPU accelerator setting is not automatically preserved from the previous version. "
         "It must be manually re-enabled each time under Settings → Accelerator → GPU T4."),
    SP(4),
    P("Fix:", fix_style),
    Body("The session was stopped within 25 minutes after seeing Device: cpu in logs. "
         "T4 GPU re-enabled in Kaggle Settings, new version created (Version 4) and rerun."),
    SP(4),
    Note("Always verify 'Device: cuda' in the first cell output before closing the laptop. "
         "If Device: cpu appears, stop immediately and check accelerator settings."),
    SP(8),

    H2("Error 8 — lr_lambda Off-by-One Epoch Indexing"),
    P("What went wrong:", error_style),
    Body("PyTorch's LambdaLR scheduler internally tracks an epoch counter starting at 0 "
         "(not 1 — it increments before calling the lambda on the first step). "
         "The original lr_lambda compared this 0-indexed value directly to decay_epoch=75 "
         "(which is 1-indexed). The result: LR decay began at internal epoch 75 "
         "which corresponds to training epoch 76 — one epoch late."),
    SP(4),
    Body("<b>Why it happened:</b> Mismatch between PyTorch's internal 0-indexed epoch counter "
         "and the 1-indexed epoch variable (range(1, 151)) used in the training loop. "
         "Easy to miss because the effect is subtle — LR decay starts one epoch late "
         "rather than crashing."),
    SP(4),
    P("Fix:", fix_style),
    Code("def lr_lambda(epoch):\n"
         "    e = epoch + 1  # convert 0-indexed to 1-indexed\n"
         "    if e < CONFIG['decay_epoch']:\n"
         "        return 1.0\n"
         "    return max(0.0, 1.0 - (e - CONFIG['decay_epoch']) /\n"
         "               (CONFIG['epochs'] - CONFIG['decay_epoch']))"),
    PageBreak(),
]

# ── SECTION 9 — RESULTS ──────────────────────────────────────
content += [
    H1("9. Results"),
    HR(),
    H2("9.1 Test Set Metrics"),
    Body("Evaluation was performed on the held-out test set (744 CT + 744 MRI images) "
         "using the best checkpoint saved at epoch 110 (Val SSIM: 0.3771)."),
    SP(4),
    make_table([
        ["Direction", "MAE (↓ better)", "SSIM (↑ better)", "PSNR dB (↑ better)"],
        ["CT → MRI", "0.169 ± 0.070", "0.264 ± 0.120", "12.40 ± 2.80"],
        ["MRI → CT", "0.173 ± 0.040", "0.486 ± 0.075", "9.78 ± 1.20"],
    ], col_widths=[4*cm, 4*cm, 4*cm, 5*cm]),
    SP(6),
    H2("9.2 Training Dynamics"),
    make_table([
        ["Metric", "Epoch 81 (session 3 start)", "Epoch 110 (best)", "Epoch 150 (final)"],
        ["Loss_G", "2.235", "2.177", "1.904"],
        ["Loss_D", "0.112", "0.104", "0.131"],
        ["Val SSIM A2B", "—", "0.274 (best)", "0.267"],
        ["Val SSIM B2A", "—", "0.480 (best)", "0.475"],
        ["LR", "0.000093", "0.000053", "0.000000"],
    ], col_widths=[4.5*cm, 4.5*cm, 4*cm, 4*cm]),
    SP(6),
    Body("Key observations from training dynamics:"),
    SP(4),
    Bullet("Generator loss declined steadily from 2.71 (epoch 1) to 1.90 (epoch 150) — "
           "healthy monotonic improvement throughout training"),
    Bullet("Discriminator loss dropped from 0.44 to ~0.07 by epoch 100, then rose back to 0.13 "
           "by epoch 150 — this rise is positive, indicating the generator became strong enough "
           "to challenge the discriminator again"),
    Bullet("Val SSIM peaked at epoch 110 for both directions — best checkpoint correctly captured"),
    Bullet("MRI→CT SSIM (0.45-0.48) consistently higher than CT→MRI (0.25-0.27) throughout all epochs"),
    Bullet("LR decay visible in loss curves: smoother, steadily declining from epoch 130 onward"),
    SP(6),
    H2("9.3 Visual Quality Analysis"),
    Body("Sample images were saved every 10 epochs during session 3 (epochs 90-150). "
         "Grid layout per image: [real CT | fake MRI | real MRI | fake CT]"),
    SP(4),
    H3("MRI → CT Direction (columns 3-4)"),
    Bullet("Clearly better quality — skull boundary well defined from epoch 90"),
    Bullet("Bone density contrast visible and improving epoch by epoch"),
    Bullet("By epoch 150: structurally very close to real CT scans"),
    Bullet("Low std of SSIM (0.075) confirms consistent quality across test images"),
    SP(4),
    H3("CT → MRI Direction (columns 1-2)"),
    Bullet("Brain structures visible but soft tissue contrast is flat and dark"),
    Bullet("MRI's bright white matter / dark grey matter distinction not fully captured"),
    Bullet("Slight left-side geometric asymmetry present from epoch 70, persists to epoch 150"),
    Bullet("High std of SSIM (0.120) indicates inconsistent quality across test images"),
    SP(6),
    H2("9.4 Why CT→MRI is Harder"),
    Body("The performance gap between directions is expected and has a clear cause:"),
    SP(4),
    Bullet("CT encodes tissue density (Hounsfield units) — bone is bright, soft tissue is grey, air is black. "
           "The mapping from CT to MRI requires the generator to hallucinate soft tissue texture "
           "that is not encoded in the CT signal"),
    Bullet("MRI encodes proton density and relaxation times — the mapping to CT bone structure "
           "is more direct because bone has predictable density in CT"),
    Bullet("MRI scan type variability (T1, T2, FLAIR) in the training data may cause "
           "the generator to average across styles rather than produce a specific MRI type"),
    Bullet("n_blocks=4 may be insufficient for the complex CT→MRI texture generation at 256px"),
    PageBreak(),
]

# ── SECTION 10 — ASSESSMENT AND NEXT STEPS ───────────────────
content += [
    H1("10. Assessment and Next Steps"),
    HR(),
    H2("10.1 Current Model Limitations"),
    make_table([
        ["Limitation", "Evidence", "Planned Fix"],
        ["CT→MRI SSIM too low (0.264)",
         "Visible in both metrics and visual inspection",
         "Increase n_blocks from 4 to 9"],
        ["Geometric asymmetry in generated images",
         "Left side of brain slightly deformed since epoch 70",
         "Increase lambda_cycle from 5.0 to 10.0"],
        ["n_blocks=4 selected at 128px/50 images",
         "Hyperparameter search not representative of 256px scale",
         "Rerun search or directly test n_blocks=9"],
        ["Identity loss may over-constrain CT→MRI",
         "CT and MRI look very different — identity "
         "loss forces generator to keep input similar",
         "Test lambda_identity=0.5 in experiment_v2"],
        ["Training stopped at epoch 150",
         "Loss still declining at epoch 150 (1.904, still dropping)",
         "Train for 200 epochs in next run"],
    ], col_widths=[4.5*cm, 5*cm, 7.5*cm]),
    SP(8),
    H2("10.2 Experiment V2 — Local Testing"),
    Body("Before committing to a full 200-epoch Kaggle run, three configurations are being "
         "tested locally on 500 images × 30 epochs to validate improvements cheaply "
         "(approximately 1 hour per config on RTX 3050 6GB Laptop GPU):"),
    SP(4),
    make_table([
        ["Config", "n_blocks", "lambda_cycle", "lambda_identity", "Flip Aug", "Purpose"],
        ["A (baseline)", "4", "5.0", "2.5", "No", "Reference — current model config"],
        ["B", "9", "10.0", "2.5", "Yes", "Architecture + cycle + augmentation"],
        ["C", "9", "10.0", "0.5", "Yes", "Same as B but reduced identity loss"],
    ], col_widths=[3*cm, 2.5*cm, 3*cm, 3*cm, 2.5*cm, 4*cm]),
    SP(4),
    Body("If Config B or C achieves higher Val SSIM than Config A at epoch 30, the winning "
         "config will be used for a full 200-epoch Kaggle training run."),
    SP(4),
    Body("Script location: E:\\code\\mri to cti\\experiment_v2\\train_experiment.py"),
    SP(8),
    H2("10.3 File Reference"),
    make_table([
        ["File / Folder", "Description"],
        [r"E:\code\mri to cti\kaggle_notebook.py",
         "Full Kaggle training notebook (Sections 1-12)"],
        [r"E:\code\mri to cti\experiment_v2\train_experiment.py",
         "Local experiment script — 3 configs × 30 epochs"],
        [r"E:\code\mri to cti\results after epoch 150",
         "Complete output from final Kaggle training run"],
        [r"E:\code\mri to cti\backup\full parameter test",
         "28-run grid search results (10 epochs each)"],
        [r"E:\code\mri to cti\backup\epooch 30 test",
         "30-epoch confirmation run results (top 4 configs)"],
        [r"E:\code\mri to cti\resume_state.pth",
         "Checkpoint file from session 1 end (epoch 70)"],
    ], col_widths=[8*cm, 9*cm]),
    SP(8),
    H2("10.4 Comparison Baseline"),
    Body("For academic comparison, this model's scores can be benchmarked against:"),
    SP(4),
    Bullet("Other notebooks on the same Kaggle dataset: darren2020/ct-to-mri-cgan → Notebooks tab"),
    Bullet("Fair comparison requires: same testA/testB split, same 256px size, same SSIM definition"),
    Bullet("Published CycleGAN papers on MRI/CT typically report SSIM of 0.30-0.55 depending on "
           "direction — MRI→CT at 0.486 is competitive; CT→MRI at 0.264 is below average"),
    Bullet("Pix2pix notebooks on the same dataset will show higher SSIM (0.6-0.8) "
           "because they use paired supervision — not a fair comparison to unpaired CycleGAN"),
    SP(8),
    HR(),
    P("End of Documentation", make_style("EndNote", fontSize=9, alignment=TA_CENTER,
                                         textColor=colors.HexColor("#888888"), italic=True)),
]

# ─────────────────────────────────────────────────────────────
# BUILD PDF
# ─────────────────────────────────────────────────────────────

doc = SimpleDocTemplate(
    OUTPUT_PATH,
    pagesize=A4,
    leftMargin=2*cm,
    rightMargin=2*cm,
    topMargin=2.2*cm,
    bottomMargin=2.2*cm,
    title="MRI CT CycleGAN — Model Documentation",
    author="CycleGAN Training Project",
)

doc.build(content)
print(f"PDF saved to: {OUTPUT_PATH}")
