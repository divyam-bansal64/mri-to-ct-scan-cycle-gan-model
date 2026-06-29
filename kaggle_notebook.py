# ============================================================
# MRI <-> CT CycleGAN — Full Training Notebook (Kaggle)
# Based on confirm_run_012 winning config, scaled to 256px
# ============================================================

# ─────────────────────────────────────────────────────────────
# SECTION 1 — SETUP & IMPORTS
# ─────────────────────────────────────────────────────────────

import os
import sys
import json
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")


# ─────────────────────────────────────────────────────────────
# SECTION 2 — EXPLORE DATASET PATHS (run this first, check output)
# ─────────────────────────────────────────────────────────────

BASE = Path("/kaggle/input/datasets/darren2020/ct-to-mri-cgan")

print("\n=== Dataset folder tree (2 levels deep) ===")
for root, dirs, files in os.walk(BASE):
    depth = str(root).replace(str(BASE), "").count(os.sep)
    if depth > 4:
        continue
    indent = "  " * depth
    print(f"{indent}{Path(root).name}/")
    if depth == 2:
        print(f"{indent}  ... ({len(files)} files)")


# ─────────────────────────────────────────────────────────────
# SECTION 3 — CONFIGURATION
# ─────────────────────────────────────────────────────────────
# Update CT_TRAIN_DIR / MRI_TRAIN_DIR / CT_TEST_DIR / MRI_TEST_DIR
# after checking the folder tree above

CONFIG = {
    # ── Paths ────────────────────────────────────────────────
    "ct_train_dir":  "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainA",
    "mri_train_dir": "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/trainB",
    "ct_test_dir":   "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/testA",
    "mri_test_dir":  "/kaggle/input/datasets/darren2020/ct-to-mri-cgan/Dataset/images/testB",
    "output_dir":    "/kaggle/working/outputs",

    # ── Image ────────────────────────────────────────────────
    "image_size":   256,
    "input_nc":     3,
    "output_nc":    3,

    # ── Model (proven from confirm_run_012) ──────────────────
    "ngf":          64,
    "ndf":          64,
    "n_blocks":     4,       # ResNet blocks in generator
    "n_layers_D":   3,       # PatchGAN depth → 30x30 patch map at 256px
    "use_spect":    True,    # Spectral norm on discriminator

    # ── Training ─────────────────────────────────────────────
    "epochs":       150,
    "batch_size":   1,
    "lr_G":         0.0001,
    "lr_D":         0.0001,
    "betas":        [0.5, 0.999],
    "decay_epoch":  75,      # Start linear LR decay at this epoch

    # ── Loss weights ─────────────────────────────────────────
    "lambda_cycle":    5.0,
    "lambda_identity": 2.5,

    # ── Misc ─────────────────────────────────────────────────
    "val_split":       0.1,  # Fraction of train data used for validation
    "save_interval":   10,   # Save checkpoint every N epochs
    "sample_interval": 10,   # Save sample images every N epochs
    "buffer_size":     50,   # Replay buffer size for discriminator
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(os.path.join(CONFIG["output_dir"], "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(CONFIG["output_dir"], "samples"), exist_ok=True)

print("\nConfig loaded.")
print(f"  image_size : {CONFIG['image_size']}")
print(f"  epochs     : {CONFIG['epochs']}")
print(f"  lr_G / lr_D: {CONFIG['lr_G']} / {CONFIG['lr_D']}")
print(f"  n_blocks   : {CONFIG['n_blocks']}")
print(f"  use_spect  : {CONFIG['use_spect']}")


# ─────────────────────────────────────────────────────────────
# SECTION 4 — DATASET
# ─────────────────────────────────────────────────────────────

class MRICTDataset(Dataset):
    """
    Loads images from a single flat directory (CT or MRI).
    Converts grayscale to RGB if needed.
    """
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        self.files = sorted([
            f for f in Path(img_dir).iterdir()
            if f.suffix.lower() in exts
        ])
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {img_dir}")
        print(f"  Loaded {len(self.files)} images from {img_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def get_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


transform = get_transforms(CONFIG["image_size"])

print("\nLoading datasets...")
full_ct_train  = MRICTDataset(CONFIG["ct_train_dir"],  transform)
full_mri_train = MRICTDataset(CONFIG["mri_train_dir"], transform)
ct_test        = MRICTDataset(CONFIG["ct_test_dir"],   transform)
mri_test       = MRICTDataset(CONFIG["mri_test_dir"],  transform)

# Carve out validation split from training data
def split_dataset(dataset, val_fraction, seed=SEED):
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    return random_split(dataset, [n_train, n_val],
                        generator=torch.Generator().manual_seed(seed))

ct_train,  ct_val  = split_dataset(full_ct_train,  CONFIG["val_split"])
mri_train, mri_val = split_dataset(full_mri_train, CONFIG["val_split"])

print(f"\nSplit summary:")
print(f"  CT  — train: {len(ct_train)}, val: {len(ct_val)}, test: {len(ct_test)}")
print(f"  MRI — train: {len(mri_train)}, val: {len(mri_val)}, test: {len(mri_test)}")

loader_ct_train  = DataLoader(ct_train,  batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
loader_mri_train = DataLoader(mri_train, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
loader_ct_val    = DataLoader(ct_val,    batch_size=1, shuffle=False, num_workers=2)
loader_mri_val   = DataLoader(mri_val,   batch_size=1, shuffle=False, num_workers=2)
loader_ct_test   = DataLoader(ct_test,   batch_size=1, shuffle=False, num_workers=2)
loader_mri_test  = DataLoader(mri_test,  batch_size=1, shuffle=False, num_workers=2)


# ─────────────────────────────────────────────────────────────
# SECTION 5 — MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────

# ── Generator ────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=4):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]
        # Downsample
        for mult in [1, 2]:
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]
        # Residual blocks
        for _ in range(n_blocks):
            layers.append(ResidualBlock(ngf * 4))
        # Upsample
        for mult in [4, 2]:
            layers += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3,
                                   stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ── Discriminator ─────────────────────────────────────────────

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_spect=True):
        super().__init__()

        def conv(in_c, out_c, stride, norm=True):
            c = nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1)
            if use_spect:
                c = nn.utils.spectral_norm(c)
            block = [c, nn.LeakyReLU(0.2, inplace=True)]
            if norm:
                block.insert(1, nn.InstanceNorm2d(out_c))
            return block

        layers = conv(input_nc, ndf, stride=2, norm=False)
        mult = 1
        for n in range(1, n_layers):
            layers += conv(ndf * mult, ndf * min(mult * 2, 8), stride=2)
            mult = min(mult * 2, 8)
        layers += conv(ndf * mult, ndf * min(mult * 2, 8), stride=1)
        mult = min(mult * 2, 8)

        final = nn.Conv2d(ndf * mult, 1, 4, padding=1)
        if use_spect:
            final = nn.utils.spectral_norm(final)
        layers.append(final)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def init_weights(net, gain=0.02):
    def _init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("InstanceNorm2d") != -1:
            if m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)
    net.apply(_init)
    return net


G_A2B = init_weights(ResnetGenerator(CONFIG["input_nc"], CONFIG["output_nc"],
                                      CONFIG["ngf"], CONFIG["n_blocks"]).to(device))
G_B2A = init_weights(ResnetGenerator(CONFIG["input_nc"], CONFIG["output_nc"],
                                      CONFIG["ngf"], CONFIG["n_blocks"]).to(device))
D_A   = init_weights(NLayerDiscriminator(CONFIG["input_nc"], CONFIG["ndf"],
                                          CONFIG["n_layers_D"], CONFIG["use_spect"]).to(device))
D_B   = init_weights(NLayerDiscriminator(CONFIG["input_nc"], CONFIG["ndf"],
                                          CONFIG["n_layers_D"], CONFIG["use_spect"]).to(device))

total_params = sum(p.numel() for p in G_A2B.parameters())
print(f"\nGenerator params : {total_params:,}")
total_params = sum(p.numel() for p in D_A.parameters())
print(f"Discriminator params: {total_params:,}")


# ─────────────────────────────────────────────────────────────
# SECTION 6 — LOSSES, OPTIMIZERS, SCHEDULERS
# ─────────────────────────────────────────────────────────────

# LSGAN adversarial loss
def gan_loss(pred, target_is_real):
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return nn.functional.mse_loss(pred, target)

criterion_cycle    = nn.L1Loss()
criterion_identity = nn.L1Loss()

optimizer_G = torch.optim.Adam(
    list(G_A2B.parameters()) + list(G_B2A.parameters()),
    lr=CONFIG["lr_G"], betas=tuple(CONFIG["betas"])
)
optimizer_D = torch.optim.Adam(
    list(D_A.parameters()) + list(D_B.parameters()),
    lr=CONFIG["lr_D"], betas=tuple(CONFIG["betas"])
)

# Linear decay: keep lr constant until decay_epoch, then decay to 0
def lr_lambda(epoch):
    # epoch here is 0-indexed (LambdaLR passes 0,1,2,... internally)
    # our training epochs are 1-indexed, so offset by 1
    e = epoch + 1
    if e < CONFIG["decay_epoch"]:
        return 1.0
    return max(0.0, 1.0 - (e - CONFIG["decay_epoch"]) /
               (CONFIG["epochs"] - CONFIG["decay_epoch"]))

scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda)


# ─────────────────────────────────────────────────────────────
# SECTION 7 — IMAGE REPLAY BUFFER
# ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Stores up to `max_size` previously generated images.
    Returns a mix of current and stored images to the discriminator
    — prevents the discriminator from over-fitting to the most
    recent generator output and stabilises GAN training.
    """
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.buffer = []

    def push_and_pop(self, images):
        result = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.buffer) < self.max_size:
                self.buffer.append(img)
                result.append(img)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    tmp = self.buffer[idx].clone()
                    self.buffer[idx] = img
                    result.append(tmp)
                else:
                    result.append(img)
        return torch.cat(result, dim=0)

buffer_fake_A = ReplayBuffer(CONFIG["buffer_size"])
buffer_fake_B = ReplayBuffer(CONFIG["buffer_size"])


# ─────────────────────────────────────────────────────────────
# SECTION 8 — VALIDATION METRIC
# ─────────────────────────────────────────────────────────────

def compute_val_ssim(G, loader_src, loader_tgt):
    """
    Runs G on all images in loader_src and computes mean SSIM
    against loader_tgt images (positional match — same index).
    Returns mean SSIM across the validation set.
    """
    G.eval()
    scores = []
    with torch.no_grad():
        for real_src, real_tgt in zip(loader_src, loader_tgt):
            real_src = real_src.to(device)
            fake_tgt = G(real_src)
            # Denormalize [-1,1] -> [0,1]
            fake_np = ((fake_tgt.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)
            real_np = ((real_tgt.squeeze().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)
            score = ssim(real_np, fake_np, data_range=1.0, channel_axis=2)
            scores.append(score)
    G.train()
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────
# SECTION 9 — TRAINING LOOP
# ─────────────────────────────────────────────────────────────

history = {"loss_G": [], "loss_D": [], "val_ssim_A2B": [], "val_ssim_B2A": []}
best_ssim = -1.0
best_epoch = 0
start_epoch = 1

# ── Resume from checkpoint if available ──────────────────────
save_resume_path = f"{CONFIG['output_dir']}/checkpoints/resume_state.pth"

# Full resume state (preferred)
full_resume_path = "/kaggle/input/datasets/divyambnsl/cycle-gan-resume-checkpoint/resume_state.pth"

# Partial resume — individual weight files from epoch 80 (no optimizer state)
partial_resume_dir = "/kaggle/input/datasets/divyambnsl/cycle-gan-resume-checkpoint-80"
partial_resume_epoch = 80

load_resume_path = None
if os.path.exists(save_resume_path):
    load_resume_path = save_resume_path
elif os.path.exists(full_resume_path):
    load_resume_path = full_resume_path

if load_resume_path:
    state = torch.load(load_resume_path, map_location=device)
    G_A2B.load_state_dict(state["G_A2B"])
    G_B2A.load_state_dict(state["G_B2A"])
    D_A.load_state_dict(state["D_A"])
    D_B.load_state_dict(state["D_B"])
    optimizer_G.load_state_dict(state["opt_G"])
    optimizer_D.load_state_dict(state["opt_D"])
    scheduler_G.load_state_dict(state["sched_G"])
    scheduler_D.load_state_dict(state["sched_D"])
    history    = state["history"]
    best_ssim  = state["best_ssim"]
    best_epoch = state["best_epoch"]
    start_epoch = state["epoch"] + 1
    print(f"Resumed from epoch {state['epoch']} (best SSIM so far: {best_ssim:.4f})")
elif os.path.exists(partial_resume_dir):
    # Partial resume: load weights only, restart optimizer, advance scheduler
    G_A2B.load_state_dict(torch.load(f"{partial_resume_dir}/G_A2B_best.pth", map_location=device))
    G_B2A.load_state_dict(torch.load(f"{partial_resume_dir}/G_B2A_best.pth", map_location=device))
    D_A.load_state_dict(torch.load(f"{partial_resume_dir}/D_A_best.pth",     map_location=device))
    D_B.load_state_dict(torch.load(f"{partial_resume_dir}/D_B_best.pth",     map_location=device))
    # Advance schedulers to correct LR for epoch 81
    # Step 79 times (not 80) because the training loop adds one more step at end of epoch 81
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(partial_resume_epoch - 1):
            scheduler_G.step()
            scheduler_D.step()
    start_epoch = partial_resume_epoch + 1
    best_ssim   = 0.3602   # known best SSIM from session 2
    best_epoch  = partial_resume_epoch
    # Pre-fill history with NaN for missing epochs so x-axis stays correct in plots
    history["loss_G"]      = [float("nan")] * partial_resume_epoch
    history["loss_D"]      = [float("nan")] * partial_resume_epoch
    history["val_ssim_A2B"] = [float("nan")] * (partial_resume_epoch // 10)
    history["val_ssim_B2A"] = [float("nan")] * (partial_resume_epoch // 10)
    print(f"Partial resume from epoch {partial_resume_epoch} (weights only, optimizer restarted)")
    print(f"  LR after advance: {scheduler_G.get_last_lr()[0]:.6f}")
else:
    print("No resume checkpoint found — starting from scratch")

print("\n" + "="*60)
print(f"Starting training from epoch {start_epoch}")
print("="*60)

for epoch in range(start_epoch, CONFIG["epochs"] + 1):
    epoch_loss_G, epoch_loss_D, n_batches = 0.0, 0.0, 0

    for real_A, real_B in zip(loader_ct_train, loader_mri_train):
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # ── Train Generators ─────────────────────────────────
        optimizer_G.zero_grad()

        # Identity losses
        loss_idt_A = criterion_identity(G_B2A(real_A), real_A) * CONFIG["lambda_identity"]
        loss_idt_B = criterion_identity(G_A2B(real_B), real_B) * CONFIG["lambda_identity"]

        # GAN losses
        fake_B = G_A2B(real_A)
        fake_A = G_B2A(real_B)
        loss_gan_A2B = gan_loss(D_B(fake_B), True)
        loss_gan_B2A = gan_loss(D_A(fake_A), True)

        # Cycle losses
        loss_cycle_A = criterion_cycle(G_B2A(fake_B), real_A) * CONFIG["lambda_cycle"]
        loss_cycle_B = criterion_cycle(G_A2B(fake_A), real_B) * CONFIG["lambda_cycle"]

        loss_G = (loss_gan_A2B + loss_gan_B2A +
                  loss_cycle_A + loss_cycle_B +
                  loss_idt_A   + loss_idt_B)
        loss_G.backward()
        optimizer_G.step()

        # ── Train Discriminators ──────────────────────────────
        optimizer_D.zero_grad()

        # D_A: real CT vs fake CT from buffer
        fake_A_buf = buffer_fake_A.push_and_pop(fake_A.detach())
        loss_D_A = 0.5 * (gan_loss(D_A(real_A), True) +
                          gan_loss(D_A(fake_A_buf), False))

        # D_B: real MRI vs fake MRI from buffer
        fake_B_buf = buffer_fake_B.push_and_pop(fake_B.detach())
        loss_D_B = 0.5 * (gan_loss(D_B(real_B), True) +
                          gan_loss(D_B(fake_B_buf), False))

        loss_D = loss_D_A + loss_D_B
        loss_D.backward()
        optimizer_D.step()

        epoch_loss_G += loss_G.item()
        epoch_loss_D += loss_D.item()
        n_batches += 1

    # ── LR Step ──────────────────────────────────────────────
    scheduler_G.step()
    scheduler_D.step()

    avg_G = epoch_loss_G / n_batches
    avg_D = epoch_loss_D / n_batches
    history["loss_G"].append(avg_G)
    history["loss_D"].append(avg_D)

    # ── Validation SSIM every 10 epochs ──────────────────────
    if epoch % 10 == 0:
        val_ssim_A2B = compute_val_ssim(G_A2B, loader_ct_val,  loader_mri_val)
        val_ssim_B2A = compute_val_ssim(G_B2A, loader_mri_val, loader_ct_val)
        avg_ssim = (val_ssim_A2B + val_ssim_B2A) / 2
        history["val_ssim_A2B"].append(val_ssim_A2B)
        history["val_ssim_B2A"].append(val_ssim_B2A)

        print(f"[Epoch {epoch:03d}/{CONFIG['epochs']}] "
              f"Loss_G: {avg_G:.4f} | Loss_D: {avg_D:.4f} | "
              f"Val SSIM A2B: {val_ssim_A2B:.4f} | B2A: {val_ssim_B2A:.4f} | "
              f"LR: {scheduler_G.get_last_lr()[0]:.6f}")

        # Save best checkpoint
        if avg_ssim > best_ssim:
            best_ssim  = avg_ssim
            best_epoch = epoch
            torch.save(G_A2B.state_dict(), f"{CONFIG['output_dir']}/checkpoints/G_A2B_best.pth")
            torch.save(G_B2A.state_dict(), f"{CONFIG['output_dir']}/checkpoints/G_B2A_best.pth")
            torch.save(D_A.state_dict(),   f"{CONFIG['output_dir']}/checkpoints/D_A_best.pth")
            torch.save(D_B.state_dict(),   f"{CONFIG['output_dir']}/checkpoints/D_B_best.pth")
            print(f"  ✓ Best model saved (SSIM: {best_ssim:.4f})")
    else:
        print(f"[Epoch {epoch:03d}/{CONFIG['epochs']}] "
              f"Loss_G: {avg_G:.4f} | Loss_D: {avg_D:.4f}")

    # ── Save periodic checkpoint + full resume state ─────────
    if epoch % CONFIG["save_interval"] == 0:
        torch.save(G_A2B.state_dict(), f"{CONFIG['output_dir']}/checkpoints/G_A2B_epoch{epoch}.pth")
        torch.save(G_B2A.state_dict(), f"{CONFIG['output_dir']}/checkpoints/G_B2A_epoch{epoch}.pth")
        torch.save({
            "epoch":      epoch,
            "G_A2B":      G_A2B.state_dict(),
            "G_B2A":      G_B2A.state_dict(),
            "D_A":        D_A.state_dict(),
            "D_B":        D_B.state_dict(),
            "opt_G":      optimizer_G.state_dict(),
            "opt_D":      optimizer_D.state_dict(),
            "sched_G":    scheduler_G.state_dict(),
            "sched_D":    scheduler_D.state_dict(),
            "history":    history,
            "best_ssim":  best_ssim,
            "best_epoch": best_epoch,
        }, save_resume_path)

    # ── Save sample images ────────────────────────────────────
    if epoch % CONFIG["sample_interval"] == 0:
        G_A2B.eval()
        G_B2A.eval()
        with torch.no_grad():
            sample_ct  = next(iter(loader_ct_val)).to(device)
            sample_mri = next(iter(loader_mri_val)).to(device)
            fake_mri   = G_A2B(sample_ct)
            fake_ct    = G_B2A(sample_mri)
            grid = make_grid(
                torch.cat([sample_ct, fake_mri, sample_mri, fake_ct], dim=0),
                nrow=4, normalize=True
            )
            save_image(grid, f"{CONFIG['output_dir']}/samples/epoch_{epoch:03d}.png")
        G_A2B.train()
        G_B2A.train()

print(f"\nTraining complete. Best SSIM: {best_ssim:.4f} at epoch {best_epoch}")


# ─────────────────────────────────────────────────────────────
# SECTION 10 — LOSS CURVES
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 4))

axes[0].plot(history["loss_G"], label="Generator")
axes[0].plot(history["loss_D"], label="Discriminator")
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True)

val_epochs = list(range(10, 10 * len(history["val_ssim_A2B"]) + 1, 10))
axes[1].plot(val_epochs, history["val_ssim_A2B"], label="CT→MRI")
axes[1].plot(val_epochs, history["val_ssim_B2A"], label="MRI→CT")
axes[1].set_title("Validation SSIM")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(val_epochs[:len(history["val_ssim_A2B"])], history["val_ssim_A2B"], label="A2B")
axes[2].axvline(x=CONFIG["decay_epoch"], color="r", linestyle="--", label="LR decay start")
axes[2].set_title("SSIM vs LR Decay")
axes[2].set_xlabel("Epoch")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/training_curves.png", dpi=150)
plt.show()


# ─────────────────────────────────────────────────────────────
# SECTION 11 — EVALUATION ON TEST SET
# ─────────────────────────────────────────────────────────────

# Load best saved weights
G_A2B.load_state_dict(torch.load(f"{CONFIG['output_dir']}/checkpoints/G_A2B_best.pth", map_location=device))
G_B2A.load_state_dict(torch.load(f"{CONFIG['output_dir']}/checkpoints/G_B2A_best.pth", map_location=device))
G_A2B.eval()
G_B2A.eval()

def evaluate(G, loader_src, loader_tgt, direction_label):
    mae_scores, ssim_scores, psnr_scores = [], [], []

    with torch.no_grad():
        for real_src, real_tgt in zip(loader_src, loader_tgt):
            real_src = real_src.to(device)
            fake_tgt = G(real_src)

            fake_np = ((fake_tgt.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)
            real_np = ((real_tgt.squeeze().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)

            mae_scores.append(np.mean(np.abs(fake_np - real_np)))
            ssim_scores.append(ssim(real_np, fake_np, data_range=1.0, channel_axis=2))
            psnr_scores.append(psnr(real_np, fake_np, data_range=1.0))

    print(f"\n── {direction_label} ──")
    print(f"  MAE  : {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"  SSIM : {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
    print(f"  PSNR : {np.mean(psnr_scores):.2f} ± {np.std(psnr_scores):.2f} dB")

    return {
        "direction": direction_label,
        "mae_mean":  float(np.mean(mae_scores)),
        "mae_std":   float(np.std(mae_scores)),
        "ssim_mean": float(np.mean(ssim_scores)),
        "ssim_std":  float(np.std(ssim_scores)),
        "psnr_mean": float(np.mean(psnr_scores)),
        "psnr_std":  float(np.std(psnr_scores)),
    }

print("\n" + "="*60)
print("Test Set Evaluation (best checkpoint)")
print("="*60)

results_A2B = evaluate(G_A2B, loader_ct_test,  loader_mri_test, "CT → MRI")
results_B2A = evaluate(G_B2A, loader_mri_test, loader_ct_test,  "MRI → CT")

# Save metrics
with open(f"{CONFIG['output_dir']}/test_metrics.json", "w") as f:
    json.dump([results_A2B, results_B2A], f, indent=2)
print(f"\nMetrics saved to {CONFIG['output_dir']}/test_metrics.json")


# ─────────────────────────────────────────────────────────────
# SECTION 12 — VISUAL RESULTS
# ─────────────────────────────────────────────────────────────

def show_results(G, loader_src, loader_tgt, title, n=6):
    fig, axes = plt.subplots(3, n, figsize=(n * 3, 9))
    fig.suptitle(title, fontsize=14)

    with torch.no_grad():
        for i, (real_src, real_tgt) in enumerate(zip(loader_src, loader_tgt)):
            if i >= n:
                break
            real_src = real_src.to(device)
            fake_tgt = G(real_src)

            def to_img(t):
                return ((t.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)

            axes[0, i].imshow(to_img(real_src), cmap="gray")
            axes[0, i].set_title("Input")
            axes[0, i].axis("off")

            axes[1, i].imshow(to_img(fake_tgt), cmap="gray")
            axes[1, i].set_title("Generated")
            axes[1, i].axis("off")

            axes[2, i].imshow(to_img(real_tgt), cmap="gray")
            axes[2, i].set_title("Real Target")
            axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/{title.replace(' ', '_')}.png", dpi=150)
    plt.show()

show_results(G_A2B, loader_ct_test,  loader_mri_test, "CT to MRI — Test Results")
show_results(G_B2A, loader_mri_test, loader_ct_test,  "MRI to CT — Test Results")
