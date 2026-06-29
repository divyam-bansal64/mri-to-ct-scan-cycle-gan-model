import os
import random
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image

from skimage.metrics import structural_similarity as ssim_metric

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

CONFIGS = {
    "A_baseline": {
        "n_blocks": 4,
        "lambda_cycle": 5.0,
        "lambda_identity": 2.5,
        "augment_flip": False,
    },
    "B_arch_aug": {
        "n_blocks": 9,
        "lambda_cycle": 10.0,
        "lambda_identity": 2.5,
        "augment_flip": True,
    },
    "C_low_identity": {
        "n_blocks": 9,
        "lambda_cycle": 10.0,
        "lambda_identity": 0.5,
        "augment_flip": True,
    },
}

BASE_CONFIG = {
    "ct_train_dir":  r"E:\code\mri to cti\Dataset\images\trainA",
    "mri_train_dir": r"E:\code\mri to cti\Dataset\images\trainB",
    "output_dir":    r"E:\code\mri to cti\experiment_v2\outputs",
    "image_size":    256,
    "input_nc":      3,
    "output_nc":     3,
    "ngf":           64,
    "ndf":           64,
    "n_layers_D":    3,
    "use_spect":     True,
    "epochs":        30,
    "batch_size":    1,
    "lr_G":          0.0001,
    "lr_D":          0.0001,
    "betas":         [0.5, 0.999],
    "decay_epoch":   15,
    "limit":         500,     # images per domain
    "val_split":     0.1,
    "save_interval": 10,
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

class MRICTDataset(Dataset):
    def __init__(self, img_dir, transform=None, limit=None):
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = sorted([f for f in Path(img_dir).iterdir() if f.suffix.lower() in exts])
        if limit:
            files = files[:limit]
        self.files = files
        self.transform = transform
        print(f"  Loaded {len(self.files)} images from {img_dir}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def get_transforms(image_size, augment_flip=False):
    ops = [transforms.Resize((image_size, image_size), transforms.InterpolationMode.BICUBIC)]
    if augment_flip:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    return transforms.Compose(ops)


# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────

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
    def forward(self, x): return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]
        for mult in [1, 2]:
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]
        for _ in range(n_blocks):
            layers.append(ResidualBlock(ngf * 4))
        for mult in [4, 2]:
            layers += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3,
                                   stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x): return self.model(x)


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

    def forward(self, x): return self.model(x)


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


# ─────────────────────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────────────────────

def gan_loss(pred, target_is_real):
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return nn.functional.mse_loss(pred, target)

criterion_cycle    = nn.L1Loss()
criterion_identity = nn.L1Loss()


# ─────────────────────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────────────────────

class ReplayBuffer:
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


# ─────────────────────────────────────────────────────────────
# VALIDATION SSIM
# ─────────────────────────────────────────────────────────────

def compute_val_ssim(G, loader_src, loader_tgt):
    G.eval()
    scores = []
    with torch.no_grad():
        for real_src, real_tgt in zip(loader_src, loader_tgt):
            real_src = real_src.to(device)
            fake_tgt = G(real_src)
            fake_np = ((fake_tgt.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)
            real_np = ((real_tgt.squeeze().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)
            scores.append(ssim_metric(real_np, fake_np, data_range=1.0, channel_axis=2))
    G.train()
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────

def run_experiment(config_name, exp_config):
    cfg = {**BASE_CONFIG, **exp_config}
    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"  n_blocks={cfg['n_blocks']} | lambda_cycle={cfg['lambda_cycle']} | "
          f"lambda_identity={cfg['lambda_identity']} | flip={cfg['augment_flip']}")
    print(f"{'='*60}")

    out_dir = os.path.join(cfg["output_dir"], config_name)
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)

    transform = get_transforms(cfg["image_size"], cfg["augment_flip"])

    full_ct  = MRICTDataset(cfg["ct_train_dir"],  transform, limit=cfg["limit"])
    full_mri = MRICTDataset(cfg["mri_train_dir"], transform, limit=cfg["limit"])

    def split(dataset):
        n_val = max(1, int(len(dataset) * cfg["val_split"]))
        return random_split(dataset, [len(dataset) - n_val, n_val],
                            generator=torch.Generator().manual_seed(SEED))

    ct_train,  ct_val  = split(full_ct)
    mri_train, mri_val = split(full_mri)

    loader_ct_train  = DataLoader(ct_train,  batch_size=cfg["batch_size"], shuffle=True,  num_workers=4, pin_memory=True)
    loader_mri_train = DataLoader(mri_train, batch_size=cfg["batch_size"], shuffle=True,  num_workers=4, pin_memory=True)
    loader_ct_val    = DataLoader(ct_val,    batch_size=1, shuffle=False, num_workers=4)
    loader_mri_val   = DataLoader(mri_val,   batch_size=1, shuffle=False, num_workers=4)

    G_A2B = init_weights(ResnetGenerator(cfg["input_nc"], cfg["output_nc"], cfg["ngf"], cfg["n_blocks"]).to(device))
    G_B2A = init_weights(ResnetGenerator(cfg["input_nc"], cfg["output_nc"], cfg["ngf"], cfg["n_blocks"]).to(device))
    D_A   = init_weights(NLayerDiscriminator(cfg["input_nc"], cfg["ndf"], cfg["n_layers_D"], cfg["use_spect"]).to(device))
    D_B   = init_weights(NLayerDiscriminator(cfg["input_nc"], cfg["ndf"], cfg["n_layers_D"], cfg["use_spect"]).to(device))

    optimizer_G = torch.optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()),
                                   lr=cfg["lr_G"], betas=tuple(cfg["betas"]))
    optimizer_D = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()),
                                   lr=cfg["lr_D"], betas=tuple(cfg["betas"]))

    def lr_lambda(epoch):
        e = epoch + 1
        if e < cfg["decay_epoch"]:
            return 1.0
        return max(0.0, 1.0 - (e - cfg["decay_epoch"]) / (cfg["epochs"] - cfg["decay_epoch"]))

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda)

    buffer_fake_A = ReplayBuffer(50)
    buffer_fake_B = ReplayBuffer(50)

    history = {"loss_G": [], "loss_D": [], "val_ssim_A2B": [], "val_ssim_B2A": []}
    best_ssim = -1.0

    for epoch in range(1, cfg["epochs"] + 1):
        epoch_loss_G, epoch_loss_D, n_batches = 0.0, 0.0, 0

        for real_A, real_B in zip(loader_ct_train, loader_mri_train):
            real_A, real_B = real_A.to(device), real_B.to(device)

            optimizer_G.zero_grad()
            loss_idt_A = criterion_identity(G_B2A(real_A), real_A) * cfg["lambda_identity"]
            loss_idt_B = criterion_identity(G_A2B(real_B), real_B) * cfg["lambda_identity"]
            fake_B = G_A2B(real_A)
            fake_A = G_B2A(real_B)
            loss_gan_A2B = gan_loss(D_B(fake_B), True)
            loss_gan_B2A = gan_loss(D_A(fake_A), True)
            loss_cycle_A = criterion_cycle(G_B2A(fake_B), real_A) * cfg["lambda_cycle"]
            loss_cycle_B = criterion_cycle(G_A2B(fake_A), real_B) * cfg["lambda_cycle"]
            loss_G = loss_gan_A2B + loss_gan_B2A + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
            loss_G.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            fake_A_buf = buffer_fake_A.push_and_pop(fake_A.detach())
            loss_D_A = 0.5 * (gan_loss(D_A(real_A), True) + gan_loss(D_A(fake_A_buf), False))
            fake_B_buf = buffer_fake_B.push_and_pop(fake_B.detach())
            loss_D_B = 0.5 * (gan_loss(D_B(real_B), True) + gan_loss(D_B(fake_B_buf), False))
            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            optimizer_D.step()

            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            n_batches += 1

        scheduler_G.step()
        scheduler_D.step()

        avg_G = epoch_loss_G / n_batches
        avg_D = epoch_loss_D / n_batches
        history["loss_G"].append(avg_G)
        history["loss_D"].append(avg_D)

        if epoch % 10 == 0:
            val_A2B = compute_val_ssim(G_A2B, loader_ct_val, loader_mri_val)
            val_B2A = compute_val_ssim(G_B2A, loader_mri_val, loader_ct_val)
            avg_ssim = (val_A2B + val_B2A) / 2
            history["val_ssim_A2B"].append(val_A2B)
            history["val_ssim_B2A"].append(val_B2A)
            if avg_ssim > best_ssim:
                best_ssim = avg_ssim
            print(f"[Epoch {epoch:03d}/{cfg['epochs']}] Loss_G: {avg_G:.4f} | Loss_D: {avg_D:.4f} | "
                  f"Val SSIM A2B: {val_A2B:.4f} | B2A: {val_B2A:.4f}")
        else:
            print(f"[Epoch {epoch:03d}/{cfg['epochs']}] Loss_G: {avg_G:.4f} | Loss_D: {avg_D:.4f}")

        if epoch % cfg["save_interval"] == 0:
            G_A2B.eval(); G_B2A.eval()
            with torch.no_grad():
                sample_ct  = next(iter(loader_ct_val)).to(device)
                sample_mri = next(iter(loader_mri_val)).to(device)
                fake_mri = G_A2B(sample_ct)
                fake_ct  = G_B2A(sample_mri)
                grid = make_grid(torch.cat([sample_ct, fake_mri, sample_mri, fake_ct], dim=0),
                                 nrow=4, normalize=True)
                save_image(grid, os.path.join(out_dir, "samples", f"epoch_{epoch:03d}.png"))
            G_A2B.train(); G_B2A.train()

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{config_name} done. Best avg SSIM: {best_ssim:.4f}")
    return history, best_ssim


# ─────────────────────────────────────────────────────────────
# MAIN — run all 3 configs sequentially
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}
    for name, exp_cfg in CONFIGS.items():
        history, best_ssim = run_experiment(name, exp_cfg)
        results[name] = {"best_ssim": best_ssim, "history": history}

    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for name, r in results.items():
        cfg = CONFIGS[name]
        print(f"{name:20s} | best_ssim: {r['best_ssim']:.4f} | "
              f"n_blocks={cfg['n_blocks']} | lambda_cycle={cfg['lambda_cycle']} | "
              f"lambda_identity={cfg['lambda_identity']}")

    with open(os.path.join(BASE_CONFIG["output_dir"], "experiment_summary.json"), "w") as f:
        summary = {k: {"best_ssim": v["best_ssim"], "config": CONFIGS[k]} for k, v in results.items()}
        json.dump(summary, f, indent=2)
    print("\nSummary saved.")
