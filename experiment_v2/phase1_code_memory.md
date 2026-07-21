# Phase 1 Code and Results Memory

This file serves as a backup memory file documenting the code and findings of Phase 1 (Single-Variable Ablations) of the CycleGAN MRI-to-CT experiments before proceeding to Phase 2.

## Phase 1 Results Summary Table

| Configuration | n_blocks | lambda_cycle | lambda_idt_A | lambda_idt_B | upsample_mode | channels | augment_flip | augment_spatial | epochs | Best Cycle-SSIM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A_baseline** | 4 | 5.0 | 2.5 | 2.5 | `ConvTranspose` | 3 | False | False | 30 | **0.7930** |
| **B_capacity** | **9** | 5.0 | 2.5 | 2.5 | `ConvTranspose` | 3 | False | False | 30 | **0.8066** |
| **D_low_idt** | 9 | 5.0 | **0.5** | **0.5** | `ConvTranspose` | 3 | False | False | 30 | **0.7743** |
| **R_resize** | 4 | 5.0 | 2.5 | 2.5 | **ResizeConv** | 3 | False | False | 30 | **0.7009** |
| **F_asym_idt** | 9 | 5.0 | 2.5 | **0.5** | `ConvTranspose` | 3 | False | False | 30 | **0.8056** |
| **S_spatial** | 4 | 5.0 | 2.5 | 2.5 | `ConvTranspose` | 3 | False | **True** | 50 | **0.7673** |

### Key Observations:
1. **Network Capacity (Winner):** Deeper generators (`n_blocks = 9` in `B_capacity`) perform better, achieving **0.8066** SSIM compared to **0.7930** SSIM of the baseline 4-block generators.
2. **Identity Loss Weight:** Lowering identity loss to `0.5` (`D_low_idt`) reduces the SSIM significantly to **0.7743**, confirming that high identity constraints (`2.5`) are necessary to preserve structure.
3. **Upsampling Mode (ConvTranspose Wins):** `ResizeConv` (`R_resize`) performed poorly on this task, dropping the SSIM to **0.7009** (nearly 10% lower than baseline). Standard `ConvTranspose` is superior.
4. **Asymmetric Identity Loss:** `F_asym_idt` performs practically identical to `B_capacity` (**0.8056** vs **0.8066**). A higher identity weight of `2.5` on the CT generator path is beneficial.
5. **Spatial Augmentation:** Adding spatial augmentations locally (`S_spatial`) led to a drop in SSIM (**0.7673**). This indicates that spatial changes (rotation, scaling) make convergence slower and harder within 50 epochs, or it might require more epochs.

---

## Phase 1 Source Code (`train_experiment.py`)

```python
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
        "lambda_identity_A": 2.5,
        "lambda_identity_B": 2.5,
        "upsample_mode": "ConvTranspose",
        "channels": 3,
        "augment_flip": False,
        "augment_spatial": False,
        "epochs": 30,
    },
    "B_capacity": {
        "n_blocks": 9,
        "lambda_cycle": 5.0,
        "lambda_identity_A": 2.5,
        "lambda_identity_B": 2.5,
        "upsample_mode": "ConvTranspose",
        "channels": 3,
        "augment_flip": False,
        "augment_spatial": False,
        "epochs": 30,
    },
    "D_low_idt": {
        "n_blocks": 9,
        "lambda_cycle": 5.0,
        "lambda_identity_A": 0.5,
        "lambda_identity_B": 0.5,
        "upsample_mode": "ConvTranspose",
        "channels": 3,
        "augment_flip": False,
        "augment_spatial": False,
        "epochs": 30,
    },
    "R_resize": {
        "n_blocks": 4,
        "lambda_cycle": 5.0,
        "lambda_identity_A": 2.5,
        "lambda_identity_B": 2.5,
        "upsample_mode": "ResizeConv",
        "channels": 3,
        "augment_flip": False,
        "augment_spatial": False,
        "epochs": 30,
    },
    "F_asym_idt": {
        "n_blocks": 9,
        "lambda_cycle": 5.0,
        "lambda_identity_A": 2.5,  # Keep high for MRI -> CT (G_B2A)
        "lambda_identity_B": 0.5,  # Lower for CT -> MRI (G_A2B)
        "upsample_mode": "ConvTranspose",
        "channels": 3,
        "augment_flip": False,
        "augment_spatial": False,
        "epochs": 30,
    },
    "S_spatial": {
        "n_blocks": 4,
        "lambda_cycle": 5.0,
        "lambda_identity_A": 2.5,
        "lambda_identity_B": 2.5,
        "upsample_mode": "ConvTranspose",
        "channels": 3,
        "augment_flip": False,
        "augment_spatial": True,    # rotation ±5°, scale 0.9–1.1
        "epochs": 50,
    },
    "G_combined": {
        "n_blocks": 9,
        "lambda_cycle": 5.0,
        "lambda_identity_A": 0.5,
        "lambda_identity_B": 0.5,
        "upsample_mode": "ResizeConv",
        "channels": 1,              # Grayscale 1 channel
        "augment_flip": False,
        "augment_spatial": False,
        "epochs": 30,
    },
    "H_full": {
        "n_blocks": 9,
        "lambda_cycle": 5.0,
        "lambda_identity_A": 0.5,
        "lambda_identity_B": 0.5,
        "upsample_mode": "ResizeConv",
        "channels": 1,
        "augment_flip": True,
        "augment_spatial": True,    # rotation ±5°, scale 0.9–1.1
        "epochs": 50,               # 50 epochs for spatial augmentations to converge
    }
}

BASE_CONFIG = {
    "ct_train_dir":  r"E:\code\mri to cti\Dataset\images\trainA",
    "mri_train_dir": r"E:\code\mri to cti\Dataset\images\trainB",
    "output_dir":    r"E:\code\mri to cti\experiment_v2\outputs",
    "image_size":    256,
    "ngf":           64,
    "ndf":           64,
    "n_layers_D":    3,
    "use_spect":     True,
    "batch_size":    1,
    "lr_G":          0.0001,
    "lr_D":          0.0001,
    "betas":         [0.5, 0.999],
    "decay_epoch":   15,
    "limit":         500,     # images per domain for local testing
    "val_split":     0.1,
    "save_interval": 10,
    "num_workers":   0,       # Windows-safe default to prevent multi-process crashes
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
    def __init__(self, img_dir, transform=None, limit=None, channels=3):
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = sorted([f for f in Path(img_dir).iterdir() if f.suffix.lower() in exts])
        if limit:
            files = files[:limit]
        self.files = files
        self.transform = transform
        self.channels = channels
        print(f"  Loaded {len(self.files)} images from {img_dir}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        mode = "RGB" if self.channels == 3 else "L"
        img = Image.open(self.files[idx]).convert(mode)
        if self.transform:
            img = self.transform(img)
        return img


def get_transforms(image_size, augment_flip=False, augment_spatial=False, channels=3):
    ops = [transforms.Resize((image_size, image_size), transforms.InterpolationMode.BICUBIC)]
    
    if augment_spatial:
        ops.append(transforms.RandomAffine(degrees=5, scale=(0.9, 1.1)))
        
    if augment_flip:
        ops.append(transforms.RandomHorizontalFlip())
        
    ops.append(transforms.ToTensor())
    
    if channels == 1:
        ops.append(transforms.Normalize([0.5], [0.5]))
    else:
        ops.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        
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
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9, upsample_mode="ConvTranspose"):
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
            if upsample_mode == "ResizeConv":
                layers += [
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(ngf * mult, ngf * mult // 2, 3, stride=1, padding=1),
                    nn.InstanceNorm2d(ngf * mult // 2),
                    nn.ReLU(inplace=True),
                ]
            else:  # ConvTranspose
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
            weight_attr = "weight_orig" if hasattr(m, "weight_orig") else "weight"
            w_data = getattr(m, weight_attr).data
            nn.init.normal_(w_data, 0.0, gain)
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
# VALIDATION CYCLE & IDENTITY METRICS
# ─────────────────────────────────────────────────────────────

def compute_val_cycle_metrics(G_A2B, G_B2A, loader_ct, loader_mri, channels=3):
    G_A2B.eval()
    G_B2A.eval()
    
    ssim_A, ssim_B = [], []
    mae_A, mae_B = [], []
    
    with torch.no_grad():
        for real_A in loader_ct:
            real_A = real_A.to(device)
            fake_B = G_A2B(real_A)
            rec_A = G_B2A(fake_B)
            
            real_A_np = ((real_A[0].cpu().numpy() + 1) / 2).clip(0, 1)
            rec_A_np = ((rec_A[0].cpu().numpy() + 1) / 2).clip(0, 1)
            
            if channels == 3:
                real_A_np = real_A_np.transpose(1, 2, 0)
                rec_A_np = rec_A_np.transpose(1, 2, 0)
                ssim_val = ssim_metric(real_A_np, rec_A_np, data_range=1.0, channel_axis=2)
            else:
                real_A_np = real_A_np[0]
                rec_A_np = rec_A_np[0]
                ssim_val = ssim_metric(real_A_np, rec_A_np, data_range=1.0)
                
            ssim_A.append(ssim_val)
            mae_A.append(float(np.mean(np.abs(real_A_np - rec_A_np))))
            
        for real_B in loader_mri:
            real_B = real_B.to(device)
            fake_A = G_B2A(real_B)
            rec_B = G_A2B(fake_A)
            
            real_B_np = ((real_B[0].cpu().numpy() + 1) / 2).clip(0, 1)
            rec_B_np = ((rec_B[0].cpu().numpy() + 1) / 2).clip(0, 1)
            
            if channels == 3:
                real_B_np = real_B_np.transpose(1, 2, 0)
                rec_B_np = rec_B_np.transpose(1, 2, 0)
                ssim_val = ssim_metric(real_B_np, rec_B_np, data_range=1.0, channel_axis=2)
            else:
                real_B_np = real_B_np[0]
                rec_B_np = rec_B_np[0]
                ssim_val = ssim_metric(real_B_np, rec_B_np, data_range=1.0)
                
            ssim_B.append(ssim_val)
            mae_B.append(float(np.mean(np.abs(real_B_np - rec_B_np))))
            
    G_A2B.train()
    G_B2A.train()
    
    return {
        "ssim_rec_A": float(np.mean(ssim_A)),
        "ssim_rec_B": float(np.mean(ssim_B)),
        "mae_rec_A": float(np.mean(mae_A)),
        "mae_rec_B": float(np.mean(mae_B)),
    }


def compute_val_identity_metrics(G_A2B, G_B2A, loader_ct, loader_mri, channels=3):
    G_A2B.eval()
    G_B2A.eval()
    
    mae_idt_A, mae_idt_B = [], []
    
    with torch.no_grad():
        for real_A in loader_ct:
            real_A = real_A.to(device)
            idt_A = G_B2A(real_A)
            
            real_A_np = ((real_A[0].cpu().numpy() + 1) / 2).clip(0, 1)
            idt_A_np = ((idt_A[0].cpu().numpy() + 1) / 2).clip(0, 1)
            
            if channels == 1:
                real_A_np = real_A_np[0]
                idt_A_np = idt_A_np[0]
                
            mae_idt_A.append(float(np.mean(np.abs(real_A_np - idt_A_np))))
            
        for real_B in loader_mri:
            real_B = real_B.to(device)
            idt_B = G_A2B(real_B)
            
            real_B_np = ((real_B[0].cpu().numpy() + 1) / 2).clip(0, 1)
            idt_B_np = ((idt_B[0].cpu().numpy() + 1) / 2).clip(0, 1)
            
            if channels == 1:
                real_B_np = real_B_np[0]
                idt_B_np = idt_B_np[0]
                
            mae_idt_B.append(float(np.mean(np.abs(real_B_np - idt_B_np))))
            
    G_A2B.train()
    G_B2A.train()
    
    return {
        "mae_idt_A": float(np.mean(mae_idt_A)),
        "mae_idt_B": float(np.mean(mae_idt_B)),
    }


# ─────────────────────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────

def run_experiment(config_name, exp_config):
    cfg = {**BASE_CONFIG, **exp_config}
    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"  n_blocks={cfg['n_blocks']} | lambda_cycle={cfg['lambda_cycle']} | "
          f"lambda_idt_A={cfg['lambda_identity_A']} | lambda_idt_B={cfg['lambda_identity_B']} | "
          f"upsample={cfg['upsample_mode']} | channels={cfg['channels']} | "
          f"flip={cfg['augment_flip']} | spatial_aug={cfg.get('augment_spatial', False)}")
    print(f"{'='*60}")

    out_dir = os.path.join(cfg["output_dir"], config_name)
    checkpoints_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    train_transform = get_transforms(cfg["image_size"], cfg["augment_flip"], cfg.get("augment_spatial", False), cfg["channels"])
    val_transform = get_transforms(cfg["image_size"], augment_flip=False, augment_spatial=False, channels=cfg["channels"])

    full_ct_train  = MRICTDataset(cfg["ct_train_dir"],  train_transform, limit=cfg["limit"], channels=cfg["channels"])
    full_ct_val    = MRICTDataset(cfg["ct_train_dir"],  val_transform, limit=cfg["limit"], channels=cfg["channels"])
    full_mri_train = MRICTDataset(cfg["mri_train_dir"], train_transform, limit=cfg["limit"], channels=cfg["channels"])
    full_mri_val   = MRICTDataset(cfg["mri_train_dir"], val_transform, limit=cfg["limit"], channels=cfg["channels"])

    def get_split_subsets(full_train, full_val):
        n_val = max(1, int(len(full_train) * cfg["val_split"]))
        n_train = len(full_train) - n_val
        indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(SEED)).tolist()
        train_subset = Subset(full_train, indices[:n_train])
        val_subset = Subset(full_val, indices[n_train:])
        return train_subset, val_subset

    ct_train, ct_val = get_split_subsets(full_ct_train, full_ct_val)
    mri_train, mri_val = get_split_subsets(full_mri_train, full_mri_val)

    loader_ct_train  = DataLoader(ct_train,  batch_size=cfg["batch_size"], shuffle=True,  num_workers=cfg["num_workers"], pin_memory=True)
    loader_mri_train = DataLoader(mri_train, batch_size=cfg["batch_size"], shuffle=True,  num_workers=cfg["num_workers"], pin_memory=True)
    loader_ct_val    = DataLoader(ct_val,    batch_size=1, shuffle=False, num_workers=cfg["num_workers"])
    loader_mri_val   = DataLoader(mri_val,   batch_size=1, shuffle=False, num_workers=cfg["num_workers"])

    G_A2B = init_weights(ResnetGenerator(cfg["channels"], cfg["channels"], cfg["ngf"], cfg["n_blocks"], cfg["upsample_mode"]).to(device))
    G_B2A = init_weights(ResnetGenerator(cfg["channels"], cfg["channels"], cfg["ngf"], cfg["n_blocks"], cfg["upsample_mode"]).to(device))
    D_A   = init_weights(NLayerDiscriminator(cfg["channels"], cfg["ndf"], cfg["n_layers_D"], cfg["use_spect"]).to(device))
    D_B   = init_weights(NLayerDiscriminator(cfg["channels"], cfg["ndf"], cfg["n_layers_D"], cfg["use_spect"]).to(device))

    optimizer_G = torch.optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()),
                                   lr=cfg["lr_G"], betas=tuple(cfg["betas"]))
    optimizer_D = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()),
                                   lr=cfg["lr_D"], betas=tuple(cfg["betas"]))

    def lr_lambda(epoch):
        e = epoch + 1
        if e < cfg["decay_epoch"]:
            return 1.0
        denom = cfg["epochs"] - cfg["decay_epoch"]
        if denom <= 0:
            return 1.0
        return max(0.0, 1.0 - (e - cfg["decay_epoch"]) / denom)

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda)

    buffer_fake_A = ReplayBuffer(50)
    buffer_fake_B = ReplayBuffer(50)

    history = {
        "loss_G": [], 
        "loss_D": [], 
        "val_rec_ssim_A": [], 
        "val_rec_ssim_B": [],
        "val_rec_mae_A": [],
        "val_rec_mae_B": [],
        "val_idt_mae_A": [],
        "val_idt_mae_B": []
    }
    best_ssim = -1.0
    start_epoch = 1

    resume_path = os.path.join(checkpoints_dir, "resume_state.pth")
    if os.path.exists(resume_path):
        print(f"Loading checkpoint '{resume_path}' to resume training...")
        state = torch.load(resume_path, map_location=device)
        G_A2B.load_state_dict(state["G_A2B"])
        G_B2A.load_state_dict(state["G_B2A"])
        D_A.load_state_dict(state["D_A"])
        D_B.load_state_dict(state["D_B"])
        optimizer_G.load_state_dict(state["opt_G"])
        optimizer_D.load_state_dict(state["opt_D"])
        scheduler_G.load_state_dict(state["sched_G"])
        scheduler_D.load_state_dict(state["sched_D"])
        history = state["history"]
        best_ssim = state["best_ssim"]
        start_epoch = state["epoch"] + 1
        print(f"Resuming successfully from epoch {start_epoch} (Best SSIM: {best_ssim:.4f})")

    log_file = os.path.join(out_dir, "train_log.txt")
    with open(log_file, "a") as f:
        f.write(f"\n--- Run Started/Resumed: Epoch {start_epoch} to {cfg['epochs']} ---\n")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        epoch_loss_G, epoch_loss_D, n_batches = 0.0, 0.0, 0

        len_ct = len(loader_ct_train)
        len_mri = len(loader_mri_train)
        max_batches = max(len_ct, len_mri)
        
        iter_ct = iter(loader_ct_train)
        iter_mri = iter(loader_mri_train)

        for _ in range(max_batches):
            try:
                real_A = next(iter_ct)
            except StopIteration:
                iter_ct = iter(loader_ct_train)
                real_A = next(iter_ct)
                
            try:
                real_B = next(iter_mri)
            except StopIteration:
                iter_mri = iter(loader_mri_train)
                real_B = next(iter_mri)

            real_A, real_B = real_A.to(device), real_B.to(device)

            optimizer_G.zero_grad()
            loss_idt_A = criterion_identity(G_B2A(real_A), real_A) * cfg["lambda_identity_A"]
            loss_idt_B = criterion_identity(G_A2B(real_B), real_B) * cfg["lambda_identity_B"]
            
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
            metrics = compute_val_cycle_metrics(G_A2B, G_B2A, loader_ct_val, loader_mri_val, channels=cfg["channels"])
            idt_metrics = compute_val_identity_metrics(G_A2B, G_B2A, loader_ct_val, loader_mri_val, channels=cfg["channels"])
            
            val_ssim_A = metrics["ssim_rec_A"]
            val_ssim_B = metrics["ssim_rec_B"]
            avg_ssim = (val_ssim_A + val_ssim_B) / 2
            
            history["val_rec_ssim_A"].append(val_ssim_A)
            history["val_rec_ssim_B"].append(val_ssim_B)
            history["val_rec_mae_A"].append(metrics["mae_rec_A"])
            history["val_rec_mae_B"].append(metrics["mae_rec_B"])
            history["val_idt_mae_A"].append(idt_metrics["mae_idt_A"])
            history["val_idt_mae_B"].append(idt_metrics["mae_idt_B"])
            
            if avg_ssim > best_ssim:
                best_ssim = avg_ssim
            
            log_str = (f"[Epoch {epoch:03d}/{cfg['epochs']}] Loss_G: {avg_G:.4f} | Loss_D: {avg_D:.4f} | "
                       f"Val Rec SSIM A: {val_ssim_A:.4f} | B: {val_ssim_B:.4f} | "
                       f"Val Rec MAE A: {metrics['mae_rec_A']:.4f} | B: {metrics['mae_rec_B']:.4f} | "
                       f"Val Idt MAE A: {idt_metrics['mae_idt_A']:.4f} | B: {idt_metrics['mae_idt_B']:.4f}")
            print(log_str)
            with open(log_file, "a") as f:
                f.write(log_str + "\n")
        else:
            log_str = f"[Epoch {epoch:03d}/{cfg['epochs']}] Loss_G: {avg_G:.4f} | Loss_D: {avg_D:.4f}"
            print(log_str)
            with open(log_file, "a") as f:
                f.write(log_str + "\n")

        # Save periodic visual samples
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

        # Save periodic model checkpoints & resume state to allow resume-from-crash
        if epoch % cfg["save_interval"] == 0 or epoch == cfg["epochs"]:
            torch.save(G_A2B.state_dict(), os.path.join(checkpoints_dir, f"G_A2B_epoch{epoch}.pth"))
            torch.save(G_B2A.state_dict(), os.path.join(checkpoints_dir, f"G_B2A_epoch{epoch}.pth"))
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
            }, resume_path)
            
            with open(os.path.join(out_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)

    print(f"\n{config_name} done. Best avg Cycle SSIM: {best_ssim:.4f}")
    return history, best_ssim


# ─────────────────────────────────────────────────────────────
# MAIN — run selected configs sequentially
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    RUN_LIST = ["A_baseline", "B_capacity", "D_low_idt", "R_resize", "F_asym_idt", "S_spatial"]
    
    results = {}
    for name in RUN_LIST:
        exp_cfg = CONFIGS[name]
        history, best_ssim = run_experiment(name, exp_cfg)
        results[name] = {"best_ssim": best_ssim, "history": history}

    print("\n" + "="*60)
    print("LOCAL EXPERIMENT MATRIX SUMMARY")
    print("="*60)
    for name in RUN_LIST:
        r = results[name]
        cfg = CONFIGS[name]
        print(f"{name:25s} | best_ssim: {r['best_ssim']:.4f} | "
              f"n_blocks={cfg['n_blocks']} | lambda_cycle={cfg['lambda_cycle']} | "
              f"lambda_idt_A={cfg['lambda_identity_A']} | lambda_idt_B={cfg['lambda_identity_B']} | "
              f"upsample={cfg['upsample_mode']} | channels={cfg['channels']}")

    os.makedirs(BASE_CONFIG["output_dir"], exist_ok=True)
    with open(os.path.join(BASE_CONFIG["output_dir"], "experiment_summary.json"), "w") as f:
        summary = {k: {"best_ssim": v["best_ssim"], "config": CONFIGS[k]} for k, v in results.items() if k in RUN_LIST}
        json.dump(summary, f, indent=2)
    print("\nSummary saved to experiment_summary.json")
```
