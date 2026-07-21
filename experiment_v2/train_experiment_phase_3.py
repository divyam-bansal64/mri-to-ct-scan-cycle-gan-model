import os
import random
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image

from skimage.metrics import structural_similarity as ssim_metric
from utils.metrics import evaluate_all_metrics, AnatomicalDiceLoss
from utils.losses_phase_3 import VGGPerceptualLossV2, compute_fft_loss_v2

# ─────────────────────────────────────────────────────────────
# PHASE 3 CONFIGURATIONS
# ─────────────────────────────────────────────────────────────

CONFIGS = {
    "Run3v2_H_full_corrected_R1": {
        "n_blocks": 9,
        "lambda_cycle": 5.0,
        "lambda_identity_A": 2.5,
        "lambda_identity_B": 2.5,
        "upsample_mode": "ConvTranspose",
        "channels": 1,
        "augment_flip": True,
        "augment_spatial": True,
        "epochs": 50,
        "lambda_R1": 1.0,           # Corrected R1 penalty
    },
    "Run4v2_H_full_normalized_FFT": {
        "n_blocks": 9,
        "lambda_cycle": 5.0,
        "lambda_identity_A": 2.5,
        "lambda_identity_B": 2.5,
        "upsample_mode": "ConvTranspose",
        "channels": 1,
        "augment_flip": True,
        "augment_spatial": True,
        "epochs": 50,
        "lambda_fft": 5.0,          # Corrected & normalized FFT
    },
    "Run6v2_H_full_deep_VGG": {
        "n_blocks": 9,
        "lambda_cycle": 5.0,
        "lambda_identity_A": 2.5,
        "lambda_identity_B": 2.5,
        "upsample_mode": "ConvTranspose",
        "channels": 1,
        "augment_flip": True,
        "augment_spatial": True,
        "epochs": 50,
        "lambda_perceptual": 0.3,   # Deep VGG on cycle-reconstructed (Option B)
        "perceptual_mode": "cycle",
        "fid_gated_stop": True,
    },
    "Run8v2_H_full_R1_FFT": {
        "n_blocks": 9,
        "lambda_cycle": 5.0,
        "lambda_identity_A": 2.5,
        "lambda_identity_B": 2.5,
        "upsample_mode": "ConvTranspose",
        "channels": 1,
        "augment_flip": True,
        "augment_spatial": True,
        "epochs": 50,
        "lambda_R1": 1.0,
        "lambda_fft": 5.0,
    },
    "Run9_H_full_VGG_FFT_combo": {
        "n_blocks": 9,
        "lambda_cycle": 5.0,
        "lambda_identity_A": 2.5,
        "lambda_identity_B": 2.5,
        "upsample_mode": "ConvTranspose",
        "channels": 1,
        "augment_flip": True,
        "augment_spatial": True,
        "epochs": 50,
        "lambda_perceptual": 0.3,   # Combo with Option B VGG
        "perceptual_mode": "cycle",
        "lambda_fft": 5.0,
        "fid_gated_stop": True,
    },
    "Run10_H_full_idt_VGG": {
        "n_blocks": 9,
        "lambda_cycle": 5.0,
        "lambda_identity_A": 2.5,
        "lambda_identity_B": 2.5,
        "upsample_mode": "ConvTranspose",
        "channels": 1,
        "augment_flip": True,
        "augment_spatial": True,
        "epochs": 50,
        "lambda_perceptual": 0.3,   # Option A: VGG on identity mappings
        "perceptual_mode": "identity",
        "fid_gated_stop": True,
    },
    "Run11_H_full_idt_VGG_FFT_combo": {
        "n_blocks": 9,
        "lambda_cycle": 5.0,
        "lambda_identity_A": 2.5,
        "lambda_identity_B": 2.5,
        "upsample_mode": "ConvTranspose",
        "channels": 1,
        "augment_flip": True,
        "augment_spatial": True,
        "epochs": 50,
        "lambda_perceptual": 0.3,   # Option A + FFT combo
        "perceptual_mode": "identity",
        "lambda_fft": 5.0,
        "fid_gated_stop": True,
    }
}

BASE_CONFIG = {
    "ct_train_dir":  os.getenv("CT_TRAIN_DIR", r"E:\code\mri to cti\Dataset\images\trainA"),
    "mri_train_dir": os.getenv("MRI_TRAIN_DIR", r"E:\code\mri to cti\Dataset\images\trainB"),
    "output_dir":    os.getenv("OUTPUT_DIR", r"E:\code\mri to cti\experiment_v2\outputs"),
    "image_size":    256,
    "ngf":           64,
    "ndf":           64,
    "n_layers_D":    3,
    "use_spect":     True,
    "batch_size":    1,
    "lr_G":          0.0001,
    "lr_D":          0.0001,  # Equal learning rates (No TTUR)
    "betas":         [0.5, 0.999],
    "decay_epoch":   15,
    "limit":         500,     # images per domain for local testing
    "val_split":     0.2,     # Reconciled to 80/20 train/val split per plan Section 3
    "save_interval": 10,
    "num_workers":   0,       # Windows-safe default to prevent multi-process crashes
    "fid_gated_stop": False,
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
# LOSSES & REPLAY BUFFER
# ─────────────────────────────────────────────────────────────

def gan_loss(pred, target_is_real):
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return nn.functional.mse_loss(pred, target)

criterion_cycle    = nn.L1Loss()
criterion_identity = nn.L1Loss()


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
# TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────

def run_experiment(config_name, exp_config):
    cfg = {**BASE_CONFIG, **exp_config}
    print(f"\n{'='*60}")
    print(f"Running Phase 3 Config: {config_name}")
    print(f"  n_blocks={cfg['n_blocks']} | lambda_cycle={cfg['lambda_cycle']} | "
          f"lambda_idt_A={cfg['lambda_identity_A']} | lambda_idt_B={cfg['lambda_identity_B']} | "
          f"upsample={cfg['upsample_mode']} | channels={cfg['channels']} | "
          f"lambda_R1={cfg.get('lambda_R1', 0.0)} | lambda_fft={cfg.get('lambda_fft', 0.0)} | "
          f"lambda_perceptual={cfg.get('lambda_perceptual', 0.0)}")
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

    from utils.metrics import InceptionFeatureExtractor, AnatomicalDiceLoss
    eval_extractor = InceptionFeatureExtractor().to(device)
    eval_dice_loss = AnatomicalDiceLoss().to(device)

    eval_perceptual_loss = None
    if cfg.get("lambda_perceptual", 0.0) > 0.0:
        eval_perceptual_loss = VGGPerceptualLossV2().to(device)

    best_ssim = -1.0
    best_fid = float('inf')
    patience_counter = 0
    start_epoch = 1

    optimizer_G = torch.optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()),
                                   lr=cfg["lr_G"], betas=tuple(cfg["betas"]))
    optimizer_D = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()),
                                   lr=cfg["lr_D"], betas=tuple(cfg["betas"]))

    def lr_lambda(epoch):
        e = epoch + 1
        if e < cfg["decay_epoch"]: return 1.0
        denom = cfg["epochs"] - cfg["decay_epoch"]
        if denom <= 0: return 1.0
        return max(0.0, 1.0 - (e - cfg["decay_epoch"]) / denom)

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda)

    buffer_fake_A = ReplayBuffer(50)
    buffer_fake_B = ReplayBuffer(50)

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
        best_fid = state.get("best_fid", float('inf'))
        patience_counter = state.get("patience_counter", 0)
        start_epoch = state["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")
    else:
        history = {
            "loss_G": [], "loss_D": [], 
            "loss_fft": [], "loss_R1": [], "loss_perceptual": [],
            "val_rec_ssim_A": [], "val_rec_ssim_B": [],
            "val_rec_mae_A": [], "val_rec_mae_B": [],
            "val_idt_mae_A": [], "val_idt_mae_B": [],
            "val_fid_A": [], "val_fid_B": [],
            "val_cycle_dice_A": [], "val_cycle_dice_B": [],
            "val_idt_dice_A": [], "val_idt_dice_B": [],
            "val_fft_ratio_fake_A": [], "val_fft_ratio_fake_B": []
        }

    log_file = os.path.join(out_dir, "train_log.txt")
    if start_epoch == 1:
        with open(log_file, "w") as f:
            f.write(f"--- Phase 3 Experiment: {config_name} started ---\n")

    # Fixed dataset sizes logic to cycle dataset loader properly
    len_ct_loader = len(loader_ct_train)
    len_mri_loader = len(loader_mri_train)
    max_batches = max(len_ct_loader, len_mri_loader)

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        epoch_loss_G, epoch_loss_D, n_batches = 0.0, 0.0, 0
        epoch_loss_fft, epoch_loss_R1, epoch_loss_perceptual = 0.0, 0.0, 0.0
        
        iter_ct = iter(loader_ct_train)
        iter_mri = iter(loader_mri_train)

        for step in range(max_batches):
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

            # ──────────────────────────────────────────────────
            # GENERATOR STEP
            # ──────────────────────────────────────────────────
            optimizer_G.zero_grad()
            
            # Identity constraints (locked at λ=2.5 symmetric, cached to save VRAM and computations)
            idt_A = G_B2A(real_A)
            idt_B = G_A2B(real_B)
            loss_idt_A = criterion_identity(idt_A, real_A) * cfg["lambda_identity_A"]
            loss_idt_B = criterion_identity(idt_B, real_B) * cfg["lambda_identity_B"]
            
            fake_B = G_A2B(real_A)
            fake_A = G_B2A(real_B)
            
            loss_gan_A2B = gan_loss(D_B(fake_B), True)
            loss_gan_B2A = gan_loss(D_A(fake_A), True)
            
            rec_A = G_B2A(fake_B)
            rec_B = G_A2B(fake_A)
            
            loss_cycle_A = criterion_cycle(rec_A, real_A) * cfg["lambda_cycle"]
            loss_cycle_B = criterion_cycle(rec_B, real_B) * cfg["lambda_cycle"]
            
            # Normalized FFT Loss (v2, applied directly to translated scans fake_A/fake_B)
            loss_fft = 0.0
            if cfg.get("lambda_fft", 0.0) > 0.0:
                loss_fft = (compute_fft_loss_v2(fake_A, real_A) + compute_fft_loss_v2(fake_B, real_B)) * cfg["lambda_fft"]
                
            # Deep VGG Perceptual Loss (v2, semantic-only, supports Option A 'identity' and Option B 'cycle')
            loss_perceptual = 0.0
            if cfg.get("lambda_perceptual", 0.0) > 0.0 and eval_perceptual_loss is not None:
                if cfg.get("perceptual_mode", "cycle") == "identity":
                    loss_perceptual = (eval_perceptual_loss(idt_B, real_B) + eval_perceptual_loss(idt_A, real_A)) * cfg["lambda_perceptual"]
                else:
                    loss_perceptual = (eval_perceptual_loss(rec_B, real_B) + eval_perceptual_loss(rec_A, real_A)) * cfg["lambda_perceptual"]
            
            loss_G = (loss_gan_A2B + loss_gan_B2A + 
                       loss_cycle_A + loss_cycle_B + 
                       loss_idt_A + loss_idt_B + 
                       loss_fft + loss_perceptual)
                       
            loss_G.backward()
            optimizer_G.step()

            # ──────────────────────────────────────────────────
            # DISCRIMINATOR STEP
            # ──────────────────────────────────────────────────
            optimizer_D.zero_grad()
            fake_A_buf = buffer_fake_A.push_and_pop(fake_A.detach())
            loss_D_A = 0.5 * (gan_loss(D_A(real_A), True) + gan_loss(D_A(fake_A_buf), False))
            
            fake_B_buf = buffer_fake_B.push_and_pop(fake_B.detach())
            loss_D_B = 0.5 * (gan_loss(D_B(real_B), True) + gan_loss(D_B(fake_B_buf), False))
            
            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            
            # Corrected R1: computed EVERY step (not lazy) with correct scaling
            loss_R1 = 0.0
            if cfg.get("lambda_R1", 0.0) > 0.0:
                real_A.requires_grad = True
                real_B.requires_grad = True
                
                pred_real_A = D_A(real_A)
                pred_real_B = D_B(real_B)
                
                grads_A = torch.autograd.grad(
                    outputs=pred_real_A.sum(), inputs=real_A, 
                    create_graph=True, retain_graph=True, only_inputs=True
                )[0]
                grads_B = torch.autograd.grad(
                    outputs=pred_real_B.sum(), inputs=real_B, 
                    create_graph=True, retain_graph=True, only_inputs=True
                )[0]
                
                r1_penalty_A = (grads_A ** 2).sum(dim=(1, 2, 3)).mean()
                r1_penalty_B = (grads_B ** 2).sum(dim=(1, 2, 3)).mean()
                
                loss_R1 = 0.5 * (r1_penalty_A + r1_penalty_B) * cfg["lambda_R1"]
                loss_R1.backward()
                
            optimizer_D.step()

            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            epoch_loss_fft += loss_fft.item() if isinstance(loss_fft, torch.Tensor) else loss_fft
            epoch_loss_R1 += loss_R1.item() if isinstance(loss_R1, torch.Tensor) else loss_R1
            epoch_loss_perceptual += loss_perceptual.item() if isinstance(loss_perceptual, torch.Tensor) else loss_perceptual
            n_batches += 1

        scheduler_G.step()
        scheduler_D.step()

        avg_G = epoch_loss_G / n_batches
        avg_D = epoch_loss_D / n_batches
        history["loss_G"].append(avg_G)
        history["loss_D"].append(avg_D)
        history["loss_fft"].append(epoch_loss_fft / n_batches)
        history["loss_R1"].append(epoch_loss_R1 / n_batches)
        history["loss_perceptual"].append(epoch_loss_perceptual / n_batches)

        if epoch % cfg.get("eval_interval", 10) == 0:
            metrics = evaluate_all_metrics(
                G_A2B, G_B2A, loader_ct_val, loader_mri_val, 
                channels=cfg["channels"], device=device, epoch=epoch, run_dir=out_dir,
                extractor=eval_extractor, dice_loss_fn=eval_dice_loss
            )
            
            val_ssim_A = metrics["cycle_ssim_A2B"]
            val_ssim_B = metrics["cycle_ssim_B2A"]
            avg_ssim = metrics["avg_ssim"]
            
            history["val_rec_ssim_A"].append(val_ssim_A)
            history["val_rec_ssim_B"].append(val_ssim_B)
            history["val_rec_mae_A"].append(metrics["cycle_mae_A2B"])
            history["val_rec_mae_B"].append(metrics["cycle_mae_B2A"])
            history["val_idt_mae_A"].append(metrics["idt_mae_A2B"])
            history["val_idt_mae_B"].append(metrics["idt_mae_B2A"])
            history["val_fid_A"].append(metrics["fid_A"])
            history["val_fid_B"].append(metrics["fid_B"])
            history["val_cycle_dice_A"].append(metrics["cycle_dice_A2B"])
            history["val_cycle_dice_B"].append(metrics["cycle_dice_B2A"])
            history["val_idt_dice_A"].append(metrics["idt_dice_A"])
            history["val_idt_dice_B"].append(metrics["idt_dice_B"])
            history["val_fft_ratio_fake_A"].append(metrics["fft_ratio_fake_A"])
            history["val_fft_ratio_fake_B"].append(metrics["fft_ratio_fake_B"])
            
            if avg_ssim > best_ssim:
                best_ssim = avg_ssim
            
            log_str = (f"[Epoch {epoch:03d}/{cfg['epochs']}] Loss_G: {avg_G:.4f} | Loss_D: {avg_D:.4f} | "
                       f"Val Rec SSIM A: {val_ssim_A:.4f} | B: {val_ssim_B:.4f} | "
                       f"Val Rec MAE A: {metrics['cycle_mae_A2B']:.4f} | B: {metrics['cycle_mae_B2A']:.4f} | "
                       f"Val Idt MAE A: {metrics['idt_mae_A2B']:.4f} | B: {metrics['idt_mae_B2A']:.4f} | "
                       f"Val FID A: {metrics['fid_A']:.2f} | B: {metrics['fid_B']:.2f} | "
                       f"Val Dice Cycle CT: {metrics['cycle_dice_A2B']:.4f} | MRI: {metrics['cycle_dice_B2A']:.4f} | "
                       f"Val Dice Idt CT: {metrics['idt_dice_A']:.4f} | MRI: {metrics['idt_dice_B']:.4f} | "
                       f"Val FFT Ratio CT: {metrics['fft_ratio_fake_A']:.4f} | MRI: {metrics['fft_ratio_fake_B']:.4f}")
            print(log_str)
            with open(log_file, "a") as f:
                f.write(log_str + "\n")
                
            # FID-Gated Early Stopping (Run 6v2 & 9)
            if cfg.get("fid_gated_stop", False):
                current_fid = metrics["fid_B"]
                if epoch < 30:
                    if current_fid < best_fid: best_fid = current_fid
                else:
                    if current_fid < (best_fid - 1.0):
                        print(f"FID improved. Resetting patience.")
                        best_fid = current_fid
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        print(f"FID did not improve. Patience: {patience_counter}/3")
                        
                    if patience_counter >= 3:
                        log_stop = f"FID-gated early stopping triggered at epoch {epoch}. Best FID: {best_fid:.2f}"
                        print(log_stop)
                        with open(log_file, "a") as f:
                            f.write(log_stop + "\n")
                        break
        else:
            log_str = f"[Epoch {epoch:03d}/{cfg['epochs']}] Loss_G: {avg_G:.4f} | Loss_D: {avg_D:.4f}"
            print(log_str)
            with open(log_file, "a") as f:
                f.write(log_str + "\n")

        # Save sample outputs
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

        # Save checkpoint (skipped during smoke tests where save_interval > epochs)
        if epoch % cfg["save_interval"] == 0 or (epoch == cfg["epochs"] and cfg["save_interval"] <= cfg["epochs"]):
            torch.save(G_A2B.state_dict(), os.path.join(checkpoints_dir, f"G_A2B_epoch{epoch}.pth"))
            torch.save(G_B2A.state_dict(), os.path.join(checkpoints_dir, f"G_B2A_epoch{epoch}.pth"))
            torch.save({
                "epoch":            epoch,
                "G_A2B":            G_A2B.state_dict(),
                "G_B2A":            G_B2A.state_dict(),
                "D_A":              D_A.state_dict(),
                "D_B":              D_B.state_dict(),
                "opt_G":            optimizer_G.state_dict(),
                "opt_D":            optimizer_D.state_dict(),
                "sched_G":          scheduler_G.state_dict(),
                "sched_D":          scheduler_D.state_dict(),
                "history":          history,
                "best_ssim":        best_ssim,
                "best_fid":         best_fid,
                "patience_counter": patience_counter,
            }, resume_path)
            
            with open(os.path.join(out_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)

    print(f"\n{config_name} done. Best avg Cycle SSIM: {best_ssim:.4f}")
    return history, best_ssim


if __name__ == "__main__":
    # Runs to execute sequentially inside a single execution block
    # Modify this list as needed per Kaggle session
    RUN_LIST = ["Run6v2_H_full_deep_VGG", "Run9_H_full_VGG_FFT_combo"]
    
    results = {}
    for name in RUN_LIST:
        exp_cfg = CONFIGS[name]
        history, best_ssim = run_experiment(name, exp_cfg)
        results[name] = {"best_ssim": best_ssim, "history": history}

    print("\n" + "="*60)
    print("PHASE 3 MATRIX SUMMARY")
    print("="*60)
    for name in RUN_LIST:
        r = results[name]
        cfg = CONFIGS[name]
        print(f"{name:25s} | best_ssim: {r['best_ssim']:.4f} | "
              f"lambda_R1={cfg.get('lambda_R1', 0.0)} | lambda_fft={cfg.get('lambda_fft', 0.0)}")

    os.makedirs(BASE_CONFIG["output_dir"], exist_ok=True)
    with open(os.path.join(BASE_CONFIG["output_dir"], "experiment_summary_phase_3.json"), "w") as f:
        summary = {k: {"best_ssim": v["best_ssim"], "config": CONFIGS[k]} for k, v in results.items() if k in RUN_LIST}
        json.dump(summary, f, indent=2)
    print("\nSummary saved to experiment_summary_phase_3.json")
