import os
import random
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image

from skimage.metrics import structural_similarity as ssim_metric
from utils.metrics import evaluate_all_metrics, AnatomicalDiceLoss
from utils.losses_phase_4 import VGGPerceptualLossV4 as VGGPerceptualLossV2, compute_fft_loss_v4 as compute_fft_loss_v2

# ─────────────────────────────────────────────────────────────
# PHASE 4 CONFIGURATIONS (FINAL 200-EPOCH RUNS)
# ─────────────────────────────────────────────────────────────

CONFIGS = {
    "Phase4_Alpha_Hybrid": {
        "n_blocks": 9,
        "lambda_cycle": 10.0,
        "lambda_identity_A": 5.0,
        "lambda_identity_B": 5.0,
        "upsample_mode": "ConvTranspose",
        "channels": 1,
        "augment_flip": True,
        "augment_spatial": True,
        "epochs": 200,
        "decay_epoch": 100,
        "lambda_R1": 1.0,           # R1 penalty — keeps D at 0.5 equilibrium
        "lambda_fft": 5.0,          # Reduced from 10.0 to avoid competing with VGG
        "lambda_perceptual": 0.5,   # Reduced from 1.0 to prevent FID-obsessed textures
        "perceptual_mode": "identity", # Identity mode proven superior
        "lr_G": 0.0002,
        "lr_D": 0.0002,
    },
    "Phase4_Beta_Control": {
        "n_blocks": 9,
        "lambda_cycle": 10.0,
        "lambda_identity_A": 5.0,
        "lambda_identity_B": 5.0,
        "upsample_mode": "ConvTranspose",
        "channels": 1,
        "augment_flip": True,
        "augment_spatial": True,
        "epochs": 200,
        "decay_epoch": 100,
        "lambda_R1": 1.0,           # Exact Run 8v2 configuration
        "lambda_fft": 10.0,         # Full FFT (Run 8v2 proven)
        "lambda_perceptual": 0.0,   # No VGG — pure structural optimization
        "perceptual_mode": "none",
        "lr_G": 0.0002,
        "lr_D": 0.0002,
    }
}

BASE_CONFIG = {
    "ct_train_dir":  os.getenv("CT_TRAIN_DIR", r"E:\code\mri to cti\Dataset\images\trainA"),
    "mri_train_dir": os.getenv("MRI_TRAIN_DIR", r"E:\code\mri to cti\Dataset\images\trainB"),
    "output_dir":    os.getenv("OUTPUT_DIR", r"E:\code\mri to cti\experiment_v2\outputs"),
    "image_size":    256,
    "ngf":           64,
    "ndf":           64,
    "lr_G":          0.0002,
    "lr_D":          0.0002,
    "betas":         (0.5, 0.999),
    "batch_size":    1,
    "num_workers":   2,
    "epochs":        200,
    "decay_epoch":   100,
    "eval_interval": 10,
    "save_interval": 10,
    "seed":          42,
    "limit":         None,
}

# ─────────────────────────────────────────────────────────────
# REPEATABLE REPRODUCIBILITY SEED
# ─────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────────────────────
# HELPER GAN LOSS FUNCTION (MSE Loss with Tensor targets)
# ─────────────────────────────────────────────────────────────
def gan_loss(pred, target_is_real):
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)

# ─────────────────────────────────────────────────────────────
# ARCHITECTURE DEFINITIONS
# ─────────────────────────────────────────────────────────────

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_blocks=9, ngf=64, upsample_mode="ConvTranspose"):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_blocks):
            model += [ResNetBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            if upsample_mode == "ResizeConv":
                model += [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_features, out_features, 3),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True)
                ]
            else:
                model += [
                    nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True)
                ]
            in_features = out_features
            out_features = in_features // 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, ndf=64):
        super().__init__()
        def block(in_f, out_f, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, ndf, normalize=False),
            *block(ndf, ndf * 2),
            *block(ndf * 2, ndf * 4),
            nn.Conv2d(ndf * 4, ndf * 8, 4, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# ─────────────────────────────────────────────────────────────
# DATASET AND REPLAY BUFFER
# ─────────────────────────────────────────────────────────────

class UnpairedDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        self.files = sorted([f for f in self.root_dir.iterdir() if f.suffix.lower() in valid_exts])
        if limit is not None:
            self.files = self.files[:limit]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image

class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

# ─────────────────────────────────────────────────────────────
# MASTER EXPERIMENT RUNNER — PHASE 4
# ─────────────────────────────────────────────────────────────

def run_experiment(run_name, custom_config):
    cfg = BASE_CONFIG.copy()
    cfg.update(custom_config)
    set_seed(cfg["seed"])

    out_dir = os.path.join(cfg["output_dir"], run_name)
    checkpoints_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    log_file = os.path.join(out_dir, "train_log.txt")
    print(f"\n========================================================")
    print(f"       STARTING PHASE 4 EXPERIMENT: {run_name}")
    print(f"========================================================\n")

    t_list = [transforms.Resize((cfg["image_size"], cfg["image_size"]))]
    if cfg.get("augment_flip", False):
        t_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if cfg.get("augment_spatial", False):
        t_list.append(transforms.RandomRotation(degrees=5))
        t_list.append(transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)))
    t_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transform_train = transforms.Compose(t_list)

    transform_val = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_ct  = UnpairedDataset(cfg["ct_train_dir"], transform=None, limit=cfg["limit"])
    full_mri = UnpairedDataset(cfg["mri_train_dir"], transform=None, limit=cfg["limit"])

    # Deterministic 80/20 train/val split
    val_len_ct = max(1, int(len(full_ct) * 0.2))
    train_len_ct = len(full_ct) - val_len_ct
    gen_split = torch.Generator().manual_seed(cfg["seed"])
    idx_ct_train, idx_ct_val = torch.utils.data.random_split(range(len(full_ct)), [train_len_ct, val_len_ct], generator=gen_split)

    val_len_mri = max(1, int(len(full_mri) * 0.2))
    train_len_mri = len(full_mri) - val_len_mri
    idx_mri_train, idx_mri_val = torch.utils.data.random_split(range(len(full_mri)), [train_len_mri, val_len_mri], generator=gen_split)

    # Subsets with respective transforms
    dataset_ct  = Subset(UnpairedDataset(cfg["ct_train_dir"], transform=transform_train, limit=cfg["limit"]), idx_ct_train.indices)
    dataset_mri = Subset(UnpairedDataset(cfg["mri_train_dir"], transform=transform_train, limit=cfg["limit"]), idx_mri_train.indices)

    val_dataset_ct  = Subset(UnpairedDataset(cfg["ct_train_dir"], transform=transform_val, limit=cfg["limit"]), idx_ct_val.indices)
    val_dataset_mri = Subset(UnpairedDataset(cfg["mri_train_dir"], transform=transform_val, limit=cfg["limit"]), idx_mri_val.indices)

    loader_ct  = DataLoader(dataset_ct, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True)
    loader_mri = DataLoader(dataset_mri, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True)

    loader_ct_val  = DataLoader(val_dataset_ct, batch_size=1, shuffle=False)
    loader_mri_val = DataLoader(val_dataset_mri, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    print(f"Train/Val split: CT ({len(dataset_ct)} train / {len(val_dataset_ct)} val) | MRI ({len(dataset_mri)} train / {len(val_dataset_mri)} val)")

    G_A2B = Generator(cfg["channels"], cfg["channels"], cfg["n_blocks"], ngf=cfg["ngf"], upsample_mode=cfg["upsample_mode"]).to(device)
    G_B2A = Generator(cfg["channels"], cfg["channels"], cfg["n_blocks"], ngf=cfg["ngf"], upsample_mode=cfg["upsample_mode"]).to(device)
    D_A   = Discriminator(cfg["channels"], ndf=cfg["ndf"]).to(device)
    D_B   = Discriminator(cfg["channels"], ndf=cfg["ndf"]).to(device)

    from utils.metrics import InceptionFeatureExtractor, AnatomicalDiceLoss
    eval_extractor = InceptionFeatureExtractor().to(device)
    eval_dice_loss = AnatomicalDiceLoss().to(device)

    # Optional VGG Perceptual Loss Initialization
    vgg_loss_fn = None
    if cfg.get("lambda_perceptual", 0.0) > 0.0:
        vgg_loss_fn = VGGPerceptualLossV2(device=device)

    # Losses
    cycle_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()
    dice_metric_fn = AnatomicalDiceLoss()

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
        optimizer_G.load_state_dict(state["optimizer_G"])
        optimizer_D.load_state_dict(state["optimizer_D"])
        start_epoch = state["epoch"] + 1
        scheduler_G.load_state_dict(state["scheduler_G"])
        scheduler_D.load_state_dict(state["scheduler_D"])
        print(f"Resumed successfully at epoch {start_epoch}.")

    history = {
        "loss_G": [], "loss_D": [], "loss_fft": [], "loss_R1": [], "loss_perceptual": [],
        "val_rec_ssim_A": [], "val_rec_ssim_B": [],
        "val_rec_mae_A": [], "val_rec_mae_B": [],
        "val_idt_mae_A": [], "val_idt_mae_B": [],
        "val_fid_A": [], "val_fid_B": [],
        "val_cycle_dice_A": [], "val_cycle_dice_B": [],
        "val_idt_dice_A": [], "val_idt_dice_B": [],
        "val_fft_ratio_fake_A": [], "val_fft_ratio_fake_B": []
    }
    history_file = os.path.join(out_dir, "history.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)

    # ─────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────
    fft_early_stop_counter = 0

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        G_A2B.train(); G_B2A.train()
        D_A.train(); D_B.train()

        running_loss_G = 0.0
        running_loss_D = 0.0
        running_loss_fft = 0.0
        running_loss_R1 = 0.0
        running_loss_perceptual = 0.0
        steps = 0

        for batch_ct, batch_mri in zip(loader_ct, loader_mri):
            real_A = batch_ct.to(device)
            real_B = batch_mri.to(device)

            # ──────────────────────────────────────────────────
            # GENERATOR STEP
            # ──────────────────────────────────────────────────
            optimizer_G.zero_grad()

            fake_B = G_A2B(real_A)
            loss_gan_A2B = gan_loss(D_B(fake_B), True)

            fake_A = G_B2A(real_B)
            loss_gan_B2A = gan_loss(D_A(fake_A), True)

            rec_A = G_B2A(fake_B)
            rec_B = G_A2B(fake_A)
            loss_cycle_A = cycle_loss(rec_A, real_A) * cfg["lambda_cycle"]
            loss_cycle_B = cycle_loss(rec_B, real_B) * cfg["lambda_cycle"]

            idt_A = G_B2A(real_A)
            idt_B = G_A2B(real_B)
            loss_idt_A = identity_loss(idt_A, real_A) * cfg["lambda_identity_A"]
            loss_idt_B = identity_loss(idt_B, real_B) * cfg["lambda_identity_B"]

            # Normalized FFT Loss
            loss_fft = torch.tensor(0.0, device=device)
            if cfg.get("lambda_fft", 0.0) > 0.0:
                loss_fft = (compute_fft_loss_v2(fake_A, real_A) + compute_fft_loss_v2(fake_B, real_B)) * cfg["lambda_fft"]

            # VGG Perceptual Loss (Option A: Identity mode or Option B: Cycle mode)
            loss_perceptual = torch.tensor(0.0, device=device)
            if cfg.get("lambda_perceptual", 0.0) > 0.0 and vgg_loss_fn is not None:
                if cfg.get("perceptual_mode", "identity") == "identity":
                    loss_perceptual = (vgg_loss_fn(idt_A, real_A) + vgg_loss_fn(idt_B, real_B)) * cfg["lambda_perceptual"]
                else:
                    loss_perceptual = (vgg_loss_fn(rec_A, real_A) + vgg_loss_fn(rec_B, real_B)) * cfg["lambda_perceptual"]

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
            
            # Enable gradients on real inputs for single-pass R1 computation
            if cfg.get("lambda_R1", 0.0) > 0.0:
                real_A.requires_grad = True
                real_B.requires_grad = True

            d_out_real_A = D_A(real_A)
            d_out_real_B = D_B(real_B)

            fake_A_buf = buffer_fake_A.push_and_pop(fake_A.detach())
            loss_D_A = 0.5 * (gan_loss(d_out_real_A, True) + gan_loss(D_A(fake_A_buf), False))

            fake_B_buf = buffer_fake_B.push_and_pop(fake_B.detach())
            loss_D_B = 0.5 * (gan_loss(d_out_real_B, True) + gan_loss(D_B(fake_B_buf), False))

            loss_D = loss_D_A + loss_D_B
            loss_D.backward(retain_graph=(cfg.get("lambda_R1", 0.0) > 0.0))

            # R1 Penalty using the same forward pass output
            loss_R1 = torch.tensor(0.0, device=device)
            if cfg.get("lambda_R1", 0.0) > 0.0:
                grad_real_A = torch.autograd.grad(outputs=d_out_real_A.sum(), inputs=real_A, create_graph=True, retain_graph=True)[0]
                grad_real_B = torch.autograd.grad(outputs=d_out_real_B.sum(), inputs=real_B, create_graph=True, retain_graph=True)[0]
                
                r1_A = grad_real_A.pow(2).reshape(grad_real_A.shape[0], -1).sum(1).mean()
                r1_B = grad_real_B.pow(2).reshape(grad_real_B.shape[0], -1).sum(1).mean()
                
                loss_R1 = 0.5 * (r1_A + r1_B) * cfg["lambda_R1"]
                loss_R1.backward()

            optimizer_D.step()

            running_loss_G += loss_G.item()
            running_loss_D += loss_D.item()
            running_loss_fft += loss_fft.item()
            running_loss_R1 += loss_R1.item()
            running_loss_perceptual += loss_perceptual.item()
            steps += 1

        scheduler_G.step()
        scheduler_D.step()

        avg_G = running_loss_G / steps
        avg_D = running_loss_D / steps
        avg_fft = running_loss_fft / steps
        avg_R1 = running_loss_R1 / steps
        avg_perceptual = running_loss_perceptual / steps

        history["loss_G"].append(avg_G)
        history["loss_D"].append(avg_D)
        history["loss_fft"].append(avg_fft)
        history["loss_R1"].append(avg_R1)
        history["loss_perceptual"].append(avg_perceptual)

        # ─────────────────────────────────────────────────────────
        # EVALUATION & LOGGING
        # ─────────────────────────────────────────────────────────
        if epoch % cfg["eval_interval"] == 0 or epoch == cfg["epochs"]:
            metrics = evaluate_all_metrics(
                G_A2B, G_B2A, loader_ct_val, loader_mri_val,
                channels=cfg["channels"], device=device, epoch=epoch, run_dir=out_dir,
                extractor=eval_extractor, dice_loss_fn=eval_dice_loss
            )

            history["val_rec_ssim_A"].append(metrics["cycle_ssim_A2B"])
            history["val_rec_ssim_B"].append(metrics["cycle_ssim_B2A"])
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

            log_str = (f"[Epoch {epoch:03d}/{cfg['epochs']}] Loss_G: {avg_G:.4f} | Loss_D: {avg_D:.4f} | "
                       f"Val Rec SSIM A: {metrics['cycle_ssim_A2B']:.4f} | B: {metrics['cycle_ssim_B2A']:.4f} | "
                       f"Val Rec MAE A: {metrics['cycle_mae_A2B']:.4f} | B: {metrics['cycle_mae_B2A']:.4f} | "
                       f"Val Idt MAE A: {metrics['idt_mae_A2B']:.4f} | B: {metrics['idt_mae_B2A']:.4f} | "
                       f"Val FID A: {metrics['fid_A']:.2f} | B: {metrics['fid_B']:.2f} | "
                       f"Val Dice Cycle CT: {metrics['cycle_dice_A2B']:.4f} | MRI: {metrics['cycle_dice_B2A']:.4f} | "
                       f"Val Dice Idt CT: {metrics['idt_dice_A']:.4f} | MRI: {metrics['idt_dice_B']:.4f} | "
                       f"Val FFT Ratio CT: {metrics['fft_ratio_fake_A']:.4f} | MRI: {metrics['fft_ratio_fake_B']:.4f}")
            print(log_str)
            with open(log_file, "a") as f:
                f.write(log_str + "\n")

            # FFT Early-Stop Guard: trigger if FFT_B > 0.015 for 2 consecutive checkpoints
            if metrics["fft_ratio_fake_B"] > 0.015:
                fft_early_stop_counter += 1
                warn_msg = f"[WARNING] High FFT artifact ratio detected ({metrics['fft_ratio_fake_B']:.4f} > 0.015). Counter: {fft_early_stop_counter}/2"
                print(warn_msg)
                with open(log_file, "a") as f:
                    f.write(warn_msg + "\n")
                if fft_early_stop_counter >= 2:
                    stop_msg = f"[EARLY STOP] Terminating run {run_name} at epoch {epoch} due to 2 consecutive high FFT artifact spikes."
                    print(stop_msg)
                    with open(log_file, "a") as f:
                        f.write(stop_msg + "\n")
                    break
            else:
                fft_early_stop_counter = 0
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

        # Save checkpoint
        if epoch % cfg["save_interval"] == 0 or (epoch == cfg["epochs"] and cfg["save_interval"] <= cfg["epochs"]):
            checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch:03d}.pth")
            torch.save({
                "epoch": epoch,
                "G_A2B": G_A2B.state_dict(),
                "G_B2A": G_B2A.state_dict(),
                "D_A": D_A.state_dict(),
                "D_B": D_B.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "optimizer_D": optimizer_D.state_dict(),
                "scheduler_G": scheduler_G.state_dict(),
                "scheduler_D": scheduler_D.state_dict(),
                "config": cfg,
            }, checkpoint_path)

            torch.save({
                "epoch": epoch,
                "G_A2B": G_A2B.state_dict(),
                "G_B2A": G_B2A.state_dict(),
                "D_A": D_A.state_dict(),
                "D_B": D_B.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "optimizer_D": optimizer_D.state_dict(),
                "scheduler_G": scheduler_G.state_dict(),
                "scheduler_D": scheduler_D.state_dict(),
                "config": cfg,
            }, resume_path)

        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n[SUCCESS] Completed {run_name}!")

if __name__ == "__main__":
    import sys
    run_target = sys.argv[1] if len(sys.argv) > 1 else "Phase4_Alpha_Hybrid"
    if run_target in CONFIGS:
        run_experiment(run_target, CONFIGS[run_target])
    else:
        print(f"Unknown config '{run_target}'. Available: {list(CONFIGS.keys())}")
