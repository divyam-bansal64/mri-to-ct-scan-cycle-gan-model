import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms

from models.generator import define_generators
from models.discriminator import define_discriminators
from utils.dataset import MRICTDataset


def gan_loss(pred, target_is_real):
    """Least Squares GAN loss (used in CycleGAN)."""
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return torch.nn.functional.mse_loss(pred, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    with open(args.config_path, "r") as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = config.get("exp_name", "test_run")
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(os.path.join(exp_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)


    # =========================
    # Dataset setup
    # =========================
    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset_A = MRICTDataset("Dataset/images", domain="A", transform=transform, limit=config["limit"])
    dataset_B = MRICTDataset("Dataset/images", domain="B", transform=transform, limit=config["limit"])
    loader_A = DataLoader(dataset_A, batch_size=config["batch_size"], shuffle=True)
    loader_B = DataLoader(dataset_B, batch_size=config["batch_size"], shuffle=True)

    # =========================
    # Model setup
    # =========================
    G_A2B, G_B2A = define_generators(
        input_nc=config["input_nc"],
        output_nc=config["output_nc"],
        ngf=config["ngf"],
        n_blocks=config["n_blocks"],
        device=device
    )

    D_A, D_B = define_discriminators(
        input_nc=config["input_nc"],
        ndf=config["ndf"],
        n_layers=config["n_layers_D"],
        device=device
    )

    # =========================
    # Loss functions
    # =========================
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # =========================
    # Optimizers
    # =========================
    optimizer_G = torch.optim.Adam(
        list(G_A2B.parameters()) + list(G_B2A.parameters()),
        lr=config["lr_G"], betas=tuple(config["betas"])
    )

    optimizer_D = torch.optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()),
        lr=config["lr_D"], betas=tuple(config["betas"])
    )

    print(f"\nTraining Full CycleGAN Model: {config['exp_name']}")
    print(f"Parameters: lambda_cycle={config['lambda_cycle']} | lambda_identity={config['lambda_identity']} | n_blocks={config['n_blocks']}\n")

    # =========================
    # Training Loop
    # =========================
    for epoch in range(config["epochs"]):
        for i, (real_A, real_B) in enumerate(zip(loader_A, loader_B)):
            real_A, real_B = real_A.to(device), real_B.to(device)

            # 1. Train Generators
            optimizer_G.zero_grad()

            # Identity loss
            idt_A = G_B2A(real_A)
            idt_B = G_A2B(real_B)
            loss_idt_A = criterion_identity(idt_A, real_A) * config["lambda_identity"]
            loss_idt_B = criterion_identity(idt_B, real_B) * config["lambda_identity"]

            # GAN loss
            fake_B = G_A2B(real_A)
            fake_A = G_B2A(real_B)
            loss_GAN_A2B = gan_loss(D_B(fake_B), True)
            loss_GAN_B2A = gan_loss(D_A(fake_A), True)

            # Cycle loss
            recov_A = G_B2A(fake_B)
            recov_B = G_A2B(fake_A)
            loss_cycle_A = criterion_cycle(recov_A, real_A) * config["lambda_cycle"]
            loss_cycle_B = criterion_cycle(recov_B, real_B) * config["lambda_cycle"]

            # Total generator loss
            loss_G = (loss_GAN_A2B + loss_GAN_B2A +
                      loss_cycle_A + loss_cycle_B +
                      loss_idt_A + loss_idt_B)
            loss_G.backward()
            optimizer_G.step()

            # 2. Train Discriminators
            optimizer_D.zero_grad()

            # Real
            loss_real_A = gan_loss(D_A(real_A), True)
            loss_real_B = gan_loss(D_B(real_B), True)

            # Fake
            loss_fake_A = gan_loss(D_A(fake_A.detach()), False)
            loss_fake_B = gan_loss(D_B(fake_B.detach()), False)

            # Total discriminator loss
            loss_D_A = (loss_real_A + loss_fake_A) * 0.5
            loss_D_B = (loss_real_B + loss_fake_B) * 0.5
            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            optimizer_D.step()

            # Logging every few batches
            if i % 10 == 0:
                print(
                    f"[Epoch {epoch+1}/{config['epochs']}] [Batch {i}] "
                    f"Loss_G: {loss_G.item():.4f} | Loss_D: {loss_D.item():.4f} "
                    f"| lambda_cycle={config['lambda_cycle']} | lambda_identity={config['lambda_identity']}"
                )

        # Save checkpoints and sample images
        if (epoch + 1) % config["save_interval"] == 0:
            save_image(fake_B * 0.5 + 0.5, f"{exp_dir}/images/fakeB_epoch{epoch+1}.png")
            save_image(fake_A * 0.5 + 0.5, f"{exp_dir}/images/fakeA_epoch{epoch+1}.png")
            torch.save(G_A2B.state_dict(), f"{exp_dir}/checkpoints/G_A2B_epoch{epoch+1}.pth")
            torch.save(G_B2A.state_dict(), f"{exp_dir}/checkpoints/G_B2A_epoch{epoch+1}.pth")
            torch.save(D_A.state_dict(), f"{exp_dir}/checkpoints/D_A_epoch{epoch+1}.pth")
            torch.save(D_B.state_dict(), f"{exp_dir}/checkpoints/D_B_epoch{epoch+1}.pth")

    print(f"\nTraining completed for {config['exp_name']}. Checkpoints and images saved in {exp_dir}.\n")


if __name__ == "__main__":
    main()
