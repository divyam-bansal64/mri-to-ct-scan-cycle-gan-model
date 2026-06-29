import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms

from models.generator import define_generators
from utils.dataset import MRICTDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        CONFIG = json.load(f)

    # === Setup ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = os.path.dirname(args.config_path)
    os.makedirs(os.path.join(exp_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

    image_size = CONFIG["image_size"]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    # === Datasets ===
    dataset_A = MRICTDataset("Dataset/images", domain="A", transform=transform, limit=CONFIG["limit"])
    dataset_B = MRICTDataset("Dataset/images", domain="B", transform=transform, limit=CONFIG["limit"])
    loader_A = DataLoader(dataset_A, batch_size=CONFIG["batch_size"], shuffle=True)
    loader_B = DataLoader(dataset_B, batch_size=CONFIG["batch_size"], shuffle=True)

    # === Models ===
    G_A2B, G_B2A = define_generators(
        input_nc=CONFIG["input_nc"],
        output_nc=CONFIG["output_nc"],
        ngf=CONFIG["ngf"],
        n_blocks=CONFIG["n_blocks"],
        device=device
    )

    criterion_cycle = torch.nn.L1Loss()
    optimizer_G = torch.optim.Adam(
        list(G_A2B.parameters()) + list(G_B2A.parameters()),
        lr=CONFIG["lr_G"], betas=tuple(CONFIG["betas"])
    )

    print(f" Training generator-only model: {CONFIG['exp_name']}")
    for epoch in range(CONFIG["epochs"]):
        for i, (real_A, real_B) in enumerate(zip(loader_A, loader_B)):
            real_A, real_B = real_A.to(device), real_B.to(device)
            optimizer_G.zero_grad()

            fake_B = G_A2B(real_A)
            fake_A = G_B2A(real_B)

            recov_A = G_B2A(fake_B)
            recov_B = G_A2B(fake_A)

            loss_cycle_A = criterion_cycle(recov_A, real_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_G = (loss_cycle_A + loss_cycle_B) * CONFIG["lambda_cycle"]

            loss_G.backward()
            optimizer_G.step()

            if i % 10 == 0:
                print(f"[Epoch {epoch+1}/{CONFIG['epochs']}] [Batch {i}] "
                      f"Loss_G: {loss_G.item():.4f} | Lambda_cyc={CONFIG['lambda_cycle']} | Lambda_id={CONFIG['lambda_identity']} | nB={CONFIG['n_blocks']}")


        # === Save every few epochs ===
        if (epoch + 1) % CONFIG["save_interval"] == 0:
            save_image(fake_B * 0.5 + 0.5, f"{exp_dir}/images/fakeB_epoch{epoch+1}.png")
            save_image(fake_A * 0.5 + 0.5, f"{exp_dir}/images/fakeA_epoch{epoch+1}.png")
            torch.save(G_A2B.state_dict(), f"{exp_dir}/checkpoints/G_A2B_epoch{epoch+1}.pth")
            torch.save(G_B2A.state_dict(), f"{exp_dir}/checkpoints/G_B2A_epoch{epoch+1}.pth")

if __name__ == "__main__":
    main()
