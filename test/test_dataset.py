from torch.utils.data import DataLoader
from utils.dataset import MRICTDataset, default_transform

if __name__ == "__main__":
    # Domain A = MRI
    dataset_A = MRICTDataset(
        root_dir='Dataset/images',
        domain='A',
        transform=default_transform,
        limit=100
    )

    # Domain B = CT
    dataset_B = MRICTDataset(
        root_dir='Dataset/images',
        domain='B',
        transform=default_transform,
        limit=100
    )

    # Create DataLoaders
    loader_A = DataLoader(dataset_A, batch_size=1, shuffle=True)
    loader_B = DataLoader(dataset_B, batch_size=1, shuffle=True)

    print(f" Dataset A (CT) loaded: {len(dataset_A)} images")
    print(f" Dataset B (MRI) loaded: {len(dataset_B)} images")

    # Fetch one batch from each loader to verify
    real_A = next(iter(loader_A))
    real_B = next(iter(loader_B))

    print(f" CT batch shape: {real_A.shape}")
    print(f" MRI batch shape: {real_B.shape}")


