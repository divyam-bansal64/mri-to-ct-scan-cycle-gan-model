import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MRICTDataset(Dataset):
    def __init__(self, root_dir, domain='A', transform=None, limit=None, mode='train'):
        """
        Args:
            root_dir (str): Path to dataset folder (e.g., 'Dataset/images')
            domain (str): 'A' for CT domain or 'B' for MRI domain
            transform (torchvision.transforms): Transformations to apply
            limit (int): Optional limit for number of images (useful for debugging)
            mode (str): 'train' or 'test'
        """
        self.path = os.path.join(root_dir, f"{mode}{domain}")
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Path not found: {self.path}")
        
        self.images = sorted([
            img for img in os.listdir(self.path)
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if limit:
            self.images = self.images[:limit]
        
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.images[idx])
        img = Image.open(img_path).convert('RGB')  # ensure 3-channel for CycleGAN
        
        if self.transform:
            img = self.transform(img)
        
        return img


# Default transform: Resize + Normalize to [-1, 1]
default_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
