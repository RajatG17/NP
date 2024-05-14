import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class RGBDDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.rgb_images = sorted([os.path.join(data_dir, 'rgb', f) for f in os.listdir(os.path.join(data_dir, 'rgb')) if f.endswith('.jpg')])
        self.depth_images = sorted([os.path.join(data_dir, 'depth', f) for f in os.listdir(os.path.join(data_dir, 'depth')) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_path = self.rgb_images[idx]
        depth_path = self.depth_images[idx]

        rgb_image = Image.open(rgb_path).convert('RGB')
        depth_image = Image.open(depth_path)

        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth_image = self.transform(depth_image)

        return rgb_image, depth_image

def get_data_loaders(data_dir, batch_size, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust these values as per your data
    ])

    dataset = RGBDDataset(data_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return data_loader