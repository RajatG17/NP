import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class RGBDDataset(Dataset):
    def __init__(self,  root_dir, split='train', rgb_transform=None, depth_transform=None, test_size=0.2, val_size=0.1):
        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.rgb_images = []
        self.depth_images = []

        rgb_dir= os.path.join(root_dir, 'rgb')
        depth_dir = os.path.join(root_dir, 'depth')

        for filename in os.listdir(rgb_dir):
            if filename.endswith('.jpg') or filename.endswith('jpeg'):
                rgb_path = os.path.join(rgb_dir, filename)
                depth_path = os.path.join(depth_dir, filename[:-4]+'-depth_raw.png')

                self.rgb_images.append(rgb_path)
                self.depth_images.append(depth_path)

        train_data, test_data = train_test_split(
            list(range(len(self.rgb_images))),
            test_size=test_size,
            random_state=42
        )
        train_data, val_data = train_test_split(
            train_data,
            test_size=val_size / (1 - test_size),
            random_state=42
        )

        if split == 'train':
            self.rgb_images = [self.rgb_images[i] for i in train_data]
            self.depth_images = [self.depth_images[i] for i in train_data]
        elif split == 'val':
            self.rgb_images = [self.rgb_images[i] for i in val_data]
            self.depth_images = [self.depth_images[i] for i in val_data]
        elif split == 'test':
            self.rgb_images = [self.rgb_images[i] for i in test_data]
            self.depth_images = [self.depth_images[i] for i in test_data]
        else:
            raise ValueError("Invalid split value. Use 'train', 'val', or 'test'.")

        assert len(self.rgb_images) == len(self.depth_images)

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_image = self.rgb_images[idx]
        depth_image = self.depth_images[idx]

        rgb_image = Image.open(rgb_image).convert('RGB')
        depth_image = Image.open(depth_image).convert('L')

        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
        if self.depth_transform:
            depth_image = self.depth_transform(depth_image)

        return rgb_image, depth_image


def get_data_loaders(root_dir, batch_size, test_size=0.2, val_size=0.1):
    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    depth_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
        transforms.Normalize((0.5), (0.5))
    ])

    train_dataset = RGBDDataset(root_dir, split='train', rgb_transform=rgb_transform, depth_transform=depth_transform,
                                test_size=test_size, val_size=val_size)
    val_dataset = RGBDDataset(root_dir, split='val', rgb_transform=rgb_transform, depth_transform=depth_transform,
                              test_size=test_size, val_size=val_size)
    test_dataset = RGBDDataset(root_dir, split='test', rgb_transform=rgb_transform, depth_transform=depth_transform,
                               test_size=test_size, val_size=val_size)

    # Create data loaders for train, val, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader