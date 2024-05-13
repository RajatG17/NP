import argparse

import torch
import torch.nn as nn
from models.generator import Generator
from models.discriminator import Discriminator
from models.loss import GANLoss
from utils.data_utils import get_data_loaders
from generate import generate_images

def train(train_loader, val_loader, generator, discriminator, loss_fn, optimizer_g, optimizer_d, device, num_epochs):
    for epoch in range(num_epochs):
        # Training loop
        generator.train()
        discriminator.train()
        train_loss_g = 0.0
        train_loss_d = 0.0
        for rgb_images, depth_images in train_loader:
            rgb_images = rgb_images.to(device)
            depth_images = depth_images.to(device)

            optimizer_d.zero_grad()
            noise = torch.randn(rgb_images.size(0), 100, 1, 1).to(device)
            generated_images = generator(rgb_images, depth_images, noise).to(device)
            real_output = discriminator(rgb_images, depth_images, rgb_images).to(device)
            fake_output = discriminator(rgb_images, depth_images, generated_images.detach()).to(device)
            discriminator_loss = loss_fn(real_output, True) + loss_fn(fake_output, False)
            discriminator_loss.backward()
            optimizer_d.step()
            train_loss_d += discriminator_loss.item()

            optimizer_g.zero_grad()
            noise = torch.randn(rgb_images.size(0), 100, 1, 1).to(device)
            generated_images = generator(rgb_images, depth_images, noise).to(device)
            fake_output = discriminator(rgb_images, depth_images, generated_images.detach()).to(device)
            generator_loss = loss_fn(fake_output, True)
            generator_loss.backward()
            optimizer_g.step()
            train_loss_g += generator_loss.item()

        # Validation loop
        generator.eval()
        discriminator.eval()
        val_loss_g = 0.0
        val_loss_d = 0.0
        with torch.no_grad():
            for rgb_images, depth_images in val_loader:
                rgb_images = rgb_images.to(device)
                depth_images = depth_images.to(device)

                noise = torch.randn(rgb_images.size(0), 100, 1, 1).to(device)
                generated_images = generator(rgb_images, depth_images, noise).to(device)
                real_output = discriminator(rgb_images, depth_images, rgb_images).to(device)
                fake_output = discriminator(rgb_images, depth_images, generated_images).to(device)
                discriminator_loss = loss_fn(real_output, True) + loss_fn(fake_output, False)
                val_loss_d += discriminator_loss.item()

                fake_output = discriminator(rgb_images, depth_images, generated_images)
                generator_loss = loss_fn(fake_output, True)
                val_loss_g += generator_loss.item()


        # Compute average losses
        train_loss_g /= len(train_loader)
        train_loss_d /= len(train_loader)
        val_loss_g /= len(val_loader)
        val_loss_d /= len(val_loader)

        generate_images(generator, 25, device, epoch)

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train - Generator Loss: {train_loss_g:.4f}, Discriminator Loss: {train_loss_d:.4f} "
              f"Validation - Generator Loss: {val_loss_g:.4f}, Discriminator Loss: {val_loss_d:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate for optimizers')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to use for training')
    parser.add_argument('--attention', type=bool, default=True, help='Use attention layers in the GAN')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    device = torch.device(args.device)
    train_loader, val_loader, _ = get_data_loaders(args.root_dir, args.batch_size)

    # Initialize models
    generator = Generator(noise_dim=100, attention=args.attention).to(device)
    discriminator = Discriminator(attention=args.attention).to(device)

    loss_fn = GANLoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    train(train_loader, val_loader, generator, discriminator, loss_fn, optimizer_g, optimizer_d, device, args.num_epochs)