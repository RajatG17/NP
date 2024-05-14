import os
import argparse
import torch
from torch.utils.data import random_split
from torchvision.utils import save_image
import cv2
from utils.data_utils import get_data_loaders
from models.generator import Generator
from models.discriminator import Discriminator
from models.loss import GeneratorLoss, DiscriminatorLoss
from utils.visualize import generate_frames

def train(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    data_loader = get_data_loaders(args.data_dir, args.batch_size)
    total_samples = len(data_loader.dataset)
    train_size = int(args.train_ratio * total_samples)
    val_size = int(args.val_ratio * total_samples)
    test_size = total_samples - train_size - val_size
    train_set, val_set, test_set = random_split(data_loader.dataset, [train_size, val_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Initialize loss functions and optimizers
    generator_loss = GeneratorLoss()
    discriminator_loss = DiscriminatorLoss()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(args.epochs):
        for rgb_images, depth_images in train_loader:
            rgb_images = rgb_images.to(device)
            depth_images = depth_images.to(device)

            # Train discriminator
            discriminator_optimizer.zero_grad()
            fake_scenes = generator(rgb_images, depth_images)
            real_output = discriminator(torch.cat((rgb_images, depth_images), 1))
            fake_output = discriminator(fake_scenes.detach())
            discriminator_loss_value = discriminator_loss(real_output, fake_output)
            discriminator_loss_value.backward()
            discriminator_optimizer.step()

            # Train generator
            generator_optimizer.zero_grad()
            fake_scenes = generator(rgb_images, depth_images)
            fake_output = discriminator(fake_scenes)
            generator_loss_value = generator_loss(fake_output)
            generator_loss_value.backward()
            generator_optimizer.step()

        # Save models and log training progress
        if (epoch + 1) % args.save_interval == 0:
            torch.save(generator.state_dict(), os.path.join(args.save_dir, f"generator_epoch_{epoch + 1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(args.save_dir, f"discriminator_epoch_{epoch + 1}.pth"))
            print(f"Epoch [{epoch + 1}/{args.epochs}] Generator Loss: {generator_loss_value.item():.4f} Discriminator Loss: {discriminator_loss_value.item():.4f}")

    # Generate video from test set
    video_frames = generate_frames(
        generator=generator,
        rgb_depth_pairs=list(test_loader),
        initial_position=args.initial_position,
        initial_orientation=args.initial_orientation,
        num_frames=args.num_frames,
        delta_degrees=args.delta_degrees,
        device=device,
        output_dir=args.save_dir
    )

    height, width, channels = video_frames[0].shape
    video = cv2.VideoWriter(os.path.join(args.save_dir, 'output.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (width, height))
    for frame in video_frames:
        video.write((frame * 255).astype('uint8'))
    video.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save models and generated video")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data to use for training")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of data to use for validation")
    parser.add_argument("--save_interval", type=int, default=10, help="Interval (in epochs) to save models")
    parser.add_argument("--initial_position", type=list, default=[0, 0, 0], help="Initial camera position for video generation")
    parser.add_argument("--initial_orientation", type=list, default=[0, 0, 0], help="Initial camera orientation (roll, pitch, yaw) for video generation")
    parser.add_argument("--num_frames", type=int, default=30, help="Number of frames to generate for the video")
    parser.add_argument("--delta_degrees", type=float, default=5.0, help="Degrees to pan the camera view between frames")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the generated video")

    args = parser.parse_args()
    train(args)