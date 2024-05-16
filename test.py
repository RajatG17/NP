import os
import argparse

import imageio
import torch
from PIL import Image
import cv2
from torch.utils.data import random_split
from torchvision.utils import make_grid

from models.generator import Generator
from models.discriminator import Discriminator
from models.loss import GeneratorLoss, DiscriminatorLoss
from utils.data_utils import get_data_loaders
from utils.visualize import generate_frames


def load_and_generate(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    data_loader = get_data_loaders(args.data_dir, args.batch_size)
    total_samples = len(data_loader.dataset)
    train_size = int(args.train_ratio * total_samples)
    val_size = int(args.val_ratio * total_samples)
    test_size = total_samples - train_size - val_size
    _, _, test_set = random_split(data_loader.dataset, [train_size, val_size, test_size])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Load trained models
    checkpoint = torch.load(os.path.join(args.save_dir, "model.pth"))
    generator = Generator(attention=args.attention).to(device)
    discriminator = Discriminator().to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    generator.eval()
    discriminator.eval()

    # Initialize loss functions
    generator_loss = GeneratorLoss()
    discriminator_loss = DiscriminatorLoss()

    test_generator_losses = []
    test_discriminator_losses = []

    with torch.no_grad():
        for i, (rgb_images, depth_images) in enumerate(test_loader):
            rgb_images = rgb_images.to(device)
            depth_images = depth_images.to(device)

            fake_scenes = generator(rgb_images, depth_images)
            real_output = discriminator(torch.cat((rgb_images, depth_images), 1))
            fake_output = discriminator(fake_scenes)

            # Calculate test losses
            test_generator_loss_value = generator_loss(fake_output)
            test_discriminator_loss_value = discriminator_loss(real_output, fake_output)

            test_generator_losses.append(test_generator_loss_value.item())
            test_discriminator_losses.append(test_discriminator_loss_value.item())

            video_frames, _ = generate_frames(
                generator=generator,
                rgb_depth_pair=(rgb_images, depth_images),
                initial_orientation=args.initial_orientation,
                num_frames=args.num_frames,
                delta_degrees=args.delta_degrees,
                device=device,
                save_dir=os.path.join(args.save_dir, f'frames_{i}'),
                attention=args.attention
            )

            gif_path = os.path.join(args.save_dir, f"output_animation_{i}.gif")
            # Write frames to the GIF file
            imageio.mimsave(gif_path, video_frames, fps=args.fps)
            print(f"GIF saved to: {gif_path}")

    # Print test losses
    print(f"Test Generator Loss: {sum(test_generator_losses) / len(test_generator_losses):.4f}")
    print(f"Test Discriminator Loss: {sum(test_discriminator_losses) / len(test_discriminator_losses):.4f}")

    # Save a grid of generated and real images for comparison
    fake_images = make_grid(fake_scenes[:8], nrow=4, normalize=True)
    real_images = make_grid(torch.cat((rgb_images[:8], depth_images[:8]), 1), nrow=4, normalize=True)
    Image.fromarray((fake_images.cpu().permute(1, 2, 0) * 255).byte().numpy(), mode='RGB').save(
        os.path.join(args.save_dir, 'fake_images.png'))
    Image.fromarray((real_images.cpu().permute(1, 2, 0) * 255).byte().numpy(), mode='RGB').save(
        os.path.join(args.save_dir, 'real_images.png'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save models and generated video")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for testing")
    parser.add_argument("--attention", type=bool, default=True, help="Enable attention layers")
    parser.add_argument("--initial_orientation", type=list, default=[0, 0, 0],
                        help="Initial camera orientation (roll, pitch, yaw) for video generation")
    parser.add_argument("--num_frames", type=int, default=30, help="Number of frames to generate for the video")
    parser.add_argument("--delta_degrees", type=float, default=5.0,
                        help="Degrees to pan the camera view between frames")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the generated video")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data to use for training")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of data to use for validation")

    args = parser.parse_args()
    load_and_generate(args)