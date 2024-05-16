import os
import argparse

import imageio
import torch
from PIL.Image import Image
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from fid_test import FIDScore
import cv2
from utils.data_utils import get_data_loaders
from models.generator import Generator
from models.discriminator import Discriminator
from models.loss import GeneratorLoss, DiscriminatorLoss
from utils.visualize import generate_frames
from torch.optim.lr_scheduler import StepLR

def train(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fid_score_calculator = FIDScore(device)

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
    generator = Generator(attention=args.attention).to(device)
    discriminator = Discriminator().to(device)

    # Initialize loss functions and optimizers
    generator_loss = GeneratorLoss()
    discriminator_loss = DiscriminatorLoss()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # g_scheduler = StepLR(generator_optimizer, step_size=10, gamma=0.1)
    # d_scheduler = StepLR(discriminator_optimizer, step_size=10, gamma=0.1)

    train_generator_losses = []
    train_discriminator_losses = []
    val_generator_losses = []
    val_discriminator_losses = []

    # Training loop
    for epoch in range(args.epochs):
        epoch_train_generator_loss = 0.0
        epoch_train_discriminator_loss = 0.0

        generator.train()
        discriminator.train()
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

            epoch_train_generator_loss += generator_loss_value.item()
            epoch_train_discriminator_loss += discriminator_loss_value.item()

        epoch_train_generator_loss /= len(train_loader)
        epoch_train_discriminator_loss /= len(train_loader)
        train_generator_losses.append(epoch_train_generator_loss)
        train_discriminator_losses.append(epoch_train_discriminator_loss)

        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            epoch_val_generator_loss = 0.0
            epoch_val_discriminator_loss = 0.0
            for rgb_images, depth_images in val_loader:
                rgb_images = rgb_images.to(device)
                depth_images = depth_images.to(device)

                fake_scenes = generator(rgb_images, depth_images)
                real_output = discriminator(torch.cat((rgb_images, depth_images), 1))
                fake_output = discriminator(fake_scenes.detach())

                # Calculate validation losses
                val_generator_loss_value = generator_loss(fake_output)
                val_discriminator_loss_value = discriminator_loss(real_output, fake_output)

                epoch_val_generator_loss += val_generator_loss_value.item()
                epoch_val_discriminator_loss += val_discriminator_loss_value.item()

            # Calculate average losses per epoch for validation set
            epoch_val_generator_loss /= len(val_loader)
            epoch_val_discriminator_loss /= len(val_loader)
            val_generator_losses.append(epoch_val_generator_loss)
            val_discriminator_losses.append(epoch_val_discriminator_loss)

            if args.fid and (epoch % 5) == 0:
                real_images_list = []
                fake_images_list = []
                for rgb_images, depth_images in val_loader:
                    rgb_images = rgb_images.to(device)
                    depth_images = depth_images.to(device)

                    with torch.no_grad():
                        fake_scenes = generator(rgb_images, depth_images)
                        real_images_list.extend(
                            [img[:3].cpu() for img in rgb_images])  # Take only the first 3 channels (RGB)
                        fake_images_list.extend(
                            [img[:3].cpu() for img in fake_scenes])  # Take only the first 3 channels (RGB)

                fid_score = fid_score_calculator.calculate_fid(real_images_list, fake_images_list)
                print(f'FID Score epoch {epoch + 1}: {fid_score}')

        # g_scheduler.step()
        # d_scheduler.step()

        if epoch % args.save_interval == 0:
            print(f"Epoch [{epoch}/{args.epochs}] Train Generator Loss: {epoch_train_generator_loss:.4f}, Train Discriminator Loss: {epoch_train_discriminator_loss:.4f}, Val Generator Loss: {epoch_val_generator_loss:.4f}, Val Discriminator Loss: {epoch_val_discriminator_loss:.4f}")

    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
    }, os.path.join(args.save_dir, f"model.pth"))

    for i, (rgb_image, depth_image) in enumerate(test_loader):
        video_frames, _ = generate_frames(
            generator=generator,
            rgb_depth_pair=(rgb_image, depth_image),
            initial_orientation=args.initial_orientation,
            num_frames=args.num_frames,
            delta_degrees=args.delta_degrees,
            device=device,
            save_dir=os.path.join(args.save_dir, f'frames_{i}'),
            attention=args.attention
        )

        # Generate video from frames

        # video_path = os.path.join(args.save_dir, f'output_{i}.mp4')
        # height, width, _ = video_frames[0].shape
        #
        # # Define the codec and create VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec as needed
        # out = cv2.VideoWriter(video_path, fourcc, args.fps, (width, height))
        #
        # # Write frames to the video
        # for frame in video_frames:
        #     out.write(frame)
        #
        # # Release the VideoWriter object
        # out.release()
        #
        # print(f"Video saved to: {video_path}")

        gif_path = os.path.join(args.save_dir, f"output_animation_{i}.gif")

        # Write frames to the GIF file
        imageio.mimsave(gif_path, video_frames, fps=args.fps)

        print(f"GIF saved to: {gif_path}")

        # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(args.epochs), train_generator_losses, label="Train Generator Loss")
    plt.plot(range(args.epochs), train_discriminator_losses, label="Train Discriminator Loss")
    plt.plot(range(args.epochs), val_generator_losses, label="Val Generator Loss")
    plt.plot(range(args.epochs), val_discriminator_losses, label="Val Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, "train_val_losses.png"))
    plt.show()

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
    parser.add_argument("--attention", type=bool, default=True, help="Enable aattention layers")
    parser.add_argument("--initial_orientation", type=list, default=[0, 0, 0], help="Initial camera orientation (roll, pitch, yaw) for video generation")
    parser.add_argument("--num_frames", type=int, default=30, help="Number of frames to generate for the video")
    parser.add_argument("--delta_degrees", type=float, default=5.0, help="Degrees to pan the camera view between frames")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the generated video")
    parser.add_argument('--fid', type=bool, default=False, help='Calculate FiD metric')

    args = parser.parse_args()
    train(args)