import os
import torch
from torch.utils.data import random_split
from torchvision.utils import save_image
import cv2
from utils.data_utils import get_data_loaders
from models.generator import Generator
from models.discriminator import Discriminator
from models.loss import GeneratorLoss, DiscriminatorLoss

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "datasets/"
batch_size = 8
num_epochs = 100
learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.999
save_model_path = "models/"
save_video_path = "video/"

# Load dataset
data_loader = get_data_loaders(data_dir, batch_size)
total_samples = len(data_loader.dataset)
train_size = int(0.8 * total_samples)
val_size = int(0.1 * total_samples)
test_size = total_samples - train_size - val_size
train_set, val_set, test_set = random_split(data_loader.dataset, [train_size, val_size, test_size])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Initialize loss functions and optimizers
generator_loss = GeneratorLoss()
discriminator_loss = DiscriminatorLoss()
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

# Training loop
for epoch in range(num_epochs):
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
    if (epoch + 1) % 10 == 0:
        torch.save(generator.state_dict(), os.path.join(save_model_path, f"generator_epoch_{epoch + 1}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(save_model_path, f"discriminator_epoch_{epoch + 1}.pth"))
        print(f"Epoch [{epoch + 1}/{num_epochs}] Generator Loss: {generator_loss_value.item():.4f} Discriminator Loss: {discriminator_loss_value.item():.4f}")

# Generate video from test set
video_frames = []
for rgb_images, depth_images in test_loader:
    rgb_images = rgb_images.to(device)
    depth_images = depth_images.to(device)
    with torch.no_grad():
        fake_scenes = generator(rgb_images, depth_images)
        for i in range(fake_scenes.size(0)):
            video_frames.append(fake_scenes[i].permute(1, 2, 0).cpu().numpy())

height, width, channels = video_frames[0].shape
video = cv2.VideoWriter(os.path.join(save_video_path, 'output.avi'), cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
for frame in video_frames:
    video.write((frame * 255).astype('uint8'))
video.release()