import torch
import torch.nn as nn
from models.generator import Generator
from models.discriminator import Discriminator
from utils.data_utils import get_data_loaders
from torchvision.utils import save_image
import os

def test(test_loader, generator, discriminator, device, save_dir='test_results'):
    generator.eval()
    discriminator.eval()

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (rgb_images, depth_images) in enumerate(test_loader):
            rgb_images = rgb_images.to(device)
            depth_images = depth_images.to(device)

            noise = torch.randn(rgb_images.size(0), 100, 1, 1).to(device)
            generated_images = generator(rgb_images, depth_images, noise)

            # Save generated images
            for j in range(generated_images.size(0)):
                img_path = os.path.join(save_dir, f'generated_{i}_{j}.png')
                save_image(generated_images[j], img_path)

            # Evaluate discriminator performance
            real_output = discriminator(rgb_images, depth_images, rgb_images)
            fake_output = discriminator(rgb_images, depth_images, generated_images)

            real_score = torch.mean(real_output).item()
            fake_score = torch.mean(fake_output).item()

            print(f'Batch {i+1}/{len(test_loader)}: Real Score: {real_score:.4f}, Fake Score: {fake_score:.4f}')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "path/to/trained/model.pth"
    test_root_dir = "path/to/test/dataset"
    batch_size = 32

    generator = Generator(noise_dim=100, attention=True).to(device)
    discriminator = Discriminator(attention=True).to(device)
    generator.load_state_dict(torch.load(model_path)['generator'])
    discriminator.load_state_dict(torch.load(model_path)['discriminator'])

    test_loader = get_data_loaders(test_root_dir, batch_size)

    test(test_loader, generator, discriminator, device)