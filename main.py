import argparse
import torch
from models.generator import Generator
from models.discriminator import Discriminator
from models.loss import GANLoss
from utils.data_utils import get_data_loaders
from train import train

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate for optimizers')
    parser.add_argument('--pretrained_path', type=str, required=True, help='Path to the pre-trained model file')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use for training')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    device = torch.device(args.device)
    data_loader = get_data_loaders(args.root_dir, args.batch_size)

    # Load pre-trained models
    generator = Generator(noise_dim=100, attention=True).to(device)
    discriminator = Discriminator(attention=True).to(device)
    pretrained_state = torch.load(args.pretrained_path)
    generator.load_state_dict(pretrained_state['models']['generator_rgb_smooth'])
    discriminator.load_state_dict(pretrained_state['models']['discriminator_rgb_smooth'])

    loss_fn = GANLoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    train(data_loader, generator, discriminator, loss_fn, optimizer_g, optimizer_d, device, args.num_epochs)