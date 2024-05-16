import torch
import torch.nn as nn
from models.attention_module import SelfAttention


class Generator(nn.Module):
    def __init__(self,rgb_channels=3, depth_channels=1, output_channels=4, attention=True):
        super(Generator, self).__init__()

        input_channels = rgb_channels + depth_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(64) if attention else nn.Identity(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(256) if attention else nn.Identity(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

        )


        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )


    def forward(self, rgb, depth):
        # Concatenate RGB and depth inputs along the channel dimension
        input_tensor = torch.cat((rgb, depth), dim=1)

        # Pass through the generator
        encoded = self.encoder(input_tensor)

        generated_scene = self.decoder(encoded)

        return generated_scene

