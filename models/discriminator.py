import torch
import torch.nn as nn

from models.attention_module import SelfAttention


class Discriminator(nn.Module):
    def __init__(self, input_channels=4, attention=True):
        super(Discriminator, self).__init__()

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(32) if attention else nn.Identity(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(64) if attention else nn.Identity(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(128) if attention else nn.Identity(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(256) if attention else nn.Identity(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_tensor):
        # Pass through the convolutional layers
        x = self.conv(input_tensor)

        # Flatten the tensor
        x = x.view(-1, 256 * 4 * 4)

        # Pass through the fully connected layers
        output = self.fc(x)

        return output