import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)
        attention_scores = torch.bmm(proj_query, proj_key)
        attention_scores = attention_scores / (channels // 8) ** 0.5
        attention_scores = torch.softmax(attention_scores, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, height * width)
        attention_map = torch.bmm(proj_value, attention_scores.permute(0, 2, 1))
        attention_map = attention_map.view(batch_size, channels, height, width)
        return attention_map + x

class Generator(nn.Module):
    def __init__(self, noise_dim, attention=True, pretrained_backbone=None):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.attention = attention

        self.fc = nn.Linear(noise_dim, 256 * 4 * 4)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(128) if attention else nn.Identity(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(64) if attention else nn.Identity(),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

        if pretrained_backbone:
            self.initialize_with_pretrained(pretrained_backbone)

    def forward(self, rgb_images, depth_images, noise):
        out = self.fc(noise)
        out = out.view(-1, 256, 4, 4)
        img = self.conv_blocks(out)
        return img

    def initialize_with_pretrained(self, pretrained_backbone):
        # Initialize the generator weights with the pre-trained backbone weights
        # You may need to adjust this part based on the pre-trained model architecture
        pass