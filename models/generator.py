import torch
import torch.nn as nn
import torch.nn.functional as F

class SPADE(nn.Module):
    def __init__(self, input_nc, norm_nc):
        super(SPADE, self).__init__()
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        self.conv_gamma = nn.Conv2d(norm_nc, input_nc, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(norm_nc, input_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        gamma = self.conv_gamma(segmap)
        beta = self.conv_beta(segmap)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class SPADEBlock(nn.Module):
    def __init__(self, input_nc, norm_nc):
        super(SPADEBlock, self).__init__()
        self.spade = SPADE(input_nc, norm_nc)
        self.conv = nn.Conv2d(input_nc, input_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        x = self.conv(x)
        x = self.spade(x, segmap)
        x = F.relu(x)
        return x

class Generator(nn.Module):
    def __init__(self, noise_dim, num_spade_blocks=4):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_spade_blocks = num_spade_blocks

        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.spade_blocks = nn.ModuleList([
            SPADEBlock(64, 64) for _ in range(num_spade_blocks)
        ])

        self.conv2 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(3)
        self.tanh = nn.Tanh()

    def forward(self, rgb_images, depth_images, noise):
        inputs = torch.cat((rgb_images, depth_images), dim=1)
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.relu(x)

        for spade_block in self.spade_blocks:
            x = spade_block(x, depth_images)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.tanh(x)

        return x
