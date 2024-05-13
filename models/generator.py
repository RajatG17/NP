import torch
import torch.nn as nn
from torchvision.models import resnet18
from models.attention_module import SelfAttention

class Generator(nn.Module):
    def __init__(self, noise_dim, attention=True):
        super().__init__()
        self.noise_dim = noise_dim
        self.attention = attention

        self.backbone = resnet18(pretrained=True)

        self.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.encoder = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        )





        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512 + noise_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            SelfAttention(256) if attention else nn.Identity(),
            # nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            SelfAttention(128) if attention else nn.Identity(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            SelfAttention(64) if attention else nn.Identity(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        # self.rgb_encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     SelfAttention(128) if attention else nn.Identity(),
        #     nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     SelfAttention(128) if attention else nn.Identity(),
        #     nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )
        #
        # self.depth_encoder = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # SelfAttention(128) if attention else nn.Identity(),
        #     nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # SelfAttention(128) if attention else nn.Identity(),
        #     nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )

        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(256+noise_dim, 256, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(True),
        #     # SelfAttention(256) if attention else nn.Identity(),
        #     # nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
        #     # nn.BatchNorm2d(256),
        #     # nn.ReLU(True),
        #     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     # SelfAttention(64) if attention else nn.Identity(),
        #     nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        #     nn.Tanh(),
        # )

    def forward(self, rgb_images, depth_images, noise):
        torch.cuda.empty_cache()
        # rgb_encoded = self.rgb_encoder(rgb_images)
        # depth_encoded = self.depth_encoder(depth_images)
        # noise = noise.repeat(1, 1, rgb_encoded.size(2), rgb_encoded.size(3))
        # encoded = torch.cat((rgb_encoded, depth_encoded, noise), dim=1)
        inputs = torch.cat((rgb_images, depth_images), dim=1)
        encoded = self.encoder(inputs)
        noise = noise.repeat(1, 1, encoded.size(2), encoded.size(3))
        encoded = torch.cat((encoded, noise), dim=1)

        return self.decoder(encoded)
    