import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, attention=True, pretrained_backbone=None):
        super(Discriminator, self).__init__()
        self.attention = attention

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        if pretrained_backbone:
            self.initialize_with_pretrained(pretrained_backbone)

    def forward(self, rgb_images, depth_images, images):
        out = self.features(images)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def initialize_with_pretrained(self, pretrained_backbone):
        # Initialize the discriminator weights with the pre-trained backbone weights
        # You may need to adjust this part based on the pre-trained model architecture
        pass