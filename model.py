import torch
import torch.nn as nn


class SRmodel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.upsample(x)
        return x


loss_function = nn.MSELoss()
