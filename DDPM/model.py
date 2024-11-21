import torch
import torch.nn as nn


# data는 cifar10으로 3,32,32의 이미지
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),  # (3,32,32) -> (64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # (64,16,16) -> (128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # (128,8,8) -> (256,4,4)
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # (256,4,4) -> (128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # (128,8,8) -> (64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),  # (64,16,16) -> (3,32,32)
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, x, t=None):
        # 타임스텝 임베딩은 생략
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return x
