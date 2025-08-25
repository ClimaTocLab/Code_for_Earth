import torch
import torch.nn as nn

# --- U-Net Model ---

class DoubleConv(nn.Module):
    """
    A convolutional block with three Conv2D layers, each followed by BatchNorm and ReLU.
    Provides deeper feature extraction than the standard 2-layer block in the original U-Net.
    """
    def __init__(self, in_channels, out_channels, dropout_prob=0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Dropout layer could be used for regularization (currently commented out)
            # nn.Dropout2d(dropout_prob),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling block: reduces spatial resolution by a factor of 2,
    then applies a DoubleConv to extract features.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),  # halve H and W
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """
    Upscaling block: increases resolution and fuses encoder features via skip connections.
    Supports bilinear upsampling or transposed convolution.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            # Bilinear upsampling + DoubleConv
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # Transposed convolution + DoubleConv
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: from deeper layer (low-res), x2: from encoder (skip connection)
        x1 = self.up(x1)

        # Handle size mismatch due to odd dimensions by padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])

        # Concatenate encoder + decoder features
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetSuperRes(nn.Module):
    """
    U-Net model adapted for super-resolution of spatial fields.
    - Input: multi-channel low-resolution features
    - Output: single-channel high-resolution prediction (e.g., PM2.5 concentration)
    """
    def __init__(self, n_channels_in, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(n_channels_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024 + 512, 512, bilinear)
        self.up2 = Up(512 + 256, 256, bilinear)
        self.up3 = Up(256 + 128, 128, bilinear)
        self.up4 = Up(128 + 64, 64, bilinear)

        # Final conv layer reduces to 1 output channel
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output prediction
        return self.outc(x)
