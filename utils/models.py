import torch
import torch.nn as nn


# -------------------------
# Residual Block
# -------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


# -------------------------
# VGG Encoder
# -------------------------
class VGGEncoder(nn.Module):
    def __init__(self, vgg_path):
        super(VGGEncoder, self).__init__()

        self.vgg = nn.Sequential(

            nn.Conv2d(3, 3, 1),

            # -----------------
            # Block 1
            # -----------------
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 64, 3),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2, ceil_mode=True),

            # -----------------
            # Block 2
            # -----------------
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2, ceil_mode=True),

            # -----------------
            # Block 3
            # -----------------
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2, ceil_mode=True),

            # -----------------
            # Block 4
            # -----------------
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(inplace=True)  # relu4_1
        )

        # Load pretrained VGG weights
        state_dict = torch.load(vgg_path)
        self.vgg.load_state_dict(state_dict, strict=False)

        enc_layers = list(self.vgg.children())

        self.enc_1 = nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])

        # Freeze encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def forward(self, x, is_test=False):

        h1 = self.enc_1(x)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)

        if is_test:
            return h4

        return h1, h2, h3, h4


# -------------------------
# Improved Decoder
# -------------------------
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            # -----------------
            # 512 -> 256
            # -----------------
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(inplace=True),

            # Residual refinement
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            # -----------------
            # Upsample 1
            # -----------------
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(inplace=True),

            # -----------------
            # Upsample 2
            # -----------------
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),

            # -----------------
            # Upsample 3
            # -----------------
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, 3),
            nn.ReLU(inplace=True),

            # -----------------
            # Final RGB output
            # -----------------
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 3, 3)
        )

    def forward(self, x):
        return torch.clamp(self.net(x), 0, 1)