from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import ResnetEncoder as res
import torchvision.models as models

# 预训练模型
resnet = models.resnet50(pretrained=True)

resnet_layers = list(resnet.children())[:-2]  # 提取卷积层


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

# Unet与resnet相结合
class UNetWithResNetEncoder(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 1,
                 bilinear: bool = True):
        super(UNetWithResNetEncoder, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 使用ResNet50作为编码器
        resnet = res.resnet50()  

        # 使用ResNet的前几层作为编码器
        self.in_conv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.down1 = resnet.layer1
        self.down2 = resnet.layer2
        self.down3 = resnet.layer3
        self.down4 = resnet.layer4

        # 解码器部分保持不变
        factor = 2 if bilinear else 1
        self.conv_x5up = nn.Conv2d(2048, 1024, kernel_size=1)
        self.up1 = Up(2048, 1024 // factor, bilinear)
        self.up2 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out_conv = OutConv(64, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        x1 = self.in_conv(x)
        # print(f"x1 shape: {x1.shape}")  # 打印 x1 形状
        x2 = self.down1(x1)
        # print(f"x2 shape: {x2.shape}")  # 打印 x2 形状
        x3 = self.down2(x2)
        # print(f"x3 shape: {x3.shape}")  # 打印 x3 形状
        x4 = self.down3(x3)
        # print(f"x4 shape: {x4.shape}")  # 打印 x4 形状
        x5 = self.down4(x4)
        # print(f"x5 shape: {x5.shape}")  # 打印 x5 形状

        # 对 x5 进行上采样
        # x5 = F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=True)
        # x5 = self.conv_x5up(x5)  # 调整通道数
        # print(f"x5up shape: {x5.shape}")  # 打印 x5up 形状

        # 解码器部分
        x = self.up1(x5, x4)
        # print(f"Up1 output shape: {x.shape}")  # 打印上采样后的输出形状
        x = self.up2(x, x3)
        # print(f"Up2 output shape: {x.shape}")  # 打印上采样后的输出形状
        x = self.up3(x, x2)
        # print(f"Up3 output shape: {x.shape}")  # 打印上采样后的输出形状
        x = self.up4(x, x1)
        # print(f"Up4 output shape: {x.shape}")  # 打印上采样后的输出形状
        
        logits = self.out_conv(x)
        logits = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=True)
        # print(f"Output shape: {logits.shape}")  # 打印输出形状

        return logits

if __name__ == "__main__":
    # model = UNetWithResNetEncoder(in_channels=1, num_classes=1)  # 创建模型
    # test_input = torch.randn(8, 1, 512, 512)  # 创建一个随机输入，批量大小为8，单通道，512x512
    # output = model(test_input)  # 通过模型前向传播
    # 预训练模型
    resnet = models.resnet50(pretrained=True)

    resnet_layers = list(resnet.children())[:-2]  # 提取卷积层

    for i, layer in enumerate(resnet_layers):
        print(f"Layer {i}: {layer}")
