import torch
import torch.nn as nn
from torchvision import models

class ResNet_UNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_UNet, self).__init__()
        
        # 使用 ResNet 作为编码器
        resnet = models.resnet50(pretrained=True)
        self.encoder1 = nn.Sequential(*list(resnet.children())[:3])  # 64个特征
        self.encoder2 = nn.Sequential(*list(resnet.children())[3:5])  # 128个特征
        self.encoder3 = nn.Sequential(*list(resnet.children())[5:6])  # 256个特征
        self.encoder4 = nn.Sequential(*list(resnet.children())[6:7])  # 512个特征
        
        # 解码器部分
        self.decoder4 = self.upconv(512, 256)
        self.decoder3 = self.upconv(256, 128)
        self.decoder2 = self.upconv(128, 64)
        self.decoder1 = self.upconv(64, num_classes)
        
            
    def upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # 解码
        dec4 = self.decoder4(enc4)
        dec4 = torch.cat((dec4, enc3), dim=1)  # 跳跃连接
        dec3 = self.decoder3(dec4)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec2 = self.decoder2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec1 = self.decoder1(dec2)
        
        return dec1

# 使用示例
model = ResNet_UNet(num_classes=2)
