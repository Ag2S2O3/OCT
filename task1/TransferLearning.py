import torch
import torch.nn as nn
import torchvision.models as models

class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet, self).__init__()
        # 使用ResNet的前几层作为编码器
        resnet = models.resnet50(pretrained=True)
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 修改输入通道为1
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # ResNet的第1个Block
        self.layer2 = resnet.layer2  # ResNet的第2个Block
        self.layer3 = resnet.layer3  # ResNet的第3个Block
        self.layer4 = resnet.layer4  # ResNet的第4个Block

        # 使用Unet作为解码器
        # 上采样和卷积层
        self.up4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)  # 上采样 + 卷积
        self.conv4 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.Conv2d(256, 64, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)  # 最终的输出层
        
    def forward(self, x):
        # 编码部分
        x0 = self.layer0(x)  # 输出大小: 1/4
        x1 = self.layer1(x0) # 输出大小: 1/4
        x2 = self.layer2(x1) # 输出大小: 1/8
        x3 = self.layer3(x2) # 输出大小: 1/16
        x4 = self.layer4(x3) # 输出大小: 1/32

        # 解码部分
        d4 = self.up4(x4)  
        d4 = torch.cat([d4, x3], dim=1)  
        d4 = self.conv4(d4)
        # d4: torch.Size([8, 1024, 32, 32])

        d3 = self.up3(d4) 
        d3 = torch.cat([d3, x2], dim=1)  
        d3 = self.conv3(d3)
        # d3: torch.Size([8, 512, 64, 64])

        d2 = self.up2(d3)  
        d2 = torch.cat([d2, x1], dim=1)  
        d2 = self.conv2(d2)
        # d2: torch.Size([8, 256, 128, 128])

        d1 = self.up1(d2)  

        d1 = torch.cat([d1, x0], dim=1)  
        d1 = self.conv1(d1)
        # d1: torch.Size([8, 64, 128, 128])

        out = self.final_up(d1)
        out = self.final_conv(out)  
        # d0: torch.Size([8, 1, 512, 512])

        return out
        
if __name__ == "__main__":
    model = ResUNet()  # 创建模型
    test_input = torch.randn(8, 1, 512, 512)  # 创建一个随机输入，批量大小为8，单通道，512x512
    output = model(test_input)  # 通过模型前向传播