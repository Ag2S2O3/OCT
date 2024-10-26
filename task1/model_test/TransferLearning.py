import torch
import torch.nn as nn
import torchvision.models as models

# 网络主体框架
class ResDecoder(nn.Module):
    def __init__(self):
        super(ResDecoder, self).__init__()
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
        
    def forward(self, x):
        # 编码部分
        x0 = self.layer0(x)  # 输出大小: 1/4
        x1 = self.layer1(x0) # 输出大小: 1/4
        x2 = self.layer2(x1) # 输出大小: 1/8
        x3 = self.layer3(x2) # 输出大小: 1/16
        x4 = self.layer4(x3) # 输出大小: 1/32

        return x0, x1, x2, x3, x4
    

# 数据前处理（融合、遗忘与更新）
'''
还需要改进处理！
'''
class ModifiedConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ModifiedConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.hidden = nn.Conv2d(in_channels=1, out_channels=self.hidden_dim, kernel_size=1)
        
        # 卷积操作用于输入和隐藏状态
        self.W_i = nn.Conv2d(in_channels=input_dim + hidden_dim, 
                             out_channels=2 * hidden_dim,  # 输入门和遗忘门
                             kernel_size=self.kernel_size, 
                             padding=self.padding, 
                             bias=self.bias)
        
        self.W_c = nn.Conv2d(in_channels=input_dim + hidden_dim, 
                             out_channels=hidden_dim,  # 候选记忆单元
                             kernel_size=self.kernel_size, 
                             padding=self.padding, 
                             bias=self.bias)
        
        self.W_o = nn.Conv2d(in_channels=input_dim + hidden_dim, 
                             out_channels=1,  # 输出门
                             kernel_size=self.kernel_size, 
                             padding=self.padding, 
                             bias=self.bias)

    def forward(self, current_input, last_output, cell):
        c_cur = cell

        last_output = self.hidden(last_output)
        
        # 拼接输入
        combined = torch.cat([current_input, last_output], dim=1)  # (batch_size, input_dim + hidden_dim, height, width)

        # 计算输入门和遗忘门
        i_f_gate = self.W_i(combined)  # (batch_size, 2 * hidden_dim, height, width)
        i, f = torch.split(torch.sigmoid(i_f_gate), self.hidden_dim, dim=1)  # sigmoid gates

        # 计算候选记忆单元
        c_hat = torch.tanh(self.W_c(combined))  # (batch_size, hidden_dim, height, width)

        # 更新记忆单元状态
        c_next = f * c_cur + i * c_hat
        
        # 计算输出门
        o = torch.sigmoid(self.W_o(combined))  # (batch_size, 1, height, width)
        print(o.shape)
        
        return o, c_next    # 返回值：o--结合后的图像，可用于后续的特征提取（但是现在的数目是hidden_dim，还需要一步卷积）c-next：记忆单元状态

    def init_hidden(self, batch_size, image_size):
        '''
        是否要使用？？
        '''
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.W_i.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.W_i.weight.device))
    
class UnetDecoder(nn.Module):
    def __init__(self) :
        super(UnetDecoder, self).__init__()
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

    def forward(self, x0, x1 ,x2 ,x3, x4):
        # 解码部分
        d4 = self.up4(x4)  
        d4 = torch.cat([d4, x3], dim=1)  
        d4 = self.conv4(d4)
        # d4: torch.Size([8, 1024, 16, 16])

        d3 = self.up3(d4) 
        d3 = torch.cat([d3, x2], dim=1)  
        d3 = self.conv3(d3)
        # d3: torch.Size([8, 512, 32, 32])

        d2 = self.up2(d3)  
        d2 = torch.cat([d2, x1], dim=1)  
        d2 = self.conv2(d2)
        # d2: torch.Size([8, 256, 64, 64])

        d1 = self.up1(d2)  

        d1 = torch.cat([d1, x0], dim=1)  
        d1 = self.conv1(d1)
        # d1: torch.Size([8, 64, 64, 64])

        out = self.final_up(d1)
        out = self.final_conv(out)  
        # d0: torch.Size([8, 1, 256, 256])

        return out

class TRUnet(nn.Module):
    def __init__(self):
        super(TRUnet, self).__init__()


        self.lstm = ModifiedConvLSTMCell(1, 64, 3, True)
        self.f_ex = ResDecoder()                      
        self.d = UnetDecoder()
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.mix = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, current_input, last_output, cell_in):

        input, cell_out = self.lstm(current_input, last_output, cell_in)

        # 提取特征
        x0, x1, x2, x3, x4 = self.f_ex(input)

        # 输出分割图
        output = self.d(x0, x1, x2, x3, x4)

        output = self.sig(output)

        cell_out = self.tanh(cell_out)

        cell_out = self.mix(cell_out)

        output = output * cell_out

        return output, cell_out

        
if __name__ == "__main__":
    # 创建输入数据
    current_input = torch.rand(8, 1, 256, 256)
    past_data = torch.rand(8, 1, 256, 256)
    cell = torch.rand(8, 1, 256, 256)

    model = TRUnet()

    output, _ = model(past_data, current_input, cell)
    print(f"output:{output.shape}")