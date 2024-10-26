import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights

class ResDecoder(nn.Module):
    def __init__(self):
        super(ResDecoder, self).__init__()
        # 使用ResNet的前几层作为编码器
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.layer0 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 修改输入通道为64*2
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
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

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
                             out_channels=hidden_dim,  # 输出门
                             kernel_size=self.kernel_size, 
                             padding=self.padding, 
                             bias=self.bias)

    def forward(self, current_input, last_hidden, last_cell):

        # 拼接当前输入和上一个隐藏状态
        combined = torch.cat([current_input, last_hidden], dim=1)  

        # 计算输入门和遗忘门
        i_f_gate = self.W_i(combined)  
        i, f = torch.split(torch.sigmoid(i_f_gate), self.hidden_dim, dim=1) 

        # 计算候选记忆单元
        c_hat = torch.tanh(self.W_c(combined)) 

        # 更新记忆单元状态
        c_next = f * last_cell + i * c_hat  
        
        # 计算输出门
        o_gate = torch.sigmoid(self.W_o(combined))  

        # 计算下一个隐藏状态
        h_next = o_gate * torch.tanh(c_next)  
        
        return h_next, c_next  
    
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
    
# CA attention
class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()
        
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
 
        self.relu   = nn.ReLU()
        self.bn     = nn.BatchNorm2d(channel//reduction)
 
        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
 
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
 
    def forward(self, x):
        _, _, h, w = x.size()
        
        x_h = torch.mean(x, dim = 3, keepdim = True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim = 2, keepdim = True)
 
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
 
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
 
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
 
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):  # x.size() 30,40,50,30
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 30,1,50,30
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 30,1,50,30
        return self.sigmoid(x)  # 30,1,50,30


class TRUnet(nn.Module):
    def __init__(self, hidden_dim):
        super(TRUnet, self).__init__()

        self.lstm = ConvLSTMCell(input_dim=1, hidden_dim=hidden_dim, kernel_size=3)
        self.channel_attention = CA_Block(hidden_dim)   
        self.spatial_attention = SpatialAttention()
        self.f_ex = ResDecoder()                      
        self.d = UnetDecoder()
        self.decoder_attention_x3 = CA_Block(1024)
        self.decoder_attention_x2 = CA_Block(512)
        self.decoder_attention_x1 = CA_Block(256)
        self.decoder_attention_x0 = CA_Block(64)

        self.expand_conv = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)

        # 可学习的初始化参数
        self.h0 = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))  # 初始化为1x1
        self.c0 = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))  # 初始化为1x1


    def forward(self, image_sequence):

        batch_size, seq_len, channels, height, width = image_sequence.size()

        # 初始化隐藏状态和记忆单元，将h0和c0扩展到输入尺寸
        h = self.h0.expand(batch_size, -1, height, width).contiguous()
        c = self.c0.expand(batch_size, -1, height, width).contiguous()

        hidden_state = []  # 用于存储每个时间步的隐藏状态

        # 遍历时序图像序列
        for t in range(seq_len - 1):
            frame = image_sequence[:, t, :, :, :]
            h, c = self.lstm(frame, h, c)
            attention_map  = self.spatial_attention(h)
            weighted_hidden_state = h * attention_map  
            hidden_state.append(weighted_hidden_state)  # 存储当前时间步的隐藏状态

        # 处理当前帧
        current_frame = image_sequence[:, -1, :, :, :]
        expanded_current_frame = self.expand_conv(current_frame)

        # 将当前帧经过通道注意力机制
        attended_features = self.channel_attention(expanded_current_frame)

        # 将 hidden_state 转换为张量以进行处理
        hidden_state_tensor = torch.stack(hidden_state)  # 形状为 [seq_len - 1, batch_size, channels, height, width]

        # 使用平均池化将时间步整合为 64 层特征
        final_hidden_state = torch.mean(hidden_state_tensor, dim=0)  # 输出形状为 [batch_size, 64, height, width]

        # combined_features = torch.cat([attended_features, aggregated_hidden], dim=1)
        combined_features = torch.cat([attended_features, final_hidden_state], dim=1)

        x0, x1, x2, x3, x4 = self.f_ex(combined_features)

        # 将编码的部分通过注意力机制
        x3 = self.decoder_attention_x3(x3)
        x2 = self.decoder_attention_x2(x2)
        x1 = self.decoder_attention_x1(x1)
        x0 = self.decoder_attention_x0(x0)

        # 输出分割图
        output = self.d(x0, x1, x2, x3, x4)

        return output

        
if __name__ == "__main__":

    model = TRUnet(hidden_dim=64)


    # 创建一个随机数据集
    batch_size = 4
    seq_len = 10
    channels = 1
    height = 256
    width = 256

    # 随机生成测试数据 (batch_size, seq_len, channels, height, width)
    data = torch.randn(batch_size, seq_len, channels, height, width)

    output = model(data)
    print(f"output:{output.shape}")