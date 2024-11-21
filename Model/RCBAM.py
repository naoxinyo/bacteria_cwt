# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
# import netron
# %%
class RamanAttention(nn.Module):                        #Raman注意力机制
    def __init__(self, kernel_size=3):                  #卷积核大小
        super(RamanAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7' 
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

   
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([max_out, avg_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)

        return x
# %%
class ChannelAttention(nn.Module):                      #通道注意力机制
    def __init__(self, channels, reduction=2):
        super(ChannelAttention, self).__init__()

        self.channels = channels
        self.inter_channels = channels // reduction

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.mlp = nn.Sequential(
            nn.Linear(self.channels, self.inter_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.inter_channels, self.channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x).squeeze(2))
        # max_out = self.mlp(self.max_pool(x))

        # out = avg_out + max_out
        out = self.sigmoid(avg_out)

        return out.unsqueeze(2)

# %%
# ChannelAttention(2)(torch.rand([11, 2, 2048])).shape
# %%
class BasicBlock(nn.Module):

    def __init__(self, inchannel, outchannel, stride=1, shift_att=None, channel_att=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(outchannel)

        self.conv2 = nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(outchannel)

        self.ShiftAttention = shift_att
        self.ChannelAttention = channel_att
        self.stride = stride

        self.relu = nn.ReLU(inplace=True)

        self.Downsample = None
        if inchannel != outchannel or stride != 1:
            self.Downsample = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )

    def forward(self, x):
        residual = x

        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))

        if self.ChannelAttention is not None:
            out = out * self.ChannelAttention(out)
        
        if self.ShiftAttention is not None:
            out = out * self.ShiftAttention(out)
        
        if self.Downsample is not None:
            residual = self.Downsample(x)
        
        out += residual
        out = self.relu(out)

        return out
# %%
class ResNet8_RamanAttention(nn.Module):
    def __init__(self) -> None:
        super(ResNet8_RamanAttention, self).__init__()

        self.ResBlock1 = BasicBlock(inchannel=1, outchannel=16, stride=3, shift_att=RamanAttention())
        self.ResBlock2 = BasicBlock(inchannel=16, outchannel=16, stride=1, shift_att=RamanAttention())
        self.ResBlock3 = BasicBlock(inchannel=16, outchannel=32, stride=3, shift_att=RamanAttention())
        self.ResBlock4 = BasicBlock(inchannel=32, outchannel=32, stride=1, shift_att=RamanAttention())
        self.ResBlock5 = BasicBlock(inchannel=32, outchannel=64, stride=3, shift_att=RamanAttention())
        self.ResBlock6 = BasicBlock(inchannel=64, outchannel=64, stride=1, shift_att=RamanAttention())
        self.ResBlock7 = BasicBlock(inchannel=64, outchannel=128, stride=3, shift_att=RamanAttention())
        self.ResBlock8 = BasicBlock(inchannel=128, outchannel=128, stride=1, shift_att=RamanAttention())

        self.LongResBlock1 = BasicBlock(inchannel=1, outchannel=32, stride=9, shift_att=RamanAttention())
        self.LongResBlock2 = BasicBlock(inchannel=32, outchannel=128, stride=9, shift_att=RamanAttention())

                # 定义分类层
        self.Dropout_1 = nn.Dropout(0.6)
        self.Dropout_2 = nn.Dropout(0.6)
        self.linear_1 = nn.Linear(128*26, 128)
        self.linear_2 = nn.Linear(128, 30)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.ResBlock1(x)
        out = self.ResBlock2(out)
        out = self.ResBlock3(out)

        long_out_1 = self.LongResBlock1(x)
        out = self.ResBlock4(out+long_out_1)

        long_out_2 = self.LongResBlock2(out)

        out = self.ResBlock5(out)
        out = self.ResBlock6(out)
        out = self.ResBlock7(out)
        out = self.ResBlock8(out+long_out_2)

        out = out.view(out.size(0), -1)
        out = self.linear_1(out)
        out = self.relu(out)
        out = self.Dropout_1(out)
        out = self.linear_2(out)
        # out = self.Dropout_2(out)

        
        return out
# %%
class ResNet8_ChannelAttention(nn.Module):
    def __init__(self) -> None:
        super(ResNet8_ChannelAttention, self).__init__()

        self.ResBlock1 = BasicBlock(inchannel=1, outchannel=16, stride=3, channel_att=ChannelAttention(16))
        self.ResBlock2 = BasicBlock(inchannel=16, outchannel=16, stride=1, channel_att=ChannelAttention(16))
        self.ResBlock3 = BasicBlock(inchannel=16, outchannel=32, stride=3, channel_att=ChannelAttention(32))
        self.ResBlock4 = BasicBlock(inchannel=32, outchannel=32, stride=1, channel_att=ChannelAttention(32))
        self.ResBlock5 = BasicBlock(inchannel=32, outchannel=64, stride=3, channel_att=ChannelAttention(64))
        self.ResBlock6 = BasicBlock(inchannel=64, outchannel=64, stride=1, channel_att=ChannelAttention(64))
        self.ResBlock7 = BasicBlock(inchannel=64, outchannel=128, stride=3, channel_att=ChannelAttention(128))
        self.ResBlock8 = BasicBlock(inchannel=128, outchannel=128, stride=1, channel_att=ChannelAttention(128))

        self.LongResBlock1 = BasicBlock(inchannel=1, outchannel=32, stride=9, channel_att=ChannelAttention(32))
        self.LongResBlock2 = BasicBlock(inchannel=32, outchannel=128, stride=9, channel_att=ChannelAttention(128))

# 定义分类层
        self.Dropout_1 = nn.Dropout(0.6)
        self.Dropout_2 = nn.Dropout(0.6)
        self.linear_1 = nn.Linear(128*26, 128)
        self.linear_2 = nn.Linear(128, 30)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.ResBlock1(x)
        out = self.ResBlock2(out)
        out = self.ResBlock3(out)

        long_out_1 = self.LongResBlock1(x)
        out = self.ResBlock4(out+long_out_1)

        long_out_2 = self.LongResBlock2(out)

        out = self.ResBlock5(out)
        out = self.ResBlock6(out)
        out = self.ResBlock7(out)
        out = self.ResBlock8(out+long_out_2)

        out = out.view(out.size(0), -1)
        out = self.linear_1(out)
        out = self.relu(out)
        out = self.Dropout_1(out)
        out = self.linear_2(out)
        # out = self.Dropout_2(out)
        
        return out
# %%
class ResNet8_RCBAM(nn.Module):
    def __init__(self) -> None:
        super(ResNet8_RCBAM, self).__init__()

        self.ResBlock1 = BasicBlock(inchannel=1, outchannel=16, stride=3, shift_att=RamanAttention(), channel_att=ChannelAttention(16))
        self.ResBlock2 = BasicBlock(inchannel=16, outchannel=16, stride=1, shift_att=RamanAttention(), channel_att=ChannelAttention(16))
        self.ResBlock3 = BasicBlock(inchannel=16, outchannel=32, stride=3, shift_att=RamanAttention(), channel_att=ChannelAttention(32))
        self.ResBlock4 = BasicBlock(inchannel=32, outchannel=32, stride=1, shift_att=RamanAttention(), channel_att=ChannelAttention(32))
        self.ResBlock5 = BasicBlock(inchannel=32, outchannel=64, stride=3, shift_att=RamanAttention(), channel_att=ChannelAttention(64))
        self.ResBlock6 = BasicBlock(inchannel=64, outchannel=64, stride=1, shift_att=RamanAttention(), channel_att=ChannelAttention(64))
        self.ResBlock7 = BasicBlock(inchannel=64, outchannel=128, stride=3, shift_att=RamanAttention(), channel_att=ChannelAttention(128))
        self.ResBlock8 = BasicBlock(inchannel=128, outchannel=128, stride=1, shift_att=RamanAttention(), channel_att=ChannelAttention(128))

        self.LongResBlock1 = BasicBlock(inchannel=1, outchannel=32, stride=9, shift_att=RamanAttention(), channel_att=ChannelAttention(32))
        self.LongResBlock2 = BasicBlock(inchannel=32, outchannel=128, stride=9, shift_att=RamanAttention(), channel_att=ChannelAttention(128))

# 定义分类层
        self.Dropout_1 = nn.Dropout(0.6)
        self.Dropout_2 = nn.Dropout(0.6)
        self.linear_1 = nn.Linear(128*26, 128)
        self.linear_2 = nn.Linear(128, 30)
        # self.linear_3 = nn.Linear(128, 15)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.ResBlock1(x)
        out = self.ResBlock2(out)
        out = self.ResBlock3(out)

        long_out_1 = self.LongResBlock1(x)
        out = self.ResBlock4(out+long_out_1)

        long_out_2 = self.LongResBlock2(out)

        out = self.ResBlock5(out)
        out = self.ResBlock6(out)
        out = self.ResBlock7(out)
        out = self.ResBlock8(out+long_out_2)
        
        
        out = out.view(out.size(0), -1)
        out = self.linear_1(out)
        out = self.relu(out)
        out = self.Dropout_1(out)
        out = self.linear_2(out)
        # out = self.Dropout_2(out)
        
        return out

