# %%# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, dropout=0.1):
        """
        Initializes a ResBlock object.

        Args:
        - inchannel (int): number of input channels
        - outchannel (int): number of output channels
        - stride (int): stride of the convolutional layers (default: 1)
        - dropout (float): dropout probability (default: 0.1)

        Returns:
        - None
        """
        super(ResBlock, self).__init__()

        # 定义ResBlock的左侧部分，包括两个卷积层、BatchNorm层、ReLU激活函数、Dropout层
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(outchannel)
        )

        # 定义ResBlock的shortcut部分，如果输入输出通道数不同或stride不为1，则需要进行卷积操作
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )
            
    def forward(self, x):
        """
        Defines the forward pass of the ResBlock.

        Args:
        - x (torch.Tensor): input tensor

        Returns:
        - out (torch.Tensor): output tensor
        """
        # 左侧部分的输出加上shortcut部分的输出，然后进行ReLU激活函数
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out
    
class ResNet_8(nn.Module):
    def __init__(self):
        """
        Initializes a ResNet_8 object.

        Args:
        - None

        Returns:
        - None
        """
        super(ResNet_8, self).__init__()

        # 定义8个ResBlock，其中第一个ResBlock的输入通道数为1，输出通道数为16，步长为3，其余ResBlock的输入通道数等于上一个ResBlock的输出通道数，输出通道数分别为16、32、64、128，步长分别为1或3
        self.ResBlock1 = ResBlock(inchannel=1, outchannel=16, stride=3)
        self.ResBlock2 = ResBlock(inchannel=16, outchannel=16, stride=1)
        self.ResBlock3 = ResBlock(inchannel=16, outchannel=32, stride=3)
        self.ResBlock4 = ResBlock(inchannel=32, outchannel=32, stride=1)
        self.ResBlock5 = ResBlock(inchannel=32, outchannel=64, stride=3)
        self.ResBlock6 = ResBlock(inchannel=64, outchannel=64, stride=1)
        self.ResBlock7 = ResBlock(inchannel=64, outchannel=128, stride=3)
        self.ResBlock8 = ResBlock(inchannel=128, outchannel=128, stride=1)

        # 定义两个ResBlock，用于处理输入通道数为1的长序列，其中第一个ResBlock的输出通道数为32，步长为9，第二个ResBlock的输入通道数等于第一个ResBlock的输出通道数，输出通道数为128，步长为9
        self.LongResBlock1 = ResBlock(inchannel=1, outchannel=32, stride=9)
        self.LongResBlock2 = ResBlock(inchannel=32, outchannel=128, stride=9)

        # 定义分类层
        self.Dropout_1 = nn.Dropout(0.6)
        self.Dropout_2 = nn.Dropout(0.6)
        self.linear_1 = nn.Linear(3328, 128)
        self.linear_2 = nn.Linear(128, 30)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Defines the forward pass of the ResNet_8.

        Args:
        - x (torch.Tensor): input tensor

        Returns:
        - out (torch.Tensor): output tensor
        """
        # 前三个ResBlock的输出作为第四个ResBlock的输入，第四个ResBlock的输出作为第五个ResBlock的输入，第五个ResBlock的输出作为第六个ResBlock的输入，第六个ResBlock的输出作为第七个ResBlock的输入，第七个ResBlock的输出作为第八个ResBlock的输入
        out = self.ResBlock1(x)
        
        out = self.ResBlock2(out)
        out = self.ResBlock3(out)
        
        # 第一个长序列的输出作为第四个ResBlock的输入，第二个长序列的输出作为第八个ResBlock的输入
        long_out_1 = self.LongResBlock1(x)
        out = self.ResBlock4(out+long_out_1)

        long_out_2 = self.LongResBlock2(out)

        # 最终输出
        out = self.ResBlock5(out)
        out = self.ResBlock6(out)
        out = self.ResBlock7(out)
        out = self.ResBlock8(out+long_out_2)  
        # print('out.shape1',out.shape)  
        out = out.view(out.size(0), -1)
        # print('out.shape2',out.shape)  
        out = self.linear_1(out)
        out = self.relu(out)
        out = self.Dropout_1(out)
        out = self.linear_2(out)
        # out = self.Dropout_2(out)
        
        return out
    

class ResNet_16(nn.Module):
    def __init__(self):
        """
        Initializes a ResNet_16 object.

        Args:
        - None

        Returns:
        - None
        """
        super(ResNet_16, self).__init__()

        # 定义8个ResBlock，其中第一个ResBlock的输入通道数为1，输出通道数为16，步长为3，其余ResBlock的输入通道数等于上一个ResBlock的输出通道数，输出通道数分别为16、32、64、128，步长分别为1或3
        self.ResBlock1 = ResBlock(inchannel=1, outchannel=16, stride=3)#为什么要
        self.ResBlock2 = ResBlock(inchannel=16, outchannel=16, stride=1)
        self.ResBlock3 = ResBlock(inchannel=16, outchannel=32, stride=3)
        self.ResBlock4 = ResBlock(inchannel=32, outchannel=32, stride=1)
        self.ResBlock5 = ResBlock(inchannel=32, outchannel=64, stride=3)
        self.ResBlock6 = ResBlock(inchannel=64, outchannel=64, stride=1)
        self.ResBlock7 = ResBlock(inchannel=64, outchannel=128, stride=3)#2
        self.ResBlock8 = ResBlock(inchannel=128, outchannel=128, stride=1)

        self.ResBlock9 = ResBlock(inchannel=128, outchannel=256, stride=3)
        self.ResBlock10= ResBlock(inchannel=256, outchannel=256, stride=1)
        self.ResBlock11= ResBlock(inchannel=256, outchannel=512, stride=3)#3
        self.ResBlock12= ResBlock(inchannel=512, outchannel=512, stride=1)
        self.ResBlock13= ResBlock(inchannel=512, outchannel=1024, stride=3)
        self.ResBlock14= ResBlock(inchannel=1024, outchannel=1024, stride=1)
        self.ResBlock15= ResBlock(inchannel=1024, outchannel=2048, stride=3)#4
        self.ResBlock16= ResBlock(inchannel=2048, outchannel=2048, stride=1)

        # 定义两个ResBlock，用于处理输入通道数为1的长序列，其中第一个ResBlock的输出通道数为32，步长为9，第二个ResBlock的输入通道数等于第一个ResBlock的输出通道数，输出通道数为128，步长为9
        self.LongResBlock1 = ResBlock(inchannel=1, outchannel=32, stride=9)
        self.LongResBlock2 = ResBlock(inchannel=32, outchannel=128, stride=9)#2
        self.LongResBlock3 = ResBlock(inchannel=128, outchannel=512, stride=9)#3
        self.LongResBlock4 = ResBlock(inchannel=512, outchannel=2048, stride=9)#4
        # 定义分类层
        self.Dropout_1 = nn.Dropout(0.6)
        self.Dropout_2 = nn.Dropout(0.6)
        self.linear_1 = nn.Linear(2048, 128)
        self.linear_2 = nn.Linear(128, 30)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Defines the forward pass of the ResNet_8.

        Args:
        - x (torch.Tensor): input tensor

        Returns:
        - out (torch.Tensor): output tensor
        """
        # 前三个ResBlock的输出作为第四个ResBlock的输入，第四个ResBlock的输出作为第五个ResBlock的输入，第五个ResBlock的输出作为第六个ResBlock的输入，第六个ResBlock的输出作为第七个ResBlock的输入，第七个ResBlock的输出作为第八个ResBlock的输入
        out = self.ResBlock1(x)
        out = self.ResBlock2(out)
        out = self.ResBlock3(out)
        # 第一个长序列的输出作为第四个ResBlock的输入，第二个长序列的输出作为第八个ResBlock的输入
        # 第三个长序列的输出作为第十二个ResBlock输入，第四个长序列输出作为十六个ResBlock输入
        long_out_1 = self.LongResBlock1(x)
        out = self.ResBlock4(out+long_out_1)

        long_out_2 = self.LongResBlock2(out)

        # 最终输出
        out = self.ResBlock5(out)
        out = self.ResBlock6(out)
        out = self.ResBlock7(out)
        out = self.ResBlock8(out+long_out_2)
        # print('out.shape1',out.shape)       
        long_out_3 = self.LongResBlock3(out)
        # print('long_out_3.shape2',long_out_3.shape)  
        out = self.ResBlock9(out)
        out = self.ResBlock10(out)
        out = self.ResBlock11(out)

        out = self.ResBlock12(out+long_out_3)
        # print('out.shape',out.shape)  
        long_out_4 = self.LongResBlock4(out)

        out = self.ResBlock13(out)
        out = self.ResBlock14(out)
        out = self.ResBlock15(out)
        out = self.ResBlock16(out+long_out_4)
        # print('out.shape',out.shape)
        out = out.view(out.size(0), -1)
        out = self.linear_1(out)
        out = self.relu(out)
        out = self.Dropout_1(out)
        out = self.linear_2(out)
        # out = self.Dropout_2(out)
        
        return out