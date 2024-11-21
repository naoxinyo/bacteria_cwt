# %%
from numpy import outer #outer函数用于计算两个数组的点积
import torch
from torch import nn, einsum #nn模块是神经网络相关的模块，einsum模块用于进行 Einstein summation，这是一种高效计算张量积分的算法。
import torch.nn.functional as F #导入F模块，这个模块包含了一些常用的激活函数和损失函数。
from torch.optim import Adam #导入Adam优化器，这个优化器是一种常用的优化器，用于在深度学习中更新模型参数。
import math

# %%
def drop_path(x, drop_prob: float = 0., training: bool = False):                    #定义一个函数drop_path，用于实现DropPath操作，这是一种用于防止过拟合的技术，并提高模型的泛化能力。
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """


    if drop_prob == 0. or not training:                 #如果drop_prob为0或者模型不在训练模式下，则直接返回输入x。
        return x
    keep_prob = 1 - drop_prob                           #计算保留的概率keep_prob，它是1减去drop_prob。
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
 
class DropPath(nn.Module):                              #定义DropPath类，继承自nn.Module。
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
 
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
 
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):                              #定义LayerNorm类，继承自nn.Module。层归一化是一种正则化方法，通常用于对神经网络中不同层的激活值进行归一化处理，帮助模型更稳定和快速地训练。
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)  # weight bias对应γ β
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:                 #forward函数是nn.Module类中的一个方法，用于定义前向传播的过程。
        if self.data_format == "channels_last":                         # channels_last对应于输入形状为(batch_size, height, width, channels)
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)         # F.layer_norm函数用于进行层归一化操作，它的参数包括输入x、归一化的形状、权重、偏置和epsilon。
        elif self.data_format == "channels_first":
            # [batch_size, channels, height]
            # 对channels 维度求均值
            mean = x.mean(1, keepdim=True)
            # 方差
            var = (x - mean).pow(2).mean(1, keepdim=True)
            # 减均值，除以标准差的操作
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x

class GRN(nn.Module):                                   # 定义GRN类，继承自nn.Module。功能是对输入进行全局响应归一化。
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))           # 定义gamma参数，用于对输入进行缩放
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))            # 定义beta参数，用于对输入进行平移

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1), keepdim=True)              # 计算输入x的L2范数，并保持维度不变
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)            # 对输入x进行归一化
        return self.gamma * (x * Nx) + self.beta + x                # 返回归一化后的结果

class ConvNextBlock(nn.Module):                                     # 定义ConvNextBlock类，继承自nn.Module。功能是定义一个卷积神经网络块，包含深度卷积、归一化、全连接层、激活函数和全局响应归一化。
    def __init__(self, dim, drop_rate=0.2, layer_scale_init_value=1e-6):            # 定义初始化函数，包括输入通道数、dropout率、层缩放初始化值
        super().__init__()                                                           # 调用父类的初始化函数
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv   深度卷积
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")           # layer norm       层归一化
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers       全连接层
        self.act = nn.GELU()                                                                # 激活函数
        self.grn = GRN(4 * dim)                                                      # 全局响应归一化
        self.pwconv2 = nn.Linear(4 * dim, dim)                                      # pointwise/1x1 convs, implemented with linear layers   全连接层
        # gamma 针对layer scale的操作
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
        #                           requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()  # nn.Identity() 恒等映射

    def forward(self, x: torch.Tensor) -> torch.Tensor:                             # 定义前向传播函数
        shortcut = x                                                                # 残差连接

        x = self.dwconv(x)                                                                                       
        x = x.permute(0, 2, 1)                      
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        # if self.gamma is not None:
        #     x = self.gamma * x
        x = x.permute(0, 2, 1)
        
        x = shortcut + self.drop_path(x)

        return x

# %%
# ConvNextBlock(8)(torch.randn(3, 8, 2048)).shape
# %%
class Downsample(nn.Module):
    def __init__(self, in_chans, dim) -> None:
        super(Downsample, self).__init__()
        self.norm = LayerNorm(in_chans, eps=1e-6, data_format="channels_first")
        self.conv = nn.Conv1d(in_chans, dim, kernel_size=4, stride=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.conv(x)
        # x = x.permute(0, 2, 1)
        return x
# %%
class ConvNext_T(nn.Module):                    # 定义ConvNext_T类，继承自nn.Module。功能是定义一个卷积神经网络，包含卷积层、归一化层、全连接层、激活函数和全局响应归一化。
    def __init__(self, depths = [1,1,1,1]) -> None:         # 定义初始化函数，包括深度列表
        super(ConvNext_T, self).__init__()

        blocks = []                                         # 定义卷积神经网络块列表
        self.stem = nn.Sequential(                          # stem层
            nn.Conv1d(1, 24, kernel_size=7, stride=4, padding=3),
            LayerNorm(24, eps=1e-6, data_format="channels_first"),
        )
        blocks.append(self.stem)                            # 添加stem层到卷积神经网络块列表

        for _ in range(depths[0]):                           # 添加卷积神经网络块到卷积神经网络块列表
            blocks.append(ConvNextBlock(24))
        
        blocks.append(Downsample(24, 72))                   # 添加下采样层到卷积神经网络块列表
        for _ in range(depths[1]):
            blocks.append(ConvNextBlock(72))
        
        blocks.append(Downsample(72, 216))
        for _ in range(depths[2]):
            blocks.append(ConvNextBlock(216))
        
        blocks.append(Downsample(216, 648))
        for _ in range(depths[3]):
            blocks.append(ConvNextBlock(648))
        

        self.blocks = nn.Sequential(*blocks)                # 定义卷积神经网络块列表

        # 定义分类层
        self.Dropout_1 = nn.Dropout(0.6)
        self.Dropout_2 = nn.Dropout(0.6)
        self.linear_1 = nn.Linear(648*19, 128)
        self.linear_2 = nn.Linear(128, 30)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:             # 定义前向传播函数
        out = self.blocks(x)
        out = out.view(out.size(0), -1)
        out = self.linear_1(out)
        out = self.relu(out)
        out = self.Dropout_1(out)
        out = self.linear_2(out)
        # out = self.Dropout_2(out)

        return out
# %%
ConvNext_T()(torch.randn(1, 1, 2048)).shape                         #torch.randn生成一个随机张量    批大小为1，通道数为1，序列长度为2048的输入数据经过卷积神经网络后的输出形状
# %%