import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ptwt

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet8_1D(nn.Module):

    def __init__(self, block, layers, num_classes=30):
        super(ResNet8_1D, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    

class CWT(nn.Module):
    def __init__(self, sig_size = 2048, widths = np.arange(1, 65), func='mexh'):
        super().__init__()
        self.sig_size = sig_size
        self.widths = widths
        self.func = func
        self.cwt = ptwt.cwt
    def forward(self, x):
        x = torch.squeeze(x, 1)
        x, freqs = self.cwt(x, self.widths, self.func)
        x = x.transpose(0, 1).float()
        return x

class SignalEncoder(nn.Module):
    def __init__(self, spectra_size=2048, patch_size=128, in_c=1, embed_dim=32, norm_layer=None):
        super(SignalEncoder, self).__init__()
        self.spectra_size = spectra_size
        self.patch_size = patch_size
        self.num_patches = (self.spectra_size // self.patch_size) 
        self.proj = nn.Conv1d(in_c, embed_dim*2, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim*2) if norm_layer else nn.Identity()
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)

    def forward(self, x):
        B, C, L = x.shape
        assert L == self.spectra_size, \
            f"Input image size ({L}) doesn't match model ({self.spectra_size})."
        x = self.proj(x).transpose(1, 2)
        x = self.norm(x)
        x = self.maxpool(x)
        return x

class ImageEncoder(nn.Module):
    def __init__(self, spectra_size=2048, patch_size=128, in_c=64, embed_dim=32, norm_layer=None):
        super(ImageEncoder, self).__init__()
        self.spectra_size = spectra_size
        self.patch_size = patch_size
        self.num_patches = (self.spectra_size // self.patch_size) 
        self.proj = nn.Conv1d(in_c, embed_dim*2, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim*2) if norm_layer else nn.Identity()
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)

    def forward(self, x):
        B, C, L = x.shape
        assert L == self.spectra_size, \
            f"Input image size ({L}) doesn't match model ({self.spectra_size})."
        x = self.proj(x).transpose(1, 2)
        x = self.norm(x)
        x = self.maxpool(x)
        return x

class SemanticAttentionModule2(nn.Module):
    def __init__(self, global_dim, local_dim, num_heads=8):
        super(SemanticAttentionModule2, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.num_heads = num_heads
        
        # Attention layers
        self.global_attention = nn.MultiheadAttention(self.global_dim, num_heads)
        self.cross_attention = nn.MultiheadAttention(self.global_dim, num_heads)
        
        # Conditionally adjust local dimensions to match global dimensions
        if self.global_dim != self.local_dim:
            self.adjust_local_dim = nn.Linear(self.local_dim, self.global_dim)
        else:
            self.adjust_local_dim = None
        
        # Layer Norms
        self.norm1 = nn.LayerNorm(self.global_dim)
        self.norm2 = nn.LayerNorm(self.global_dim)
        
        # Feedforward layers
        self.feedforward = nn.Sequential(
            nn.Linear(self.global_dim * 2, self.global_dim * 2),
            nn.ReLU(),
            nn.Linear(self.global_dim * 2, self.global_dim + self.local_dim)
        )

    def forward(self, global_features, local_features):
        # Ensure correct shape for MultiheadAttention (T, B, C)
        global_features = global_features.permute(1, 0, 2)
        local_features = local_features.permute(1, 0, 2)

        # Adjust local features dimension if necessary
        if self.adjust_local_dim:
            adjusted_local_features = self.adjust_local_dim(local_features)
        else:
            adjusted_local_features = local_features

        # Self-attention on global features
        global_self_attn, _ = self.global_attention(global_features, global_features, global_features)
        global_self_attn = self.norm1(global_self_attn + global_features)
        
        # Cross-attention, global on adjusted local
        global_cross_attn, _ = self.cross_attention(adjusted_local_features, global_features, global_features)
        global_cross_attn = self.norm2(global_cross_attn + adjusted_local_features)
        
        # Concatenate both Attention results
        concatenated_features = torch.cat((global_self_attn, global_cross_attn), dim=2)
        
        # Process through feedforward network
        output = self.feedforward(concatenated_features)
        
        # Reshape output back to (B, T, C)
        output = output.permute(1, 0, 2)
        return output


class CWT_fusion(nn.Module):
    def __init__(self, num_classes, spectra_size, patch_size, global_dim, local_dim, num_heads=8):
        super(CWT_fusion, self).__init__()
        # Initialize the feature extractor
        self.cwt = CWT()
        self.singal_extractor = SignalEncoder()
        self.image_extractor = ImageEncoder()
        # Assuming global_dim and local_dim can be deduced or are fixed. Adjust accordingly.
        self.attention_module = SemanticAttentionModule2(global_dim, local_dim, num_heads)
        self.resnet8_1d = ResNet8_1D(BasicBlock1D, [1, 1], num_classes=num_classes)
        # Redefine the final classifier to match the dimensions after the attention module
        # self.fc = nn.Linear((global_dim + local_dim) * spectra_size // patch_size, num_classes)  # Adjust dimensions if needed

    def forward(self, x):
        # Feature extraction
        signal_input = x
        image_input = self.cwt(x)
        singal_features = self.singal_extractor(signal_input)
        image_fetures = self.image_extractor(image_input)
        # Here we use extracted features as both local and global inputs to the attention module
        # In practice, you might want to differentiate how local and global features are prepared
        enhanced_features = self.attention_module(singal_features, image_fetures)
        # Final classification
        enhanced_features = torch.flatten(enhanced_features, start_dim=1)
        enhanced_features = torch.unsqueeze(enhanced_features, 1)  # (batch_size, 1, 1024)
        output = self.resnet8_1d(enhanced_features)
        return output

def fusion_model(num_classes=30, spectra_size=2048, patch_size=128, global_dim=32, local_dim=32):
    model = CWT_fusion(num_classes=num_classes, spectra_size=spectra_size, patch_size=patch_size, global_dim=global_dim, local_dim=local_dim)
    return model

# signal_input = torch.randn(10, 1, 2048).cuda()  # batch_size=10, channels=1, length=1000
# image_input = torch.randn(10, 64, 2048).cuda()  # batch_size=10, channels=64, length=1000
# input = torch.randn(10, 1, 2048)
# model = CWT_fusion(num_classes=30, spectra_size=2048, patch_size=128, global_dim=32, local_dim=32)
# fused_output = model(input)
# print(fused_output.shape)  # 输出融合后特征的尺寸
