import torch
from torch import nn
import torch.nn.functional as F


# Taken from https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs**2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs**2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


class SpaceToDepth(nn.Module):
    # Expects the following shape: Batch, Channel, Height, Width
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs**2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super().__init__()
        # In Tensorflow DepthwiseConv2D has depth_multiplier instead of kernels_per_layer
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding = 0, kernels_per_layer = 1, stride = 1):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            stride=stride
        )
        self.pointwise = nn.Conv3d(in_channels * kernels_per_layer, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class DoubleDense(nn.Module):
    def __init__(self, in_channels, hidden_neurons, output_channels):
        super().__init__()
        self.dense1 = nn.Linear(in_channels, out_features=hidden_neurons)
        self.dense2 = nn.Linear(in_features=hidden_neurons, out_features=hidden_neurons // 2)
        self.dense3 = nn.Linear(in_features=hidden_neurons // 2, out_features=output_channels)

    def forward(self, x):
        out = F.relu(self.dense1(x.view(x.size(0), -1)))
        out = F.relu(self.dense2(out))
        out = self.dense3(out)
        return out


class DoubleDSConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_ds_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_ds_conv(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super().__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels),
        )

    def forward(self, x):
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out

class CBAM3D(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super(CBAM3D, self).__init__()
        # Channel Attention Module
        self.channel_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool3d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv3d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channels // reduction_ratio, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        
        # Spatial Attention Module
        self.conv_spatial = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.shared_MLP(self.channel_avg_pool(x))
        max_out = self.shared_MLP(self.channel_max_pool(x))
        channel_out = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_out

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.sigmoid_spatial(self.conv_spatial(spatial_out))
        x = x * spatial_out

        return x

class Interpolate(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.size = size
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, size=self.size, mode=self.mode)
        