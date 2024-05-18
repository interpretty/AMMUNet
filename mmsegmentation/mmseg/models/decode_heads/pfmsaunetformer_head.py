import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.cnn import ConvModule, Scale
from mmengine.utils import to_2tuple
import torch.utils.checkpoint as cp
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from ..utils import resize
from .aspp_head import ASPPHead, ASPPModule
from ..utils import UpConvBlock, Upsample
from mmcv.cnn.bricks import DropPath
from mmengine.model.weight_init import trunc_normal_
import torch.nn.functional as F
from einops import rearrange
from ..backbones.swin import ShiftWindowMSA


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()

        head_embed_dims = dim // num_heads
        self.attn = ShiftWindowMSA(
            embed_dims=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            qkv_bias=qkv_bias,
            qk_scale=head_embed_dims ** -0.5,
            attn_drop_rate=0.,
            proj_drop_rate=0.,
            dropout_layer=dict(type='DropPath', drop_prob=0.),
            init_cfg=None
        )

        self.local1 = ConvModule(dim, dim, kernel_size=3, padding=1, norm_cfg=dict(type='BN'), act_cfg=None)
        self.local2 = ConvModule(dim, dim, kernel_size=1, norm_cfg=dict(type='BN'), act_cfg=None)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        local = self.local2(x) + self.local1(x)

        hw_shape = (x.shape[2], x.shape[3])
        attn = x.flatten(2).transpose(1, 2)
        attn = self.attn(attn, hw_shape)
        attn = attn.transpose(1, 2)
        attn = attn.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())

        return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, scale_factor=2, eps=1e-8):
        super(WF, self).__init__()
        self.scale_factor = scale_factor
        self.pre_conv = ConvModule(in_channels, decode_channels, kernel_size=1, norm_cfg=None, act_cfg=None)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvModule(decode_channels, decode_channels, kernel_size=3, padding=1,
                                    norm_cfg=dict(type='BN'),
                                    act_cfg=dict(type='ReLU'))

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


# 和WF一起用的FeatureRefinementHead实现
class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
            nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                ConvModule(decode_channels, decode_channels // 16, kernel_size=1, norm_cfg=None,
                                           act_cfg=None),
                                nn.ReLU6(),
                                ConvModule(decode_channels // 16, decode_channels, kernel_size=1, norm_cfg=None,
                                           act_cfg=None),
                                nn.Sigmoid())

        self.shortcut = ConvModule(decode_channels, decode_channels, kernel_size=1, norm_cfg=dict(type='BN'))
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x):
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)
        return x


# # FeatureRefinementHead原始实现
# class FeatureRefinementHead(nn.Module):
#     def __init__(self, in_channels=64, decode_channels=64):
#         super().__init__()
#         self.pre_conv = ConvModule(in_channels, decode_channels, kernel_size=1, norm_cfg=None, act_cfg=None)
#
#         self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         self.eps = 1e-8
#         self.post_conv = ConvModule(decode_channels, decode_channels, kernel_size=3, padding=1, norm_cfg=dict(type='BN'),
#                                     act_cfg=dict(type='ReLU'))
#
#         self.pa = nn.Sequential(
#             nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
#             nn.Sigmoid())
#         self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
#                                 ConvModule(decode_channels, decode_channels // 16, kernel_size=1, norm_cfg=None,
#                                            act_cfg=None),
#                                 nn.ReLU6(),
#                                 ConvModule(decode_channels // 16, decode_channels, kernel_size=1, norm_cfg=None,
#                                            act_cfg=None),
#                                 nn.Sigmoid())
#
#         self.shortcut = ConvModule(decode_channels, decode_channels, kernel_size=1, norm_cfg=dict(type='BN'))
#         self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
#         self.act = nn.ReLU6()
#
#     def forward(self, x, res):
#         x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
#         weights = nn.ReLU()(self.weights)
#         fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
#         x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
#         x = self.post_conv(x)
#         shortcut = self.shortcut(x)
#         pa = self.pa(x) * x
#         ca = self.ca(x) * x
#         x = pa + ca
#         x = self.proj(x) + shortcut
#         x = self.act(x)
#
#         return x


@MODELS.register_module()
class PfMSAUNetFormerHead(BaseDecodeHead):
    def __init__(self,
                 # encoder_channels=(64, 128, 256, 512),
                 # decode_channels=64,
                 encoder_channels=(256, 512, 1024, 2048),
                 decode_channels=256,
                 dropout=0.1,
                 window_size=8,
                 **kwargs):
        super(PfMSAUNetFormerHead, self).__init__(**kwargs)

        self.pre_conv = ConvModule(
            encoder_channels[-1],
            decode_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN'))
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=16)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=32)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=64)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        # self.b1 = Block(dim=decode_channels, num_heads=8, window_size=16)  # 暂时定为16
        self.p1 = WF(encoder_channels[-4], decode_channels)

        # for ori_img
        self.p0 = WF(in_channels=3, decode_channels=decode_channels, scale_factor=4)
        self.fr = FeatureRefinementHead(decode_channels, decode_channels)

        # self.segmentation_head = nn.Sequential(
        #     ConvModule(decode_channels, decode_channels, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
        #     nn.Dropout2d(p=dropout, inplace=True),
        #     ConvModule(decode_channels, self.num_classes, kernel_size=1))

    def forward(self, inputs):
        # 实现原始图片信息传入的判断
        if len(inputs) == 2:
            ori_img = inputs[1]
            inputs = inputs[0]
        else:
            ori_img = None

        x = self.b4(self.pre_conv(inputs[-1]))
        h4 = x

        x = self.p3(x, inputs[-2])
        x = self.b3(x)
        h3 = x

        x = self.p2(x, inputs[-3])
        x = self.b2(x)
        h2 = x

        x = self.p1(x, inputs[-4])

        if ori_img is not None:
            # 实现原始图片信息的传入

            # 实现最终结果为512*512大小
            # x = self.b1(x)

            # 将ori_img进行卷积
            x = self.p0(x, ori_img)
            # 把feature_refinement里的内容往后调整

        # x = self.segmentation_head(x)
        x = self.fr(x)
        x = self.cls_seg(x)
        return x
