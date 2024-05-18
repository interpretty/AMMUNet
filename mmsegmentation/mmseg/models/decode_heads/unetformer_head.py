import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.cnn import ConvModule, Scale
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
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = ConvModule(dim, 3 * dim, kernel_size=1, bias=qkv_bias, norm_cfg=None, act_cfg=None)
        self.local1 = ConvModule(dim, dim, kernel_size=3, padding=1, norm_cfg=dict(type='BN'), act_cfg=None)
        self.local2 = ConvModule(dim, dim, kernel_size=1, norm_cfg=dict(type='BN'), act_cfg=None)
        # self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.proj = DepthwiseSeparableConvModule(dim, dim, kernel_size=window_size, stride=1, dilation=1,
                 padding=((1 - 1) + 1 * (window_size - 1)) // 2,
                 dw_norm_cfg=dict(type='BN'), dw_act_cfg=None, pw_act_cfg=None)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            # x = F.pad(x, (0, ps - W % ps), mode='reflect')
            x = F.pad(x, (0, ps - W % ps, 0, 0), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

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
        self.post_conv = ConvModule(decode_channels, decode_channels, kernel_size=3, padding=1, norm_cfg=dict(type='BN'),
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
        # self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)

        self.proj = DepthwiseSeparableConvModule(decode_channels, decode_channels, kernel_size=3, stride=1, dilation=1,
                 padding=((1 - 1) + 1 * (3 - 1)) // 2,
                 dw_norm_cfg=dict(type='BN'), dw_act_cfg=None, pw_act_cfg=None)

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
class UNetFormerHead(BaseDecodeHead):
    def __init__(self,
                 encoder_channels=(256, 512, 1024, 2048),
                 decode_channels=256,
                 dropout=0.1,
                 window_size=8,
                 **kwargs):
        super(UNetFormerHead, self).__init__(**kwargs)

        self.pre_conv = ConvModule(
            encoder_channels[-1],
            decode_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN'))
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        # self.b1 = Block(dim=decode_channels, num_heads=8, window_size=window_size)  # 暂时定为16
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
        x = self.fr(x)
        if ori_img is not None:
            # 实现原始图片信息的传入

            # 实现最终结果为512*512大小
            # x = self.b1(x)

            # 将ori_img进行卷积
            x = self.p0(x, ori_img)
            # 把feature_refinement里的内容往后调整

        # x = self.segmentation_head(x)
        # x = self.fr(x)
        x = self.cls_seg(x)
        return x
