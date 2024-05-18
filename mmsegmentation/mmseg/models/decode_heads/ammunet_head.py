import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
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
from ..backbones.ammm_gmsa import SwinBlock
from mmcv.cnn.bricks.transformer import FFN


# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
#         self.act = act_layer()
#         self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
#         self.drop = nn.Dropout(drop, inplace=True)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


# class GlobalLocalAttention(nn.Module):
#     def __init__(self,
#                  dim=256,
#                  num_heads=16,
#                  qkv_bias=False,
#                  window_size=8,
#                  relative_pos_embedding=True
#                  ):
#         super().__init__()
#
#         head_embed_dims = dim // num_heads
#         self.attn = ShiftWindowMSA(
#             embed_dims=dim,
#             num_heads=num_heads,
#             window_size=window_size,
#             shift_size=0,
#             qkv_bias=qkv_bias,
#             qk_scale=head_embed_dims ** -0.5,
#             attn_drop_rate=0.,
#             proj_drop_rate=0.,
#             dropout_layer=dict(type='DropPath', drop_prob=0.),
#             init_cfg=None
#         )
#
#         # self.local1 = ConvModule(dim, dim, kernel_size=3, padding=1, norm_cfg=dict(type='BN'), act_cfg=None)
#         # self.local2 = ConvModule(dim, dim, kernel_size=1, norm_cfg=dict(type='BN'), act_cfg=None)
#         # self.proj = DepthwiseSeparableConvModule(dim, dim, kernel_size=window_size, stride=1, dilation=1,
#         #                                          padding=((1 - 1) + 1 * (window_size - 1)) // 2,
#         #                                          dw_norm_cfg=dict(type='BN'), dw_act_cfg=None, pw_act_cfg=None)
#
#         # self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
#         # self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))
#
#     # def pad_out(self, x):
#     #     x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
#     #     return x
#
#     def forward(self, x):
#         # local = self.local2(x) + self.local1(x)
#
#         hw_shape = (x.shape[2], x.shape[3])
#         attn = x.flatten(2).transpose(1, 2)
#         attn = self.attn(attn, hw_shape)
#         attn = attn.transpose(1, 2)
#         out = attn.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
#
#         # out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
#         #       self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))
#
#         # out = out + local
#         # out = self.pad_out(out)
#         # out = self.proj(attn)
#         # print(out.size())
#
#         return out


# class Block(SwinBlock):
#     def __init__(self, dim=256, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)
#
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
#         #                drop=drop)
#         self.ffn = FFN(
#             embed_dims=dim,
#             feedforward_channels=mlp_hidden_dim,
#             num_fcs=2,
#             ffn_drop=0.,
#             dropout_layer=dict(type='DropPath', drop_prob=0.),
#             act_cfg=dict(type='GELU')
#         )
#         self.norm2 = norm_layer(dim)
#
#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.ffn(self.norm2(x)))
#
#         return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, scale_factor=2, eps=1e-8):
        super(WF, self).__init__()
        self.scale_factor = scale_factor
        self.pre_conv = ConvModule(in_channels, decode_channels, kernel_size=1, norm_cfg=None, act_cfg=None)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvModule(2*decode_channels, decode_channels, kernel_size=3, padding=1,
                                    norm_cfg=dict(type='BN'),
                                    act_cfg=dict(type='ReLU'))

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        # weights = nn.ReLU()(self.weights)
        # fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        # x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        # x = self.post_conv(x)
        res = self.pre_conv(res)
        x = torch.cat((x, res), dim=1)
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


class PosBias(nn.Module):
    def __init__(self, window_size=64, num_heads=8):
        super(PosBias, self).__init__()
        self.window_size = (window_size, window_size)
        self.num_heads = num_heads
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        if window_size == 2:
            self.register_buffer('relative_position_index_2', rel_position_index)
        else:
            self.register_buffer('relative_position_index_1', rel_position_index)

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)

    def forward(self, window_size):
        if window_size == 2:
            relative_position_index = self.relative_position_index_2
        else:
            relative_position_index = self.relative_position_index_1
        # 生成相对位置特征
        relative_position_bias = self.relative_position_bias_table[
            relative_position_index.view(-1)].view(
            window_size * window_size,
            window_size * window_size,
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias


@MODELS.register_module()
class AMMUNetHead(BaseDecodeHead):
    def __init__(self,
                 encoder_channels=(256, 512, 1024, 2048),
                 decode_channels=256,
                 dropout=0.1,
                 window_size=(64, 32, 16),
                 **kwargs):
        super(AMMUNetHead, self).__init__(**kwargs)

        self.window_size = window_size
        self.pb4 = PosBias(window_size=window_size[-1], num_heads=8)
        self.pb3 = PosBias(window_size=2, num_heads=8)
        self.pb2 = PosBias(window_size=2, num_heads=8)

        self.pre_conv = ConvModule(
            encoder_channels[-1],
            decode_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN'))
        self.b4 = SwinBlock(embed_dims=decode_channels, num_heads=8, window_size=window_size[-1],
                            feedforward_channels=decode_channels * 4, shift=False)

        self.b3 = SwinBlock(embed_dims=decode_channels, num_heads=8, window_size=window_size[-2],
                            feedforward_channels=decode_channels * 4, shift=False)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = SwinBlock(embed_dims=decode_channels, num_heads=8, window_size=window_size[-3],
                            feedforward_channels=decode_channels * 4, shift=False)
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
        x = self.pre_conv(inputs[-1])
        global_tuple = [self.pb4(window_size=self.window_size[-1])]
        ori_shape = x.size()
        hw_shape = x.size()[-2:]
        x = x.flatten(2).transpose(1, 2)
        x, global_tuple = self.b4(x, hw_shape, global_tuple)
        x = x.transpose(1, 2)
        x = x.reshape(ori_shape)

        x = self.p3(x, inputs[-2])
        global_tuple[0] = self.pb3(window_size=2)
        ori_shape = x.size()
        hw_shape = x.size()[-2:]
        x = x.flatten(2).transpose(1, 2)
        x, global_tuple = self.b3(x, hw_shape, global_tuple)
        x = x.transpose(1, 2)
        x = x.reshape(ori_shape)

        x = self.p2(x, inputs[-3])
        global_tuple[0] = self.pb2(window_size=2)
        ori_shape = x.size()
        hw_shape = x.size()[-2:]
        x = x.flatten(2).transpose(1, 2)
        x, _ = self.b2(x, hw_shape, global_tuple)
        x = x.transpose(1, 2)
        x = x.reshape(ori_shape)

        x = self.p1(x, inputs[-4])
        # x = self.fr(x)

        # x = self.segmentation_head(x)
        # x = self.fr(x)
        x = self.cls_seg(x)
        return x
