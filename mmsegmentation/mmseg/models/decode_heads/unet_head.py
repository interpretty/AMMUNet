import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
import torch.utils.checkpoint as cp
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from ..utils import resize
from .aspp_head import ASPPHead, ASPPModule
from ..utils import UpConvBlock, Upsample


class BasicConvBlock(nn.Module):
    """Basic convolutional block for UNet.

    This module consists of several plain convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers. Default: 2.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolutional layer to downsample the input feature
            map. Options are 1 or 2. Default: 1.
        dilation (int): Whether use dilated convolution to expand the
            receptive field. Set dilation rate of each convolutional layer and
            the dilation rate of the first convolutional layer is always 1.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 dcn=None,
                 plugins=None):
        super().__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.with_cp = with_cp
        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out


@MODELS.register_module()
class UNetHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.decoder = nn.ModuleList()
        self.num_stages = 4
        self.channels = 256
        self.dec_num_convs = (2, 2, 2, 2)
        self.dec_dilations = (1, 1, 1, 1)
        self.with_cp = False
        self.conv_cfg = None
        self.norm_cfg = dict(type='BN')
        self.act_cfg = dict(type='ReLU')
        self.input_transform = 'multiple_select'

        for i in range(self.num_stages):
            if i != 0:
                self.decoder.append(
                    UpConvBlock(
                        conv_block=BasicConvBlock,
                        in_channels=self.channels * 2 ** i,
                        skip_channels=self.channels * 2 ** (i - 1),
                        out_channels=self.channels * 2 ** (i - 1),
                        num_convs=self.dec_num_convs[i - 1],
                        stride=1,
                        dilation=self.dec_dilations[i - 1],
                        with_cp=self.with_cp,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        upsample_cfg=dict(type='InterpConv'),
                        dcn=None,
                        plugins=None))

    def _forward_feature(self, inputs):
        enc_outs = self._transform_inputs(inputs)
        # enc_outs[0] enc_outs[1] enc_outs[2] enc_outs[3]分别是resnet的第1、2、3、4层的输出
        # enc_outs[3]为2048层
        x = enc_outs[3]
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)
        return x

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
