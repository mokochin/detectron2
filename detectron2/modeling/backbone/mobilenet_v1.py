import logging

import torch.nn as nn
import torch.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.plugins import GeneralizedAttention
from mmdet.ops import ContextBlock, DeformConv, ModulatedDeformConv
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer

class BasicBlock(nn.Module):

    def __init__(self,
                 inplanes, #输入通道
                 planes,   #输出通道
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),):
        super(BasicBlock, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, inplanes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        #修改成mbnet的dw_conv
        self.conv1 = build_conv_layer( #depthwise_conv
            conv_cfg,
            inplanes,
            inplanes,
            3,
            stride=stride,
            padding=1,
            bias=False, #后面是bn，不需要bias
            groups=inplanes
            )
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(   #point_conv
            conv_cfg, inplanes, planes, 1, stride=1, padding=0, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name) #返回一个对象的属性值 这里相当于一个norm层

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        return out


def make_mb_layer(block, #block类型 这个函数可以把多个block合成一个stage
                   inplanes,
                   planes_stage_index, #tuple类型
                   num_blocks, #第几个stage的block数目
                   strides_stage_index,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   ):

    layers=[]
    for i in range(0, num_blocks):     #每个stage里面block的数目，这里遍历每个block
        planes=inplanes*planes_stage_index[i]
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=strides_stage_index[i],
                dilation=dilation,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                ))
        inplanes=planes

    return nn.Sequential(*layers),inplanes #这是一个stage的所有layer

@BACKBONES.register_module
class MbNet_V1(nn.Module):

    arch_settings = {
        28: (BasicBlock, (3,2,6,2))
    }

    stride_settings={
        0: (1,2,1),
        1: (2,1),
        2: (2,1,1,1,1,1),
        3: (2,1,)
    }

    plane_settings={
        0: (2,2,1),
        1: (2,1),
        2: (2,1,1,1,1,1),
        3: (2,1)
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 strides_index=stride_settings,
                 planes_index=plane_settings,
                 dilations=1,
                 out_indices=(0, 1, 2, 3), #第几个stage
                 style='pytorch',
                 frozen_stages=-1, #不冻结
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 with_cp=False,
                 zero_init_residual=True):
        super(MbNet_V1, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for mbnet'.format(depth))
        self.depth = depth
        self.num_stages = num_stages
        self.strides_index = strides_index
        self.dilations = dilations
        self.plane_index = planes_index
        self.out_indices = out_indices
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 32

        self._make_stem_layer()

        self.mb_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            strides_stage_index = strides_index[i] #这里strides_index，plane_index是tuple
            planes_stage_index = planes_index[i]
            dilation = 1
            mb_layer, planes_scale = make_mb_layer(
                self.block,
                self.inplanes,
                planes_stage_index, #tuple
                num_blocks, #这个stage里面block的总数
                strides_stage_index,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                )
            self.inplanes=planes_scale
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, mb_layer)
            self.mb_layers.append(layer_name)

        self._freeze_stages()

        # self.feat_dim = self.block.expansion * 64 * 2**(
        #     len(self.stage_blocks) - 1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            3,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 32, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            #if self.zero_init_residual:
                #for m in self.modules():
                    #if isinstance(m, BottleBlock):
                    #    constant_init(m.norm3, 0)
                    #elif isinstance(m, BasicBlock):
                    #constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        outs = []
        for i, layer_name in enumerate(self.mb_layers):
            mb_layer = getattr(self, layer_name)
            x = mb_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super(MbNet_V1, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()