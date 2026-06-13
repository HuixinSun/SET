import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from mmdet.models.builder import NECKS

@NECKS.register_module()
class CTNeck_FPN_ADDR(BaseModule):
    """
    as centernet-FPN neck, we want three feature maps
    """

    def __init__(self,
                 in_channel,
                 use_dcn=True,
                 init_cfg=None):
        super(CTNeck_SR_WMZ, self).__init__(init_cfg)
        self.fp16_enabled = False
        self.use_dcn = use_dcn
        self.in_channel = in_channel    #512+q
        self.deconv1 = self._make_single_deconv_layer(self.in_channel, 256)
        self.deconv2 = self._make_single_deconv_layer(512, 128)
        self.deconv3 = self._make_single_deconv_layer(256, 64)

    def _make_single_deconv_layer(self, in_channel, feat_channel):
        layers = []
        conv_module = ConvModule(
            in_channel,
            feat_channel,
            3,
            padding=1,
            conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
            norm_cfg=dict(type='BN'))
        layers.append(conv_module)
        upsample_module = ConvModule(
            feat_channel,
            feat_channel,
            4,              # same to the original centernet
            stride=2,
            padding=1,
            conv_cfg=dict(type='deconv'),
            norm_cfg=dict(type='BN'))
        layers.append(upsample_module)

        return nn.Sequential(*layers)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = \
                            (1 - math.fabs(i / f - c)) * (
                                    1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # self.use_dcn is False
            elif not self.use_dcn and isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    @auto_fp16()
    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple))
        outs = self.deconv1(inputs[2])
        outs1 = torch.cat((outs, inputs[1]), 1)
        outs = self.deconv2(outs1)
        outs2 = torch.cat((outs, inputs[0]), 1)
        det_outs = self.deconv3(outs2)
        return det_outs, outs2, outs1,
    
    
    
@NECKS.register_module()
class CTNeck_FPN(BaseModule):
    """
    as centernet-FPN neck, we want three feature maps
    """

    def __init__(self,
                 in_channel,
                 use_dcn=True,
                 init_cfg=None):
        super(CTNeck_FPN, self).__init__(init_cfg)
        self.fp16_enabled = False
        self.use_dcn = use_dcn
        self.in_channel = in_channel    #512+q
        self.deconv1 = self._make_single_deconv_layer(self.in_channel, 256)
        self.deconv2 = self._make_single_deconv_layer(512, 128)
        self.deconv3 = self._make_single_deconv_layer(256, 64)

    def _make_single_deconv_layer(self, in_channel, feat_channel):
        layers = []
        conv_module = ConvModule(
            in_channel,
            feat_channel,
            3,
            padding=1,
            conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
            norm_cfg=dict(type='BN'))
        layers.append(conv_module)
        upsample_module = ConvModule(
            feat_channel,
            feat_channel,
            4,              # same to the original centernet
            stride=2,
            padding=1,
            conv_cfg=dict(type='deconv'),
            norm_cfg=dict(type='BN'))
        layers.append(upsample_module)

        return nn.Sequential(*layers)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = \
                            (1 - math.fabs(i / f - c)) * (
                                    1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # self.use_dcn is False
            elif not self.use_dcn and isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    @auto_fp16()
    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple))
        outs = self.deconv1(inputs[2])
        outs1 = torch.cat((outs, inputs[1]), 1)
        outs = self.deconv2(outs1)
        outs2 = torch.cat((outs, inputs[0]), 1)
        det_outs = self.deconv3(outs2)
        print(det_outs.size())
        return det_outs,   