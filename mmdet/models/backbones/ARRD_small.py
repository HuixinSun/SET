import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
#from torch.nn.functional import interpolate, conv2d
from ..builder import BACKBONES
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule


def conv_block(in_channels, filters, kernel_size, strides, padding , mode='cba'):
    
    conv = nn.Conv2d(in_channels, filters, kernel_size, strides, padding, bias=False)
    bn = nn.BatchNorm2d(filters)
    act = nn.LeakyReLU(0.2)

    if mode == 'cba':  
        return nn.Sequential(conv, bn, act)
    elif mode == 'cb':
        return nn.Sequential(conv, bn)
    elif mode == 'cab':
        return nn.Sequential(conv, act, bn)
    elif mode == 'ca':
        return nn.Sequential(conv, act)
    elif mode == 'c':
        return conv

class Res_block(nn.Module):
    def __init__(self, in_c=48, mid_c=16):
        super(Res_block, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(in_c, mid_c, 3, 1, 1, bias=True), nn.ReLU(),
                                        nn.Conv2d(mid_c, in_c, 3, 1, 1, bias=True))

    def forward(self, x):
        add = x
        x = self.conv_block(x)
        x = x + add
        return x

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=256, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  
        x = self.proj(x)  # 展平空间特征，转换维度，eg:[3,256,128,128] -> [3,16384,256] -> [3,16384,768]
        return x
    
    
@BACKBONES.register_module()
class ARRD_Multi_Level_FPN_SMALL(BaseModule):
    def __init__(self, 
        in_channel=256,
        mid_channel=128,
        scale=0.4,
        scale_in=2,
        scale_pixel=2,
        scale_linear=2,
        #embedding_dim=1024, 
        mode='bilinear',
        input_number=4, # Number of level features use for reconstruction
        init_cfg=dict(type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super(ARRD_Multi_Level_FPN_SMALL, self).__init__(init_cfg)
        '''
        mode (str) – algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' 
        | 'bicubic' | 'trilinear' | 'area'. Default: 'bilinear'
        '''
        self.mode = mode 
        self.scale_in = scale_in
        self.input_number = input_number
        self.scale_pixel = scale_pixel
        self.scale = scale
        self.scale_linear = scale_linear
        ## Level of features to use


        if self.input_number >= 4:
            #self.linear_c4 = MLP(input_dim=in_channel, embed_dim=embedding_dim)d
            self.linear_c4 = MLP(input_dim=in_channel, embed_dim=in_channel*scale_linear*scale_linear)
        if self.input_number >= 3:
            self.linear_c3 = MLP(input_dim=in_channel, embed_dim=in_channel*scale_linear*scale_linear)
        self.linear_c2 = MLP(input_dim=in_channel, embed_dim=in_channel*scale_linear*scale_linear)
        self.linear_c1 = MLP(input_dim=in_channel, embed_dim=in_channel*scale_linear*scale_linear)

        
        self.linear_fuse = ConvModule(
            in_channels=in_channel*input_number,
            out_channels=mid_channel,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)   # 这里有更改，BN
        )
        self.res_block = Res_block(mid_channel)  # C + A + C
        self.conv_final = conv_block(mid_channel, 3*self.scale_pixel*self.scale_pixel, 3, 1, 1, 'c')
        #self.dropout = nn.Dropout2d(0.1)
       
        self.ps = nn.PixelShuffle(self.scale_pixel)
    
    def forward(self, x):
        
        # c1, c2, c3, c4, _ = x
        _, c2, c3, c4, c5 = x
        
        #print(c1.shape, c4.shape)
        n = c3.shape[0]
           
        
        if self.input_number >= 4:
            _c5 = self.linear_c4(c5).permute(0,2,1).reshape(n, -1, c5.shape[2]*self.scale_linear, c5.shape[3]*self.scale_linear)
            _c5 = F.interpolate(_c5, size=(c2.shape[2]*self.scale_linear, c2.shape[3]*self.scale_linear),mode='bilinear',align_corners=False)

        if self.input_number >= 3:
            _c4 = self.linear_c3(c4).permute(0,2,1).reshape(n, -1, c4.shape[2]*self.scale_linear, c4.shape[3]*self.scale_linear)
            _c4 = F.interpolate(_c4, size=(c2.shape[2]*self.scale_linear, c2.shape[3]*self.scale_linear),mode='bilinear',align_corners=False)

        _c3 = self.linear_c2(c3).permute(0,2,1).reshape(n, -1, c3.shape[2]*self.scale_linear, c3.shape[3]*self.scale_linear)
        _c3 = F.interpolate(_c3, size=(c2.shape[2]*self.scale_linear, c2.shape[3]*self.scale_linear),mode='bilinear',align_corners=False)

        _c2 = self.linear_c1(c2).permute(0,2,1).reshape(n, -1, c2.shape[2]*self.scale_linear, c2.shape[3]*self.scale_linear)

        # Linear Fusion
        if self.input_number == 5:
            pass
        elif self.input_number == 4:
            x = self.linear_fuse(torch.cat([_c2, _c3, _c4, _c5], dim=1))
        elif self.input_number == 3:
            x = self.linear_fuse(torch.cat([_c2, _c3, _c4], dim=1))
        else:
            x = self.linear_fuse(torch.cat([_c2, _c3], dim=1))
        
        if self.scale_in == 1:
            res = x
            x = self.res_block(x)
        else:
            res = F.interpolate(x, scale_factor=self.scale_in, mode=self.mode, align_corners=False)    # up-scale
            x = self.res_block(x)
            x = F.interpolate(x, scale_factor=self.scale_in, mode=self.mode, align_corners=False)      # down-scale

        x += res
        x = nn.Dropout2d(0.1)(x)
        
        x_final = self.conv_final(x)
        x_final = self.ps(x_final)

        return x_final