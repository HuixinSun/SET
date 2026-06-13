# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import torchvision.transforms.functional as F
import torch.nn.functional as F2

from mmcv.image import tensor2imgs
import matplotlib.pyplot as plt

# kornia: a GPU-based torch library, here we use it to adopt gaussian blur on tensor
# source paper: 'https://arxiv.org/pdf/1910.02190.pdf'
# from kornia.filters import filter2D
import random
import numpy as np


from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS, build_backbone

from ...core.utils import flip_tensor
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class RetinaNet(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained, init_cfg)
    #     self.feature_grad = 1
    #     self.i = 0
    #     self.box = 0
    #     self.file_name = ''
    #     self.size = 0
    
    # def extract_feat(self, img):
    #     """Directly extract features from the backbone+neck."""
    #     x = self.backbone(img)
    #     if self.with_neck:
    #         x = self.neck(x)

    #     return x

    # def hook(self, grad):
    #     self.feature_grad = grad
    
    # def forward_train(self,
    #                   img,
    #                   img_metas,
    #                   gt_bboxes,
    #                   gt_labels,
    #                   gt_bboxes_ignore=None):
    #     """
    #     Args:
    #         img (Tensor): Input images of shape (N, C, H, W).
    #             Typically these should be mean centered and std scaled.
    #         img_metas (list[dict]): A List of image info dict where each dict
    #             has: 'img_shape', 'scale_factor', 'flip', and may also contain
    #             'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
    #             For details on the values of these keys see
    #             :class:`mmdet.datasets.pipelines.Collect`.
    #         gt_bboxes (list[Tensor]): Each item are the truth boxes for each
    #             image in [tl_x, tl_y, br_x, br_y] format.
    #         gt_labels (list[Tensor]): Class indices corresponding to each box
    #         gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
    #             boxes can be ignored when computing the loss.

    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #     """
    #     super(SingleStageDetector, self).forward_train(img, img_metas)
    #     x = self.extract_feat(img)
    #     losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
    #                                           gt_labels, gt_bboxes_ignore)
    #     x[0].retain_grad()
    #     x[0].register_hook(self.hook)
    #     # print(self.feature_grad)
    #     # 
    #     from matplotlib import pyplot as plt
    #     if self.i != 0:
    #         if self.size > 12 :
    #             plt.imshow(self.feature_grad[0][0].detach().cpu().numpy(), cmap='Blues')
    #             plt.savefig(f'pictures/bad/{self.file_name}',dpi=1000)
    #             for j in range(self.box.shape[0]):
    #                 ax = plt.gca()
    #                 box = self.box[j].cpu().numpy()
    #                 ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='lime', linewidth=1)) # xmax - xmin
    #             plt.savefig(f'pictures/bad_with_bboxes/{self.file_name}',dpi=1000)
    #             plt.clf()
                
    #     self.i = self.i + 1
              
    #     self.size =  gt_bboxes[0].shape[0]
    #     self.file_name = img_metas[0]['ori_filename']
    #     if self.size > 12 :
    #         img_plt = plt.imread(img_metas[0]['filename'])
    #         plt.imshow(img_plt)
    #         for p in range(gt_bboxes[0].shape[0]):
    #             ax = plt.gca()
    #             box = gt_bboxes[0][p].cpu().numpy()
    #             ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='lime', linewidth=1)) # xmax - xmin
            
    #         plt.savefig(f'pictures/img_with_bboxes/{self.file_name}',dpi=1000)
    #         plt.clf()

        
    #     self.box = gt_bboxes[0] / 8
    #     return losses