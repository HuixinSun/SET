import math

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from mmdet.models.builder import DETECTORS

from .single_stage import SingleStageDetector


class DNResBlock(nn.Module):
    """HBS: scale-adaptive background smoothing block."""

    def __init__(self, in_c=256, reduction=4, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_c, in_c // reduction, kernel_size, 1, padding,
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(in_c // reduction, in_c, kernel_size, 1, padding,
                      bias=True))

    def forward(self, x):
        return x + self.conv_block(x)


@DETECTORS.register_module()
class FCOS_set(SingleStageDetector):
    """FCOS with SET (HBS + API). Training-only enhancement, zero inference cost."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 set_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained, init_cfg)

        self.reg_factor_range = set_cfg.reg_factor_range
        self.reg_factors = set_cfg.reg_factors
        self.scale = set_cfg.scale

        kernel_sizes = [(int(math.log2(s)) // 2 * 2) + 1
                        for s in bbox_head['strides']]
        self.denoisers = nn.ModuleList([
            DNResBlock(in_c=bbox_head['in_channels'],
                       reduction=4,
                       kernel_size=k) for k in kernel_sizes
        ])

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)

        binary_mask = torch.zeros(
            [img.shape[0], 1, img.shape[2], img.shape[3]]).to(img.device)
        for img_index in range(img.shape[0]):
            for bbox_index in range(gt_bboxes[img_index].shape[0]):
                y1 = gt_bboxes[img_index][bbox_index][1].int()
                y2 = gt_bboxes[img_index][bbox_index][3].int()
                x1 = gt_bboxes[img_index][bbox_index][0].int()
                x2 = gt_bboxes[img_index][bbox_index][2].int()
                binary_mask[img_index, 0, y1:y2, x1:x2] = 1

        x_set = []
        for i, feature in enumerate(x):
            binary_mask_ = interpolate(
                binary_mask,
                size=(feature.shape[2], feature.shape[3]),
                mode='nearest')
            background_mask = 1 - binary_mask_

            fea_grad_cls = torch.autograd.grad(
                losses['loss_cls'],
                feature,
                retain_graph=True,
                allow_unused=True)[0]
            fea_grad_cls = fea_grad_cls / (torch.norm(fea_grad_cls) + 1e-12)

            fea_grad_reg = torch.autograd.grad(
                losses['loss_bbox'],
                feature,
                retain_graph=True,
                allow_unused=True)[0]
            fea_grad_reg = fea_grad_reg / (torch.norm(fea_grad_reg) + 1e-12)

            fea_grad_centerness = torch.autograd.grad(
                losses['loss_centerness'],
                feature,
                retain_graph=True,
                allow_unused=True)[0]
            fea_grad_centerness = fea_grad_centerness / (
                torch.norm(fea_grad_centerness) + 1e-12)

            # HBS: background smoothing
            dn_feature_bg = self.denoisers[i](feature * background_mask)
            dn_feature = dn_feature_bg * background_mask + feature * binary_mask_

            # API: adversarial perturbation
            reg_factor = self.reg_factor_range[self.reg_factors[i]]
            noise = (fea_grad_reg + fea_grad_cls +
                     fea_grad_centerness) / 3.0 * reg_factor
            x_set.append(dn_feature + noise)

        losses_set = self.bbox_head.forward_train(x_set, img_metas, gt_bboxes,
                                                  gt_labels, gt_bboxes_ignore)

        losses['loss_set_cls'] = (
            self.bbox_head.loss_cls.loss_weight * losses_set['loss_cls'] *
            self.scale)
        losses['loss_set_bbox'] = (
            self.bbox_head.loss_bbox.loss_weight * losses_set['loss_bbox'] *
            self.scale)
        losses['loss_set_centerness'] = (
            self.bbox_head.loss_bbox.loss_weight *
            losses_set['loss_centerness'] * self.scale)

        return losses
