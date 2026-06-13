"""Plot feature saliency for FCOS vs FCOS w/ SET (paper Fig. 5)."""

import argparse
import os
import os.path as osp
import sys

import mmcv
import torch
from mmcv.parallel import collate
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from vis_utils import (build_set_features, compute_multibranch_saliency,
                       save_saliency_figure, save_saliency_grid_figure,
                       scatter_data, unwrap_batch)


LOSS_KEYS = ['loss_cls', 'loss_bbox', 'loss_centerness']
ADV_SCALE = 3.0  # API perturbation scale for SET saliency (visualization only)


def parse_args():
    parser = argparse.ArgumentParser(description='Plot API saliency visualization')
    parser.add_argument('baseline_config', help='FCOS baseline config')
    parser.add_argument('baseline_checkpoint', help='FCOS baseline checkpoint')
    parser.add_argument('set_config', help='FCOS w/ SET config')
    parser.add_argument('set_checkpoint', help='FCOS w/ SET checkpoint')
    parser.add_argument('--layer', type=int, default=0, help='FPN level index (0=P3)')
    parser.add_argument('--index', type=int, default=0, help='Sample index in train set')
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                        help='Multiple sample indices for 2xN paper grid')
    parser.add_argument('--out-dir', default='vis/saliency', help='Output directory')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def prepare_sample(cfg, index):
    dataset = build_dataset(cfg.data.train)
    data = dataset[index]
    return collate([data], samples_per_gpu=1)


def forward_saliency_baseline(model, data, device, layer):
    data = scatter_data(data, device)
    img, img_metas, gt_bboxes, gt_labels = unwrap_batch(data)
    if isinstance(img_metas, dict):
        img_metas = [img_metas]
    if isinstance(gt_bboxes, torch.Tensor):
        gt_bboxes = [gt_bboxes]
    if isinstance(gt_labels, torch.Tensor):
        gt_labels = [gt_labels]

    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    feats = model.extract_feat(img)
    feature = feats[layer]
    feature.retain_grad()

    losses = model.bbox_head.forward_train(
        feats, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None)
    saliency = compute_multibranch_saliency(feature, losses, LOSS_KEYS)
    return img_metas[0], gt_bboxes[0], saliency


def forward_saliency_set(model, data, device, layer, adv_scale):
    data = scatter_data(data, device)
    img, img_metas, gt_bboxes, gt_labels = unwrap_batch(data)
    if isinstance(img_metas, dict):
        img_metas = [img_metas]
    if isinstance(gt_bboxes, torch.Tensor):
        gt_bboxes = [gt_bboxes]
    if isinstance(gt_labels, torch.Tensor):
        gt_labels = [gt_labels]

    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    feats = model.extract_feat(img)
    losses = model.bbox_head.forward_train(
        feats, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None)

    x_set = build_set_features(
        model, feats, losses, img, gt_bboxes, adv_scale=adv_scale)
    feature = x_set[layer]
    feature.retain_grad()

    losses_set = model.bbox_head.forward_train(
        x_set, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None)
    saliency = compute_multibranch_saliency(feature, losses_set, LOSS_KEYS)
    return img_metas[0], gt_bboxes[0], saliency


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    cfg_base = mmcv.Config.fromfile(args.baseline_config)
    cfg_set = mmcv.Config.fromfile(args.set_config)

    model_base = build_detector(cfg_base.model)
    load_checkpoint(model_base, args.baseline_checkpoint, map_location='cpu')
    model_base.to(device)

    model_set = build_detector(cfg_set.model)
    load_checkpoint(model_set, args.set_checkpoint, map_location='cpu')
    model_set.to(device)

    indices = args.indices if args.indices else [args.index]
    img_paths, boxes_list, sal_base_list, sal_set_list = [], [], [], []

    for idx in indices:
        data = prepare_sample(cfg_set, idx)
        meta, boxes, sal_base = forward_saliency_baseline(
            model_base, data, device, args.layer)
        _, _, sal_set = forward_saliency_set(
            model_set, data, device, args.layer, ADV_SCALE)
        img_paths.append(meta['filename'])
        boxes_list.append(boxes.cpu().numpy())
        sal_base_list.append(sal_base)
        sal_set_list.append(sal_set)

    set_score_scale = ADV_SCALE
    if len(indices) == 1:
        out_path = osp.join(args.out_dir, f'sample{indices[0]}_layer{args.layer}_saliency.pdf')
        png_path = save_saliency_figure(
            img_paths[0], boxes_list[0], sal_base_list[0], sal_set_list[0],
            out_path, set_score_scale=set_score_scale)
    else:
        tag = '_'.join(str(i) for i in indices)
        out_path = osp.join(args.out_dir, f'samples{tag}_layer{args.layer}_saliency.pdf')
        png_path = save_saliency_grid_figure(
            img_paths, boxes_list, sal_base_list, sal_set_list, out_path,
            set_score_scale=set_score_scale)

    print(f'Saved saliency figure to {out_path}')
    print(f'Saved saliency preview to {png_path}')


if __name__ == '__main__':
    main()
