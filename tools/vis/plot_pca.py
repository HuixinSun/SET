"""Plot background PCA maps for FCOS vs FCOS w/ SET (paper Fig. 4)."""

import argparse
import os
import os.path as osp
import sys

import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from vis_utils import (apply_hbs, build_binary_mask, compute_pca,
                       get_denoiser, save_pca_figure, scatter_data,
                       unwrap_batch)


def parse_args():
    parser = argparse.ArgumentParser(description='Plot HBS PCA visualization')
    parser.add_argument('baseline_config', help='FCOS baseline config')
    parser.add_argument('baseline_checkpoint', help='FCOS baseline checkpoint')
    parser.add_argument('set_config', help='FCOS w/ SET config')
    parser.add_argument('set_checkpoint', help='FCOS w/ SET checkpoint')
    parser.add_argument('--layer', type=int, default=0, help='FPN level index (0=P3)')
    parser.add_argument('--index', type=int, default=0, help='Sample index in train set')
    parser.add_argument('--out-dir', default='vis/pca', help='Output directory')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def prepare_sample(cfg, index):
    dataset = build_dataset(cfg.data.train)
    data = dataset[index]
    return collate([data], samples_per_gpu=1)


def forward_features(model, data, device):
    data = scatter_data(data, device)
    img, img_metas, gt_bboxes, _ = unwrap_batch(data)

    model.train()
    img.requires_grad_(False)
    feats = model.extract_feat(img)
    return img, img_metas, gt_bboxes, feats


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    cfg_base = mmcv.Config.fromfile(args.baseline_config)
    cfg_set = mmcv.Config.fromfile(args.set_config)

    model_base = build_detector(cfg_base.model, test_cfg=cfg_base.get('test_cfg'))
    load_checkpoint(model_base, args.baseline_checkpoint, map_location='cpu')
    model_base.to(device)

    model_set = build_detector(cfg_set.model, test_cfg=cfg_set.get('test_cfg'))
    load_checkpoint(model_set, args.set_checkpoint, map_location='cpu')
    model_set.to(device)

    data = prepare_sample(cfg_set, args.index)
    _, meta_base, boxes, feats_base = forward_features(model_base, data, device)
    _, _, _, feats_set = forward_features(model_set, data, device)

    layer = args.layer
    feat_base = feats_base[layer][0].detach().cpu().numpy()
    feat_set = feats_set[layer][0]

    binary_mask = build_binary_mask(boxes, (meta_base['img_shape'][0], meta_base['img_shape'][1]), device)
    denoiser = get_denoiser(model_set, layer)
    if denoiser is None:
        raise RuntimeError('SET model has no HBS denoisers; check set checkpoint/config.')

    with torch.no_grad():
        feat_hbs = apply_hbs(feats_set[layer], binary_mask, denoiser)[0].cpu().numpy()

    pca_before = compute_pca(feat_base)
    pca_after = compute_pca(feat_hbs)
    boxes_np = boxes.cpu().numpy()

    out_path = osp.join(args.out_dir, f'sample{args.index}_layer{layer}_pca.pdf')
    png_path = save_pca_figure(meta_base['filename'], boxes_np, pca_before, pca_after, out_path)
    print(f'Saved PCA figure to {out_path}')
    print(f'Saved PCA preview to {png_path}')


if __name__ == '__main__':
    main()
