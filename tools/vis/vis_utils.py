"""Visualization helpers for SET paper figures (PCA and saliency)."""

import os

import cv2
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from mmcv.parallel import scatter
from sklearn.decomposition import PCA
from torch.nn.functional import interpolate

TINY_MAX = 16 * 16
SMALL_MAX = 32 * 32
MEDIUM_MAX = 96 * 96


def scatter_data(data, device):
    """Move collated data to device (mmcv scatter expects GPU index)."""
    if device.type == 'cuda':
        return scatter(data, [device.index])[0]
    return data


def unwrap_batch(data):
    """Unwrap mmcv DataContainer fields after collate/scatter."""
    def _get(field):
        value = data[field]
        if hasattr(value, 'data'):
            return value.data[0]
        return value[0] if isinstance(value, list) else value

    img = _get('img')
    if img.dim() == 3:
        img = img.unsqueeze(0)
    img_metas = _get('img_metas')
    gt_bboxes = _get('gt_bboxes')
    gt_labels = _get('gt_labels') if 'gt_labels' in data else None
    return img, img_metas, gt_bboxes, gt_labels


def compute_pca(features, n_components=1):
    """Project CxHxW feature to 1 channel PCA map (HxW)."""
    flat_features = features.reshape((features.shape[0], -1)).T
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(flat_features)
    return pca_result.reshape(features.shape[1:3])


def build_binary_mask(gt_bboxes, img_shape, device):
    """Build foreground mask from GT boxes on the input image grid."""
    h, w = img_shape
    binary_mask = torch.zeros((1, 1, h, w), device=device)
    for box in gt_bboxes:
        x1, y1, x2, y2 = box.int()
        binary_mask[0, 0, y1:y2, x1:x2] = 1
    return binary_mask


def apply_hbs(feature, binary_mask, denoiser):
    """Apply HBS on one FPN level."""
    mask = interpolate(
        binary_mask,
        size=(feature.shape[2], feature.shape[3]),
        mode='nearest')
    background_mask = 1 - mask
    dn_bg = denoiser(feature * background_mask)
    return dn_bg * background_mask + feature * mask


def get_denoiser(model, layer_idx):
    """Return HBS block for FPN level if model is FCOS_set."""
    if hasattr(model, 'denoisers'):
        return model.denoisers[layer_idx]
    return None


def compute_multibranch_saliency(feature, losses, loss_keys):
    """L2 norm of averaged cls/bbox/centerness gradients (API saliency map)."""
    grads = []
    for key in loss_keys:
        grad = torch.autograd.grad(
            losses[key],
            feature,
            retain_graph=True,
            allow_unused=True)[0]
        if grad is not None:
            grads.append(grad)
    if not grads:
        raise RuntimeError('No gradients available for saliency.')
    grad_mean = sum(grads) / len(grads)
    if grad_mean.dim() == 4:
        grad_mean = grad_mean[0]
    return grad_mean.norm(dim=0).detach().cpu().numpy()


def build_set_features(model, feats, losses, img, gt_bboxes, adv_scale=1.0):
    """Build API perturbed features for SET saliency visualization."""
    binary_mask = torch.zeros(
        [img.shape[0], 1, img.shape[2], img.shape[3]], device=img.device)
    for img_index in range(img.shape[0]):
        for bbox_index in range(gt_bboxes[img_index].shape[0]):
            y1 = gt_bboxes[img_index][bbox_index][1].int()
            y2 = gt_bboxes[img_index][bbox_index][3].int()
            x1 = gt_bboxes[img_index][bbox_index][0].int()
            x2 = gt_bboxes[img_index][bbox_index][2].int()
            binary_mask[img_index, 0, y1:y2, x1:x2] = 1

    x_set = []
    for i, feature in enumerate(feats):
        binary_mask_ = interpolate(
            binary_mask,
            size=(feature.shape[2], feature.shape[3]),
            mode='nearest')
        background_mask = 1 - binary_mask_

        fea_grad_cls = torch.autograd.grad(
            losses['loss_cls'], feature, retain_graph=True, allow_unused=True)[0]
        fea_grad_cls = fea_grad_cls / (torch.norm(fea_grad_cls) + 1e-12)

        fea_grad_reg = torch.autograd.grad(
            losses['loss_bbox'], feature, retain_graph=True, allow_unused=True)[0]
        fea_grad_reg = fea_grad_reg / (torch.norm(fea_grad_reg) + 1e-12)

        fea_grad_centerness = torch.autograd.grad(
            losses['loss_centerness'], feature, retain_graph=True,
            allow_unused=True)[0]
        fea_grad_centerness = fea_grad_centerness / (
            torch.norm(fea_grad_centerness) + 1e-12)

        dn_feature_bg = model.denoisers[i](feature * background_mask)
        dn_feature = dn_feature_bg * background_mask + feature * binary_mask_

        reg_factor = model.reg_factor_range[model.reg_factors[i]]
        noise = (fea_grad_reg + fea_grad_cls + fea_grad_centerness) / 3.0
        noise = noise * reg_factor * adv_scale
        x_set.append(dn_feature + noise)
    return x_set


def _resize_map(arr, size_wh):
    w, h = size_wh
    return cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def _box_area(box):
    return max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)


def _box_saliency_scores(saliency_map, boxes, img_shape, score_scale=1.0):
    """Per GT box saliency score (sum over box region)."""
    h, w = img_shape
    sal_full = _resize_map(saliency_map, (w, h))
    scores = []
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        patch = sal_full[y1:y2, x1:x2]
        scores.append(float(patch.sum()) * score_scale if patch.size else 0.0)
    return np.asarray(scores), sal_full


def _category_averages(boxes, scores):
    areas = np.array([_box_area(box) for box in boxes])
    stats = {}
    tiny = scores[areas < TINY_MAX]
    small = scores[(areas >= TINY_MAX) & (areas < SMALL_MAX)]
    medium = scores[(areas >= SMALL_MAX) & (areas < MEDIUM_MAX)]
    if tiny.size:
        stats['tiny'] = float(tiny.mean())
    if small.size:
        stats['small'] = float(small.mean())
    if medium.size:
        stats['medium'] = float(medium.mean())
    return stats


def _save_figure(fig, out_path):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    png_path = os.path.splitext(out_path)[0] + '.png'
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return png_path


def save_pca_figure(img_path, boxes, pca_before, pca_after, out_path):
    """Fig. 4: input / FCOS PCA / FCOS w/ HBS PCA, stacked vertically with right colorbar."""
    vmin = min(pca_before.min(), pca_after.min())
    vmax = max(pca_before.max(), pca_after.max())

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3, 1)

    ax0 = fig.add_subplot(gs[0, 0])
    img = plt.imread(img_path)
    ax0.imshow(img)
    for box in boxes:
        x1, y1, x2, y2 = box
        ax0.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='lime', linewidth=2))
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[1, 0])
    im_pca = ax1.imshow(pca_before, cmap='Blues', vmin=vmin, vmax=vmax)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[2, 0])
    ax2.imshow(pca_after, cmap='Blues', vmin=vmin, vmax=vmax)
    ax2.axis('off')

    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im_pca, cax=cbar_ax, orientation='vertical')
    fig.subplots_adjust(left=0.05, right=0.86, top=0.98, bottom=0.02, hspace=0.02)
    return _save_figure(fig, out_path)


def _draw_box_text(ax, x, y, text, fontsize):
    ax.text(
        x, y, text, color='yellow', ha='center', va='center',
        fontsize=fontsize, weight='bold',
        path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])


def _render_saliency_panel(ax, img, boxes, saliency_map, title, score_scale=1.0):
    """Single panel: original image + per box saliency value + category averages."""
    h, w = img.shape[:2]
    scores, _ = _box_saliency_scores(saliency_map, boxes, (h, w), score_scale)
    stats = _category_averages(boxes, scores)

    ax.imshow(img)

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
        ax.add_patch(
            Rectangle((x1, y1), bw, bh, fill=False, edgecolor='lime', linewidth=2))
        fs = max(11, min(18, int(min(bw, bh) * 0.45 + 8)))
        _draw_box_text(ax, x1 + bw / 2, y1 + bh / 2, f'{score:.2f}', fs)

    lines = []
    if 'tiny' in stats:
        lines.append(f'Average Tiny Object Saliency: {stats["tiny"]:.2f}')
    if 'small' in stats:
        lines.append(f'Average Small Object Saliency: {stats["small"]:.2f}')
    if 'medium' in stats:
        lines.append(f'Average Medium Object Saliency: {stats["medium"]:.2f}')
    if lines:
        ax.text(
            0.01, 0.01, '\n'.join(lines), transform=ax.transAxes,
            color='white', fontsize=13, va='bottom', ha='left', weight='bold',
            bbox=dict(facecolor='black', alpha=0.55, pad=5, edgecolor='none'))

    ax.set_title(title, loc='left', fontsize=14, weight='bold', pad=8)
    ax.axis('off')


def save_saliency_figure(img_path, boxes, saliency_base, saliency_set, out_path,
                         set_score_scale=1.0):
    """Fig. 5: original image with per box values, FCOS vs FCOS w/ SET."""
    img = plt.imread(img_path)
    boxes = np.asarray(boxes)

    fig, axes = plt.subplots(2, 1, figsize=(9, 16))
    _render_saliency_panel(axes[0], img, boxes, saliency_base, '(a) Vanilla FCOS')
    _render_saliency_panel(
        axes[1], img, boxes, saliency_set, '(b) FCOS w/ SET (API)',
        score_scale=set_score_scale)
    fig.subplots_adjust(hspace=0.06)
    return _save_figure(fig, out_path)


def save_saliency_grid_figure(img_paths, boxes_list, saliency_base_list,
                              saliency_set_list, out_path, set_score_scale=1.0):
    """Fig. 5 grid: 2 rows (FCOS / SET) x N columns (scenes)."""
    n = len(img_paths)
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 10))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, (img_path, boxes, sal_base, sal_set) in enumerate(
            zip(img_paths, boxes_list, saliency_base_list, saliency_set_list)):
        img = plt.imread(img_path)
        title_top = '(a) Vanilla FCOS' if col == 0 else ''
        title_bot = '(b) FCOS w/ SET (API)' if col == 0 else ''
        _render_saliency_panel(axes[0, col], img, np.asarray(boxes), sal_base, title_top)
        _render_saliency_panel(
            axes[1, col], img, np.asarray(boxes), sal_set, title_bot,
            score_scale=set_score_scale)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    return _save_figure(fig, out_path)
