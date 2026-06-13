<h1 align="center">SET: Spectral Enhancement for Tiny Object Detection</h1>

<p align="center" style="font-size: 1.05em">
  Huixin Sun, Runqi Wang, Yanjing Li, Linlin Yang, Shaohui Lin, Xianbin Cao, Baochang Zhang
</p>

Official PyTorch implementation of [SET](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.pdf), built on [MMDetection](https://github.com/open-mmlab/mmdetection) 2.23.

Tiny object detection remains challenging because tiny instances become less distinct after feature encoding and are easily overwhelmed by high frequency background noise. SET amplifies the frequency signatures of tiny objects via a heterogeneous architecture with two modules: HBS suppresses high frequency background noise through adaptive smoothing, and API injects feature level adversarial perturbations to increase saliency during training. SET is applied during training only and introduces no extra cost at inference.

<p align="center" style="margin-top: 1.2em">
  <a href="https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.pdf"><img src="https://img.shields.io/badge/CVPR-2025-blue.svg" alt="CVPR 2025"/></a>
</p>

<p align="center">
  <img src="assets/figs/motivation_p2.png" width="720"/>
</p>

## Why SET?

- Smooths cluttered backgrounds via scale adaptive spectral smoothing (HBS)
- Sharpens tiny instance features through adversarial perturbation during training (API)
- Training only enhancement with no additional overhead at inference

## Results on AI-TOD (Table 1)

ResNet-50, 800×800, 12 epochs, trainval to test.

| Method | AP | AP50 | AP75 | APvt | APt | APs |
|--------|-----|------|------|------|-----|-----|
| FCOS | 12.0 | 29.0 | 8.0 | 2.5 | 11.9 | 17.1 |
| FCOS w/ SET | 14.2 | 34.9 | 9.8 | 2.9 | 13.0 | 20.2 |

Pretrained checkpoints: [`checkpoints/`](checkpoints/)

## Environment

```bash
conda create -n set python=3.9 -y && conda activate set
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch

pip install -U openmim && mim install mmcv-full==1.6.0
pip install -v -e .
pip install -r requirements/runtime.txt

cd cocoapi-aitod-master/aitodpycocotools && pip install -v -e .
```

Download [AI-TOD](https://github.com/jwwangchn/AI-TOD) to `data/aitod/`.

## Training & Evaluation

```bash
# Train
bash scripts/train.sh configs/aitod/fcos_r50_baseline.py 4 output/fcos_baseline
bash scripts/train.sh configs/aitod/fcos_r50_set.py 4 output/fcos_set

# Eval
bash scripts/eval.sh configs/aitod/fcos_r50_baseline.py checkpoints/aitod_fcos_r50_baseline_epoch12.pth 1
bash scripts/eval.sh configs/aitod/fcos_r50_set.py checkpoints/aitod_fcos_set_epoch12.pth 1
```

## Visualization

The two visualization tools directly mirror the two core modules:

| Module | Role | Tool |
|--------|------|------|
| HBS | Background smoothing | [`run_pca.sh`](run_pca.sh) — background feature PCA |
| API | Feature saliency enhancement | [`run_saliency.sh`](run_saliency.sh) — per instance saliency on original images |

Requires `scikit-learn`, `matplotlib`, and `opencv-python`.

```bash
# HBS: background smoothing (Fig. 4)
bash run_pca.sh

# API: feature saliency enhancement (Fig. 5)
bash run_saliency.sh
```

## Citation

If you find SET useful in your research, please cite:

```bibtex
@inproceedings{sun2025set,
  title={SET: Spectral Enhancement for Tiny Object Detection},
  author={Sun, Huixin and Wang, Runqi and Li, Yanjing and Yang, Linlin and Lin, Shaohui and Cao, Xianbin and Zhang, Baochang},
  booktitle={CVPR},
  year={2025}
}
```

## Acknowledgements

[MMDetection](https://github.com/open-mmlab/mmdetection) and [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod).
