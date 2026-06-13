<h1 align="center">SET: Spectral Enhancement for Tiny Object Detection</h1>

<p align="center" style="font-size: 1.05em">
  <strong>Huixin Sun, Runqi Wang, Yanjing Li, Linlin Yang, Shaohui Lin, Xianbin Cao, Baochang Zhang</strong><br/>
  <a href="https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.pdf"><img src="https://img.shields.io/badge/CVPR-2025-blue.svg" alt="CVPR 2025"/></a>
</p>

Official PyTorch implementation of [**SET**](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.pdf) (CVPR 2025), built on [MMDetection](https://github.com/open-mmlab/mmdetection) 2.23.

**SET** amplifies the frequency signatures of tiny objects through a heterogeneous, training only architecture with two complementary modules: **HBS** (Hierarchical Background Smoothing) suppresses high frequency background noise, while **API** (Adversarial Perturbation Injection) enhances tiny object feature saliency. On AI-TOD, FCOS with SET improves AP from **12.0** to **14.2** with **zero extra cost at inference**.

<p align="center">
  <img src="assets/figs/motivation_p2.png" width="720"/>
</p>

## Why SET?

- **Smooths cluttered backgrounds** via scale adaptive spectral smoothing (HBS)
- **Sharpens tiny instance features** through adversarial perturbation during training (API)
- **Training only enhancement** with no additional overhead at inference

## What We Provide

We release a complete open source stack for SET research on tiny object detection:

| Component | Description |
|-----------|-------------|
| **SET detector** | FCOS with HBS + API (`mmdet/models/detectors/fcos_set.py`) |
| **AI-TOD configs** | Baseline and SET training / evaluation configs |
| **Visualization tools** | [`run_pca.sh`](run_pca.sh) for HBS (Fig. 4) and [`run_saliency.sh`](run_saliency.sh) for API (Fig. 5) |
| **Pretrained checkpoints** | FCOS baseline and FCOS w/ SET on AI-TOD ([`checkpoints/`](checkpoints/)) |

The two visualization tools directly mirror the two core modules:

| Module | Role | Tool |
|--------|------|------|
| **HBS** | Background smoothing | [`run_pca.sh`](run_pca.sh) — background feature PCA |
| **API** | Feature saliency enhancement | [`run_saliency.sh`](run_saliency.sh) — per instance saliency on original images |

## Results on AI-TOD (Table 1)

ResNet-50, 800×800, 12 epochs, trainval to test.

| Method | AP | AP50 | AP75 | APvt | APt | APs |
|--------|-----|------|------|------|-----|-----|
| FCOS | 12.0 | 29.0 | 8.0 | 2.5 | 11.9 | 17.1 |
| **FCOS w/ SET** | **14.2** | **34.9** | **9.8** | **2.9** | **13.0** | **20.2** |

## Quick Start

```bash
conda create -n set python=3.9 -y && conda activate set
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch

pip install -U openmim && mim install mmcv-full==1.6.0
pip install -v -e .
pip install -r requirements/runtime.txt

cd cocoapi-aitod-master/aitodpycocotools && pip install -v -e .
```

Download [AI-TOD](https://github.com/jwwangchn/AI-TOD) to `data/aitod/`.

```bash
# Train
bash scripts/train.sh configs/aitod/fcos_r50_baseline.py 4 output/fcos_baseline
bash scripts/train.sh configs/aitod/fcos_r50_set.py 4 output/fcos_set

# Eval
bash scripts/eval.sh configs/aitod/fcos_r50_baseline.py checkpoints/aitod_fcos_r50_baseline_epoch12.pth 1
bash scripts/eval.sh configs/aitod/fcos_r50_set.py checkpoints/aitod_fcos_set_epoch12.pth 1
```

## Visualization

Requires `scikit-learn`, `matplotlib`, and `opencv-python`.

```bash
# HBS: background smoothing (Fig. 4)
bash run_pca.sh

# API: feature saliency enhancement (Fig. 5)
bash run_saliency.sh
```

Outputs are saved to `vis/pca/` and `vis/saliency/` as PDF and PNG.

<p align="center">
  <img src="assets/figs/vis_pca.png" width="720"/>
  <br/>
  <em>Fig. 4. Background feature PCA. HBS compresses high frequency background variance.</em>
</p>

<p align="center">
  <img src="assets/figs/vis_saliency.png" width="720"/>
  <br/>
  <em>Fig. 5. Per instance saliency on original images with tiny/small object averages. API enhances saliency over vanilla FCOS.</em>
</p>

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
