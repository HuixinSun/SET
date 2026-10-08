# Checkpoints

| Model | File | Config |
|-------|------|--------|
| FCOS baseline | `aitod_fcos_r50_baseline_epoch12.pth` | `configs/aitod/fcos_r50_baseline.py` |
| FCOS w/ SET | [`AI-TOD_FCOS_R50_SET_epoch_12.pth`](https://github.com/HuixinSun/SET/releases/download/model-checkpoints/AI-TOD_FCOS_R50_SET_epoch_12.pth) | `configs/aitod/fcos_r50_set.py` |

```bash
python tools/test.py configs/aitod/fcos_r50_set.py checkpoints/AI-TOD_FCOS_R50_SET_epoch_12.pth --eval bbox
```
