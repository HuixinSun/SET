# Checkpoints

| Model | File | Config |
|-------|------|--------|
| FCOS baseline | `aitod_fcos_r50_baseline_epoch12.pth` | `configs/aitod/fcos_r50_baseline.py` |
| FCOS w/ SET | `aitod_fcos_set_epoch12.pth` | `configs/aitod/fcos_r50_set.py` |

```bash
python tools/test.py configs/aitod/fcos_r50_set.py checkpoints/aitod_fcos_set_epoch12.pth --eval bbox
```
