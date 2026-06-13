_base_ = ['./fcos_r50_baseline.py']

model = dict(
    type='FCOS_set',
    set_cfg=dict(
        reg_factor_range=[0, 0.01, 0.1, 0.5, 1, 2, 5],
        reg_factors=[4, 4, 4, 4, 4],
        scale=1.0))
