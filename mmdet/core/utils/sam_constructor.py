from mmcv.utils import build_from_cfg

from mmcv.runner.optimizer import OPTIMIZER_BUILDERS, OPTIMIZERS, DefaultOptimizerConstructor
from mmdet.utils import get_root_logger
from .sam import SAM


@OPTIMIZER_BUILDERS.register_module()
class SAM_Constructor(DefaultOptimizerConstructor):

        
    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            optimizer_cfg['params'] = model.parameters()
            return build_from_cfg(optimizer_cfg, OPTIMIZERS)

        # set param-wise lr and weight decay recursively
        params: List[Dict] = []
        self.add_params(params, model)
        optimizer_cfg['params'] = params

        return build_from_cfg(optimizer_cfg, OPTIMIZERS)
    
        # return my_optimizer
