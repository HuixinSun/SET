import torch
from mmcv.runner import OPTIMIZERS
from torch.optim import Optimizer

import json
import numpy as np

@OPTIMIZERS.register_module()
class SAM(Optimizer):
    def __init__(self, 
                 params, 
                 lr,  
                 momentum, 
                 weight_decay,
                 base_optimizer='SGD', 
                 rho=0.05, 
                 adaptive=False, 
                 **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        if base_optimizer == 'SGD':
            self.base_optimizer =  torch.optim.SGD(self.param_groups, lr=lr,  momentum=momentum, weight_decay=weight_decay)
        self.param_groups = self.base_optimizer.param_groups
        self.epoch = 0
        self.param_info = { "parameters": []}

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)" # Update parameter: w_new = w_old + e(w)

                # Calculate and log the shape, sharpness, e_w_norm of the parameter
                # param_grad_norm = torch.norm(p.grad.data).item()
                # try:
                #     shape = [p.shape[0],p.shape[1],p.shape[2],p.shape[3]]
                # except:
                #     try:
                #         shape = [p.shape[0],p.shape[1]]
                #     except:
                #         shape = 0
                # e_w_norm = torch.norm(e_w).item()
                
                # self.param_info["parameters"].append({
                #     "epoch": self.epoch,
                #     "shape": shape,
                #     "sharpness": param_grad_norm,
                #     'e_w_norm': e_w_norm
                # })
              
        if zero_grad: self.zero_grad()
    

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                
        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()
   
    
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
   
    def log_params(self, epoch):
        print('Saving per epoch sharpness...')
        self.epoch = epoch
                
        with open("results/test_parameter_info.json", 'a') as json_file:
            json.dump(self.param_info, json_file, indent=4)
        self.param_info = { "parameters": []}