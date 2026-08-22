import torch
import torch.nn as nn
from typing import Iterable, Tuple

def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power (orthogonalization) of a matrix G.
    Returns G(G^T G)^{-1/2}.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.to(torch.float32)
    # Scale X to have spectral norm <= 1
    X = X / (X.norm() + 1e-7)
    if G.size(0) > G.size(1):
        X = X.T
    
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
        
    if G.size(0) > G.size(1):
        X = X.T
        
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    """
    Muon optimizer (MomentUm Orthogonalized by Newton-Schulz).
    Applies Newton-Schulz orthogonalization to momentum of 2D parameters.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            wd = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(p.grad)
                
                if p.ndim == 2:
                    # Apply Muon for 2D parameters
                    update = zeropower_via_newtonschulz5(buf)
                    # Scale update to match the scale of the original gradients roughly
                    update *= max(1, p.size(0) / p.size(1))**0.5
                else:
                    # Fallback to standard SGD with momentum for 1D/non-2D params
                    update = buf
                
                if wd > 0:
                    p.mul_(1 - lr * wd)
                
                p.add_(update, alpha=-lr)
