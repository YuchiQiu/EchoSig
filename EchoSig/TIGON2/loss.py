import torch
import torch.nn as nn
import numpy as np
import ot

class OT_loss(nn.Module):
    _valid = 'emd sinkhorn sinkhorn_knopp_unbalanced'.split()

    def __init__(self, which='emd', use_cuda=True):
        if which not in self._valid:
            raise ValueError(f'{which} not known ({self._valid})')
        elif which == 'emd':
            self.fn = lambda m, n, M: ot.emd(m, n, M)
        elif which == 'sinkhorn':
            self.fn = lambda m, n, M : ot.sinkhorn(m, n, M, 2.0)
        elif which == 'sinkhorn_knopp_unbalanced':
            self.fn = lambda m, n, M : ot.unbalanced.sinkhorn_knopp_unbalanced(m, n, M, 1.0, 1.0)
        else:
            pass
        self.use_cuda = use_cuda

    def __call__(self, source, target, rho0=None, rho1=None, use_cuda=None):
        if use_cuda is None:
            use_cuda = self.use_cuda
        M = torch.cdist(source, target) ** 2
        # if rho0 is None:
        #     rho0 = torch.ones(source.shape[0])
        # if rho1 is None:
        #     rho1 = torch.ones(target.shape[0])
        if len(rho0.shape)==2:
            rho0=rho0.squeeze(1)
        if len(rho1.shape)==2:
            rho1=rho1.squeeze(1)
        pi = self.fn(rho0, rho1, M.detach().cpu())
        if type(pi) is np.ndarray:
            pi = torch.tensor(pi)
        elif type(pi) is torch.Tensor:
            pi = pi.clone().detach()
        pi = pi.cuda() if use_cuda else pi
        M = M.to(pi.device)
        loss = torch.sum(pi * M)
        return loss

