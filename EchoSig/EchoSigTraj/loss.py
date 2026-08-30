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
        # Keep the transport plan on the same device as the inputs. Calling
        # .cuda() without a device silently selects the default GPU (usually
        # cuda:0), even when this loss is evaluated on cuda:1.
        if use_cuda:
            pi = pi.to(device=M.device, dtype=M.dtype)
        else:
            pi = pi.to(dtype=M.dtype)
        loss = torch.sum(pi * M)
        return loss
class MMD_loss(nn.Module):
    '''
    https://github.com/ZongxianLee/MMD_Loss.Pytorch/blob/master/mmd_loss.py
    '''
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

class WMMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(WMMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        L2_distance = torch.cdist(total, total, p=2)**2

        del total
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        del L2_distance
        return sum(kernel_val)

    def forward(self, source, target, Ws, Wt=None):
        batch_size = int(source.size()[0])
        if Wt is None:
            Wt = torch.ones(batch_size, 1).to(source.device) / batch_size

        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]

        Ws = Ws.view(batch_size, 1)
        Wt = Wt.view(batch_size, 1)

        Ws_norm = Ws / torch.sum(Ws)
        Wt_norm = Wt / torch.sum(Wt)

        mmd_loss = torch.sum(Ws_norm * Ws_norm.t() * XX) + torch.sum(Wt_norm * Wt_norm.t() * YY) - torch.sum(Ws_norm * Wt_norm.t() * XY) - torch.sum(Wt_norm * Ws_norm.t() * YX)
        
        return mmd_loss
    
