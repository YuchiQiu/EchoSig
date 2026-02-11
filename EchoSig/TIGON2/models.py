import torch
torch.cuda.empty_cache()
import torch.nn as nn
import numpy as np
from TorchDiffEqPack import odesolve
from EchoSig.TIGON2.utility import MultimodalGaussian_density, Sampling, trans_loss
import sys
import os
# import matplotlib.pyplot as plt
import scipy.io as sio
import random
from torchdiffeq import odeint
from functools import partial
import getpass
# from mpl_toolkits import mplot3d
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.patches import FancyArrowPatch
# from mpl_toolkits.mplot3d import proj3d
# import seaborn as sns
import ot
import time
import warnings
# import umap
#from geomloss import SamplesLoss
import ot
from EchoSig.utility import create_activation
# from src.TIGON.utility import MultimodalGaussian_density, Sampling, trans_loss,diffeq_args
warnings.filterwarnings("ignore")



class UOT(nn.Module):
    def __init__(self, in_out_dim, dim_hiddens, n_layers, activation, odesolver,):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.dim_hiddens = dim_hiddens
        self.hyper_net1 = HyperNetwork1(in_out_dim, dim_hiddens, n_layers,activation) #v= dx/dt
        self.hyper_net2 = HyperNetwork2(in_out_dim, dim_hiddens, activation) #g
        self.apply(initialize_weights)
        self.odesolver = odesolver
    
    def forward(self, t, states):
        z = states[0]
        g_z = states[1]
        logp_z = states[2]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            dz_dt = self.hyper_net1(t, z)
            
            g = self.hyper_net2(t, z)

            dlogp_z_dt = g - trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, g, dlogp_z_dt)
    def trajectory(self,num_sample, t_eval,time_idx,data_train,sigma,device='gpu'):
        # options = diffeq_args()
        # options.update({'t0': t_eval[0]})
        # options.update({'t1': t_eval[-1]})
        # options.update({'t_eval':t_eval}) 
        options=self.odesolver
        self.eval()
        if num_sample is not None:
            x = Sampling(num_sample,time_idx,data_train,sigma,device)   
        else:
            x = torch.tensor(data_train[time_idx]).type(torch.float32).to(device)
        x.requires_grad=True
        logp_diff_t1 = torch.zeros(x.shape[0], 1).type(torch.float32).to(device)
        g_t1 = logp_diff_t1
        # z, _, _ = odesolve(self,y0=(x, g_t1, logp_diff_t1),options=options)
        z, _, _ = odeint(self,y0=(x, g_t1, logp_diff_t1),
                            t=torch.tensor(t_eval).type(torch.float32).to(device),
                            method = options['method'],
                            rtol = options['rtol'],
                            atol = options['atol'],
                            options={'step_size':options['step_size']})
        return z
def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Adapted from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()


class HyperNetwork1(nn.Module):
    # input x, t to get v= dx/dt
    def __init__(self, in_out_dim, dim_hiddens, n_layers, activation='tanh'):
        super().__init__()
        Layers = [in_out_dim+1]
        for i in range(n_layers):
            Layers.append(dim_hiddens)
        Layers.append(in_out_dim)
        
        self.activation=create_activation(activation)
        

        self.net = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(Layers[i], Layers[i + 1]),
                self.activation,
            )
                for i in range(len(Layers) - 2)
            ]
        )
        self.out = nn.Linear(Layers[-2], Layers[-1])

    def forward(self, t, x):
        # x is N*2
        batchsize = x.shape[0]
        if t.dim()==0:
            t = torch.tensor(t).repeat(batchsize).reshape(batchsize, 1) 
        t=t.detach()
        t.requires_grad=True
        state  = torch.cat((t,x),dim=1)
        
        ii = 0
        for layer in self.net:
            if ii == 0:
                x = layer(state)
            else:
                x = layer(x)
            ii =ii+1
        x = self.out(x)
        return x
    

class HyperNetwork2(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, dim_hiddens, activation='tanh'):
        super().__init__()
        self.activation=create_activation(activation)
        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, dim_hiddens),
            self.activation,
            nn.Linear(dim_hiddens,dim_hiddens),
            self.activation,
            nn.Linear(dim_hiddens,dim_hiddens),
            self.activation,
            nn.Linear(dim_hiddens,1))
    def forward(self, t, x):
        # x is N*2
        batchsize = x.shape[0]
        if t.dim()==0:
            t = torch.tensor(t).repeat(batchsize).reshape(batchsize, 1)
        t=t.detach()
        t.requires_grad=True
        state  = torch.cat((t,x),dim=1)
        return self.net(state)
        
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
        


def divide_intervals(timepoints, n_pieces):
    """
    Divide intervals between consecutive time points into n_pieces sub-intervals and round to two decimal places.
    
    Parameters:
    timepoints (list of float): List of original time points.
    n_pieces (int): Number of sub-intervals to divide each interval into.
    
    Returns:
    list of float: New list of time points with sub-intervals rounded to two decimal places.
    """
    new_timepoints = []
    
    for i in range(len(timepoints) - 1):
        t0, t1 = timepoints[i], timepoints[i + 1]
        step_size = (t1 - t0) / n_pieces
        new_timepoints.extend([round(t0 + j * step_size, 2) for j in range(n_pieces)])
    
    # Add the last original timepoint, rounded to two decimal places
    new_timepoints.append(round(timepoints[-1], 2))
    
    return new_timepoints


def loaddata(args,device):
    data=np.load(os.path.join(args.input_dir,(args.dataset+'.npy')),allow_pickle=True)
    data_train=[]
    for i in range(data.shape[1]):
        data_train.append(torch.from_numpy(data[0,i]).type(torch.float32).to(device))
    return data_train


    




         
            


