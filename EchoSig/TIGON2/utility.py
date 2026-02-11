import torch
import torch.nn as nn
import random
from functools import partial
from TorchDiffEqPack import odesolve
from torchdiffeq import odeint
import numpy as np

# def diffeq_args():
#     # configure training options
#     options = {}
#     options.update({'method': 'euler'})
#     options.update({'h': 0.01})
#     options.update({'rtol': 1e-3})
#     options.update({'atol': 1e-5})
#     options.update({'print_neval': False})
#     options.update({'neval_max': 1000000})
#     options.update({'safety': None})
#     return options
def MultimodalGaussian_density(x, time_idx, data_train, sigma, device):
    """density function for MultimodalGaussian
    """
    mu = data_train[time_idx]
    num_gaussian = mu.shape[0] # mu is number_sample * dimension
    dim = mu.shape[1]
    sigma_matrix = sigma * torch.eye(dim).type(torch.float32).to(device)
    
    # Expand dimensions of x and mu to allow broadcasting
    x_expanded = x[:, None, :]
    mu_expanded = mu[None, :, :]
    
    # Compute the exponent part of the Gaussian formula in a vectorized manner
    diff = x_expanded - mu_expanded
    exponent = -0.5 * torch.einsum('bik,kl,bil->bi', diff, torch.inverse(sigma_matrix), diff)
    
    # Compute the normalization constant
    norm_constant = torch.sqrt((2 * torch.pi) ** dim * torch.det(sigma_matrix))
    
    # Compute the unnormalized probabilities
    p_unn = torch.sum(torch.exp(exponent) / norm_constant, dim=-1)
    
    # Normalize the probabilities
    p_n = p_unn / num_gaussian
    
    return p_n



def Sampling(num_samples,time_idx,data_train,sigma,device):
    # sample data from discrete data points
    # we perturb the  coordinate x with Gaussian noise N (0, sigma*I )
    mu = data_train[time_idx]
    mu = torch.tensor(mu).type(torch.float32).to(device)
    num_gaussian = mu.shape[0] # mu is number_sample * dimension
    dim = mu.shape[1]
    sigma_matrix = sigma * torch.eye(dim)
    m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim), sigma_matrix)
    noise_add = m.rsample(torch.Size([num_samples])).type(torch.float32).to(device)
    # check if number of points is <num_samples
    
    if num_gaussian < num_samples:
        samples = mu[random.choices(range(0,num_gaussian), k=num_samples)] + noise_add
    else:
        samples = mu[random.sample(range(0,num_gaussian), num_samples)] + noise_add
    return samples
def ggrowth(t,y,func,device):
    y_0 = torch.zeros(y[0].shape).type(torch.float32).to(device)                    
    gg = func.forward(t, y)[1]
    return (y_0,gg,torch.zeros_like(gg))

def trans_loss(t,y,func,device,odeint_setp, t_start):
    v, g, _ = func.forward(t, y)
    y_0 = torch.zeros(g.shape).type(torch.float32).to(device)
    y_00 = torch.zeros(v.shape).type(torch.float32).to(device)
    g_growth = partial(ggrowth,func=func,device=device)
    if t != t_start:
        _, exp_g, _ = odeint(g_growth, (y_00,y_0,y_0), torch.tensor([t_start,t]).type(torch.float32).to(device),atol=1e-5,rtol=1e-5,method='midpoint',options = {'step_size': odeint_setp})
        f_int = (torch.norm(v,dim=1)**2+torch.norm(g,dim=1)**2).unsqueeze(1)*torch.exp(exp_g[-1])
        return (y_00,y_0,f_int)
    else:
        return (y_00,y_0,y_0)
    
