import torch
import torch.nn as nn
import random
from functools import partial
from TorchDiffEqPack import odesolve
from torchdiffeq import odeint
import numpy as np

def time2tensor(time_flatten):
    # Convert list or numpy array to torch tensor
    if isinstance(time_flatten, list):
        time_flatten = torch.tensor(time_flatten, dtype=torch.float32)
    elif isinstance(time_flatten, np.ndarray):
        time_flatten = torch.from_numpy(time_flatten).float()
    elif not isinstance(time_flatten, torch.Tensor):
        raise TypeError("Input must be a list, numpy array, or torch tensor.")

    # Ensure shape is (N, 1)
    if time_flatten.dim() == 1:
        time_flatten = time_flatten.view(-1, 1)
    elif time_flatten.dim() == 2 and time_flatten.shape[1] == 1:
        pass  # already the right shape
    else:
        raise ValueError(f"Unexpected shape for tensor: {time_flatten.shape}")

    return time_flatten
def mtx2tensor(mtx):
    if isinstance(mtx, list):
        mtx=torch.tensor(mtx, dtype=torch.float32)
    elif isinstance(mtx, np.ndarray):
        mtx = torch.from_numpy(mtx).float()
    elif isinstance(mtx,float):
        mtx=torch.tensor(mtx, dtype=torch.float32)
    elif not isinstance(mtx, torch.Tensor):
        raise TypeError("Input must be a list, numpy array, or torch tensor.")

    return mtx


def Jacobian(f, z):
    """Calculates Jacobian df/dz.
    """
    jac = []
    for i in range(f.shape[1]):
        df_dz = torch.autograd.grad(f[:, i], z, torch.ones_like(f[:, i]),retain_graph=True, create_graph=True)[0].view(z.shape[0], -1)
        df_dz = df_dz.detach().cpu().numpy()
        # jac.append(torch.unsqueeze(df_dz, 1))
        jac.append(np.expand_dims(df_dz, axis=1))
    jac = np.concat(jac,1)
    # jac = torch.cat(jac, 1)
    return jac
def get_J(model_ae,func,z_flatten,time_flatten,dt,device='cuda'):
    """Calculate GRN from embedding

    Args:
        model_ae (_type_): _description_
        vel_net (_type_): _description_
        z_flatten (_type_): _description_
        time_flatten (_type_): _description_
        dt (_type_): _description_

    Returns:
        J_vx_x: GRN with shape (N_sample,N_target_gene,N_source_gene)
    """
    model_ae.to(device)
    func.to(device)
    dt=mtx2tensor(dt).to(device)
    z_flatten = mtx2tensor(z_flatten).to(device)
    time_flatten=time2tensor(time_flatten).to(device)

    z_flatten=z_flatten.detach()
    z_flatten.requires_grad=True


    z_original_flatten = model_ae.generate(z_flatten,eval=True)
    # time_flatten = torch.tensor(np.tile(time[:, None], (1, num_sample)),dtype=torch.float32).to(device).unsqueeze(2)
    vel_net = func.hyper_net1
    vel_net.eval()

    vel_flatten=vel_net(time_flatten, z_flatten)
    z_delta_flatten = z_flatten+dt*vel_flatten
    z_delta_original_flatten = model_ae.generate(z_delta_flatten,eval=True)
    vel_original_flatten = (z_delta_original_flatten-z_original_flatten)/dt
    J_vx_z = Jacobian(vel_original_flatten,z_flatten)
    z_hat = model_ae.emb(z_original_flatten)
    J_z_x = Jacobian(z_hat,z_original_flatten)
    J_vx_x =  np.einsum('nmk,nkl->nml', J_vx_z, J_z_x)

    # z_original_flatten = z_original_flatten.detach().cpu().numpy()
    # vel_flatten = vel_flatten.detach().cpu().numpy()
    # vel_original_flatten = vel_original_flatten.detach().cpu().numpy()
    return J_vx_x #,z_original_flatten,vel_flatten,vel_original_flatten

def get_v(model_ae,func,z_flatten,time_flatten,dt,device='cuda'):
    """Calculate velocity from embedding;

    Args:
        model_ae (_type_): _description_
        vel_net (_type_): _description_
        z_flatten (_type_): _description_
        time_flatten (_type_): _description_
        dt (_type_): _description_

    Returns:
        z_original_flatten: reconstructed features (X)
        vel_flatten: velocity at the embedding space
        vel_original_flatten: velocity at the recontructed space
    """
    model_ae.to(device)
    func.to(device)
    dt=mtx2tensor(dt).to(device)
    z_flatten = mtx2tensor(z_flatten).to(device)
    time_flatten=time2tensor(time_flatten).to(device)

    # z_flatten.requires_grad=True
    z_original_flatten = model_ae.generate(z_flatten,eval=True)
    # time_flatten = torch.tensor(np.tile(time[:, None], (1, num_sample)),dtype=torch.float32).to(device).unsqueeze(2)
    vel_net = func.hyper_net1
    vel_net.eval()

    vel_flatten=vel_net(time_flatten, z_flatten)
    z_delta_flatten = z_flatten+dt*vel_flatten
    z_delta_original_flatten = model_ae.generate(z_delta_flatten,eval=True)
    vel_original_flatten = (z_delta_original_flatten-z_original_flatten)/dt
    #J_vx_z = Jacobian(vel_original_flatten,z_flatten).detach()

    z_hat = model_ae.emb(z_original_flatten)
    #J_z_x = Jacobian(z_hat,z_original_flatten).detach()
    #J_vx_x =  torch.einsum('nmk,nkl->nml', J_vx_z, J_z_x).detach().cpu().numpy()
    z_original_flatten = z_original_flatten.detach().cpu().numpy()
    vel_flatten = vel_flatten.detach().cpu().numpy()
    vel_original_flatten = vel_original_flatten.detach().cpu().numpy()
    return z_original_flatten,vel_flatten,vel_original_flatten

def get_g(model_ae,func,z_flatten,time_flatten,dt,device='cuda'):
    """Calculate g from embedding

    Args:
        model_ae (_type_): _description_
        g_net (_type_): _description_
        z_flatten (_type_): _description_
        time_flatten (_type_): _description_
        dt (_type_): _description_
    """

    ############
    model_ae.to(device)
    func.to(device)
    dt=mtx2tensor(dt).to(device)
    z_flatten = mtx2tensor(z_flatten).to(device)
    time_flatten=time2tensor(time_flatten).to(device)

    g_net = func.hyper_net2

    g_net.eval()

    g=g_net(time_flatten, z_flatten)
    g=g.detach().cpu().numpy()
    return g

def get_dg(model_ae,func,z_flatten,time_flatten,dt,device='cuda'):
    """Calculate derivative of g from embedding

    Args:
        model_ae (_type_): _description_
        g_net (_type_): _description_
        z_flatten (_type_): _description_
        time_flatten (_type_): _description_
        dt (_type_): _description_
    """
    model_ae.to(device)
    func.to(device)
    dt=mtx2tensor(dt).to(device)
    z_flatten = mtx2tensor(z_flatten).to(device)
    time_flatten=time2tensor(time_flatten).to(device)

    z_flatten=z_flatten.detach()
    # if not z_flatten.requires_grad:
    z_flatten.requires_grad=True
    z_original_flatten = model_ae.generate(z_flatten,eval=True)
    # time_flatten = torch.tensor(np.tile(time[:, None], (1, num_sample)),dtype=torch.float32).to(device).unsqueeze(2)
    # vel_net = func.hyper_net1
    g_net = func.hyper_net2
    g_net.eval()
    z_original_flatten = model_ae.generate(z_flatten,eval=True)
    z_hat = model_ae.emb(z_original_flatten)


    g=g_net(time_flatten, z_hat)
    dg = Jacobian(g,z_original_flatten)
    assert dg.shape[1]==1, "the input G needs to have dim 1, but get {dg.shape[1]}"
    dg=dg.reshape(dg.shape[0],dg.shape[2])
    return dg


def get_trajectory(num_sample,time,time_GRN,data_train,model_ae,func,sigma,device,dt=None,time_ascend=True,batch_size=1000):
    """_summary_

    Args:
        num_sample (_type_): _description_
        time (_type_): _description_
        time_GRN (_type_): lower resolution of time for GRN
        data_train (_type_): _description_
        model_ae (_type_): _description_
        func (_type_): _description_
        sigma (_type_): _description_
        device (_type_): _description_
        dt (_type_, optional): _description_. Defaults to None.
        time_ascend (bool, optional): _description_. Defaults to True.

    Raises:
        ValueError: _description_

    Returns:
        J_vx_x: GRNs with shape (N_sample,N_time,N_target_gene,N_source_gene)
    """
    # batch_size=1000
    z = func.trajectory(num_sample,time,0,data_train,sigma,device).detach()#.cpu().numpy()
    z.requires_grad=True    
    if num_sample is None:
        num_sample = data_train[0].shape[0]
    num_time = z.shape[0]
    dim_latent = z.shape[2]    
    time_flatten = torch.tensor(np.tile(time[:, None], (1, num_sample)),dtype=torch.float32).to(device).unsqueeze(2)
    time_flatten = time_flatten.reshape(num_time*num_sample,1)   
    if dt is None:
        dt=torch.tensor(np.abs(time[1]-time[0]),dtype=torch.float32).to(device)
    else:
        dt=torch.tensor(dt,dtype=torch.float32).to(device)    
    


    z_flatten = z.reshape(z.shape[0]*z.shape[1],z.shape[2])

    indices = torch.arange(z_flatten.shape[0])   
    # J_vx_x_flatten=[]
    z_original_flatten=[]
    vel_flatten=[]
    vel_original_flatten=[]
    g_flatten = []
    # abs_max=[]
    for batch_idx in torch.split(indices,batch_size):
        a, b, c = get_v(model_ae,func,z_flatten[batch_idx,:],time_flatten[batch_idx,:],dt,device=device)
        z_original_flatten.append(a)
        vel_flatten.append(b)
        vel_original_flatten.append(c)     
        g_flatten.append(get_g(model_ae,func,z_flatten[batch_idx,:],time_flatten[batch_idx,:],dt,device=device))

    #     a,b,c,d = get_J_vx_x(model_ae,vel_net,z_flatten[batch_idx,:],time_flatten[batch_idx,:],dt,device=device)
    #     tmp_J=get_J(model_ae,func,z_flatten[batch_idx,:],time_flatten[batch_idx,:],dt,device=device)
    #     # abs_max.append(np.max(np.abs(tmp_J-a)))
    #     abs_max.append(np.max(np.abs(tmp_J-J_vx_x_flatten2[batch_idx,:,:])))

    #     J_vx_x_flatten.append(a)

    # J_vx_x_flatten=np.concatenate(J_vx_x_flatten, axis=0)
    z_original_flatten = np.concatenate(z_original_flatten, axis=0)
    vel_flatten = np.concatenate(vel_flatten, axis=0)
    vel_original_flatten = np.concatenate(vel_original_flatten, axis=0)
    g_flatten=np.concatenate(g_flatten, axis=0)

    time_GRN_idx=[]
    J_vx_x = []
    dg=[]
    for t in time_GRN:
        idx=np.argmin(abs(time-t))
        time_GRN_idx.append(idx)
    
    for i,t in enumerate(time_GRN):
        z_sub=z[time_GRN_idx[i],:,:]
        time_sub = torch.full((z_sub.shape[0], 1), t,dtype=torch.float32).to(device)
        J_vx_x.append(get_J(model_ae,func,z_sub,time_sub,dt,device=device))
        dg.append(get_dg(model_ae,func,z_sub,time_sub,dt,device=device))
    J_vx_x=np.array(J_vx_x)
    dg=np.array(dg)

    # J_vx_x_flatten = get_J(model_ae,func,z_flatten,time_flatten,dt,device=device)
    # g_flatten = get_g(model_ae,func,z_flatten,time_flatten,dt)
    # dg_flatten = get_dg(model_ae,func,z_flatten,time_flatten,dt)



    num_gene = z_original_flatten.shape[1]

    # J_vx_x = J_vx_x_flatten.reshape(num_time, num_sample,num_gene,num_gene)
    z_original = z_original_flatten.reshape(num_time, num_sample, num_gene)
    vel = vel_flatten.reshape(num_time, num_sample,dim_latent)
    vel_original = vel_original_flatten.reshape(num_time, num_sample,num_gene)
    g = g_flatten.reshape(num_time, num_sample,1)
    # dg = dg_flatten.reshape(num_time,num_sample,num_gene)
    z = z.detach().cpu().numpy()

    if time_ascend:
        d_time = time[1:]-time[0:-1]
        if (d_time<0).all():
            # time is listed in descending order; reverse them
            vel_original = vel_original[::-1,:,:]
            vel = vel[::-1,:,:]
            z = z[::-1,:,:]
            z_original = z_original[::-1,:,:]
            g=g[::-1,:,:]
            time = time[::-1].copy()
        elif (d_time>0).all():
            pass # do nothing if time is already in ascending order
        else:
            raise ValueError("Time is not strictly increasing or decreasing.")
        d_time = time_GRN[1:]-time_GRN[0:-1]
        if (d_time<0).all():
            # time is listed in descending order; reverse them
            J_vx_x=J_vx_x[::-1,:,:,:]
            dg=dg[::-1,:,:]
            time_GRN = time_GRN[::-1].copy()
        elif (d_time>0).all():
            pass # do nothing if time is already in ascending order
        else:
            raise ValueError("Time is not strictly increasing or decreasing.")
    z=np.transpose(z,(1,0,2))
    z_original=np.transpose(z_original,(1,0,2))
    vel = np.transpose(vel,(1,0,2))
    vel_original = np.transpose(vel_original,(1,0,2))
    J_vx_x = np.transpose(J_vx_x,(1,0,2,3))
    g=np.transpose(g,(1,0,2))
    dg=np.transpose(dg,(1,0,2))
    return z,z_original,vel,vel_original,g,J_vx_x,dg,time,time_GRN



