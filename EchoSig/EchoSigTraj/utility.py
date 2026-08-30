import torch
import random
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



def Sampling(num_samples,time_idx,data_train,sigma=0,device='cpu'):
    #perturb the  coordinate x with Gaussian noise N (0, sigma*I )
    mu = data_train[time_idx]
    mu = torch.tensor(mu).type(torch.float32).to(device)
    num_gaussian = mu.shape[0] # mu is number_sample * dimension
    dim = mu.shape[1]

    # check if number of points is <num_samples
    
    if num_gaussian < num_samples:
        samples = mu[random.choices(range(0,num_gaussian), k=num_samples)]
    else:
        samples = mu[random.sample(range(0,num_gaussian), num_samples)]

    if sigma==0:
        return samples
    elif sigma>0:
        sigma_matrix = sigma * torch.eye(dim)
        m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim), sigma_matrix)
        noise_add = m.rsample(torch.Size([num_samples])).type(torch.float32).to(device)
        return samples+noise_add
    else:
        raise ValueError('sigma must not be negative')
def wfr_dynamics(t, states, func, alpha_wfr):
    """Augment the particle dynamics with the accumulated WFR action.

    The states are ``(x, logm, action)``. ``logm`` is the relative log-mass
    along the particle trajectory, and ``action`` accumulates

        exp(logm) * (||v||^2 + alpha_wfr * ||g||^2).
    """
    x, logm, _ = states
    v, g = func(t, (x, logm))

    wfr_rate = torch.exp(logm) * (
        torch.sum(v ** 2, dim=1, keepdim=True)
        + alpha_wfr * g**2
    )

    return v, g, wfr_rate

# ---------------------------------------------------------------------------
# Legacy trajectory/Jacobian helpers below are retained temporarily for
# reference.  The current pipeline uses ``generate.py`` for these operations.
# They are commented out so accidental imports cannot keep the duplicate
# implementation alive; delete this section when preparing the GitHub copy.
# ---------------------------------------------------------------------------
# def Jacobian(f, z):
#     """Calculates Jacobian df/dz.
#     """
#     jac = []
#     for i in range(f.shape[1]):
#         df_dz = torch.autograd.grad(f[:, i], z, torch.ones_like(f[:, i]),retain_graph=True, create_graph=True)[0].view(z.shape[0], -1)
#         jac.append(torch.unsqueeze(df_dz, 1))
#     jac = torch.cat(jac, 1)
#     return jac
# # def GRN_sc(vel_net,model_ae,
# #            z_latent, time_pt,device):
# 
# # def Jac_ave_sub(vel_net,model_ae
# #                 ,z_t0,time_pt,gene_index):
# #     device=next(model_ae.parameters()).device
# #     dim = len(gene_index)
# #     dim2 = z_t0.shape[1]
# #     jac = np.zeros((dim,dim))
# #     g_xt0 = torch.zeros(1, 1).type(torch.float32).to(device)
# #     for i in range(viz_samples):
# #         x_t = z_t0[i,:].reshape([1,dim2])
# #         v_xt = vel_net(torch.tensor(time_pt).type(torch.float32).to(device),(x_t,g_xt0, g_xt0))[0]
# #         v_t_gene2 = model_ae.get_generative(x_t+v_xt)
# #         x_gene = model_ae.get_generative(x_t)
# #         v_t_gene = v_t_gene2 - x_gene
# #         jac1 = Jacobian(v_t_gene[:,gene_index], x_t).reshape(dim,dim2).detach().cpu().numpy()
# #         x_latent = model_ae.get_latent_representation(x_gene)
# #         jac2 = Jacobian(x_latent, x_gene).reshape(dim2,x_gene.shape[1]).detach().cpu().numpy()
# #         jac2 = jac2[:,gene_index]
# #         jac = jac + np.matmul(jac1,jac2)
# #     jac = jac/viz_samples
# #     return jac
# def get_J_vx_x(model_ae,vel_net,z_flatten,time_flatten,dt):
#     """Calculate GRN
# 
#     Args:
#         model_ae (_type_): _description_
#         vel_net (_type_): _description_
#         z_flatten (_type_): _description_
#         time_flatten (_type_): _description_
#         dt (_type_): _description_
# 
#     Returns:
#         _type_: _description_
#     """
#     # z_flatten.requires_grad=True
#     z_original_flatten = model_ae.generate(z_flatten,eval=True)
#     # time_flatten = torch.tensor(np.tile(time[:, None], (1, num_sample)),dtype=torch.float32).to(device).unsqueeze(2)
#     # vel_net = func.hyper_net1
#     vel_net.eval()
# 
#     vel_flatten=vel_net(time_flatten, z_flatten)
#     z_delta_flatten = z_flatten+dt*vel_flatten
#     z_delta_original_flatten = model_ae.generate(z_delta_flatten,eval=True)
#     vel_original_flatten = (z_delta_original_flatten-z_original_flatten)/dt
#     J_vx_z = Jacobian(vel_original_flatten,z_flatten).detach()
# 
#     z_hat = model_ae.emb(z_original_flatten)
#     J_z_x = Jacobian(z_hat,z_original_flatten).detach()
#     J_vx_x =  torch.einsum('nmk,nkl->nml', J_vx_z, J_z_x).detach().cpu().numpy()
#     z_original_flatten = z_original_flatten.detach().cpu().numpy()
#     vel_flatten = vel_flatten.detach().cpu().numpy()
#     vel_original_flatten = vel_original_flatten.detach().cpu().numpy()
#     return J_vx_x,z_original_flatten,vel_flatten,vel_original_flatten
# 
# def get_vx_x(model_ae,vel_net,z_flatten,time_flatten,dt):
#     """Calculate GRN
# 
#     Args:
#         model_ae (_type_): _description_
#         vel_net (_type_): _description_
#         z_flatten (_type_): _description_
#         time_flatten (_type_): _description_
#         dt (_type_): _description_
# 
#     Returns:
#         _type_: _description_
#     """
#     # z_flatten.requires_grad=True
#     z_original_flatten = model_ae.generate(z_flatten,eval=True)
#     # time_flatten = torch.tensor(np.tile(time[:, None], (1, num_sample)),dtype=torch.float32).to(device).unsqueeze(2)
#     # vel_net = func.hyper_net1
#     vel_net.eval()
# 
#     vel_flatten=vel_net(time_flatten, z_flatten)
#     z_delta_flatten = z_flatten+dt*vel_flatten
#     z_delta_original_flatten = model_ae.generate(z_delta_flatten,eval=True)
#     vel_original_flatten = (z_delta_original_flatten-z_original_flatten)/dt
#     #J_vx_z = Jacobian(vel_original_flatten,z_flatten).detach()
# 
#     z_hat = model_ae.emb(z_original_flatten)
#     #J_z_x = Jacobian(z_hat,z_original_flatten).detach()
#     #J_vx_x =  torch.einsum('nmk,nkl->nml', J_vx_z, J_z_x).detach().cpu().numpy()
#     z_original_flatten = z_original_flatten.detach().cpu().numpy()
#     vel_flatten = vel_flatten.detach().cpu().numpy()
#     vel_original_flatten = vel_original_flatten.detach().cpu().numpy()
#     return z_original_flatten,vel_flatten,vel_original_flatten
# 
#     
# def get_trajectory_only(num_sample,time,data_train,model_ae,func,sigma,device,dt=None,time_ascend=True):
#     """_summary_
# 
#     Args:
#         num_sample (_type_): _description_
#         time (_type_): _description_
#         data_train (_type_): _description_
#         model_ae (_type_): _description_
#         func (_type_): _description_
#         sigma (_type_): _description_
#         device (_type_): _description_
#         dt (_type_, optional): _description_. Defaults to None.
#         time_ascend (bool, optional): _description_. Defaults to True.
# 
#     Raises:
#         ValueError: _description_
# 
#     Returns:
#         J_vx_x:(N_sample,N_time,N_target_gene,N_source_gene)
#     """
#     batch_size=1000
#     z = func.trajectory(num_sample,time,0,data_train,sigma,device).detach()#.cpu().numpy()
#     # z: time × sample × latent_dim
#     z.requires_grad=True    
#     
#     num_time = z.shape[0]
#     dim_latent = z.shape[2]    
#     time_flatten = torch.tensor(np.tile(time[:, None], (1, num_sample)),dtype=torch.float32).to(device).unsqueeze(2)
#     time_flatten = time_flatten.reshape(num_time*num_sample,1)   
#     if dt is None:
#         dt=torch.tensor(np.abs(time[1]-time[0]),dtype=torch.float32).to(device)
#     else:
#         dt=torch.tensor(dt,dtype=torch.float32).to(device)    
#     
# 
# 
#     z_flatten = z.reshape(z.shape[0]*z.shape[1],z.shape[2]) # time × sample, latent_dim
#     vel_net = func.hyper_net1
# 
#     indices = torch.arange(z_flatten.shape[0])   
#     #J_vx_x_flatten=[]
#     z_original_flatten=[]
#     vel_flatten=[]
#     vel_original_flatten=[]
#     for batch_idx in torch.split(indices,batch_size):
#         b,c,d = get_vx_x(model_ae,vel_net,z_flatten[batch_idx,:],time_flatten[batch_idx,:],dt)
#         #J_vx_x_flatten.append(a)
#         z_original_flatten.append(b)
#         vel_flatten.append(c)
#         vel_original_flatten.append(d)
#     #J_vx_x_flatten=np.concatenate(J_vx_x_flatten, axis=0)
#     z_original_flatten = np.concatenate(z_original_flatten, axis=0)
#     vel_flatten = np.concatenate(vel_flatten, axis=0)
#     vel_original_flatten = np.concatenate(vel_original_flatten, axis=0)
# 
#     num_gene = z_original_flatten.shape[1]
# 
#     #J_vx_x = J_vx_x_flatten.reshape(num_time, num_sample,num_gene,num_gene)
#     z_original = z_original_flatten.reshape(num_time, num_sample, num_gene)
#     vel = vel_flatten.reshape(num_time, num_sample,dim_latent)
#     vel_original = vel_original_flatten.reshape(num_time, num_sample,num_gene)
# 
#     z = z.detach().cpu().numpy()
# 
# 
#     if time_ascend:
#         d_time = time[1:]-time[0:-1]
#         if (d_time<0).all():
#             # time is listed in descending order; reverse them
#             vel_original = vel_original[::-1,:,:]
#             vel = vel[::-1,:,:]
#             z = z[::-1,:,:]
#             z_original = z_original[::-1,:,:]
#             #J_vx_x=J_vx_x[::-1,:,:,:]
#             time = time[::-1].copy()
#         elif (d_time>0).all():
#             pass # do nothing if time is already in ascending order
#         else:
#             raise ValueError("Time is not strictly increasing or decreasing.")
#     z=np.transpose(z,(1,0,2))
#     z_original=np.transpose(z_original,(1,0,2))
#     vel = np.transpose(vel,(1,0,2))
#     vel_original = np.transpose(vel_original,(1,0,2))
#     #J_vx_x = np.transpose(J_vx_x,(1,0,2,3))
#     return z,z_original,vel,vel_original,time
# 
# # def get_trajectory_by_fate_sampling(
# #     adata,
# #     time,
# #     model_ae,
# #     func,
# #     sigma,
# #     device,
# #     target_time='E16.5',
# #     fate_key='annotation_group',
# #     n_per_fate=10,
# #     dt=None,
# #     time_ascend=True
# # ):
# #     """
# #     Sample fixed number of cells from each final cell fate group and compute trajectories.
# 
# #     Returns:
# #         z: (n_cells, n_time, latent_dim)
# #         z_original: (n_cells, n_time, gene_dim)
# #         vel: (n_cells, n_time, latent_dim)
# #         vel_original: (n_cells, n_time, gene_dim)
# #         time: sorted time vector
# #     """
# 
# #     batch_size = 1000
# #     embedding = adata.obsm["ae"]  # full latent embedding
# 
# #     # Step 1: Find the cells at target_time and their indices
# #     adata_final = adata[adata.obs['Time'] == target_time]
# #     global_indices_final = np.where(adata.obs['Time'] == target_time)[0]
# #     fates = adata_final.obs[fate_key].unique()
# 
# #     selected_global_indices = []
# #     for fate in fates:
# #         fate_local_idx = np.where(adata_final.obs[fate_key] == fate)[0]
# #         if len(fate_local_idx) < n_per_fate:
# #             raise ValueError(f"Not enough cells in fate {fate}, only {len(fate_local_idx)} available.")
# #         selected_local = np.random.choice(fate_local_idx, n_per_fate, replace=False)
# #         selected_global = global_indices_final[selected_local]
# #         selected_global_indices.extend(selected_global)
# 
# #     # Step 2: Collect latent embeddings
# #     data_tensor = torch.tensor(embedding[selected_global_indices], dtype=torch.float32).to(device)
# #     data_train = [data_tensor]  # make it a list of one tensor, as required
# #     num_sample = data_tensor.shape[0]
# 
# #     # Step 3: Trajectory in latent space
# #     z = func.trajectory(num_sample, time, 0, data_train, sigma, device).detach()
# #     z.requires_grad=True    
#     
# #     num_time = z.shape[0]
# #     dim_latent = z.shape[2]    
# #     time_flatten = torch.tensor(np.tile(time[:, None], (1, num_sample)),dtype=torch.float32).to(device).unsqueeze(2)
# #     time_flatten = time_flatten.reshape(num_time*num_sample,1)   
# #     if dt is None:
# #         dt=torch.tensor(np.abs(time[1]-time[0]),dtype=torch.float32).to(device)
# #     else:
# #         dt=torch.tensor(dt,dtype=torch.float32).to(device)    
#     
# 
# 
# #     z_flatten = z.reshape(z.shape[0]*z.shape[1],z.shape[2]) # time × sample, latent_dim
# #     vel_net = func.hyper_net1
# 
# #     indices = torch.arange(z_flatten.shape[0])   
# #     #J_vx_x_flatten=[]
# #     z_original_flatten=[]
# #     vel_flatten=[]
# #     vel_original_flatten=[]
# #     for batch_idx in torch.split(indices,batch_size):
# #         b,c,d = get_vx_x(model_ae,vel_net,z_flatten[batch_idx,:],time_flatten[batch_idx,:],dt)
# #         #J_vx_x_flatten.append(a)
# #         z_original_flatten.append(b)
# #         vel_flatten.append(c)
# #         vel_original_flatten.append(d)
# #     #J_vx_x_flatten=np.concatenate(J_vx_x_flatten, axis=0)
# #     z_original_flatten = np.concatenate(z_original_flatten, axis=0)
# #     vel_flatten = np.concatenate(vel_flatten, axis=0)
# #     vel_original_flatten = np.concatenate(vel_original_flatten, axis=0)
# 
# #     num_gene = z_original_flatten.shape[1]
# 
# #     #J_vx_x = J_vx_x_flatten.reshape(num_time, num_sample,num_gene,num_gene)
# #     z_original = z_original_flatten.reshape(num_time, num_sample, num_gene)
# #     vel = vel_flatten.reshape(num_time, num_sample,dim_latent)
# #     vel_original = vel_original_flatten.reshape(num_time, num_sample,num_gene)
# 
# #     z = z.detach().cpu().numpy()
# 
# 
# #     if time_ascend:
# #         d_time = time[1:]-time[0:-1]
# #         if (d_time<0).all():
# #             # time is listed in descending order; reverse them
# #             vel_original = vel_original[::-1,:,:]
# #             vel = vel[::-1,:,:]
# #             z = z[::-1,:,:]
# #             z_original = z_original[::-1,:,:]
# #             #J_vx_x=J_vx_x[::-1,:,:,:]
# #             time = time[::-1].copy()
# #         elif (d_time>0).all():
# #             pass # do nothing if time is already in ascending order
# #         else:
# #             raise ValueError("Time is not strictly increasing or decreasing.")
# #     z=np.transpose(z,(1,0,2))
# #     z_original=np.transpose(z_original,(1,0,2))
# #     vel = np.transpose(vel,(1,0,2))
# #     vel_original = np.transpose(vel_original,(1,0,2))
# #     fate_labels = adata.obs[fate_key][selected_global_indices].values
# #     return z,z_original,vel,vel_original,time,fate_labels
# 
# 
# def get_trajectory_by_fate_sampling(
#     adata,
#     time,
#     model_ae,
#     func,
#     sigma,
#     device,
#     target_time='E16.5',
#     fate_key='annotation_group',
#     n_per_fate=10,
#     dt=None,
#     time_ascend=True,
#     batch_size=1000
# ):
#     import pandas as pd
#     """
#     Sample fixed number of cells from each final cell fate group and compute trajectories.
# 
#     Returns:
#         z: (n_cells, n_time, latent_dim)
#         z_original: (n_cells, n_time, gene_dim)
#         vel: (n_cells, n_time, latent_dim)
#         vel_original: (n_cells, n_time, gene_dim)
#         time: sorted time vector (np.ndarray)
#         fate_labels: (n_cells,) labels for the sampled trajectories
#     """
#     if isinstance(time, list):
#         time = np.asarray(time, dtype=float)
#     elif isinstance(time, np.ndarray):
#         time = time.astype(float)
#     else:
#         raise TypeError("`time` must be list or numpy.ndarray of floats")
# 
#     if 'ae' not in adata.obsm_keys():
#         raise KeyError("adata.obsm['ae'] not found. Please provide latent embedding in .obsm['ae'].")
# 
#     embedding = adata.obsm["ae"]  # full latent embedding
# 
#     # Step 1: find target_time cells
#     mask_final = (adata.obs['Time'].values == target_time)
#     if mask_final.sum() == 0:
#         raise ValueError(f"No cells found at target_time={target_time!r} in adata.obs['Time'].")
# 
#     adata_final = adata[mask_final]
#     global_indices_final = np.where(mask_final)[0]
# 
#     if fate_key not in adata.obs.columns:
#         raise KeyError(f"{fate_key!r} not in adata.obs columns.")
# 
#     fates = sorted(pd.unique(adata_final.obs[fate_key].values))  # Sort for deterministic order
# 
#     selected_global_indices = []
#     fate_labels_ordered = []  # Track labels in sampling order
#     for fate in fates:
#         fate_local_idx = np.where(adata_final.obs[fate_key].values == fate)[0]
#         if len(fate_local_idx) < n_per_fate:
#             raise ValueError(f"Not enough cells in fate {fate}, only {len(fate_local_idx)} available.")
#         selected_local = np.random.choice(fate_local_idx, n_per_fate, replace=False)
#         selected_global = global_indices_final[selected_local]
#         selected_global_indices.extend(selected_global)
#         fate_labels_ordered.extend([fate] * n_per_fate)  # Add n_per_fate labels for this fate
# 
#     # Step 2: get latent embeddings
#     data_tensor = torch.tensor(embedding[selected_global_indices], dtype=torch.float32, device=device)
#     data_train = [data_tensor]  # func.trajectory 的输入形式
#     num_sample = data_tensor.shape[0]
# 
#     # Step 3: genereate latent trajecotry
#     #  z: shape (n_time, n_sample, dim_latent)
#     z = func.trajectory(num_sample, time, 0, data_train, 0.00000001, device).detach()
#     # z.requires_grad = True
# 
#     num_time = z.shape[0]
#     dim_latent = z.shape[2]
# 
#     # construct time_flatten
#     time_flatten = torch.tensor(
#         np.tile(time[:, None], (1, num_sample)),
#         dtype=torch.float32,
#         device=device
#     ).reshape(num_time * num_sample, 1)
# 
#     if dt is None:
#         if len(time) < 2:
#             raise ValueError("Need at least two time points to infer dt.")
#         dt_val = abs(time[1] - time[0])
#         dt = torch.tensor(dt_val, dtype=torch.float32, device=device)
#     else:
#         dt = torch.tensor(dt, dtype=torch.float32, device=device)
# 
#     z_flatten = z.reshape(num_time * num_sample, dim_latent)  # (time × sample, latent_dim)
#     vel_net = func.hyper_net1
# 
#     indices = torch.arange(z_flatten.shape[0], device=device)
# 
#     
#     z_original_flatten = []
#     vel_flatten = []
#     vel_original_flatten = []
# 
#     for batch_idx in torch.split(indices, batch_size):
#         #  get_vx_x return: (J_vx_x?, x_recon, v_latent, v_original)
#         # a, b, c, d = get_vx_x(...)
#         x_recon, v_lat, v_ori = get_vx_x(model_ae, vel_net, z_flatten[batch_idx, :], time_flatten[batch_idx, :], dt)
#         z_original_flatten.append(x_recon)
#         vel_flatten.append(v_lat)
#         vel_original_flatten.append(v_ori)
# 
#     z_original_flatten = np.concatenate(z_original_flatten, axis=0)
#     vel_flatten = np.concatenate(vel_flatten, axis=0)
#     vel_original_flatten = np.concatenate(vel_original_flatten, axis=0)
# 
#     num_gene = z_original_flatten.shape[1]
# 
#     z_original = z_original_flatten.reshape(num_time, num_sample, num_gene)
#     vel = vel_flatten.reshape(num_time, num_sample, dim_latent)
#     vel_original = vel_original_flatten.reshape(num_time, num_sample, num_gene)
# 
#     z = z.detach().cpu().numpy()
# 
#     # assure order of time consistent
#     if time_ascend:
#         d_time = time[1:] - time[:-1]
#         if (d_time < 0).all():
#             vel_original = vel_original[::-1, :, :]
#             vel = vel[::-1, :, :]
#             z = z[::-1, :, :]
#             z_original = z_original[::-1, :, :]
#             time = time[::-1].copy()
#         elif (d_time > 0).all():
#             pass
#         else:
#             raise ValueError("`time` is neither strictly increasing nor strictly decreasing.")
# 
#     # reshape (n_cells, n_time, dim) 
#     z = np.transpose(z, (1, 0, 2))
#     z_original = np.transpose(z_original, (1, 0, 2))
#     vel = np.transpose(vel, (1, 0, 2))
#     vel_original = np.transpose(vel_original, (1, 0, 2))
# 
#     # Use the ordered labels instead of looking up by indices to avoid order mismatch
#     fate_labels = np.array(fate_labels_ordered)
# 
#     return z, z_original, vel, vel_original, time, fate_labels
# 
# def get_trajectory(num_sample,time,data_train,model_ae,func,sigma,device,dt=None,time_ascend=True):
#     """_summary_
# 
#     Args:
#         num_sample (_type_): _description_
#         time (_type_): _description_
#         data_train (_type_): _description_
#         model_ae (_type_): _description_
#         func (_type_): _description_
#         sigma (_type_): _description_
#         device (_type_): _description_
#         dt (_type_, optional): _description_. Defaults to None.
#         time_ascend (bool, optional): _description_. Defaults to True.
# 
#     Raises:
#         ValueError: _description_
# 
#     Returns:
#         J_vx_x:(N_sample,N_time,N_target_gene,N_source_gene)
#     """
#     batch_size=1000
#     z = func.trajectory(num_sample,time,0,data_train,sigma,device).detach()#.cpu().numpy()
#     z.requires_grad=True    
#     
#     num_time = z.shape[0]
#     dim_latent = z.shape[2]    
#     time_flatten = torch.tensor(np.tile(time[:, None], (1, num_sample)),dtype=torch.float32).to(device).unsqueeze(2)
#     time_flatten = time_flatten.reshape(num_time*num_sample,1)   
#     if dt is None:
#         dt=torch.tensor(np.abs(time[1]-time[0]),dtype=torch.float32).to(device)
#     else:
#         dt=torch.tensor(dt,dtype=torch.float32).to(device)    
#     
# 
# 
#     z_flatten = z.reshape(z.shape[0]*z.shape[1],z.shape[2])
#     vel_net = func.hyper_net1
# 
#     indices = torch.arange(z_flatten.shape[0])   
#     J_vx_x_flatten=[]
#     z_original_flatten=[]
#     vel_flatten=[]
#     vel_original_flatten=[]
#     for batch_idx in torch.split(indices,batch_size):
#         a,b,c,d = get_J_vx_x(model_ae,vel_net,z_flatten[batch_idx,:],time_flatten[batch_idx,:],dt)
#         J_vx_x_flatten.append(a)
#         z_original_flatten.append(b)
#         vel_flatten.append(c)
#         vel_original_flatten.append(d)
#     J_vx_x_flatten=np.concatenate(J_vx_x_flatten, axis=0)
#     z_original_flatten = np.concatenate(z_original_flatten, axis=0)
#     vel_flatten = np.concatenate(vel_flatten, axis=0)
#     vel_original_flatten = np.concatenate(vel_original_flatten, axis=0)
# 
#     num_gene = z_original_flatten.shape[1]
# 
#     J_vx_x = J_vx_x_flatten.reshape(num_time, num_sample,num_gene,num_gene)
#     z_original = z_original_flatten.reshape(num_time, num_sample, num_gene)
#     vel = vel_flatten.reshape(num_time, num_sample,dim_latent)
#     vel_original = vel_original_flatten.reshape(num_time, num_sample,num_gene)
# 
#     z = z.detach().cpu().numpy()
# 
# 
#     # z_original_flatten = model_ae.generate(z_flatten,eval=True)
# 
#     # vel_flatten=vel_net(time_flatten,
#     #              z_flatten)
#     # vel=vel.reshape(num_time,num_sample,dim_latent)
# 
#     # z_delta_flatten = z_flatten+dt*vel_flatten
#     # z_delta_original_flatten = model_ae.generate(z_delta_flatten,eval=True)
#     # vel_original_flatten = (z_delta_original_flatten-z_original_flatten)/dt
#     
#     # df_dz = torch.autograd.grad(f[:, i], z, torch.ones_like(f[:, i]),retain_graph=True, create_graph=True)[0].view(z.shape[0], -1)
#     # J_vx_z = Jacobian(vel_original_flatten,z_flatten)
#     
#     
#     # J_G_z = torch.autograd.grad(vel_original_flatten,)
#     # vel_original_flatten
#     
#     #####
#     
#     # z_original=z_original.reshape(z.shape[0],z.shape[1],z_original.shape[1])
# 
#     # vel_net = func.hyper_net1
#     # vel_net.eval()
#     # vel=[]
#     # GRN = []
#     # dg =[]
#     
#     # for i in range(z.shape[0]):
#     #     x=torch.tensor(z[i,:,:]).type(torch.float32).to(device)
#     #     t=(torch.tensor(time[i]).type(torch.float32)*torch.ones(x.shape[0])).to(device).unsqueeze(1)
#     #     v = vel_net(t,x)
#     #     vel.append(v)
#     # vel = torch.stack(vel).detach().cpu().numpy()
#     # vel = np.transpose(vel,(1,0,2))
#     # z_delta = z+vel
#     # z_original_delta = model_ae.generate(z_delta.reshape(z_delta.shape[0]*
#     #                                                      z_delta.shape[1],
#     #                                                      z_delta.shape[2]),eval=True)
#     # z_original_delta=z_original_delta.reshape(z_delta.shape[0],z_delta.shape[1],z_original_delta.shape[1])
# 
#     # vel_original = z_original_delta-z_original
#     if time_ascend:
#         d_time = time[1:]-time[0:-1]
#         if (d_time<0).all():
#             # time is listed in descending order; reverse them
#             vel_original = vel_original[::-1,:,:]
#             vel = vel[::-1,:,:]
#             z = z[::-1,:,:]
#             z_original = z_original[::-1,:,:]
#             J_vx_x=J_vx_x[::-1,:,:,:]
#             time = time[::-1].copy()
#         elif (d_time>0).all():
#             pass # do nothing if time is already in ascending order
#         else:
#             raise ValueError("Time is not strictly increasing or decreasing.")
#     z=np.transpose(z,(1,0,2))
#     z_original=np.transpose(z_original,(1,0,2))
#     vel = np.transpose(vel,(1,0,2))
#     vel_original = np.transpose(vel_original,(1,0,2))
#     J_vx_x = np.transpose(J_vx_x,(1,0,2,3))
#     return z,z_original,vel,vel_original,J_vx_x,time
# 
# 
# def get_vel_GRN(model_ae,func,X,Time_float,dt=0.1,batch_size=1000,device='cuda'):
#     embedding=model_ae.emb(X)
#     embedding=torch.tensor(embedding,dtype=torch.float32).to(device)
#     embedding.requires_grad=True
#     Time_float=torch.tensor(Time_float,dtype=torch.float32).unsqueeze(-1).to(device)
#     dt=torch.tensor(dt,dtype=torch.float32).to(device)
#     indices = torch.arange(embedding.shape[0])   
#     J_vx_x=[]
#     z_original=[]
#     vel=[]
#     vel_original=[]
#     model_ae.to(device)
#     func.to(device)
#     for batch_idx in torch.split(indices,batch_size):
#         a,b,c,d = get_J_vx_x(model_ae,
#                              func.hyper_net1,
#                              embedding[batch_idx,:],
#                              Time_float[batch_idx,:],dt)
#         J_vx_x.append(a)
#         z_original.append(b)
#         vel.append(c)
#         vel_original.append(d)    
#     J_vx_x=np.concatenate(J_vx_x, axis=0)
#     z_original = np.concatenate(z_original, axis=0)
#     vel = np.concatenate(vel, axis=0)
#     vel_original = np.concatenate(vel_original, axis=0)
#     
# 
#     return vel,vel_original,J_vx_x
# 
# 
# def get_vel_g_GRN(model_ae,func,X,Time_float,dt=0.1,batch_size=1000,device='cuda'):
#     embedding=model_ae.emb(X)
#     embedding=torch.tensor(embedding,dtype=torch.float32).to(device)
#     embedding.requires_grad=True
#     Time_float=torch.tensor(Time_float,dtype=torch.float32).unsqueeze(-1).to(device)
#     dt=torch.tensor(dt,dtype=torch.float32).to(device)
#     indices = torch.arange(embedding.shape[0])   
#     J_vx_x=[]
#     z_original=[]
#     vel=[]
#     vel_original=[]
#     g=[]
#     model_ae.to(device)
#     func.to(device)
#     func.hyper_net2.eval()
#     for batch_idx in torch.split(indices,batch_size):
#         a,b,c,d = get_J_vx_x(model_ae,
#                              func.hyper_net1,
#                              embedding[batch_idx,:],
#                              Time_float[batch_idx,:],dt)
#         J_vx_x.append(a)
#         z_original.append(b)
#         vel.append(c)
#         vel_original.append(d)  
#         g_batch=func.hyper_net2(Time_float[batch_idx,:],embedding[batch_idx,:])
#         g_batch=g_batch.detach().cpu().numpy()
#         g.append(g_batch)
# 
#     J_vx_x=np.concatenate(J_vx_x, axis=0)
#     z_original = np.concatenate(z_original, axis=0)
#     vel = np.concatenate(vel, axis=0)
#     vel_original = np.concatenate(vel_original, axis=0)
#     g=np.concatenate(g,axis=0)
# 
#     return vel,vel_original,g,J_vx_x
# def get_vel_g(model_ae,func,embedding,Time_float,batch_size=1000,device='cuda'):
#     """Get velocity and growth (g) 
# 
#     Args:
#         model_ae (_type_): _description_
#         func (_type_): UOT class
#         embedding (_type_): _description_
#         Time_float (_type_): _description_
#         batch_size (int, optional): _description_. Defaults to 1000.
#         device (str, optional): _description_. Defaults to 'cuda'.
# 
#     Returns:
#         _type_: _description_
#     """
#     embedding=torch.tensor(embedding,dtype=torch.float32).to(device)
#     # embedding.requires_grad=True
#     Time_float=torch.tensor(Time_float,dtype=torch.float32).unsqueeze(-1).to(device)
#     indices = torch.arange(embedding.shape[0])   
#     vel=[]
#     g=[]
#     model_ae.to(device)
#     func.to(device)
#     hyper_net1=func.hyper_net1
#     hyper_net2=func.hyper_net2
# 
#     for batch_idx in torch.split(indices,batch_size):
#         v_batch=hyper_net1(Time_float[batch_idx,:],embedding[batch_idx,:])
#         v_batch=v_batch.detach().cpu().numpy()
#         vel.append(v_batch)
#         g_batch=hyper_net2(Time_float[batch_idx,:],embedding[batch_idx,:])
#         g_batch=g_batch.detach().cpu().numpy()
#         g.append(g_batch)
#     vel = np.concatenate(vel, axis=0)
#     g=np.concatenate(g,axis=0)
#         
#     return vel,g
# 
# 
# 
# 
# def get_vel_o_g(model_ae, func, embedding, Time_float, dt=0.1, batch_size=1000, device='cuda'):
#     """
#     Get velocity (latent + original space) and growth (g), skipping Jacobian to save computation.
# 
#     Args:
#         model_ae: Autoencoder model.
#         func: UOT class with hyper_net1 (velocity) and hyper_net2 (growth).
#         X: Input data (original space).
#         Time_float: Time points as float array.
#         dt: Time step size.
#         batch_size: Batch size for processing.
#         device: 'cuda' or 'cpu'.
# 
#     Returns:
#         vel: Velocity in latent space.
#         vel_original: Velocity in original space.
#         g: Growth values.
#     """
#     embedding=torch.tensor(embedding,dtype=torch.float32).to(device)
#     Time_float = torch.tensor(Time_float, dtype=torch.float32).unsqueeze(-1).to(device)
#     dt = torch.tensor(dt, dtype=torch.float32).to(device)
#     indices = torch.arange(embedding.shape[0])
# 
#     vel_list = []
#     vel_original_list = []
#     g_list = []
# 
#     model_ae.to(device)
#     func.to(device)
#     hyper_net1 = func.hyper_net1
#     hyper_net2 = func.hyper_net2
# 
#     for batch_idx in torch.split(indices, batch_size):
#         # Get latent velocity
#         vel_batch = hyper_net1(Time_float[batch_idx, :], embedding[batch_idx, :])
# 
#         # Map back to original space
#         z_original = model_ae.generate(embedding[batch_idx, :], eval=True)
#         z_delta = embedding[batch_idx, :] + dt * vel_batch
#         z_delta_original = model_ae.generate(z_delta, eval=True)
#         vel_original_batch = (z_delta_original - z_original) / dt
# 
#         # Get growth
#         g_batch = hyper_net2(Time_float[batch_idx, :], embedding[batch_idx, :])
# 
#         # Detach and move to CPU
#         vel_list.append(vel_batch.detach().cpu().numpy())
#         vel_original_list.append(vel_original_batch.detach().cpu().numpy())
#         g_list.append(g_batch.detach().cpu().numpy())
# 
#     vel = np.concatenate(vel_list, axis=0)
#     vel_original = np.concatenate(vel_original_list, axis=0)
#     g = np.concatenate(g_list, axis=0)
# 
#     return vel, vel_original, g
