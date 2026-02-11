import torch
import torch.nn as nn
import random
import numpy as np
import scipy
import scanpy as sc
import os
import torch
import EchoSig
import yaml
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
def read_config_file(filepath):
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config
def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.2, True)
    elif name == 'tanh':
        return nn.Tanh()
    elif name =='softplus':
        return nn.Softplus()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError(f"{name} is not implemented.")
def retrieval_data(config,):
    save_dir=config['save_dir']
    n_genes=config['preprocessing']['n_genes']
    dim_latent = config['AE']['model']['dim_latent']
    device = config['device']
    adata=sc.read_h5ad(os.path.join(save_dir,'adata.h5ad'))
    Time = adata.obs['Time'].values
    time_str = config['Time']['str'] #['0d', '8h', '1d', '3d', '7d']
    time_float = config['Time']['float'] #[0, 0.1, 0.3, 0.9, 2.1]
    time_map = {time_str[i]:time_float[i] for i in range(len(time_float))}
    Time_float = [time_map[Time[i]] for i in range(len(Time))]
    # X=np.array(adata.X.todense())
    if adata.shape[1] != n_genes:
        #pick only HVGs
        adata = adata[:, adata.var['highly_variable']].copy()
    if scipy.sparse.issparse(adata.X):
        X=np.array(adata.X.todense())
    else:
        X=adata.X
    model_ae = EchoSig.AE.create_AE(n_genes=n_genes,
                    **config['AE']['model']
                    )
    ae_ckpt=torch.load(os.path.join(save_dir,'AE.pth'),weights_only=False)
    model_ae.load_state_dict(ae_ckpt['ae_state_dict'])
    model_ae.scaler = ae_ckpt['scaler']
    model_ae.to(device)
    # trainer_ae = src.AE.Trainer(model=model_ae,X=X,device=device,
    #                         **config['AE']['trainer'])
    X_hat=model_ae.generate(model_ae.emb(X))
    embedding=model_ae.emb(X)
    sigma = config['TIGON']['trainer']['sigma']

    data_train = []
    for k in range(len(time_str)):
        indices = [i for i, l in enumerate(Time) if l == time_str[k]]
        samples = embedding[indices,]
        samples = torch.from_numpy(samples).type(torch.float32).to(device)
        data_train.append(samples)
    func = EchoSig.TIGON2.UOT(in_out_dim=dim_latent, **config['TIGON']['model'],
                          odesolver=config['TIGON']['odesolver'])
    uot_ckpt = torch.load(os.path.join(save_dir, 'ckpt.pth'))
    func.load_state_dict(uot_ckpt['func_state_dict'])
    func.to(device)
    return adata,data_train,X,embedding,np.array(Time_float),model_ae,func,sigma,device,n_genes


def cluster_trajectory(n_fates,z,z_original_norm):
    num_sample = z.shape[0]
    if n_fates==1:
        fate_idx=[np.array([True]*num_sample)]
        z_fate=[]
        z_original_fate=[]
        for fate_id in range(n_fates):
            z_fate.append(np.mean(z[fate_idx[fate_id],:,:],axis=0))
            z_original_fate.append(np.mean(z_original_norm[fate_idx[fate_id],:,:],axis=0))
        z_fate = np.array(z_fate)
        z_original_fate = np.array(z_original_fate)
    else:
        z_flatten = z.reshape(z.shape[0],z.shape[1]*z.shape[2])
        kmeans = KMeans(n_clusters=n_fates).fit(z_flatten)
        fates = kmeans.predict(z_flatten)
        # cluster_center=kmeans.cluster_centers_
        fate_idx=[]
        for fate_id in range(n_fates):
            fate_idx.append(fates==fate_id)
        z_fate=[]
        z_original_fate=[]
        for fate_id in range(n_fates):
            z_fate.append(np.mean(z[fate_idx[fate_id],:,:],axis=0))
            z_original_fate.append(np.mean(z_original_norm[fate_idx[fate_id],:,:],axis=0))
        z_fate = np.array(z_fate)
        z_original_fate = np.array(z_original_fate)
    return fate_idx, z_fate, z_original_fate

def spatial_neighbor_sampling(adata,
                              focus_group,
                              annotation_key='annotation',
                              dis_max=10,
                              n_neighbors=10,
                              min_cell=500,
                              num_sample_focus = 100,
                              num_sample_neighbor = 20):
    """
    Perform spatial sampling for a selected cell type (focus) and its neighborhood cells within a distance

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial coordinates in `obsm['spatial']`
        and cell-type annotations in `annotation_key`.
    focus_group : str
        The cell annotation category to be treated as the focused cell type to infer its communication with other cell types.
        
    annotation_key : str, optional (default: 'annotation')
        Column name in `adata.obs` that specifies cell-type or group identity.
        Used to define cell types
    dis_max : float, optional (default: 10)
        Maximum spatial radius used to search for neighboring cells 
        around each sender cell.

    n_neighbors : int, optional (default: 10)
        Minimum number of neighbors required for a sender cell to be kept.

    min_cell : int, optional (default: 500)
        Minimum number of cells required in a sender group to perform sampling.

    num_sample_focus : int, optional (default: 100)
        Number of cells in focused cell type to sample using the `point_sampling` method.

    num_sample_neighbor : int, optional (default: 20)
        Number of neighbor cells to sample per neighboring group.

    Returns
    -------
    focus_index : np.ndarray
        The indices of all focused cells (original AnnData index).
    neighbor_index : np.ndarray
        The indices of all neighboring cells of the focused cell type
    focus_sample_index : np.ndarray
        The indices of sampled focused cells (original AnnData index).

    neighbor_sample_index : dict
        Dictionary mapping each neighboring group name to the sampled sender indices
        (original AnnData index).

    Notes
    -----
    - Uses a cKDTree query to find neighboring cells that are within `dis_max` of
      at least `n_neighbors` focused cells.
    - Performs spatial sampling via the external function `point_sampling()`.
    """
    def farthest_point_sampling(x, k):
        """
        x: (N,2) array of coordinates
        k: number of points to sample
        """
        N = x.shape[0]
        sampled_idx = np.zeros(k, dtype=int)
        
        # random starting point
        sampled_idx[0] = np.random.randint(N)
        distances = np.full(N, np.inf)

        for i in range(1, k):
            # update distances to nearest selected point
            last_p = x[sampled_idx[i-1]]
            dist = np.linalg.norm(x - last_p, axis=1)
            distances = np.minimum(distances, dist)

            # pick the farthest point
            sampled_idx[i] = np.argmax(distances)

        return sampled_idx
    def point_sampling(x,k):
        """
            Sample `k` points from data using its coordinate `x`. 
            We try to keep points well-seperated and also have similar distribution with `x`. 
            Procedure: perform random sampling for `5k` points, 
                then search `k` points with farthest distance to

        Args:
            x: (N,2) array of coordinates
            k: number of points to sample
        """
        N = x.shape[0]
        idx=np.random.choice(N,5*k,replace=False)
        idx2=farthest_point_sampling(x[idx,:],k)
        return idx[idx2]    

    focus_index=adata.obs[annotation_key]==focus_group
    neighbor_index = adata.obs[annotation_key]!=focus_group
    Y_focus = adata[focus_index].obsm['spatial']
    Y_neighbor = adata[neighbor_index].obsm['spatial']


    tree = cKDTree(Y_focus)
    neighbors = tree.query_ball_point(Y_neighbor, r=dis_max)
    mask = np.array([len(n) > n_neighbors for n in neighbors])
    neighbor_index=np.where(neighbor_index)[0][mask]
    focus_index=np.where(focus_index)[0]
    sc.pl.embedding(adata[np.append(focus_index,neighbor_index)],basis='spatial',color=annotation_key)
    Y_neighbor = adata[neighbor_index].obsm['spatial']



    idx=point_sampling(Y_focus,num_sample_focus)
    focus_sample_index=focus_index[idx]


    neighbor_sample_index={}
    adata_sender = adata[neighbor_index]
    for g in list(adata_sender.obs[annotation_key].cat.categories):
        if sum(adata_sender.obs[annotation_key]==g) > min_cell:
            idx_group = np.where(adata_sender.obs[annotation_key]==g)[0]
            x=adata_sender[idx_group].obsm['spatial']
            idx=point_sampling(x,num_sample_neighbor)
            neighbor_sample_index[g]=neighbor_index[idx_group[idx]]

    return focus_index, neighbor_index, focus_sample_index,neighbor_sample_index


def get_configs(dataset,focus_group=None):
    if dataset =='EMT':
        time_scale=80
        time_unit='h'
        n_fates=1
        num_sample=32
        seed=20
        save_CCC_dir=dataset+'/'
        config = read_config_file('configs/config_emt.yaml')
        # save_dir=config['save_dir']+'seed='+str(seed)+'/'
        # config['save_dir']=save_dir
        time_float=config['Time']['float']

        ## d7 data
        time = np.linspace(time_float[0],time_float[-1],10080+1) # one time step is 1 min
        time_GRN = np.linspace(time_float[0],time_float[-1],168+1)
        lag_list = np.linspace(-800,800,801).astype(int)
        max_lag = 0.06

        ## d3 data
        ## reverse integration
        # time = np.linspace(time_float[0],time_float[-1],4320+1) # one time step is 1 min
        # time_GRN = np.linspace(time_float[0],time_float[-1],18+1)

        # lag_list = np.linspace(-800,800,801).astype(int) # 800: this needs to exceed range of `max_lag`*len(time). So we will conclude no causal effect if lag hits maximal (800 this case)
        # max_lag = 0.16  # maximal allowed lag: `max_lag`*len(time)
        
    elif dataset == 'iPSC':
        time_scale=24
        time_unit='h'
        n_fates=2
        num_sample=64
        seed=2
        n_layers=3
        dim_hiddens=64
        config = read_config_file('configs/config_iPSC.yaml')
        config['TIGON']['model']['n_layers']=n_layers
        config['TIGON']['model']['dim_hiddens']=dim_hiddens
        config['seed']=seed
        # save_dir=config['save_dir']+'seed='+str(seed)+'/n_lay='+str(n_layers)+'_'+'n_hid='+str(dim_hiddens)+'/'
        # config['save_dir']=save_dir 
        save_CCC_dir=dataset+'/'
        time_float=config['Time']['float']
        time = np.linspace(time_float[0],time_float[-1],4320+1) # one time step is 1 min
        time_GRN = np.linspace(time_float[0],time_float[-1],72+1)
        lag_list = np.linspace(-800,800,801).astype(int)  # 800: this needs to exceed range of `max_lag`*len(time). So we will conclude no causal effect if lag hits maximal (800 this case)
        max_lag = 0.16  


    elif dataset == 'MOSTA':
        time_scale=24
        time_unit='h'
        n_fates = None
        num_sample = None
        # seed=42
        # dim_latent=4
        # n_layers=3
        # dim_hiddens=64
        config = read_config_file('configs/config_MOSTA.yaml')
        # config['TIGON']['model']['n_layers']=n_layers
        # config['TIGON']['model']['dim_hiddens']=dim_hiddens
        dim_latent =config['AE']['model']['dim_latent']
        seed = config['seed']
        # save_dir=config['save_dir']+'_seed='+str(seed)+'/n_lay='+str(n_layers)+'_'+'n_hid='+str(dim_hiddens)+'/'
        # save_dir=config['save_dir']+'/AE_n_latent='+str(dim_latent)+'/seed='+str(seed)+'/'

        # config['save_dir']=save_dir 
        if focus_group is not None:
            save_CCC_dir=dataset+'/'+focus_group+'/'
        else:
            save_CCC_dir=dataset+'/'
        time_float=config['Time']['float']
        time = np.linspace(time_float[0],time_float[-1],10080+1) # one time step is 1 hour
        time_GRN = np.linspace(time_float[0],time_float[-1],14+1)
        lag_list = np.linspace(-800,800,801).astype(int)  # 800: this needs to exceed range of `max_lag`*len(time). So we will conclude no causal effect if lag hits maximal (800 this case)
        max_lag = 0.06  
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    source_id_lst = []
    target_id_lst = []
    if n_fates is not None:
        for source_id in range(n_fates):
            for target_id in range(n_fates):
                source_id_lst.append(source_id)
                target_id_lst.append(target_id)

    return config,save_CCC_dir,time_scale,time_unit,n_fates,num_sample,time,time_GRN,time_float,lag_list,max_lag, source_id_lst,target_id_lst
def get_traj_data(save_dir):
    data=np.load(save_dir+'/traj.npz',allow_pickle=True)
    J_vx_x = data['J_vx_x']
    dg=data['dg']
    # g = data['g']
    z = data['z']
    vel = data['vel']
    vel_original = data['vel_original']
    z_original = data['z_original']
    z_original_norm = data['z_original_norm']
    fate_idx = data['fate_idx']
    z_fate=data['z_fate']
    pathway_list=list(data['pathway_list'])
    gene_list=list(data['gene_list'])
    time_GRN=data['time_GRN']
    time=data['time']
    cell_name= data['cell_name'].item()
    return z,z_original,z_original_norm,vel,vel_original,fate_idx,z_fate,J_vx_x,dg,pathway_list,gene_list,time_GRN,time,cell_name


def get_dataset_config(dataset,focus_group=None):
    if dataset=='EMT':
        fig_save_dir='Fig4/'
        fate_list=['Trajectory']
        # cell_name=None
        FDR=0.01
        species='human'
    elif dataset=='iPSC':
        fig_save_dir='Fig3/'
        fate_list=['End','Mes']
        # cell_name={1:'Mes', 0:'End'}
        FDR=0.05
        species='human'
    elif dataset == 'MOSTA':
        if focus_group is not None:
            fig_save_dir = 'Fig5/'+focus_group+'/'
        else:
            fig_save_dir = 'Fig5/'
        # fate_list = ['Adipose', 'Epithelial', 'Mesodermal', 'Nervous', 'Urogenital', 'Visceral']
        fate_list=None
        FDR=0.05
        species='mouse'
    os.makedirs(fig_save_dir,exist_ok=True)

    stat_test_map = {'ftest':'F-test',
                    'chi2':'Chi2 test',
                    }
    cellchat=pd.read_csv('src/CCC/CellChatDB/CellChatDB.ligrec.'+species+'.csv',index_col=0)
    signal_type_map = dict(zip(cellchat.iloc[:,2], cellchat.iloc[:,3]))
    return fig_save_dir,fate_list,FDR,species,stat_test_map, cellchat,signal_type_map

    