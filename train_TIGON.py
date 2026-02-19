import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, sys
import scanpy as sc
import EchoSig
import EchoSig.TIGON2
import EchoSig.TIGON2.trainer
import random
from scipy.sparse import issparse
from EchoSig.utility import read_config_file


if __name__=="__main__":
    dataset = sys.argv[1]
    # dataset = 'iPSC'
    config = read_config_file('configs/config_'+dataset+'.yaml')
    seed=config['seed']
    save_dir = config['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    n_genes=config['preprocessing']['n_genes']
    dim_latent = config['AE']['model']['dim_latent']
    device = config['device']
    adata = sc.read_h5ad(config['input_h5ad'])

    # sc.pp.highly_variable_genes(adata,
    #                             n_top_genes=n_genes,
    #                             flavor='seurat_v3')
    # adata = adata[:, adata.var['highly_variable']].copy()
    Time = adata.obs['Time'].values
    time_str = config['Time']['str'] #['0d', '8h', '1d', '3d', '7d']
    time_float = config['Time']['float'] #[0, 0.1, 0.3, 0.9, 2.1]
    time_map = {time_str[i]:time_float[i] for i in range(len(time_float))}
    Time_float = [time_map[Time[i]] for i in range(len(Time))]

    adata.obs['Time'] = pd.Categorical(adata.obs['Time'], categories=time_str, ordered=True)
    if issparse(adata.X):
        X=np.array(adata.X.todense())
    else:
        X=adata.X
    
    model_ae = EchoSig.AE.create_AE(n_genes=n_genes,
                    **config['AE']['model']
                    )
    trainer_ae = EchoSig.AE.Trainer(model=model_ae,X=X,device=device,
                            **config['AE']['trainer'])
    ## Train AE model and save the embedding
    trainer_ae.train()
    embedding=model_ae.emb(X)
    adata.obsm['ae']=embedding
    adata.write(os.path.join(save_dir,'adata.h5ad'))

    torch.save({'config':config},
            os.path.join(save_dir,'config.pth'))
    torch.save({'ae_state_dict':model_ae.state_dict(),
                'scaler':model_ae.scaler}, 
            os.path.join(save_dir,'AE.pth'))


    sigma = config['TIGON']['trainer']['sigma']
    
    data_train = []
    for k in range(len(time_str)):
        indices = [i for i, l in enumerate(Time) if l == time_str[k]]
        samples = embedding[indices,]
        samples = torch.from_numpy(samples).type(torch.float32).to(device)
        data_train.append(samples)

    ## Train TIGON2 model
    func = EchoSig.TIGON2.UOT(in_out_dim=dim_latent, 
                          **config['TIGON']['model'],
                          odesolver=config['TIGON']['odesolver'])
    trainer = EchoSig.TIGON2.trainer.Trainer(func,device=config['device'],**config['TIGON']['trainer'])
    time_dic =  trainer.train(data_train,time_float,save_dir=save_dir)

