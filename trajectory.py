import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, sys
import scanpy as sc
import EchoSig
import yaml
# from EchoSig.TIGON.models import *
import EchoSig.EchoSigTraj
import EchoSig.EchoSigTraj.trainer
import EchoSig.utility
import random
from scipy.sparse import issparse
from EchoSig.utility import read_config_file

if __name__=="__main__":
    seed=9
    config = read_config_file('configs/config_iPSC.yaml')
    config['seed']=seed
    save_dir=config['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    n_genes=config['preprocessing']['n_genes']
    dim_latent = config['AE']['model']['dim_latent']
    device = config['device']
    adata = sc.read_h5ad(config['input_h5ad'])


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
    trainer_ae.train()
    embedding=model_ae.emb(X)
    adata.obsm['ae']=embedding
    adata.write(os.path.join(save_dir,'adata.h5ad'))
    EchoSig.pl.embedding_trajectory(adata,
                            palette='Set2')
    torch.save({'config':config},
            os.path.join(save_dir,'config.pth'))
    torch.save({'ae_state_dict':model_ae.state_dict(),
                'scaler':model_ae.scaler}, 
            os.path.join(save_dir,'AE.pth'))

    sigma = config['EchoSigTraj']['trainer']['sigma']
    
    data_train = []
    for k in range(len(time_str)):
        indices = [i for i, l in enumerate(Time) if l == time_str[k]]
        samples = embedding[indices,]
        samples = torch.from_numpy(samples).type(torch.float32).to(device)
        data_train.append(samples)

    
    func = EchoSig.EchoSigTraj.UOT(in_out_dim=dim_latent, 
                          **config['EchoSigTraj']['model'],
                          odesolver=config['EchoSigTraj']['odesolver'])
    # options = EchoSig.TIGON.diffeq_args()
    trainer = EchoSig.EchoSigTraj.trainer.Trainer(func,device=config['device'],**config['EchoSigTraj']['trainer'])
    time_dic =  trainer.train(data_train,time_float,save_dir=save_dir)
    pd.Series(time_dic, name='time_seconds').rename_axis('metric').to_csv(
        os.path.join(save_dir, 'time_dic.csv')
    )
    print(time_dic)

