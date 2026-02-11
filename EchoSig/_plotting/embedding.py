import anndata
import matplotlib.pyplot as plt
import matplotlib as mpl
import scanpy as sc
import numpy as np
# import networkx as nx
# import matplotlib.patches as mpatches
# import matplotlib.lines as mlines
# import plotly.graph_objects as go
import seaborn as sns
import pandas as pd
# from collections import defaultdict
import os
import itertools
from mpl_toolkits.mplot3d import Axes3D
# add new function to draw in the original space 

def embedding_trajectory(adata_input,trajectory=None,basis='ae',color='Time',traj_color='grey',traj_legend=None,components=None,n_components=2,
                         alpha=0.2,linestyle='-',save=None,**kwarg):
    """_summary_

    Args:
        adata (_type_): _description_
        trajectory (_type_): shape in (N_time,N_samples, N_dim)
        basis (str, optional): _description_. Defaults to 'ae'.
        color (str, optional): _description_. Defaults to 'Time'.
        traj_color (str or list, optional): _description_. Defaults to 'grey'. 
            1. (str), all trajectories take the same color
            2. (list), a list of str give each trajectory a color
        components (_type_, optional): 
            For instance, ['1,2', '2,3']. To plot all available components use. components='all'.
            Defaults to None.
        n_components (int, optional): _description_. Defaults to 2.
        alpha (float, optional): _description_. Defaults to 0.2.
        save (_type_, optional): _description_. Defaults to None.
    """
    adata=adata_input.copy()
    if trajectory is not None:
        # trajectory=trajectory.transpose(1,0,2)
        if not isinstance(traj_color,str):
            if (traj_legend is None):
                traj_legend=['_nolegend_' for _ in range(trajectory.shape[1])]
    # if trajectory.shape[2]==2:
    if adata.obsm[basis].shape[1]==2:
        axes = sc.pl.embedding(adata,#[adata.obs['Time'].isin(['1d','3d','7d'])],
                       basis=basis,color=color,show=False,**kwarg)
        if not isinstance(axes,np.ndarray):
            axes = [axes]
        if trajectory is not None:
            for i,ax in enumerate(axes):
                if isinstance(traj_color,str):
                    ax.plot(trajectory[:,:,0],trajectory[:,:,1],traj_color,linestyle=linestyle,alpha=alpha,
                            label=traj_legend)
                else:
                    for k,c in enumerate(traj_color):
                        ax.plot(trajectory[:,k,0],trajectory[:,k,1],traj_color[k],linestyle=linestyle[k],alpha=alpha,label=traj_legend[k])
    elif adata.obsm[basis].shape[1]>2 and components is not None:
    # elif trajectory.shape[2]>2 and components is not None:
        axes = sc.pl.embedding(adata,#[adata.obs['Time'].isin(['1d','3d','7d'])],
                basis=basis,color=color,components=components,show=False,**kwarg)
        if not isinstance(axes,np.ndarray):
            axes = [axes]        
        for i,ax in enumerate(axes):
            comps=list(map(int,components[i].split(',')))
            if trajectory is not None:
                if isinstance(traj_color,str):
                    ax.plot(trajectory[:,:,comps[0]-1],trajectory[:,:,comps[1]-1],
                            traj_color,alpha=alpha,linestyle=linestyle,
                            label=traj_legend)
                else:
                    for k,c in enumerate(traj_color):
                        ax.plot(trajectory[:,k,comps[0]-1],trajectory[:,k,comps[1]-1],
                                traj_color[k],alpha=alpha,linestyle=linestyle[k],
                                label=traj_legend[k])
    #     new_adata = anndata.AnnData(X=adata.X,
    #                                 obs=adata.obs,
    #                                 uns=adata.uns)
        
    #     new_adata.obs[basis] = adata.obs[ba]
    #     sc.pl.embedding(adata,#[adata.obs['Time'].isin(['1d','3d','7d'])],
    #             basis=basis,color=color,show=False)
        
    #     plt.plot(trajectory[:,:,0],trajectory[:,:,1],'grey',alpha=alpha)
    elif basis!='pca':
        # index_observed = np.arange(adata.obsm['ae'].shape[0])
        X=adata.obsm[basis]
        if 'pca' in adata.obsm:
            del adata.obsm['pca']
        # index_tractory = np.arange(z_reshape.shape[0])+index_observed[-1]
        # new_adata=anndata.AnnData(X=adata.obsm['ae'],
        #                           obs=adata.obs,
        #                           uns=adata.uns
        #                         )
        # new_adata.var_names = ['AE'+str(i) for i in range(new_adata.shape[1])]
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        # if trajectory is not None:
        #     z_reshape = trajectory.transpose(1,0,2).reshape(trajectory.shape[0]*trajectory.shape[1],trajectory.shape[2])
        #     z_scaled_reshape = scaler.transform(z_reshape)
        #     z_pca_reshape=pca.transform(z_scaled_reshape)
        #     z_pca = z_pca_reshape.reshape(trajectory.shape[1],trajectory.shape[0],n_components).transpose(1,0,2)
        # new_adata = anndata.AnnData(X=X_pca,
        #                             obs=adata.obs,
        #                             uns=adata.uns)
        # new_adata.obsm['X_pca']=X_pca
        # sc.pl.embedding(new_adata,#[adata.obs['Time'].isin(['1d','3d','7d'])],
        #         basis='pca',color=color,show=False)
        adata.obsm['X_pca']=X_pca
        axes=sc.pl.embedding(adata,#[adata.obs['Time'].isin(['1d','3d','7d'])],
                basis='pca',color=color,show=False,**kwarg)        
        if not isinstance(axes,np.ndarray):
            axes = [axes]
        if trajectory is not None:
            z_reshape = trajectory.transpose(1,0,2).reshape(trajectory.shape[0]*trajectory.shape[1],trajectory.shape[2])
            z_scaled_reshape = scaler.transform(z_reshape)
            z_pca_reshape=pca.transform(z_scaled_reshape)
            z_pca = z_pca_reshape.reshape(trajectory.shape[1],trajectory.shape[0],n_components).transpose(1,0,2)
            for i,ax in enumerate(axes):
                if isinstance(traj_color,str):
                    ax.plot(z_pca[:,:,0],z_pca[:,:,1],traj_color,alpha=alpha,linestyle=linestyle,
                            label=traj_legend)
                else:
                    for k,c in enumerate(traj_color):
                        ax.plot(z_pca[:,k,0],z_pca[:,k,1],traj_color[k],alpha=alpha,linestyle=linestyle[k],
                                label=traj_legend[k])
    else:
        raise NotImplementedError(f"This for {basis} is pca in this case has not been implemented yet.")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.tight_layout()
    
    if save is not None:
        plt.savefig(save)
    plt.show()


