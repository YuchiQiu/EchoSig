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


def embedding_time_panels(adata_input, basis='ae', time_key='Time', color=None,
                          ncols=2, palette='Set2', point_size=5, alpha=0.5,
                          save=None):
    """Plot each time point in a separate panel using shared 2D coordinates."""
    adata = adata_input.copy()
    embedding = np.asarray(adata.obsm[basis])

    if embedding.shape[1] == 2:
        coordinates = embedding
    elif embedding.shape[1] > 2:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        coordinates = PCA(n_components=2).fit_transform(
            StandardScaler().fit_transform(embedding)
        )
    else:
        raise ValueError(f"Embedding '{basis}' must have at least two dimensions.")

    times = (list(adata.obs[time_key].cat.categories)
             if isinstance(adata.obs[time_key].dtype, pd.CategoricalDtype)
             else sorted(adata.obs[time_key].unique()))
    nrows = int(np.ceil(len(times) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows),
        sharex=True, sharey=True, squeeze=False
    )
    axes = axes.ravel()

    if color is not None:
        color_levels = (list(adata.obs[color].cat.categories)
                        if isinstance(adata.obs[color].dtype, pd.CategoricalDtype)
                        else sorted(adata.obs[color].dropna().unique()))
        colors = sns.color_palette(palette, n_colors=len(color_levels))
        color_map = dict(zip(color_levels, colors))

    for index, time in enumerate(times):
        ax = axes[index]
        selected = np.asarray(adata.obs[time_key] == time)
        if color is None:
            ax.scatter(
                coordinates[selected, 0], coordinates[selected, 1],
                s=point_size, alpha=alpha, color=sns.color_palette(palette)[index]
            )
        else:
            for level in color_levels:
                cells = selected & np.asarray(adata.obs[color] == level)
                if cells.any():
                    ax.scatter(
                        coordinates[cells, 0], coordinates[cells, 1],
                        s=point_size, alpha=alpha, color=color_map[level],
                        label=str(level)
                    )
        ax.set_title(f"{time} (n={selected.sum():,})")
        ax.set_xlabel(f"{basis} PC1" if embedding.shape[1] > 2 else f"{basis} 1")
        ax.set_ylabel(f"{basis} PC2" if embedding.shape[1] > 2 else f"{basis} 2")

    for index in range(len(times), len(axes)):
        axes[index].set_visible(False)

    if color is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title=color, loc='center left',
                   bbox_to_anchor=(1, 0.5), frameon=False)

    fig.tight_layout()
    if save is not None:
        fig.savefig(save, bbox_inches='tight')
    plt.show()
    return axes[:len(times)]

def embedding_trajectory(adata_input,trajectory=None,basis='ae',color='Time',traj_color='grey',traj_legend=None,components=None,n_components=2,
                         alpha=0.2,alpha_pts=0.5,linestyle='-',save=None,**kwarg):
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
                       basis=basis,color=color,show=False,alpha=alpha_pts,**kwarg)
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
                basis=basis,color=color,components=components,alpha=alpha_pts,show=False,**kwarg)
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
                basis='pca',color=color,alpha=alpha_pts,show=False,**kwarg)        
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



def embedding_trajectory_ori_space(adata_input, trajectory=None, basis='ae', color='Time', 
                                    traj_color='grey', traj_legend=None, alpha=0.2, 
                                    linestyle='-', ncols=2, save=None, **kwarg):
    """
    Plot embedding trajectory on original high-dimensional space by projecting onto all 2D axis pairs.

    Parameters:
        adata_input (AnnData): AnnData object with high-dimensional embedding in .obsm[basis]
        trajectory (np.ndarray): Trajectory array of shape (T, K, D)
        basis (str): Key for the embedding in adata.obsm (e.g., 'ae')
        color (str): Column in adata.obs to color cells
        traj_color (str or list): Color(s) for trajectory lines
        traj_legend (list or str): Labels for trajectories
        alpha (float): Transparency for trajectory lines
        linestyle (str or list): Linestyle(s) for trajectory
        ncols (int): Number of subplot columns
        save (str): Path to save the figure
        **kwarg: Extra keyword arguments for seaborn scatterplot
    """
    adata = adata_input.copy()
    emb = adata.obsm[basis]
    D = emb.shape[1]
    assert D >= 2, "Embedding dimension must be at least 2."

    dim_pairs = list(itertools.combinations(range(D), 2))
    n_panels = len(dim_pairs)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten() if n_panels > 1 else [axes]

    for i, (x_idx, y_idx) in enumerate(dim_pairs):
        ax = axes[i]
        sns.scatterplot(
            x=emb[:, x_idx], y=emb[:, y_idx], 
            hue=adata.obs[color], ax=ax, s=2, alpha=0.3, **kwarg
        )
        ax.set_xlabel(f'{basis}_{x_idx}')
        ax.set_ylabel(f'{basis}_{y_idx}')
        ax.set_title(f'{basis} axis {x_idx} vs {y_idx}')
        ax.legend(loc='best',markerscale=5, frameon=False)

        if trajectory is not None:
            if isinstance(traj_color, str):
                ax.plot(trajectory[:, :, x_idx], trajectory[:, :, y_idx],
                        color=traj_color, linestyle=linestyle, alpha=alpha,
                        label=traj_legend)
            else:
                for k, c in enumerate(traj_color):
                    ax.plot(trajectory[:, k, x_idx], trajectory[:, k, y_idx],
                            color=c, linestyle=linestyle[k] if isinstance(linestyle, list) else linestyle,
                            alpha=alpha, label=traj_legend[k] if traj_legend is not None else None)

    # Hide any unused subplots
    for j in range(n_panels, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    plt.show()


def embedding_trajectory_3d(adata_input, trajectory=None, basis='ae', color='Time',
                            traj_color='grey', traj_legend=None, alpha=0.2,
                            linestyle='-', elev=20, azim=135,
                            save=None, **kwarg):
    """
    Plot 3D embedding trajectory.

    Parameters:
        adata_input (AnnData): AnnData object with .obsm[basis] of at least 3D
        trajectory (np.ndarray): Trajectory array of shape (T, K, D)
        basis (str): Key for the embedding in adata.obsm (e.g., 'ae')
        color (str): Column in adata.obs to color cells
        traj_color (str or list): Color(s) for trajectory lines
        traj_legend (list or str): Labels for trajectories
        alpha (float): Transparency for trajectory lines
        linestyle (str or list): Linestyle(s) for trajectory
        elev (float): Elevation angle for 3D view
        azim (float): Azimuth angle for 3D view
        save (str): Path to save the figure
        **kwarg: Extra keyword arguments for scatter plot
    """
    adata = adata_input.copy()
    emb = adata.obsm[basis]
    assert emb.shape[1] >= 3, f"Embedding dimension must be at least 3, got {emb.shape[1]}"

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)

    # Convert categorical color to numeric codes and map to a colormap
    colors = adata.obs[color]
    if colors.dtype.name == 'category' or colors.dtype == object:
        colors = colors.astype('category')
        categories = colors.cat.categories
        color_codes = colors.cat.codes
        sc = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2],
                        c=color_codes, cmap='Set2', s=1, alpha=0.3, **kwarg)

        # Add colorbar with labels
        mappable = plt.cm.ScalarMappable(cmap='Set2')
        mappable.set_array([])
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_ticks(np.linspace(0, 1, len(categories)))
        cbar.set_ticklabels(categories)
        cbar.ax.tick_params(labelsize=8)
    else:
        sc = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2],
                        c=colors, cmap='viridis', s=1, alpha=0.3, **kwarg)
        plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)

    # Plot trajectory
    if trajectory is not None:
        if isinstance(traj_color, str):
            for k in range(trajectory.shape[1]):
                ax.plot(trajectory[:, k, 0], trajectory[:, k, 1], trajectory[:, k, 2],
                        color=traj_color, linestyle=linestyle, alpha=alpha,
                        label=traj_legend if traj_legend else None)
        else:
            for k, c in enumerate(traj_color):
                ax.plot(trajectory[:, k, 0], trajectory[:, k, 1], trajectory[:, k, 2],
                        color=c, linestyle=linestyle[k] if isinstance(linestyle, list) else linestyle,
                        alpha=alpha, label=traj_legend[k] if traj_legend else None)

    ax.set_xlabel(f'{basis}_0')
    ax.set_ylabel(f'{basis}_1')
    ax.set_zlabel(f'{basis}_2')
    ax.set_title(f'3D Trajectory on {basis} space')

    if traj_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches='tight')
    plt.show()


def embedding_fate_trajectory(
    adata_input,
    trajectory,           # (n_time, n_traj, n_dim)
    fate_labels,          # len = n_traj
    basis='ae',
    color='annotation_group',
    components=None,      # 显式给如 [(0,1),(1,2),(2,3)] 则不做 PCA
    n_components=2,       # 仅在 components=None 且 ndim>2 时使用
    alpha=0.35,
    linestyle='-',
    save=None,
    nan_cell_color="#D3D3D3",
    nan_traj_color="#000000",
    label2color=None,     # <--- 新增：若不传会按当前 cats/uns 组装
    force_new_palette=False  # True 则忽略 uns 重建颜色（与 _prepare_palette 一致）
):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA as _PCA
    from matplotlib.lines import Line2D

    adata = adata_input.copy()
    if basis not in adata.obsm_keys():
        raise KeyError(f"Embedding '{basis}' not found in adata.obsm.")

    # ------- 类别与顺序 -------
    if not pd.api.types.is_categorical_dtype(adata.obs[color]):
        cats = list(pd.unique(adata.obs[color].dropna()))
        adata.obs[color] = pd.Categorical(adata.obs[color], categories=cats, ordered=True)
    cats = list(adata.obs[color].cat.categories)
    colors_key = f"{color}_colors"

    # ------- 颜色：构建 idx2color & label2color -------
    if label2color is None:
        if (not force_new_palette) and (colors_key in adata.uns) and (len(adata.uns[colors_key]) == len(cats)):
            idx2color = list(adata.uns[colors_key])
        else:
            base = mpl.cm.get_cmap('tab20')
            idx2color = [mpl.colors.to_hex(base(i % base.N)) for i in range(len(cats))]
            adata.uns[colors_key] = idx2color
        label2color = {lab: idx2color[i] for i, lab in enumerate(cats)}
    else:
        # 若传入字典，派生 idx2color 仅用于点云（按 cats 顺序取）
        idx2color = [label2color.get(lab, nan_cell_color) for lab in cats]
        adata.uns[colors_key] = idx2color  # 同步保存

    # 点云颜色按 codes→idx2color（和 cats 完全对齐）
    cell_codes = adata.obs[color].cat.codes
    cell_colors = [idx2color[i] if i >= 0 else nan_cell_color for i in cell_codes]
    # 轨迹颜色直接 label→color（避免任何顺序/类型差异）
    traj_colors = []
    for lbl in fate_labels:
        if lbl in label2color:
            traj_colors.append(label2color[lbl])
        elif str(lbl) in label2color:
            traj_colors.append(label2color[str(lbl)])
        else:
            print(f"Warning: No color found for label '{lbl}' (type: {type(lbl)})")
            traj_colors.append(nan_traj_color)

    # ------- 坐标准备（是否做 PCA） -------
    X = adata.obsm[basis]
    ndim = X.shape[1]
    scaler = None
    pca = None
    if components is not None:
        basis_used = basis
        X_used = X
    else:
        if ndim == 2:
            basis_used = basis
            X_used = X
            components = [(0, 1)]
        elif ndim > 2:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = _PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            adata.obsm['X_pca'] = X_pca
            basis_used = 'pca'
            X_used = X_pca
            components = [(0, 1)]
        else:
            raise ValueError("Embedding must have >=2 dimensions.")

    # ------- 轨迹投影 -------
    n_time, n_traj, tdim = trajectory.shape
    if basis_used == 'pca':
        if scaler is None or pca is None:
            raise RuntimeError("Scaler/PCA not initialized; unexpected state.")
        z = trajectory.reshape(n_time * n_traj, tdim)
        z_scaled = scaler.transform(z)
        z_pca = pca.transform(z_scaled)
        z_projected = z_pca.reshape(n_time, n_traj, -1)
    else:
        z_projected = trajectory

    # ------- 线型 -------
    if isinstance(linestyle, str):
        linestyle = [linestyle] * n_traj
    elif isinstance(linestyle, (list, tuple)):
        if len(linestyle) != n_traj:
            raise ValueError("Length of `linestyle` must equal number of trajectories.")
    else:
        raise TypeError("`linestyle` must be a string or a list/tuple of strings.")

    # ------- 维度检查 -------
    ncols = X_used.shape[1]
    for (cx, cy) in components:
        if cx >= ncols or cy >= ncols:
            raise IndexError(
                f"components includes ({cx},{cy}) but embedding has only {ncols} columns in '{basis_used}'."
            )

    # ------- 绘图 -------
    for comp_x, comp_y in components:
        fig, ax = plt.subplots(figsize=(6.2, 5.4))

        ax.scatter(
            X_used[:, comp_x],
            X_used[:, comp_y],
            c=cell_colors,
            s=2,
            alpha=0.10,
            rasterized=True
        )

        for k in range(n_traj):
            ax.plot(
                z_projected[:, k, comp_x],
                z_projected[:, k, comp_y],
                color=traj_colors[k],
                alpha=alpha,
                linestyle=linestyle[k],
                linewidth=1.4
            )

        legend_handles = [
            Line2D([0], [0], color=idx2color[i], marker='o', markersize=6, lw=2, label=lab)
            for i, lab in enumerate(cats)
        ]
        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, ncol=1)

        ax.set_xlabel(f"{basis_used}:{comp_x + 1}")
        ax.set_ylabel(f"{basis_used}:{comp_y + 1}")
        ax.set_title("Embedding with Fate Trajectories", pad=8)
        plt.tight_layout()

        if save is not None:
            prefix, ext = os.path.splitext(save)
            suffix = f"_{comp_x + 1}_{comp_y + 1}"
            plt.savefig(prefix + suffix + (ext if ext else ".png"), dpi=300, bbox_inches='tight')
        plt.show()
