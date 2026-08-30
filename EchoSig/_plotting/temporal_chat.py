import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from .base import *
from .hierarchy_graph import *
palette = list(sns.color_palette("tab10") \
                + sns.color_palette("Set1")\
                + sns.color_palette("Set2")\
                + sns.color_palette('Set3')\
                + sns.color_palette('Dark2')\
                + sns.color_palette('Pastel1')\
                + sns.color_palette('Pastel2'))
# def get_cell_pair_name(cell_pair,cell_name):    
#     if cell_name is not None:
#         cell_pair_str = cell_name[cell_pair[0]]+'-'+cell_name[cell_pair[1]]
#     else:
#         cell_pair_str = str(cell_pair[0])+'-'+str(cell_pair[1])
#     return cell_pair_str

# def plot_delay_single(grapher,dataset,source_id,target_id,path,cell_pair,cell_name,
#                       z_original_norm,fate_idx,gene_list,time,time_unit,
#                       data_signal,FDR=0.05,color='blue',
#                       xlabel=None,ylabel=None,xtick=False,ytick=False,ax=None):
#     cell_pair_str=get_cell_pair_name(cell_pair,cell_name,)
#     df_CCC = pd.read_csv(dataset+'/'+str(source_id)+'to'+str(target_id)+'/'+'results.csv',
#                     index_col=0)
#     grapher.add_CCC_data(df_CCC,(source_id,target_id),
#                         pathway=[path],FDR=FDR)
#     if ax is None:
#         fig,ax=plt.subplot()
#     if len(grapher.CCC_data[cell_pair])>0:
#         L_list=grapher.CCC_data[cell_pair]['L'].values
#         Ligands = np.mean(z_original_norm[fate_idx[source_id],:,:],axis=0)[:,[gene_list.index(l) for l in L_list]]
#         L=np.mean(Ligands,axis=1)

#         lag=np.mean(grapher.CCC_data[cell_pair]['SPT'].values)
#         strength = np.sum(grapher.CCC_data[cell_pair]['stat '+grapher.test])
#         max_signal_time = time[np.argmax(L)]
        
#         ax.plot(time,L,color=color,label=path)
#         x_start = max_signal_time
#         x_end = max_signal_time + lag
#         if x_end>time[-1]:
#             x_end=time[-1]
#             x_start=x_end-lag
#         data_signal[cell_pair][path]=np.array([x_start,x_end,strength])

#         mask = (time >= x_start) & (time <= x_end)
#         ax.fill_between(time[mask], 0, L[mask], color=color, alpha=0.2)
#         ax.text((x_start + x_end)/2, max(L[mask].min()-0.2,0.05), f'{lag:.1f}'+time_unit, 
#                 ha='center', va='top', fontsize=10)
#         ax.set_ylim(0,1)
#         if not ytick:
#             ax.set_yticks([])
#         if not xtick:
#             ax.set_xticks([])
#         ax.set_xlabel('time (h)')
#         if ylabel is None:
#             ax.set_ylabel(path+ ' '+cell_pair_str)
#         else:
#             ax.set_ylabel(ylabel, rotation=0, labelpad=20,ha='right')
#         if xlabel is None:
#             ax.set_xlabel('time (h)')
#         else:
#             ax.set_xlabel(xlabel)
#         return True
#     else:
#         return False


# def plot_delay(dataset,cell_name,cell_pairs,cell_list_all,pathway_list,
#                z_original_norm,fate_idx,gene_list,time,
#                fig_save_dir=None,xticks=None,time_unit='day',FDR=0.05,figsize=(4,4),nrows=8,test='ftest'):
#     data_signal={}
#     color_id=0
#     palette_map={}
#     for cell_pair in cell_pairs:
#         data_signal[cell_pair]={}
#         cell_pair_str = get_cell_pair_name(cell_pair,cell_name)
#         target_gene=None
#         grapher = Hierarchy_L_R_SPG_SDG(test=test,cell_list_all=cell_list_all)
#         source_id=cell_pair[0]
#         target_id=cell_pair[1]
#         fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=figsize, 
#                                 gridspec_kw={'hspace': 0})
#         fig_id=0
        
#         for path in pathway_list:
#             # save_sub_dir=dataset+'/'+str(source_id)+'to'+str(target_id)+'/'
#             if not path in palette_map:
#                 # palette_map[path]=palette[color_id]
#                 color=palette[color_id]
#                 # pop=True
#             else:
#                 color=palette_map[path]
#                 # pop = False

#             output=plot_delay_single(grapher,dataset,source_id,target_id,path,cell_pair,cell_name,
#                                      z_original_norm,fate_idx,gene_list,time,
#                                      data_signal,FDR,color,xlabel='',ylabel=path,xtick=True,
#                                      ax=axes[fig_id],)
                 

#             if output:
#                 fig_id+=1 
#                 color_id+=1 
#                 palette_map[path]=color
#             # else: 
#             #     if pop:
#             #         palette_map.pop(path,None)
#         for i in range(fig_id-1):
#             axes[i].set_xticklabels([])
#         for i in range(fig_id, len(axes)):
#             fig.delaxes(axes[i])
#         if xticks is None:
#             xticks = axes[fig_id-1].get_xticks()
#         axes[fig_id-1].set_xticks(xticks)
#         axes[fig_id - 1].set_xticklabels([str(x // 24) for x in xticks])
#         axes[fig_id-1].set_xlabel('time ('+time_unit+')')
#         axes[0].set_title(cell_pair_str)
#         if fig_save_dir is not None:
#             os.makedirs(fig_save_dir+'/delay/',exist_ok=True)
#             plt.savefig(fig_save_dir+'/delay/'+cell_pair_str+'.pdf')
#         plt.show()
#     return data_signal,palette_map




def draw_self_chat(g,
                    arrowstyle,
                    direct='out',
                    height=1,
                    min_height=0.5,
                    alpha=0.6,
                    width_scale=1.,
                    arrowsize=10,
                    ax=None):
    """_summary_

    Args:
        g (_type_): _description_
        arrowstyle (_type_): _description_
        r (_type_, optional): _description_. Defaults to 1..
        direct (str, optional): direction of the curve. 
            Options: 'out' outward: with the same direction of the curve
                        'in' inward: opposite direction with the curve
            Defaults to 'out'.
        height (float,optional): 
    """
    from matplotlib.path import Path
    import matplotlib.patches as patches
    if ax is None:
        fig, ax = plt.subplots()
    # if height is None:
    #     height = 2 if direct == 'out' else 2
    # if arrowstyle=='-[':
    #     arrowsize=15
    # elif arrowstyle=='-|>':
    #     arrowsize=20


    for u,v,atr in g.edges(data=True):
        u=np.array(u)
        v=np.array(v)
        l=np.linalg.norm(u-v)
        mid = np.array(atr['mid'])
        orth_vec1 = mid - 0.5*(u+v)
        diff = v-u
        orth_vec2 = np.array([-diff[1],diff[0]])
        if np.linalg.norm(orth_vec1)>np.linalg.norm(orth_vec2):
            orth_vec = orth_vec1
        else:
            orth_vec = orth_vec2
        scale_height = max(min_height,height*l)
        if direct=='out':
            add_pt1 = u+orth_vec*scale_height
            add_pt2 = v+orth_vec*scale_height
        elif direct=='in':
            add_pt1 = u-orth_vec*scale_height
            add_pt2 = v-orth_vec*scale_height
        verts = [u, add_pt1,add_pt2, v]
        # plt.plot(u[0],u[1],'.',color='black')
        # plt.plot(v[0],v[1],'.',color='black')
        # plt.plot(add_pt[0],add_pt[1],'.',color='black')
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4,Path.CURVE4,]
        path = Path(verts, codes)
        patch = patches.FancyArrowPatch(
            path=path,
            lw=np.log10(1+atr['strength'])*width_scale,
            arrowstyle=arrowstyle,
            color=atr['color'],
            alpha=alpha,
            mutation_scale=arrowsize  # arrowsize in draw_networkx_edges()
        )
        ax.add_patch(patch)



def temporal_talks(data_signal,traj,traj_label,time,
                   palette_map,
                   linestyle='-',
                   linecolor=None,
                   width_scale=4,
                   min_height=0.5,
                   height=1.,
                   direct_dic=None,
                   figsize=(6,6)):
    """_summary_

    Args:
        traj (numpy.array): (num_fate,num_time,num_components)
    """
    if isinstance(linestyle,str):
        linestyle=[linestyle]*len(traj_label)
    if linecolor is None:
        linecolor = sns.color_palette("tab10")
    elif isinstance(linecolor,str):
        linecolor=[linecolor]*len(traj_label)
    fig, ax = plt.subplots(figsize=figsize)
    # plt.plot(time, np.zeros(time.shape),label = 'End',color='black',)
    # plt.plot(time, np.ones(time.shape),'--',label = 'Mes',color='black')
    for fate_id in range(traj.shape[0]):
        ax.plot(traj[fate_id,:,0],traj[fate_id,:,1], 
                 label = traj_label[fate_id],
                 linestyle=linestyle[fate_id],
                 color=linecolor[fate_id])
    if direct_dic is None:
         direct_dic={}
         for cell_pair, _ in data_signal.items():
             direct_dic[cell_pair]='out'
    
    G = {}
    for cell_pair, signal_dict in data_signal.items():
        G[cell_pair]={}
        G[cell_pair]['act']=nx.DiGraph()
        G[cell_pair]['inh']=nx.DiGraph()
        for signal, (start, end,strength) in signal_dict.items():
            idx_start=np.argmin(np.abs(time-start))
            idx_end=np.argmin(np.abs(time-end))

            start = tuple(traj[cell_pair[0],idx_start,0:2])
            end = tuple(traj[cell_pair[1],idx_end,0:2])
            if cell_pair[0]==cell_pair[1]:
                mid = tuple(traj[cell_pair[1],int(0.5*(idx_start+idx_end)),0:2])
            else:
                mid=tuple(0.5*traj[cell_pair[0],idx_start,0:2]+0.5*traj[cell_pair[1],idx_end,0:2])
            if strength>0:
                G[cell_pair]['act'].add_edge(start, end, 
                                            color=palette_map[str(signal)],
                                            strength=strength,
                                            mid=mid)
            else:
                G[cell_pair]['inh'].add_edge(start, end, 
                                            color=palette_map[str(signal)],
                                            strength=-strength,
                                            mid=mid)                
    # Get edge colors
    # edge_colors = [G[u][v]['color'] for u, v in G.edges()]

    # Draw only edges
    # pos = {node: node for node in G.nodes()}
    arrow_dic = {'act':'->','inh':'-['}
    arrow_size_dic={'act':10,'inh':5}
    # ax = plt.gca()
    for cell_pair in G.keys():
        for reg in ['act','inh']:
            g = G[cell_pair][reg]
            if len(g)>0:
                pos = {node: node for node in g.nodes()}
                edge_colors = [g[u][v]['color'] for u, v in g.edges()]
                edge_widths = [np.log10(1+g[u][v]['strength'])*width_scale for u, v in g.edges()]
                
                if cell_pair[0]!=cell_pair[1]:
                    connectionstyle = "arc3,rad=0.2"
                else:
                    connectionstyle = 'arc3,rad=-0.6'
                # elif y_start==0:
                #     connectionstyle = "arc3,rad=0.6"
                # elif y_start == 1:
                #     connectionstyle = "arc3,rad=0.2"
                if cell_pair[0]!=cell_pair[1]:
                    nx.draw_networkx_edges(g, 
                                        pos, 
                                        edge_color=edge_colors, 
                                        arrows=True, 
                                        arrowstyle=arrow_dic[reg],
                                        arrowsize=arrow_size_dic[reg],
                                        connectionstyle=connectionstyle,
                                        width=edge_widths,
                                        node_size=10,
                                        alpha=0.6,
                                        ax=ax)

                else:
                    draw_self_chat(g, 
                                arrowstyle=arrow_dic[reg],
                                width_scale=width_scale,
                                height=height,
                                min_height=min_height,
                                # direct='out',
                                arrowsize=arrow_size_dic[reg],
                                direct=direct_dic[cell_pair],
                                ax=ax)
    return ax,G
