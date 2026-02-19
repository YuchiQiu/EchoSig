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
