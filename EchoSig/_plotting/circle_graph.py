import anndata
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import networkx as nx
import seaborn as sns
import pandas as pd
from collections import defaultdict
import os
from .base import BaseGraph
from typing import Optional
from matplotlib.path import Path as MplPath 
import matplotlib.patches as patches
import matplotlib.lines as mlines
# from .chord_diagram import *

def get_edge_color(edge, node_colors, edge_color_cross_layers):
    cell_0 = int(edge[0].split(' ')[0])
    cell_1 = int(edge[1].split(' ')[0])
    cell_pair=(cell_0,cell_1)
    if cell_0==cell_1:
        color=node_colors[edge[0]]
    else:
        color=edge_color_cross_layers[cell_pair][edge[0]]
    return color
def normalize_vector(vector: np.array, normalize_to: float) -> np.array:
    """Make `vector` norm equal to `normalize_to`

    vector: np.array
        Vector with 2 coordinates
    normalize_to: float
        A norm of the new vector

    Returns
    -------
    Vector with the same direction, but length normalized to `normalize_to`
    """

    vector_norm = np.linalg.norm(vector)

    return vector * normalize_to / vector_norm
def orthogonal_vector(point: np.array, width: float,
                      normalize_to: Optional[float] = None) -> np.array:
    """Get orthogonal vector to a `point`

    point: np.array
        Vector with x and y coordinates of a point
    width: float
        Distance of the x-coordinate of the new vector from the `point` (in orthogonal direction)
    normalize_to: Optional[float] = None
        If a number is provided, normalize a new vector length to this number

    Returns
    -------
    Array with x and y coordinates of the vector, which is orthogonal to the vector
    from (0, 0) to the `point`
    """
    EPSILON = 0.000001

    x = width
    y = -x * point[0] / (point[1] + EPSILON)

    ort_vector = np.array([x, y])

    if normalize_to is not None:
        ort_vector = normalize_vector(ort_vector, normalize_to)

    return ort_vector
def draw_self_loop(
        point: np.array,
        arrowstyle="-|>",
        arrowsize=10,
        ax: Optional[plt.Axes] = None,
        padding: float = 1.5,
        width: float = 0.3,
        plot_size: int = 10,
        linewidth=0.2,
        alpha=0.7,
        color: str = "pink"
):
    """Draw a loop from `point` to itself

    !Important! By "center" we assume a (0, 0) point. If your data is centered around a different
    point, it is strongly recommended to center it around zero. Otherwise, you will probably
    get ugly plots

    Parameters
    ----------
    point: np.array
        1D array with 2 coordinates of the point. Loop will be drawn from and to these coordinates.
    padding: float = 1.5
        Controls how the distance of the loop from the center. If `padding` > 1, the loop will be
        from the outside of the `point`. If `padding` < 1, the loop will be closer to the center
    width: float = 0.3
        Controls the width of the loop
    linewidth: float = 0.2
        Width of the line of the loop
    ax: Optional[matplotlib.pyplot.Axes]:
        Axis on which to draw a plot. If None, a new Axis is generated
    plot_size: int = 7
        Size of the plot sides in inches. Ignored if `ax` is provided
    color: str = "pink"
        Color of the arrow

    Returns
    -------
    Matplotlib axes with the self-loop drawn
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(plot_size, plot_size))

    point_with_padding = padding * point

    ort_vector = orthogonal_vector(point, width, normalize_to=width)

    first_anchor = ort_vector + point_with_padding
    second_anchor = -ort_vector + point_with_padding

    verts = [point, first_anchor, second_anchor, point]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]

    path = MplPath(verts, codes)
    patch = patches.FancyArrowPatch(
        path=path,
        lw=linewidth,
        arrowstyle=arrowstyle,
        color=color,
        alpha=alpha,
        mutation_scale=arrowsize  # arrowsize in draw_networkx_edges()
    )
    ax.add_patch(patch)

    return ax
def draw_graph_edge(graph, pos, edge, edge_weight, ax, color, arc_radius=0.2,arrowstyle="-|>",arrowsize=10,alpha=0.7):
    """Draw the given edge of the network"""

    nx.draw_networkx_edges(
        graph,
        pos=pos,
        width=edge_weight,
        edgelist=[edge],
        alpha=alpha,
        edge_color=color,
        ax=ax,
        arrowsize=arrowsize,
        connectionstyle=f"arc3,rad={arc_radius}",
        # node_size=1000
        arrowstyle=arrowstyle,
    )

    return ax
    # """Create a dictionary with the weights of the graph edges

    # Parameters
    # ----------
    # graph: nx.Graph
    #     Graph, which edges' weights you want to extract
    # weight_key: str = "weight"
    #     What property of the edges to use as a weight. Other functions assume that this is a number

    # Returns
    # -------
    # Dictionary, where keys are edges and values are their weights
    # """
    # return {edge: graph.edges[edge][weight_key] for edge in graph.edges}

def draw_graph_edges(graph,edges,pos, node_colors,edge_color_cross_layers,ax,width_scale,width_thred=0.01,arrowstyle="-|>",alpha=0.7,arc_radius=0.2):
    """Draw graph edges so that edges in the opposite directions look differently

    graph: nx.Graph
        Graph, edges of which you want to draw
    pos: dict
        Dictionary, where keys are nodes and values are their positions. Can be obtained
        through networkx layout algorithms (e. g. nx.circular_layout())
    node_colors: dict
        Dictionary, node color is given. edge color will share the same color with starting node
    ax: plt.Axes
        Axis on which draw the edges

    Returns
    -------
    Axis with the edges drawn
    """
    if arrowstyle=='-[':
        arrowsize=10
    elif arrowstyle=='-|>':
        arrowsize=10
    # edge_weights = graph_edges_weights(graph)

    edges_to_draw = set(list(edges))
    edges_copy=edges.copy()
    for edge in edges:
        if edge not in edges_to_draw:
            continue

        if edge[0] == edge[1]:  # By default, networkx doesn't draw self loops correctly
            w=0 if width_scale * abs(edges[edge]) < width_thred else width_scale * abs(edges[edge])
            if w>0:
                draw_self_loop(point=np.array(pos[edge[0]]), 
                            ax=ax, 
                            linewidth=w,
                            arrowstyle=arrowstyle,
                            arrowsize=arrowsize,
                            color=node_colors[edge[0]],
                            alpha=alpha,
                            )
            edges_to_draw.remove(edge)
            edges_copy.pop(edge)
            continue

    #     nx.draw_networkx_edges(self.G,self.pos,#edge_labels=self.edge_LR_lag,
    #                            arrowstyle='-[',
    #                                  edgelist=list(self.edge_LL_negative.keys()),
    #                                  edge_color=[self.node_colors.get(u) for (u,v) in list(self.edge_LL_negative.keys())],
    #                                  ax=ax,
    #                                  width=[-width_scale*self.edge_LL_negative[(u, v)] for (u, v) in list(self.edge_LL_negative.keys())],
    #                                  alpha=0.7)
    
    # width = [0 if width_scale * abs(edges[edge]) < width_thred else width_scale * abs(edges[edge]) for edge in edges_copy.keys()]
    
    #### exclude 0 width edges
    filtered_edges = []
    filtered_widths = []
    for edge in edges_copy.keys():
        w = width_scale * abs(edges_copy[edge])
        if w >= width_thred:
            filtered_edges.append(edge)
            filtered_widths.append(w)

    edge_color=[get_edge_color(edge, node_colors, edge_color_cross_layers) for edge in filtered_edges]

    nx.draw_networkx_edges(
        graph,
        pos=pos,
        # width=width,
        # edgelist=list(edges_copy.keys()),
        width=filtered_widths,
        edgelist=filtered_edges,
        alpha=alpha,
        edge_color=edge_color,
        ax=ax,
        arrowsize=arrowsize,
        connectionstyle=f"arc3,rad={arc_radius}",
        # node_size=1000
        arrowstyle=arrowstyle,
    )        
        # color=get_edge_color(edge, node_colors, edge_color_cross_layers)
        # draw_graph_edge(graph, 
        #                 pos, 
        #                 edge, 
        #                 edge_weight=width_scale*edges[edge], 
        #                 ax=ax, 
        #                 color=color,
        #                 arrowstyle=arrowstyle,
        #                 arrowsize=arrowsize,
        #                 alpha=alpha)
        # edges_to_draw.remove(edge)

        # # Edges between the same vertices look confusing, if they have the same style
        # # So we draw such edges with different colors and curvature
        # reverse_edge = (edge[1], edge[0])
        
        # if reverse_edge in edges and reverse_edge in edges_to_draw:
        #     color=get_edge_color(reverse_edge, node_colors, edge_color_cross_layers)
        #     draw_graph_edge(
        #         graph,
        #         pos,
        #         reverse_edge,
        #         edge_weight=width_scale*edges[reverse_edge],
        #         ax=ax,
        #         color=color,
        #         arrowstyle=arrowstyle,
        #         arrowsize=arrowsize,
        #         alpha=alpha,
        #         # arc_radius=0.2
        #     )
        #     edges_to_draw.remove(reverse_edge)

    return ax


class BaseCircleGraph(BaseGraph):
    def __init__(self,test='ftest',cell_name=None):
         super().__init__(test,cell_name=cell_name)
    def edge_color_cross_layers(self,palette_dic):
        if palette_dic is not None:
            # cell_pair=(0,1)
            # cell_id=0
            palette={}
            cell_pairs=list(palette_dic.keys())
            for cell_pair in cell_pairs:
                cell_id=cell_pair[0]
                tmp_node_list= [itm for node_type in self.signal_net for itm in self.node_list[node_type][cell_id]]
                palette[cell_pair]=self.palette_values(gene_list=tmp_node_list,
                                                            palette=palette_dic[cell_pair])



            self.edge_color={}
            for cell_pair in cell_pairs:
                self.edge_color[cell_pair]={}
            # for cell_id in self.cell_list:
                cell_id=cell_pair[0]
                color_id=0
                for node_type in self.signal_net:
                    for gene in self.node_list[node_type][cell_id]:
                        node=[str(cell_id),node_type,gene]
                        node=' '.join(node)
                        self.edge_color[cell_pair][node]=palette[cell_pair][color_id]
                        color_id+=1
        else:
            self.edge_color = {}
            for cell_pair in self.cell_pairs:
                self.edge_color[cell_pair]={}
                cell_id=cell_pair[0]
                for node_type in self.signal_net:
                    for gene in self.node_list[node_type][cell_id]:
                        node=[str(cell_id),node_type,gene]
                        node=' '.join(node)
                        self.edge_color[cell_pair][node] = self.node_colors[node]

            # self.edge_color=None
    def node_layout(self,palette_mode='gene',node_colors=None,outer_radius=3,**kwargs):
        """_summary_

        Args:
            # grn_center (dict): for subgraph of individual graph, get the center for each node type.
            #     e.g.,  grn_center={'SPG':np.array([0.,0]),
            #                         'R':np.array([-inner_w,-inner_h]),
            #                         'L':np.array([inner_w,-inner_h])}
            palette_mode (str, optional): 1) 'gene' to have different palette for L, R and SPG.
                                          2) 'cell' to have different paleette for different cells
                                          3) 'cell_gene' to have different palette for different cells and different types of genes (L,R,SPG)
                Defaults to 'gene'.
            node_colors (dict, optional): Dictionary of colors for each node. If given, directory pass this dictionary to self.node_colors.
                Defaults to None.
            layout (str, or dict): layout of subgraph of individual cell. Defaults to 'horizontal'.
            inner_h (int, optional): _description_. Defaults to 1.
            inner_w (int, optional): _description_. Defaults to 2.
            outer_radius (int, optional): _description_. Defaults to 3.
        """
        
        # if isinstance(layout,str):
        #     for node_type in self.signal_net:
        #         layout_dic[node_type]=layout
        # elif isinstance(layout,dict):
        #     layout_dic=layout
        self.create_palette(palette_mode=palette_mode,**kwargs)

        if node_colors is None:
            self.node_colors={}
            if palette_mode=='gene':
                for node in self.G.nodes():
                    tmp = node.split(' ')
                    cell_id=int(tmp[0])
                    node_type=tmp[1]
                    gene = tmp[2]
                    self.node_colors[node] = self.palette[node_type][self.node_uniques[node_type].index(gene)]
            elif palette_mode=='cell':
                for cell_id in self.cell_list:
                    color_id=0
                    for node_type in self.signal_net:
                        for gene in self.node_list[node_type][cell_id]:
                            node=[str(cell_id),node_type,gene]
                            node=' '.join(node)
                            self.node_colors[node]=self.palette[cell_id][color_id]
                            color_id+=1
            elif palette_mode=='cell_gene':
                color_id=0
                for cell_id in self.cell_list:
                    for node_type in self.signal_net:

                        for gene in self.node_list[node_type][cell_id]:
                            node=[str(cell_id),node_type,gene]
                            node=' '.join(node)
                            color=self.palette[color_id]
                            self.node_colors[node]=color
                        color_id+=1
        else:
            self.node_colors=node_colors


        pos_idx = {}
        node_idx=0

        for cell in self.cell_list:
            # node_idx+=1
            for node_type in self.signal_net:
                # node_idx+=1
                H=self.subgraph({'cell':cell, 'node_type':node_type})
                if len(H) == 0:
                    node_idx=node_idx-1
                    continue
                for node in sorted(H.nodes()):
                    # node_name=node.split(' ')[2]
                    pos_idx[node] = node_idx
                    node_idx+=1
                node_idx+=1
            node_idx+=1
        if len(self.cell_list)==1:
            node_idx=node_idx-2
        theta=2*np.pi/(node_idx)
        pos = {node:[outer_radius*np.cos(theta*pos_idx[node]),
                     outer_radius*np.sin(theta*pos_idx[node])] 
                     for node in pos_idx.keys()}
        self.pos=pos
        self.pos_theta = {node: theta*pos_idx[node]
            for node in pos_idx.keys()}



    def draw_nodes(self,node_size=100,alpha=0.7):
        nx.draw_networkx_nodes(self.G,pos=self.pos,node_color=[self.node_colors.get(node) for node in self.G.nodes],
                               alpha=alpha,node_size=node_size)
        description=nx.draw_networkx_labels(self.G,self.pos,self.node_label,
                                font_size=10,horizontalalignment='left')
        for key,t in description.items():
            position=self.pos[key]
            position[0]*=1.08
            position[1]*=1.08
            t.set_position((position[0],position[1]))
            if 90 < self.pos_theta[key]*360/(2.0*np.pi) or self.pos_theta[key]*360/(2.0*np.pi) < -90 :
                angle = 180 + self.pos_theta[key]*360/(2.0*np.pi)
                t.set_ha('right')
            else:
                angle = self.pos_theta[key]*360/(2.0*np.pi)
                t.set_ha('left')
            t.set_va('center')
            t.set_rotation(angle)
            t.set_rotation_mode('anchor')
            t.set_clip_on(False)
    


class Circle_L_SPG_L(BaseCircleGraph):
    ## aggregated graph: pathway/ligand to SPG to pathway/ligand network
    def __init__(self,test='ftest',net=['L','SPG','L']):
        """_summary_
        Args:
            test (str, optional): statistical test used for FDR. Defaults to 'ftest'.
            net (list, optional): (list, optional): a list with length of three. It can be: ['L','SPG','L] or ['pathway','SPG','pathway']
                Defaults: ['L','SPG','L]
        """
        super().__init__(test)
        self.net=net
        self.signal_net=net[0:2] # the source and target node types in the signaling pathways
        self.grn_net = net[1:]
    def create_edges(self,time_scale=1,time_unit='h',abs_LRSPG=False):
        """create edges for all networks

        Args:
            time_scale (int, optional): _description_. Defaults to 1.
            time_unit (str, optional): _description_. Defaults to 'h'.
            abs_LRSPG (bool, optional): Whether to aggregate L-R-SPG causal strength with `abs()`. Defaults to False.
        """
        signal_net=self.signal_net
        self.edge_LRSPG_lag={}
        self.edge_LRSPG={}
        for key in self.cell_pairs:
            source_cell=str(key[0])
            target_cell=str(key[1])
            CCC_data=self.CCC_data[key]

            CCC_tmp = CCC_data.copy()
            if abs_LRSPG:
                CCC_tmp['flow'] = CCC_tmp['stat ' + self.test].abs()
            else:
                CCC_tmp['flow'] = CCC_tmp['stat ' + self.test]#.abs()
            signal_group = CCC_tmp.groupby(signal_net, as_index=False)
            signal_lag = signal_group[['lag']].mean()
            signal_flow = signal_group[['flow']].sum()
            signal = pd.merge(signal_lag, signal_flow, on=signal_net, how='inner')
            for i in range(len(signal)):
                itm=signal.iloc[i]
                u=itm[signal_net[0]]
                v=itm[signal_net[1]]
                cell_pair_new = (source_cell+' '+signal_net[0]+' '+u,\
                    target_cell+' '+signal_net[1]+' ' + v)
                self.edge_LRSPG_lag[cell_pair_new]=itm['lag']
                self.edge_LRSPG[cell_pair_new]=itm['flow']

        self.edge_SPGSDG={}
        for key in self.GRN.keys():
            cell=str(key)
            for idx_grn in range(len(self.GRN[key])):
                itm = self.GRN[key][idx_grn]
                self.edge_SPGSDG[cell+' SPG '+itm['source'],cell + ' '+signal_net[0]+' '+itm['target']] =itm['score']
        
        # self.edge_LR_lag = {node:f"{time_scale*self.edge_LR.get(node):.0f}{time_unit}" for node in list(self.edge_LR.keys())}
        # self.edge_RSPG_positive = {node:f"{self.edge_RSPG.get(node):.1f}" for node in list(self.edge_RSPG.keys()) if self.edge_RSPG.get(node)>0}
        # self.edge_RSPG_negative = {node:f"{self.edge_RSPG.get(node):.1f}" for node in list(self.edge_RSPG.keys()) if self.edge_RSPG.get(node)<0}
        self.edge_SPGSDG_positive =  {node:self.edge_SPGSDG.get(node) for node in list(self.edge_SPGSDG.keys()) if self.edge_SPGSDG.get(node)>0}
        self.edge_SPGSDG_negative =  {node:self.edge_SPGSDG.get(node) for node in list(self.edge_SPGSDG.keys()) if self.edge_SPGSDG.get(node)<0}
        self.edge_LRSPG_positive = {node:self.edge_LRSPG.get(node) for node in list(self.edge_LRSPG.keys()) if self.edge_LRSPG.get(node)>0}
        self.edge_LRSPG_negative = {node:self.edge_LRSPG.get(node) for node in list(self.edge_LRSPG.keys()) if self.edge_LRSPG.get(node)<0}
        self.edge_LRSPG_lag = {node:f"{time_scale*self.edge_LRSPG_lag.get(node):.0f}{time_unit}" for node in list(self.edge_LRSPG_lag.keys())}

    def draw_G(self,palette_mode='cell_gene',node_colors=None,width_scale=1.,ax=None,**kwargs):
        """_summary_

        Args:
            ax (_type_, optional): _description_. Defaults to None.
            time_scale (float,optional): how to scale the original time data. e.g., `time_scale`=1, scale days to hours.
                Default: 1
            time_unit (str,optional): unit of time showing up
                Default: 'h' hours
            layout (str,optional): layout of node. 'horizontal' or 'vertical'
                Default: 'horizontal'
            inner_h (float,optional): for subnetwork of each cell, distance between different layers.
                Default: 1
            inner_w (float,optional): for subnetwork of each cell, width of a layer nodes spreading
                Default: 2
            outer_radius (float,optional): multiple cells are distributed to a circle. Radius of this circle.
                Default: 3   
            palette (dict,optional): palette for different sets of nodes. 
                Default: {'L': 'Blues_r','R': 'Reds_r','SPG': 'Greens_r','pathway': 'Blues_r'}         
        Returns:
            ax: _description_
        """
       
        self.node_layout(palette_mode=palette_mode,node_colors=node_colors,**kwargs)

        # self.create_G(**kwargs)
        # self.node_layout(**kwargs)

        # pos = self.pos
        if ax is None:
            ax = plt.gca()

        nx.draw_networkx_edges(self.G,self.pos,
                                     edgelist=list(self.edge_LRSPG_positive.keys()),
                                     edge_color=[self.node_colors.get(u) for (u,v) in list(self.edge_LRSPG_positive.keys())],
                                     ax=ax,
                                     width=[width_scale*self.edge_LRSPG_positive[(u, v)] for (u, v) in list(self.edge_LRSPG_positive.keys())],
                                     alpha=0.7)
        nx.draw_networkx_edges(self.G,self.pos,
                               arrowstyle='-[',
                                     edgelist=list(self.edge_LRSPG_negative.keys()),
                                     edge_color=[self.node_colors.get(u) for (u,v) in list(self.edge_LRSPG_negative.keys())],
                                     ax=ax,
                                     width=[-width_scale*self.edge_LRSPG_negative[(u, v)] for (u, v) in list(self.edge_LRSPG_negative.keys())],
                                     alpha=0.7)
        nx.draw_networkx_edge_labels(self.G,self.pos,edge_labels=self.edge_LRSPG_lag,
                                    ax=ax,font_size=7,alpha=0.7,font_color='black')
        
        nx.draw_networkx_edges(self.G,self.pos,#edge_labels=self.edge_LR_lag,
                                     edgelist=list(self.edge_SPGSDG_positive.keys()),
                                     edge_color=[self.node_colors.get(u) for (u,v) in list(self.edge_SPGSDG_positive.keys())],
                                     width = [width_scale*self.edge_SPGSDG_positive[(u, v)] for (u, v) in list(self.edge_SPGSDG_positive.keys())],
                                     ax=ax,
                                     alpha=0.7)

        nx.draw_networkx_edges(self.G,self.pos,#edge_labels=self.edge_LR_lag,
                               arrowstyle='-[',
                                     edgelist=list(self.edge_SPGSDG_negative.keys()),
                                     edge_color=[self.node_colors.get(u) for (u,v) in list(self.edge_SPGSDG_negative.keys())],
                                     width = [-width_scale*self.edge_SPGSDG_negative[(u, v)] for (u, v) in list(self.edge_SPGSDG_negative.keys())],
                                     ax=ax,
                                     alpha=0.7)
        
        self.draw_nodes()   

        plt.axis('equal') 
        plt.box(False)
        # nx.draw_networkx_edge_labels(self.G,pos,edge_labels=self.edge_SPGSDG_negative,
        #                             ax=ax,font_size=6,alpha=0.6)
        # plt.xlim(self.xlim)
        # plt.ylim(self.ylim)
        legend_patches = self.create_legend(palette_mode=palette_mode,**kwargs)
        legend= ax.legend(handles=legend_patches,
                          loc='upper center',
                          bbox_to_anchor=(1.3, 1),
                          #ncol=len(self.signal_net)
                          )
        return ax

    def save_graph(self,save_dir):
        assert list(self.edge_LRSPG.keys())==list(self.edge_LRSPG_lag.keys())
        data1=[]
        for (u,v) in self.edge_LRSPG.keys():
            source_cell, source_type,source_gene = u.split(' ')
            target_cell, target_type,target_gene = v.split(' ')
            data1.append({
                'source': u,
                'target': v,
                'source_cell':source_cell,
                'source_type':source_type,
                'source_gene':source_gene,
                'target_cell':target_cell,
                'target_type':target_type,
                'target_gene':target_gene,
                'pathway':self.L2P[source_gene],
                'score': self.edge_LRSPG[(u,v)],
                'lag': self.edge_LRSPG[(u,v)],
            })
        data2=[]
        for (u,v) in self.edge_SPGSDG.keys():
            source_cell, source_type,source_gene = u.split(' ')
            target_cell, target_type,target_gene = v.split(' ')
            data2.append({
                'source': u,
                'target': v,
                'source_cell': source_cell,
                'source_type':source_type,
                'source_gene':source_gene,
                'target_cell':target_cell,
                'target_type':target_type,
                'target_gene':target_gene,
                'score': self.edge_SPGSDG[(u,v)]
            })
        data1=pd.DataFrame(data1)
        data2=pd.DataFrame(data2)
        with pd.ExcelWriter(os.path.join(save_dir,'multicell_graph.xlsx')) as writer:
            data1.to_excel(writer, sheet_name='L_SPG', index=False)
            data2.to_excel(writer, sheet_name='SPG_L', index=False)        



class Circle_L_L(Circle_L_SPG_L):
    ## aggregated graph: pathway/ligand to pathway/ligand network
    def __init__(self,test='ftest',net=['L','L'],cell_name=None):
        """_summary_
        Args:
            test (str, optional): statistical test used for FDR. Defaults to 'ftest'.
            net (list, optional): (list, optional): a list with length of three. It can be: ['L','L] or ['pathway','pathway']
                Defaults: ['L','L']
        """
        BaseCircleGraph.__init__(self,test=test,cell_name=cell_name,)
        self.net=net
        self.signal_net=net[0:1] # the source and target node types in the signaling pathways
        self.grn_net = ['SPG',net[1]]#net[1:]


    def create_edges(self,time_scale=1,time_unit='h',edge_scaling_func=None,abs_LRSPG=False):
        """create edges for all networks

        Args:
            time_scale (int, optional): _description_. Defaults to 1.
            time_unit (str, optional): _description_. Defaults to 'h'.
            abs_LRSPG (bool, optional): Whether to aggregate L-R-SPG causal strength with `abs()`. Defaults to False.
            edge_scaling_func (function, optional): function to scale edge weight. Default to None
        """
        self.signal_net.insert(1,'SPG')
        super().create_edges(time_scale=time_scale,
                             time_unit=time_unit,
                             abs_LRSPG=abs_LRSPG,
                             )
        
        self.signal_net.pop(1)
        self.edge_LL=defaultdict(float)
        for u in self.edge_LRSPG.keys():
            for v in self.edge_SPGSDG.keys():
                if u[1]==v[0]:
                    node_pair = (u[0],v[1])
                    strength = self.edge_LRSPG[u] * self.edge_SPGSDG[v]
                    self.edge_LL[node_pair] += strength
        if edge_scaling_func is None:
            edge_scaling_func=lambda x: x

        self.edge_LL_positive = {node:edge_scaling_func(self.edge_LL.get(node)) for node in list(self.edge_LL.keys()) if self.edge_LL.get(node)>0}
        self.edge_LL_negative = {node:edge_scaling_func(self.edge_LL.get(node)) for node in list(self.edge_LL.keys()) if self.edge_LL.get(node)<0}

        self.edge_L_lag={}
        for key in self.cell_pairs:
            # source_cell=str(key[0])
            # target_cell=str(key[1])
            CCC_data=self.CCC_data[key]
            CCC_tmp = CCC_data.copy()

            signal_group = CCC_tmp.groupby(self.signal_net, as_index=False)
            signal_lag = signal_group[['lag']].mean()

            self.edge_L_lag[key] = {}
            # signal = pd.merge(signal_lag, signal_flow, on=signal_net, how='inner')
            for i in range(len(signal_lag)):
                itm=signal_lag.iloc[i]
                u=itm[self.signal_net[0]]
                self.edge_L_lag[key][u] = time_scale*itm['lag']

        self.influx=defaultdict(lambda: defaultdict(float))
        self.outflux=defaultdict(lambda: defaultdict(float))
        for (u,v) in self.edge_LL:
            cell_pair = (int(u.split(' ')[0]),int(v.split(' ')[0]))
            uu = u.split(' ')[2]
            vv = v.split(' ')[2]
            self.influx[cell_pair[1]][vv]+=self.edge_LL[(u,v)]
            self.outflux[cell_pair[0]][uu]+=self.edge_LL[(u,v)]

        edge_list=list(self.edge_LL.keys())
        self.prune_nodes_from_edges(edge_list)    

    
    def draw_G(self,ax=None,palette_mode='cell',node_colors=None,width_scale=0.1,width_thred=0.01,node_size=150,
               alpha=0.7,palette_edge_dic=None,#{(0,1):'Purples_r',(1,0):'Greens_r'},
               **kwargs):
        """_summary_

        Args:
            ax (_type_, optional): _description_. Defaults to None.
            time_scale (float,optional): how to scale the original time data. e.g., `time_scale`=1, scale days to hours.
                Default: 1
            time_unit (str,optional): unit of time showing up
                Default: 'h' hours
            layout (str,optional): layout of node. 'horizontal' or 'vertical'
                Default: 'horizontal'
            inner_h (float,optional): for subnetwork of each cell, distance between different layers.
                Default: 1
            inner_w (float,optional): for subnetwork of each cell, width of a layer nodes spreading
                Default: 2
            outer_radius (float,optional): multiple cells are distributed to a circle. Radius of this circle.
                Default: 3   
            palette (dict,optional): palette for different sets of nodes. 
                Default: {0: 'Blues_r',1: 'Reds_r',2: 'Greens_r',3: 'Purples_r',}      
            palette_edge_dic (dict, optional): palette for edges cross cells.
                Default: None. 
                An example of its values: {(0,1):'Purples_r',(1,0):'Greens_r'}
        Returns:
            ax: _description_
        """
        
        self.node_layout(palette_mode=palette_mode,node_colors=node_colors,**kwargs)        
        self.edge_color_cross_layers(palette_dic = palette_edge_dic)
        # self.create_G(**kwargs)
        # self.node_layout(**kwargs)

        # pos = self.pos
        if ax is None:
            ax = plt.gca()
        draw_graph_edges(self.G,
                         self.edge_LL_positive,
                         self.pos,
                         node_colors=self.node_colors,
                         edge_color_cross_layers=self.edge_color,
                         ax=ax,
                         width_scale=width_scale,
                         width_thred=width_thred,
                         alpha=alpha)
        draw_graph_edges(self.G,
                         self.edge_LL_negative,
                         self.pos,
                         node_colors=self.node_colors,
                         edge_color_cross_layers=self.edge_color,
                         arrowstyle='-[',
                         ax=ax,
                         width_scale=width_scale,
                         width_thred=width_thred,
                         alpha=alpha)

        self.draw_nodes(alpha=alpha,node_size=node_size) 
        plt.axis('equal') 
        plt.box(False)
        legend_patches = self.create_legend(palette_mode=palette_mode)

        for cell_id in self.cell_list:
            if self.cell_name is not None:
                cell_pair_str=self.cell_name[cell_id]+'->'+self.cell_name[cell_id]
            else:
                cell_pair_str = str(cell_id)+'->'+str(cell_id)
            color=self.palette[cell_id][0]
            new_line = mlines.Line2D(
            [], [], 
            color=color, 
            linestyle='-', 
            linewidth=2, 
            label=cell_pair_str
            )
            legend_patches.append(new_line)
        if palette_edge_dic is not None:    
            for cell_pair in self.edge_color.keys():
                if self.cell_name is not None:
                    cell_pair_str=self.cell_name[cell_pair[0]]+'->'+self.cell_name[cell_pair[1]]
                else:
                    cell_pair_str = str(cell_pair[0])+'->'+str(cell_pair[1])

                color=self.edge_color[cell_pair]
                color=color[next(iter(color))]
                new_line = mlines.Line2D(
                [], [], 
                color=color, 
                linestyle='-', 
                linewidth=2, 
                label=cell_pair_str
                )
                legend_patches.append(new_line)

            
        legend= ax.legend(handles=legend_patches,
                          loc='upper center',
                          bbox_to_anchor=(1.1, 1),
                          ncol=len(self.signal_net))
        return ax    
    
    

    def save_graph(self,save_dir):
        data1=[]
        for (u,v) in self.edge_LL.keys():
            source_cell, source_type,source_gene = u.split(' ')
            target_cell, target_type,target_gene = v.split(' ')
            data1.append({
                'source': u,
                'target': v,
                'source_cell':source_cell,
                'source_type':source_type,
                'source_gene':source_gene,
                'target_cell':target_cell,
                'target_type':target_type,
                'target_gene':target_gene,
                'pathway':self.L2P[source_gene],
                'score': self.edge_LL[(u,v)],
                # 'lag': self.edge_LL[(u,v)],
            })
        data1=pd.DataFrame(data1)
        with pd.ExcelWriter(os.path.join(save_dir,'multicell_LL_graph.xlsx')) as writer:
            data1.to_excel(writer, sheet_name='L_L', index=False)





class graph_L_g(Circle_L_SPG_L):
    ## aggregated graph: pathway/ligand to pathway/ligand network
    def __init__(self,test='ftest',net=['L','growth'],cell_name=None):
        """_summary_
        Args:
            test (str, optional): statistical test used for FDR. Defaults to 'ftest'.
            net (list, optional): (list, optional): a list with length of three. It can be: ['L','L] or ['pathway','pathway']
                Defaults: ['L','L']
        """
        BaseCircleGraph.__init__(self,test=test,cell_name=cell_name,)
        self.net=net
        self.signal_net=net[0:1] # the source and target node types in the signaling pathways
        self.grn_net = ['SPG',net[1]]#net[1:]
    def add_dg(self,dg_data,time,time_idx,cell_id,fate_idx):

        if not hasattr(self, 'time_list'):
            self.time_list = []
        if time[time_idx] not in self.time_list:
            self.time_list.append(time[time_idx])
        self.default_time = time[time_idx]
        dg = np.mean(dg_data[fate_idx[cell_id],:,:],axis=0)[time_idx,:]
        # GRN = np.mean(GRN_mtx[fate_idx[cell_id],:,:,:],axis=0)[time_idx,:,:]
        # thred = np.percentile(np.abs(GRN),q=percentile,axis=(0,1))
        # medium = np.median(np.abs(GRN),axis=(0,1))
        self.dg.setdefault(time[time_idx], {})
        self.dg[cell_id]=dg
 
    def create_edges(self,time_scale=1,time_unit='h',abs_LRSPG=False):
        """create edges for all networks

        Args:
            time_scale (int, optional): _description_. Defaults to 1.
            time_unit (str, optional): _description_. Defaults to 'h'.
            abs_LRSPG (bool, optional): Whether to aggregate L-R-SPG causal strength with `abs()`. Defaults to False.
        """
        signal_net=self.signal_net
        signal_net.insert(1,'SPG')

        self.edge_LRSPG={}

        for key in self.cell_pairs:
            source_cell=str(key[0])
            target_cell=str(key[1])
            CCC_data=self.CCC_data[key]
            CCC_tmp = CCC_data.copy()
            if abs_LRSPG:
                CCC_tmp['flow'] = CCC_tmp['stat ' + self.test].abs()
            else:
                CCC_tmp['flow'] = CCC_tmp['stat ' + self.test]#.abs()
            signal_group = CCC_tmp.groupby(signal_net, as_index=False)
            signal_lag = signal_group[['lag']].mean()
            signal_flow = signal_group[['flow']].sum()
            signal = pd.merge(signal_lag, signal_flow, on=signal_net, how='inner')
            for i in range(len(signal)):
                itm=signal.iloc[i]
                u=itm[signal_net[0]]
                v=itm[signal_net[1]]
                cell_pair_new = (source_cell+' '+signal_net[0]+' '+u,\
                    target_cell+' '+signal_net[1]+' ' + v)
                # self.edge_LRSPG_lag[cell_pair_new]=itm['lag']
                self.edge_LRSPG[cell_pair_new]=itm['flow']
        signal_net.pop(1)
        

        self.edge_SPGg={}
        for key in self.GRN.keys():
            cell=str(key)
            for idx_grn in range(len(self.GRN[key])):
                itm = self.GRN[key][idx_grn]
                self.edge_SPGg[cell+' SPG '+itm['source'],
                              cell + ' '+self.grn_net[1]+' '+itm['target']] \
                =itm['score']*self.dg[key][self.gene_list.index(itm['target'])]
        self.L_g = defaultdict(float)
        for u in self.edge_LRSPG.keys():
            for v in self.edge_SPGg.keys():
                if u[1]==v[0]:
                    # node_pair = (u[0],v[1])
                    strength = self.edge_LRSPG[u] * self.edge_SPGg[v]
                    self.L_g[u[0]] += strength                
        self.L_lag={}
        for key in self.cell_pairs:
            # source_cell=str(key[0])
            # target_cell=str(key[1])
            CCC_data=self.CCC_data[key]
            CCC_tmp = CCC_data.copy()

            signal_group = CCC_tmp.groupby(self.signal_net, as_index=False)
            signal_lag = signal_group[['lag']].mean()

            self.L_lag[key] = {}
            # signal = pd.merge(signal_lag, signal_flow, on=signal_net, how='inner')
            for i in range(len(signal_lag)):
                itm=signal_lag.iloc[i]
                u=itm[self.signal_net[0]]
                self.L_lag[key][u] = time_scale*itm['lag']        

