import anndata
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import networkx as nx
import seaborn as sns
import pandas as pd
from collections import defaultdict
import os
# from .base import BaseGraph
from .base import *
from .GO_analysis import *
palette_spg = list(sns.color_palette('husl')\
                  + sns.color_palette("Set1"))

# list(sns.color_palette("pastel") \
#                 + sns.color_palette("tab10")\
#                 + sns.color_palette('husl') \
#                 + sns.color_palette("tab20") \
#                 + sns.color_palette("tab20b") \
#                 + sns.color_palette("tab20c")\
#                 + sns.color_palette("Set1")\
#                 + sns.color_palette("Set2")\
#                 + sns.color_palette('Set3')\
#                 + sns.color_palette('Dark2')\
#                 + sns.color_palette('Pastel1')\
#                 + sns.color_palette('Pastel2'))
def rescale_pos(pos, num_layers,layout='vertical'):
    # print(pos)
    axis_idx = 1 if layout == 'vertical' else 0  # 1 = y, 0 = x
    spread_idx = 1 - axis_idx
    pos_value=np.array(list(pos.values()))
    layer_coords = sorted(set(pos_value[:,spread_idx]))
    layer_map = {cord:i/(num_layers-1) for i,cord in enumerate(layer_coords)}
    # print(layer_coords)
    inlayer_map={}
    for layer_cor in list(layer_coords):
        inlayer_map[layer_cor]={}
        idx=pos_value[:,spread_idx]==layer_cor
        # layer_max=pos_value[idx,axis_idx].max()
        # layer_min=pos_value[idx,axis_idx].min()
        # print(idx)
        # print(pos_value[idx,axis_idx])
        for cor in pos_value[idx,axis_idx]:
            inlayer_map[layer_cor][cor]=cor ## the location of nodes in the same layer is kept unchanged
            # if layer_max>layer_min:
            #     inlayer_map[layer_cor][cor]=(cor-layer_min)/(layer_max-layer_min)/1.6
            # else:
            #     inlayer_map[layer_cor][cor]=0.5
    # print(inlayer_map)
    new_pos = {}
    for layer_old, cor in pos.items():
        spread_cor=cor[spread_idx]
        inlayer_cor=cor[axis_idx]
        new_spread_cor = layer_map[spread_cor]
        new_inlayer_cor = inlayer_map[spread_cor][inlayer_cor]
        new_cor=[0,0]
        new_cor[spread_idx]=new_spread_cor
        new_cor[axis_idx]=new_inlayer_cor
        new_pos[layer_old] = new_cor
    return new_pos
class BaseHierachyGraph(BaseGraph):
    ## This graph will have L-R-SPG-L. Ligand nodes will appear twice in two layers. 
    ## So we need different node names to distinguish Ligand in two layers:
    ## 'sL', and 'tL'.
    ## several functions are revised:
    ## 1) create_G(); 2) prune_nodes_from_edges(); 3) add_node_names()
    
    def __init__(self,test='ftest',cell_list_all=None,cell_name=None):
        super().__init__(test,cell_list_all=cell_list_all,cell_name=cell_name)
        self.node_list['SDG']={}
    def add_GRN(self, GRN_mtx, time,time_idx,cell_id,fate_idx,gene_list,percentile = 75,abs_edge=False,
                source_gene=None,target_gene=None, time_GRN=None,**kwargs):
        """add GRN for cell with `cell_id` at `time_idx` time point.

        Args:
            GRN_mtx (_type_): _description_
            time (_type_): _description_
            time_idx (_type_): _description_
            cell_id (_type_): _description_
            fate_idx (_type_): _description_
            gene_list (_type_): _description_
            percentile (int, optional): _description_. Defaults to 75.
            abs_edge (bool, optional): Whether to take `abs()` for GRN edges when aggregating them. 
                Defaults to False.
            source_gene (list, optional): If it is given, GRN will consider genes in this list for source genes in GRN extraction.
                Defaults to None.
            target_gene (list, optional): 
                1. None: consider all genes as target_gene;
                2. str: select genes in one layer of self.net. must an element in self.net (e.g., 'L': taget genes are ligand that has causal effects in the same cell)
                3. list: a given list of target gene 
                Defaults to None.   
            time_GRN (np.array, optional):
                if given, the time_idx is the GRN's time given by `time_GRN[time_idx]`
                Defaults: None            
        """
        # assert (source_gene is None) == (target_gene is None), "source_gene and target_gene must both be None or both not None"
        aggregate_id=-1 # whether to aggreate the weights for an axis.
        # if source_gene is None and target_gene is None:
        #     grn_net=self.grn_net
        #     grn_net_L = []
            
        #     for i,itm in enumerate(grn_net):
        #         if itm=='pathway':
        #             grn_net_L.append('L')
        #             aggregate_id=i
        #         else:
        #             grn_net_L.append(itm)
        if time_GRN is not None:
            time=time_GRN 
        if not hasattr(self, 'time_list'):
            self.time_list = []
        if time[time_idx] not in self.time_list:
            self.time_list.append(time[time_idx])
        self.default_time = time[time_idx]
        GRN = np.mean(GRN_mtx[fate_idx[cell_id],:,:,:],axis=0)[time_idx,:,:]
        thred = np.percentile(np.abs(GRN),q=percentile,axis=(0,1))
        medium = np.median(np.abs(GRN),axis=(0,1))
        self.GRN_all.setdefault(time[time_idx], {})
        # self.GRN_all[time[time_idx]][cell_id]=[]
        GRN_list = []
        if source_gene is None:
            source_gene=self.node_list[self.grn_net[0]][cell_id]
        if target_gene is None:
            target_gene=gene_list.copy()
        elif isinstance(target_gene,str):
            assert target_gene in self.net, 'str target_gene not found in self.net'
            target_gene=self.node_list[target_gene][cell_id] # SDG is taken as Ligand

            
        for source in source_gene:
            for target in target_gene:
                score=GRN[gene_list.index(target),gene_list.index(source)]

                if np.abs(score) >thred:
                    # net_edge = {'source': source,'target': target,'score': score}
                    net_edge = {'source': source,'target': target,'score': score/medium}
                    if aggregate_id==0:
                        pathway=self.L2P[source]
                        net_edge['pathway']=pathway
                    elif aggregate_id==1:
                        pathway=self.L2P[target]
                        net_edge['pathway']=pathway
                    GRN_list.append(net_edge)        
        if aggregate_id>=0:
            grouped = defaultdict(float)
            if aggregate_id==1: #L is target
                for d in GRN_list:
                    key = (d['source'], d['pathway'])
                    if abs_edge:
                        grouped[key] += np.abs(d['score'])
                    else:
                        grouped[key] += d['score']
                result = [{'source': k[0], 'target': k[1], 'score': v} for k, v in grouped.items()]
            elif aggregate_id==0: # L is source      
                for d in GRN_list:
                    key = (d['target'], d['pathway'])
                    if abs_edge:
                        grouped[key] += np.abs(d['score'])
                    else:
                        grouped[key] += d['score']
                result = [{'target': k[0], 'source': k[1], 'score': v} for k, v in grouped.items()]
        else:
            result = GRN_list
        self.GRN_all[time[time_idx]][cell_id] = result
        self.default_time = time[time_idx]
        self.GRN = self.GRN_all[self.default_time]
        self.node_list['SDG'] = self.add_list(self.node_list['SDG'],cell_id,target_gene)



    def runGO(self, cell_id,species='human',layer='SDG'):
        """run GO analysis for a specific cell `cell_id` and layer `SDG`.
        Args:
            cell_id (_type_): _description_
            species (str, optional): options: ['human','mouse'].
                Defaults to 'human'.
            layer (str, optional): _description_. Defaults to 'SDG'.
        """
        test_genes = list(set(self.node_list[layer][cell_id]))
        if len(test_genes)>0:
            df_GO =  runGO(species=species,
                        test_genes=test_genes)
        else:
            df_GO = pd.DataFrame({})
        return df_GO, test_genes

    def create_G(self,cell_pair,**kwargs):
        labels = {}
        G = nx.DiGraph()
        node_type_list=self.net
        # for cell in self.cell_list:
        for layer_id,node_type in enumerate(node_type_list):
            if node_type in ['L', 'pathway']:
                cell=cell_pair[0]
            else:
                cell=cell_pair[1]
            G,labels = self.add_node_names(G,labels, cell, self.node_list,node_type=node_type,layer=layer_id)

            # G,labels = self.add_node_names(G,labels, cell, self.R_list,node_type='R')
            # G,labels = self.add_node_names(G,labels, cell, self.SDG_list,node_type='SDG')
        self.G=G
        self.node_label=labels


        self.create_edges(cell_pair=cell_pair,**kwargs)

    def prune_nodes_from_edges(self,edge_list):
        node_list=[node for edge in edge_list for node in edge]
        node_list=sorted(list(dict.fromkeys(node_list)))
        G_node_list = list(self.G.nodes())
        
        for node in G_node_list:
            if not node in node_list:
                self.G.remove_node(node)
        self.node_label = {node: self.node_label[node] for node in self.G.nodes()}
        

        self.node_list={}
        for gene_type in self.net:
            # gene_type_short=gene_type.replace('sL','L').replace('tL','L')
            if gene_type not in self.node_list:
                self.node_list[gene_type]={}
            for cell_id in self.cell_list:
                if cell_id not in self.node_list[gene_type]:
                    self.node_list[gene_type][cell_id]=[]
        for node in node_list:
            cell_id,gene_type,gene=node.split(' ')
            # gene_type_short=gene_type.replace('sL','L').replace('tL','L')
            if gene not in self.node_list[gene_type][int(cell_id)]:
                self.node_list[gene_type][int(cell_id)].append(gene)


    def node_layout(self,cell_pair,subset_key,palette_mode='gene',node_colors=None,layout='vertical',outer_radius=3,**kwargs):
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
            layout (str, or dict): layout of subgraph of individual cell. Defaults to 'vertical'.
            inner_h (int, optional): _description_. Defaults to 1.
            inner_w (int, optional): _description_. Defaults to 2.
            outer_radius (int, optional): _description_. Defaults to 3.
        """
        self.node_colors={}
        # if isinstance(layout,str):
        #     for node_type in self.signal_net:
        #         layout_dic[node_type]=layout
        # elif isinstance(layout,dict):
        #     layout_dic=layout
        self.create_palette(palette_mode=palette_mode,**kwargs)
        # if layout=='horizontal':
        num_layers=len(self.net)
        existing_layers = set(nx.get_node_attributes(self.G, subset_key).values())
        G = self.G.copy()
        for layer in range(num_layers):
            if layer not in existing_layers:
                dummy_node = f'dummy_layer_{layer}'
                G.add_node(dummy_node)
                G.nodes[dummy_node][subset_key] = layer

        pos = nx.multipartite_layout(self.G,subset_key=subset_key,
                            # subset_key={'layer1':[str(cell_pair[0])+' '+self.signal_net[0]+' '+self.node_list[self.signal_net[0]][cell_pair[0]][i] for i in range(len(self.node_list[self.signal_net[0]][cell_pair[0]]))],
                            #             'layer2':[str(cell_pair[1])+' '+self.signal_net[1]+' '+self.node_list[self.signal_net[1]][cell_pair[1]][i] for i in range(len(self.node_list[self.signal_net[1]][cell_pair[1]]))],
                            #             },
                            align=layout)
        new_pos={}
        if layout=='vertical':
            pos_value=np.array(list(pos.values()))
            x_cat = sorted(set(pos_value[:,0]))
            for x in x_cat:
                idx=(pos_value[:,0]==x)
                pos_key_sub = sorted([node for i, node in enumerate(pos.keys()) if idx[i]])
                pos_value_sub= pos_value[idx,:]
                for iii in range(len(pos_key_sub)):
                    new_pos[pos_key_sub[iii]]=pos_value_sub[iii,:]
            # x_maps={x[i]:i for i in range(len(x))}
            # x_maps={}
            # pos = {key: [x_maps.get(pos[key][0]), pos[key][1]] for key in pos.keys()}
        elif layout == 'horizontal':
            pos_value=np.array(list(pos.values()))
            y_cat = sorted(set(pos_value[:,1]))
            for y in y_cat:
                idx=(pos_value[:,1]==y)
                pos_key_sub = sorted([node for i, node in enumerate(pos.keys()) if idx[i]])
                pos_value_sub= pos_value[idx,:]
                for iii in range(len(pos_key_sub)):
                    new_pos[pos_key_sub[iii]]=pos_value_sub[iii,:]
        new_pos = rescale_pos(new_pos,num_layers=len(self.net),layout=layout)
        assert layout=='vertical'; 'code for horizontal not done yet'
        a=list(new_pos.keys())
        node_4th = [item for item in a if item.split()[1] == 'SDG']
        node_3rd = [item for item in a if item.split()[1] == 'SPG']
        pos_3rd=np.array([new_pos[node] for node in node_3rd])
        if len(node_4th)>6:
            theta=np.linspace(1,0,len(node_4th))*np.pi-np.pi/2
            # radius=pos_3rd[:,1].max()-pos_3rd[:,1].min()
            # radius*=len(node_4th)/12
            # if radius==0:
            #     node_2nd = [item for item in a if item.split()[1] == 'R']
            #     pos_2nd=np.array([new_pos[node] for node in node_2nd])
            #     radius=pos_3rd[:,0].max()-pos_2nd[:,0].max()
            #     radius*=len(node_4th)/12
            radius=0.5
            if len(node_3rd)>4:
                radius = (pos_3rd[:,1].max()-pos_3rd[:,1].min())/len(node_3rd)*(len(node_3rd)/2+2)
            center=np.array([pos_3rd[0,0], 0.5*pos_3rd[:,1].max()+0.5*pos_3rd[:,1].min()])
            for i,node in enumerate(node_4th):
                new_pos[node]=center+np.array([np.cos(theta[i]), np.sin(theta[i])])*radius                              
            self.center_4th=center
            self.radius_4th=radius
            self.theta_4th={node:theta[i] for i,node in enumerate(node_4th)}

        self.pos = new_pos 
        if node_colors is None:
            color_id=0
            color_spg_id=0
            for cell_id in self.cell_list_all:
                for node_type in self.net:
                    if cell_id in self.node_list[node_type].keys():
                        for gene in self.node_list[node_type][cell_id]:
                            node=[str(cell_id),node_type,gene]
                            node=' '.join(node)
                            color=self.palette[color_id]
                            self.node_colors[node]=color
                            
                            if node.split(' ')[1]=='SPG' and len(self.node_list['SPG'][cell_id])<=len(palette_spg):
                                color=palette_spg[color_spg_id]
                                self.node_colors[node]=color
                                color_spg_id+=1
                            # elif node.split(' ')[1]=='L':
                            #     self.node_colors[node]=sns.color_palette('Blues_r')[2]
                            # elif node.split(' ')[1]=='L':
                                

                    color_id+=1
        else:
            self.node_colors=node_colors

    def create_legend(self,palette_mode='gene',**kwargs):
        legend_patches=[]

        color_id=0
        for cell_id in self.cell_list_all:
            for node_type in self.net:
                # node_type=node_type.replace('sL','L')
                color=self.palette[color_id]
                label=str(cell_id)+' '+node_type_map[node_type] if self.cell_name is None else self.cell_name[cell_id]+' '+node_type_map[node_type]
                legend_patches+=[mlines.Line2D([],[],color=color,
                                             label=label,
                                             marker='o',                      
                                             linestyle='None',
                                             markersize=10,  )
                                             ]
                
                # [mpatches.Patch(color=color,
                #                             label='Fate '+str(cell_id)+' '+node_type_map[node_type], )
                #                             ]
                color_id+=1

        return legend_patches
    def create_palette(self,palette={},palette_mode='gene',
                       palette_order=['Blues_r','Reds_r','Greens_r','Purples_r'],
                       **kwargs):
        """_summary_

        Args:
            palette (dict or str, optional): _description_. Defaults to {}.
            palette_mode (str, optional): 1) 'gene' to have different palette for L, R and SPG.
                                          2) 'cell' to have different paleette for different cells
                Defaults to 'gene_type'.
        """

        self.palette=self.palette_values(num_cell= len(self.cell_list_all),
                                            net_hierarchy=len(self.net),
                                            palette_order=palette_order)
    @staticmethod
    def add_node_names(G,labels,cell,node_list, node_type='L',layer=0):
        for a in node_list[node_type][cell]:
            node_name = str(cell)+' ' +node_type+' '+a
            G.add_node(node_name,cell=cell,node_type=node_type,layer=layer)
            labels[node_name]=a
        return G,labels

class Hierarchy_L_R_SPG_SDG(BaseHierachyGraph):
    def __init__(self,test='ftest',net=['L','R','SPG','SDG'],cell_list_all=None,cell_name=None,):
        super().__init__(test,cell_list_all=cell_list_all,cell_name=cell_name)
        self.net=net
        self.signal_net=net[0:3]
        self.grn_net = net[2:]   
        # self.grn_net=[n.replace('sL','L').replace('tL','L') for n in self.grn_net]
    def create_edges(self,cell_pair,time_scale=1,time_unit='h',**kwargs):
        self.edge_LR={}
        self.edge_RSPG={}
        # for key in self.cell_pairs:
        source_cell=str(cell_pair[0])
        target_cell=str(cell_pair[1])
        CCC_data=self.CCC_data[cell_pair]
        LR_avg = CCC_data.groupby('LR')[['lag']].mean().reset_index()
        RSPG_avg = CCC_data.groupby('RSPG')[['stat '+self.test]].sum().reset_index()

        # edge_lag = []
        for i in range(len(LR_avg)):
            u,v = LR_avg.iloc[i]['LR'].split('-')
            # G.add_edge('L '+u,'R ' + v,lag = LR_avg.iloc[i]['lag'])
            cell_pair_name = source_cell+' '+self.signal_net[0]+' '+u,target_cell+' '+self.signal_net[1]+' '+v
            self.edge_LR[cell_pair_name] = LR_avg.iloc[i]['lag']
        for i in range(len(RSPG_avg)):
            v,w = RSPG_avg.iloc[i]['RSPG'].split('-')
            # if v in G.nodes and w in G.nodes:
            # G.add_edge('R '+v,'SPG '+w,stat=RSPG_avg.iloc[i]['stat '+test])
            cell_pair_name = target_cell+' '+self.signal_net[1]+' '+v,target_cell+' '+self.signal_net[2]+' '+w
            self.edge_RSPG[cell_pair_name] = RSPG_avg.iloc[i]['stat '+self.test]
        self.edge_SPGSDG={}
        # for key in self.GRN.keys():
        cell=str(cell_pair[1])
        for idx_grn in range(len(self.GRN[cell_pair[1]])):
            itm = self.GRN[cell_pair[1]][idx_grn]
            self.edge_SPGSDG[cell+' '+self.net[2]+' '+itm['source'],cell +' '+self.net[3]+' '+itm['target']] =itm['score']

        self.edge_LR_lag = {node:f"{time_scale*self.edge_LR.get(node):.0f}{time_unit}" for node in list(self.edge_LR.keys())}
        self.edge_RSPG_positive = {node:self.edge_RSPG.get(node) for node in list(self.edge_RSPG.keys()) if self.edge_RSPG.get(node)>0}
        self.edge_RSPG_negative = {node:self.edge_RSPG.get(node) for node in list(self.edge_RSPG.keys()) if self.edge_RSPG.get(node)<0}
        self.edge_SPGSDG_positive =  {node:self.edge_SPGSDG.get(node) for node in list(self.edge_SPGSDG.keys()) if self.edge_SPGSDG.get(node)>0}
        self.edge_SPGSDG_negative =  {node:self.edge_SPGSDG.get(node) for node in list(self.edge_SPGSDG.keys()) if self.edge_SPGSDG.get(node)<0}

        edge_list = list(self.edge_LR.keys())+list(self.edge_RSPG.keys())+list(self.edge_SPGSDG.keys())
        self.prune_nodes_from_edges(edge_list)
    def draw_G(self,cell_pair,palette_mode='cell_gene',layout='vertical',width_scale=4.,width_scale_grn=1.,node_size=150,ax=None,**kwargs):
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
        if len(self.G.nodes())>0:
            subset_key='layer'
            self.node_layout(cell_pair=cell_pair,subset_key=subset_key,layout=layout,palette_mode=palette_mode,**kwargs)
            # pos = self.pos
            if ax is None:
                ax = plt.gca()
            nx.draw_networkx_nodes(self.G,pos=self.pos,
                                node_color=[self.node_colors.get(node) for node in self.G.nodes],
                                alpha=1.,node_size=node_size)

            nx.draw_networkx_edges(self.G,self.pos,
                                #    arrowstyle='-[',
                                        edgelist=list(self.edge_LR.keys()),
                                        #  edge_color=[self.node_colors.get(u) for (u,v) in list(self.edge_LR.keys())],
                                        edge_color='black',
                                        ax=ax,
                                        width=2,
                                        alpha=0.7)          
            nx.draw_networkx_edge_labels(self.G,self.pos,edge_labels=self.edge_LR_lag,
                                        ax=ax,font_size=7,alpha=0.7,font_color='red')  
            nx.draw_networkx_edges(self.G,self.pos,
                                arrowstyle='-[',
                                        edgelist=list(self.edge_RSPG_negative.keys()),
                                        edge_color=[self.node_colors.get(u) for (u,v) in list(self.edge_RSPG_negative.keys())],
                                        ax=ax,
                                        width=[-width_scale*self.edge_RSPG_negative[(u, v)] for (u, v) in list(self.edge_RSPG_negative.keys())],
                                        alpha=0.7)        
            nx.draw_networkx_edges(self.G,self.pos,
                                #    arrowstyle='-[',
                                        edgelist=list(self.edge_RSPG_positive.keys()),
                                        edge_color=[self.node_colors.get(u) for (u,v) in list(self.edge_RSPG_positive.keys())],
                                        ax=ax,
                                        width=[width_scale*self.edge_RSPG_positive[(u, v)] for (u, v) in list(self.edge_RSPG_positive.keys())],
                                        alpha=0.7)  
            nx.draw_networkx_edges(self.G,self.pos,
                                arrowstyle='-[',
                                        edgelist=list(self.edge_SPGSDG_negative.keys()),
                                        edge_color=[self.node_colors.get(u) for (u,v) in list(self.edge_SPGSDG_negative.keys())],
                                        ax=ax,
                                        width=[-width_scale_grn*self.edge_SPGSDG_negative[(u, v)] for (u, v) in list(self.edge_SPGSDG_negative.keys())],
                                        alpha=0.7)        
            nx.draw_networkx_edges(self.G,self.pos,
                                #    arrowstyle='-[',
                                        edgelist=list(self.edge_SPGSDG_positive.keys()),
                                        edge_color=[self.node_colors.get(u) for (u,v) in list(self.edge_SPGSDG_positive.keys())],
                                        ax=ax,
                                        width=[width_scale_grn*self.edge_SPGSDG_positive[(u, v)] for (u, v) in list(self.edge_SPGSDG_positive.keys())],
                                        alpha=0.7)        
            # nx.draw_networkx_labels(self.G,self.pos,self.node_label, font_size=10)

            description=nx.draw_networkx_labels(self.G,self.pos,self.node_label,
                                    font_size=10,horizontalalignment='center')
            for key,t in description.items():
                if key.split(' ')[1]=='SDG':
                    if hasattr(self, 'theta_4th'):
                        position=self.pos[key]
                        position[0] = position[0]*1.08-0.08*self.center_4th[0]
                        position[1] = position[1]*1.08-0.08*self.center_4th[1]
                        t.set_position((position[0],position[1]))
                        # if np.pi/2 < self.theta_4th[key] or self.theta_4th[key] < -np :
                        #     angle = np.pi + self.pos_theta[key]*360/(2.0*np.pi)
                        #     t.set_ha('right')
                        # else:
                        #     angle = self.theta_4th[key]*360/(2.0*np.pi)
                        #     t.set_ha('left')
                        angle=self.theta_4th[key]
                        t.set_va('center')
                        t.set_rotation(angle/(2*np.pi)*360)
                        t.set_rotation_mode('anchor')
                        t.set_clip_on(False)
                        t.set_ha('left')


            plt.box(False)
            # plt.axis('equal')
            # pos_value=np.array(list(self.pos.values()))
            if layout=='vertical':
                if np.max(np.array(list(self.pos.values())),axis=0)[0]<1.2:
                    plt.xlim([-0.1,1.3])
            elif layout=='horizontal':
                if np.max(np.array(list(self.pos.values())),axis=0)[1]<1.2:
                    plt.ylim([-0.1,1.3])

            legend_patches = self.create_legend(palette_mode=palette_mode,**kwargs)

            if palette_mode=='cell_gene':  
                ncol=1
            else:
                ncol=len(self.signal_net)
            # legend= ax.legend(handles=legend_patches,
            #                 loc='upper center',
            #                 bbox_to_anchor=(1.3, 1),
            #                 ncol=ncol)
            # bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
            # x0, y0 = bbox.x0, bbox.y0
            # column_width = (bbox.x1 - bbox.x0) / 3
            # ax.text(x0 + column_width * 0.5, y0 + 0.05, 'Ligand', ha='center', va='bottom', transform=ax.transAxes)
            # ax.text(x0 + column_width * 1.5, y0 + 0.05, 'Receptor', ha='center', va='bottom', transform=ax.transAxes)
            # ax.text(x0 + column_width * 2.5, y0 + 0.05, 'SPG', ha='center', va='bottom', transform=ax.transAxes)
            return ax,legend_patches
        else:
            return None


