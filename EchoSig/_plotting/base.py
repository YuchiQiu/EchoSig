import anndata
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import networkx as nx
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
from collections import defaultdict
import os
palette = list(sns.color_palette("tab20") \
                + sns.color_palette("tab20b") \
                + sns.color_palette("tab20c")\
                + sns.color_palette("Set1")\
                + sns.color_palette("Set2")\
                + sns.color_palette('Set3')\
                + sns.color_palette('Dark2')\
                + sns.color_palette('Pastel1')\
                + sns.color_palette('Pastel2'))
# blue_palette=list(
#     sns.color_palette(palette='Blues_r',n_colors=20)
#     )
# red_palette=list(
#     sns.color_palette(palette='Reds_r',n_colors=20))
# purple_palette =list(sns.color_palette(palette='Purples_r',n_colors=20))
node_type_map={'L':'Ligand',
               'R':'Receptor',
               'SPG':'SPG',
               'SDG':'SDG',
               'LR':'L-R pair',
               'pathway':'Pathway'}
stat_test_map = {'ftest':'F-test',
                'chi2':'Chi2 test',
                }

def circular_bipartite_layout(G, inner_nodes, outer_nodes, inner_radius=1, outer_radius=2):
    pos = {}
    # Inner circle for x nodes
    if len(inner_nodes)>1:
        theta_x = np.linspace(0, 2 * np.pi, len(inner_nodes), endpoint=False)+np.pi/len(outer_nodes)
        for i, node in enumerate(inner_nodes):
            pos[node] = (inner_radius * np.cos(theta_x[i]), inner_radius * np.sin(theta_x[i]))
    else:
        pos[inner_nodes[0]]=(0,0)
    # Outer circle for y nodes
    theta_y = np.linspace(0, 2 * np.pi, len(outer_nodes), endpoint=False)
    for i, node in enumerate(outer_nodes):
        pos[node] = (outer_radius * np.cos(theta_y[i]), outer_radius * np.sin(theta_y[i]))
    return pos
    

class BaseGraph:
    ## Base class for create the graph
    ## draw causal graph between multiple cells trajectories
    def __init__(self,test='ftest',cell_list_all=None,cell_name=None):
        self.test=test
        self.CCC_data={}
        self.cell_pairs=[]
        self.cell_list = []
        if cell_list_all is not None:
            self.cell_list_all=cell_list_all
        else:
            self.cell_list_all=self.cell_list
        self.node_list={}
        self.node_list['L']={}
        self.node_list['R']={}
        self.node_list['SPG']={}
        self.node_list['pathway']={}
        self.node_list['RSPG']={}
        self.node_list['LR']={}
        self.cell_name=cell_name
        # self.RSPG_list={}
        self.LR_list={}
        self.GRN_all={}
        self.dg={}
        self.L2P=defaultdict(list)
    
    def add_CCC_data(self,CCC_data,cell_pair,FDR=0.01,pathway=None):
        CCC_data=CCC_data[CCC_data['q '+self.test]<FDR]
        split_index = CCC_data.index.str.split('-')

        CCC_data['LR'] = [f"{x[0]}-{x[1]}" for x in split_index]
        CCC_data['RSPG'] = [f"{x[1]}-{x[2]}" for x in split_index]

        if pathway is not None:
            if isinstance(pathway,str):
                CCC_data=CCC_data[CCC_data['pathway']==pathway]
            elif isinstance(pathway, list):
                CCC_data=CCC_data[CCC_data['pathway'].isin(pathway)]
        self.CCC_data[cell_pair] = CCC_data

        for i in range(len(CCC_data)):
            self.L2P[CCC_data['L'][i]].append(CCC_data['pathway'][i])       

        # self.L2P = {CCC_data['L'][i]:CCC_data['pathway'][i] for i in range(len(CCC_data))}
        self.cell_pairs.append(cell_pair)
        for cell in cell_pair:
            if cell not in self.cell_list:
                self.cell_list.append(cell)
        self.cell_list=sorted(list(dict.fromkeys(self.cell_list)))

        if len(self.CCC_data[cell_pair])>0:
            df = self.CCC_data[cell_pair]
            L_list = sorted(list(df['L']))
            R_list = sorted(list(df['R']))
            SPG_list = sorted(list(df['SPG']))
            path_list = sorted(list(df['pathway']))
            LR_list = sorted(list(df['LR']))
            RSPG_list = sorted(list(df['RSPG']))
            # key = list(self.CCC_data[cell_pair].index)
            # L_list, R_list, SPG_list = zip(*(s.split('-') for s in key))
            
            # RSPG_list=sorted(list(dict.fromkeys([R_list[i]+'-'+SPG_list[i] for i in range(len(L_list))])))
            # L_list=sorted(list(dict.fromkeys(L_list)))
            # R_list=sorted(list(dict.fromkeys(R_list)))
            # SPG_list=sorted(list(dict.fromkeys(SPG_list)))
            # path_list = sorted(list(dict.fromkeys(self.CCC_data[cell_pair]['pathway'])))


            self.node_list['L']=self.add_list(self.node_list['L'],cell_pair[0],L_list)
            self.node_list['R']=self.add_list(self.node_list['R'],cell_pair[1],R_list)
            self.node_list['SPG']=self.add_list(self.node_list['SPG'],cell_pair[1],SPG_list)
            self.node_list['LR'] = self.add_list(self.node_list['LR'],cell_pair[0],LR_list)
            self.node_list['RSPG'] = self.add_list(self.node_list['RSPG'],cell_pair[1],RSPG_list)
            self.node_list['pathway'] = self.add_list(self.node_list['pathway'],cell_pair[0],path_list)
            
        else:
            self.node_list['L'][cell_pair[0]]=[]
            self.node_list['R'][cell_pair[1]]=[]
            self.node_list['SPG'][cell_pair[1]]=[]
            self.node_list['LR'][cell_pair[0]]=[]
            self.node_list['RSPG'][cell_pair[1]]=[]
            self.node_list['pathway'][cell_pair[0]]=[]
            # self.inL_list[cell_pair[0]]=list(dict.fromkeys(L_list))
            # self.R_list[cell_pair[1]]=list(dict.fromkeys(R_list))
            # self.SPG_list[cell_pair[1]]=list(dict.fromkeys(SPG_list))
            # self.RSPG_list[cell_pair[1]] = \
            #     list(dict.fromkeys([R_list[i]+'-'+SPG_list[i] for i in range(len(L_list))]))

    def get_node_list(self,):
        self.node_uniques={}
        for node_type in self.node_list.keys():
            self.node_uniques[node_type]=[]
        for node_type in self.node_list.keys():
            for cell_id in self.node_list[node_type]:
                self.node_uniques[node_type].extend(self.node_list[node_type][cell_id])
        for node_type in self.node_list.keys():
            self.node_uniques[node_type]=sorted(list(dict.fromkeys(self.node_uniques[node_type])))

        L2P=self.L2P.copy()
        self.L2P = {}
        for k, v in L2P.items():
            unique_v = list(set(v))
            if len(unique_v) > 1:
                raise ValueError(f"Ligand '{k}' maps to multiple pathways: {unique_v}")
            else:
                self.L2P[k] = unique_v[0]  # Just the single value, not a list

    def add_GRN(self, GRN_mtx, time,time_idx,cell_id,fate_idx,gene_list,percentile = 95,abs_edge=False,
                source_gene=None,target_gene=None, **kwargs):
        """add GRN for cell with `cell_id` at `time_idx` time point.

        Args:
            GRN_mtx (_type_): _description_
            time (_type_): _description_
            time_idx (_type_): _description_
            cell_id (_type_): _description_
            fate_idx (_type_): _description_
            gene_list (_type_): _description_
            percentile (int, optional): _description_. Defaults to 95.
            abs_edge (bool, optional): Whether to take `abs()` for GRN edges when aggregating them. 
                Defaults to False.
            source_gene (list, optional): If it is given, GRN will consider genes in this list for source genes in GRN extraction.
                Defaults to None.
            target_gene (list, optional): If it is given, GRN will consider genes in this list for target genes in GRN extraction.
                Defaults to None.                
        """
        self.gene_list=gene_list
        assert (source_gene is None) == (target_gene is None), "source_gene and target_gene must both be None or both not None"
        aggregate_id=-1 # whether to aggreate the weights for an axis.
        if source_gene is None and target_gene is None:
            grn_net=self.grn_net
            grn_net_L = []
            
            for i,itm in enumerate(grn_net):
                if itm=='pathway':
                    grn_net_L.append('L')
                    aggregate_id=i
                else:
                    grn_net_L.append(itm)
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
        if source_gene is None and target_gene is None:
            source_gene=self.node_list[grn_net_L[0]][cell_id]
            target_gene=self.node_list[grn_net_L[1]][cell_id]
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

    def create_edges(self,**kwargs):
        raise NotImplementedError
    def prune_nodes_from_edges(self,edge_list):
        node_list=[node for edge in edge_list for node in edge]
        node_list=sorted(list(dict.fromkeys(node_list)))
        G_node_list = list(self.G.nodes())
        
        for node in G_node_list:
            if not node in node_list:
                self.G.remove_node(node)
        self.node_label = {node: self.node_label[node] for node in self.G.nodes()}
        

        self.node_list={}
        for gene_type in self.signal_net:
            self.node_list[gene_type]={}
            for cell_id in self.cell_list:
                self.node_list[gene_type][cell_id]=[]
        for node in node_list:
            cell_id,gene_type,gene=node.split(' ')
            self.node_list[gene_type][int(cell_id)].append(gene)




        self.node_uniques={}
        for node_type in self.node_list.keys():
            self.node_uniques[node_type]=[]
        for node_type in self.node_list.keys():
            for cell_id in self.node_list[node_type]:
                self.node_uniques[node_type].extend(self.node_list[node_type][cell_id])
        for node_type in self.node_list.keys():
            self.node_uniques[node_type]=sorted(list(dict.fromkeys(self.node_uniques[node_type])))
    def create_G(self,**kwargs):
        
        labels = {}
        G = nx.DiGraph()
        node_type_list=self.signal_net
        for cell in self.cell_list:
            for node_type in node_type_list:
                G,labels = self.add_node_names(G,labels, cell, self.node_list,node_type=node_type)
            # G,labels = self.add_node_names(G,labels, cell, self.R_list,node_type='R')
            # G,labels = self.add_node_names(G,labels, cell, self.SPG_list,node_type='SPG')
        self.G=G
        self.node_label=labels
        # self.node_colors={}


        self.create_edges(**kwargs)





    def reset_GRN_time(self, time_pt):
        ## set default GRN in `self.GRN` for a selected time points `time_pt`
        self.default_time=time_pt
        self.GRN = self.GRN_all[self.default_time]

    def subgraph(self, attr_list):
        node_list=[]
        for node,attrs in self.G.nodes(data=True):
            keep = all(attrs.get(attr) == value for attr, value in attr_list.items())
            if keep: 
                node_list.append(node)
        return self.G.subgraph(node_list)
    def create_palette(self,palette={},palette_mode='gene',
                       palette_order=['Blues_r','Reds_r','Greens_r','Purples_r'],
                       **kwargs):
        """_summary_

        Args:
            palette (dict or str, optional): _description_. Defaults to {}.
            palette_mode (str, optional): 1) 'gene' to have different palette for L, R and SPG.
                                          2) 'cell' to have different palette for different cells
                                          3) 'cell_gene' to have different palette for different cells and different types of genes (L,R,SPG)
                Defaults to 'gene_type'.
        """
        if palette_mode=='gene':
            default_palette = {
                'L': 'Blues_r',
                'R': 'Reds_r',
                'SPG': 'Greens_r',
                'pathway': 'Blues_r'
            }
            # user_palette = palette.kwargs.get('palette', {})
            final_palette = {**default_palette, **palette}
            self.palette={}
            for key in final_palette.keys():
                if key in self.signal_net:
                    self.palette[key]=self.palette_values(gene_list=self.node_uniques[key],
                                                          palette=final_palette[key])
        elif palette_mode=='cell':
            default_palette = {
                0: 'Blues_r',
                1: 'Reds_r',
                2: 'Greens_r',
                3: 'Purples_r',
            }
            final_palette = {**default_palette, **palette}
            self.palette={}
            for cell_id in self.cell_list:
                tmp_node_list= [itm for node_type in self.signal_net for itm in self.node_list[node_type][cell_id]]
                self.palette[cell_id]=self.palette_values(gene_list=tmp_node_list,
                                                          palette=final_palette[cell_id])
        elif palette_mode=='cell_gene':
            # if isinstance(palette,dict):
            #     palette='self_paired'
            self.palette=self.palette_values(num_cell= len(self.cell_list_all),
                                             net_hierarchy=len(self.signal_net),
                                             palette_order=palette_order)
        
    def create_legend(self,palette_mode='gene',node_colors=None,**kwargs):
        if node_colors is None: 
            if palette_mode=='gene':
                legend={}
                for node_type in self.signal_net:
                    # legend[node_type]=[mpatches.Patch(color='none', label=node_type_map[node_type])]
                    # legend[node_type]+=[mpatches.Patch(color=self.palette[node_type][self.node_uniques[node_type].index(node)],label=node) for node in self.node_uniques[node_type]]
                    legend[node_type]=[mlines.Line2D([],[],color='none',marker='o',  linestyle='None', markersize=10,   label=node_type_map[node_type])]
                    legend[node_type]+=[mlines.Line2D([],[],
                                                        color=self.palette[node_type][self.node_uniques[node_type].index(node)],
                                                        marker='o',  
                                                        linestyle='None', 
                                                        markersize=10,label=node) 
                                        for node in self.node_uniques[node_type]]
                max_len = max([len(legend[node_type]) for  node_type in self.signal_net])
                for node_type in self.signal_net:
                    # legend[node_type]+=[mpatches.Patch(color='none', label='')] * (max_len - len(legend[node_type]))
                    legend[node_type]+=[mlines.Line2D([],[],color='none',marker='o',  linestyle='None',markersize=10, label='')] * (max_len - len(legend[node_type]))

                legend_patches=[]
                for node_type in self.signal_net:
                    legend_patches+=legend[node_type]
            elif palette_mode=='cell':
                # legend_patches=[]
                # # for cell_id in self.cell_list:
                # if self.cell_name is None:
                #     legend_patches = [mpatches.Patch(color=self.palette[cell_id][0],
                #                                     label='Fate '+str(cell_id)) 
                #                                     for cell_id in self.cell_list]
                # else:
                #     legend_patches = [mpatches.Patch(color=self.palette[cell_id][0],
                #                                     label=self.cell_name[cell_id]) 
                #                                     for cell_id in self.cell_list]    
                legend_patches = []
                for cell_id in self.cell_list:
                    label = f'Fate {cell_id}' if self.cell_name is None else self.cell_name[cell_id]
                    
                    node_marker = mlines.Line2D(
                        [], [], 
                        color=self.palette[cell_id][0],     # edge color
                        marker='o',                         # circle marker
                        linestyle='None',
                        markersize=10,                      # adjust size as needed
                        label=label                         # <-- label goes here
                    )
                    
                    legend_patches.append(node_marker)      
            elif palette_mode == 'cell_gene':
                legend_patches=[]

                color_id=0
                for cell_id in self.cell_list:
                    for node_type in self.signal_net:
                        color=self.palette[color_id]
                        label = 'Fate '+str(cell_id)+' '+node_type_map[node_type] if self.cell_name is None else self.cell_name[cell_id]+' '+node_type_map[node_type]

                        legend_patches+=[mlines.Line2D([],[],color=color,
                                                label=label,
                                                marker='o',                      
                                                linestyle='None',
                                                markersize=10,  )
                                                ]
                        color_id+=1
        else:
            legend_patches=[]
        return legend_patches

    @staticmethod
    def add_list(dict_,cell_id,new_list):
        if not cell_id in dict_.keys():
            dict_[cell_id]=[]
        dict_[cell_id]=sorted(list(dict.fromkeys(dict_[cell_id]+new_list)))
        return dict_
    @staticmethod
    def add_node_names(G,labels,cell,node_list, node_type='L'):
        for a in node_list[node_type][cell]:
            node_name = str(cell)+' ' +node_type+' '+a
            G.add_node(node_name,cell=cell,node_type=node_type)
            labels[node_name]=a
        return G,labels
    @staticmethod
    def node_horizontal_layout(H):
        num_nodes=len(H.nodes())
        x = np.linspace(-1,1,num_nodes)
        node_list=sorted(list(H.nodes()))
        pos={node:np.array([x[i],0])
            for i,node in enumerate(node_list)
        }
        return pos
    @staticmethod
    def palette_values(gene_list=None,
                       palette=None,
                       num_cell= None,
                       net_hierarchy=None,
                       palette_order=None):
        if gene_list is not None:
            if palette in ['Blues_r','Reds_r','Greens_r','Purples_r','Oranges_r','Blues','Reds','Greens','Purples','Oranges']:
                cmap = plt.get_cmap(palette)
                colors = [cmap(x) for x in np.linspace(0.1, 0.6, len(gene_list))]
            else:
                palettes = palette.split('+')
                num_palettes=len(palettes)
                if len(gene_list)<8:
                    colors = list(sns.color_palette(palette=palettes[0],n_colors=len(gene_list)))
                else:
                    colors = [c for p in palettes for c in sns.color_palette(palette=p)]
                    colors+=colors
                    colors+=colors
            # return list(sns.color_palette(palette=palette,n_colors=len(gene_list)))
            return colors
        else:
            # palettes_name = [palette[i] for i in range(num_cell)]
            palettes = [sns.color_palette(palette_order[i], n_colors=num_cell)[j]
                        for j in range(num_cell)
                        for i in range(net_hierarchy)]
                # palette1=sns.color_palette(palette='Set1',n_colors=gene_list)
                # palette2=sns.color_palette(palette='Pastel1',n_colors=gene_list)
                # return [color for pair in zip(palette1, palette2) for color in pair]
            return palettes



