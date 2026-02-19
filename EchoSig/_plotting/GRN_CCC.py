import anndata
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import networkx as nx
import seaborn as sns
import pandas as pd
from collections import defaultdict
import os
from .base import *
import matplotlib as mpl
palette_short = list(sns.color_palette("Set1") \
                + sns.color_palette("pastel") \
                + sns.color_palette("tab10")\
                + sns.color_palette('husl') \
                + sns.color_palette("tab20") \
                + sns.color_palette("tab20b") \
                + sns.color_palette("tab20c")\
                + sns.color_palette("Set1")\
                + sns.color_palette("Set2")\
                + sns.color_palette('Set3')\
                + sns.color_palette('Dark2')\
                + sns.color_palette('Pastel1')\
                + sns.color_palette('Pastel2'))

def unique_or_single(x):
    unique_vals = list(x.unique())
    if len(unique_vals) == 1:
        return unique_vals[0]
    else:
        return unique_vals
class DotsPlots_L_R_SPG(BaseGraph):
    def __init__(self,test='ftest',net=['L','R','SPG','L'],cell_name=None):
        super().__init__(test,cell_name=cell_name)
        self.net=net
        self.signal_net=net[0:3]
        self.grn_net = net[2:]
    def create_df_CCC(self,cell_pairs=None):
        if cell_pairs is None:
            cell_pairs = self.cell_pairs
        merged_df = pd.concat(
            [self.CCC_data[cell_pair].assign(cell_pair=f"{cell_pair[0]}-{cell_pair[1]}") for cell_pair in cell_pairs],
            ignore_index=False
            )
        self.CCC_df=merged_df
    def create_df_GRN(self,cell_list = None,time_list=None):
        if cell_list is None:
            cell_list = self.cell_list
        if time_list is None:
            time_list = self.time_list
        merged_df = pd.concat(
            [pd.DataFrame(self.GRN_all[time_pt][cell_id]).assign(cell_id=cell_id, time=time_pt) 
            for cell_id in cell_list 
            for time_pt in time_list],
            ignore_index=True
            )
        self.GRN_df=merged_df
    def draw_LRSPG_dots(self,ax=None,cell_pairs=None,cell_pair_order=None,pathway_order=None,color_map='Spectral',time_unit='h',time_scale=1,**kwargs):
        self.create_df_CCC(cell_pairs=cell_pairs)
        merged_df=self.CCC_df.copy()
        merged_df['scaled_lag'] = merged_df['lag'] * time_scale
        merged_df['regulation'] = np.where(
            merged_df['stat ' + self.test] > 0,
            'Act.',
            'Inh.'
        )
        merged_df['stat ' + self.test] = merged_df['stat ' + self.test].abs()
        merged_df['cell_reg'] = merged_df['cell_pair'] + ',' + merged_df['regulation']
        merged_df['cell_reg'] = pd.Categorical(merged_df['cell_reg'],categories=sorted(merged_df['cell_reg'].unique()))
        if cell_pair_order is None:
            cell_pair_order=sorted(merged_df['cell_pair'].unique())
        merged_df['cell_pair'] = pd.Categorical(merged_df['cell_pair'],categories=cell_pair_order)
        merged_df['index'] = merged_df.index        
        # merged_df['index'] = pd.Categorical(merged_df['index'],categories=sorted(merged_df['index'].unique()))
        if pathway_order is not None:
            merged_df['pathway'] = pd.Categorical(
                merged_df['pathway'],
                categories=pathway_order,
                ordered=True
            )
        merged_df = merged_df.sort_values(by=['pathway', 'index'])
        # self.merged_df=merged_df
        if ax==None:
            ax=plt.gca()
        unique_indices = merged_df['index'].unique()

        # Create mapping: index → pathway
        index_to_pathway = merged_df.drop_duplicates('index').set_index('index')['pathway']

        boundaries = []
        label_positions = []
        current_pathway = None
        start_i = 0

        for i, idx in enumerate(unique_indices):
            pathway = index_to_pathway[idx]
            if pathway != current_pathway:
                if current_pathway is not None:
                    # midpoint between start_i and i-1
                    mid_i = (start_i + i - 1) / 2
                    label_positions.append((mid_i, current_pathway))
                    boundaries.append(i - 0.5)
                start_i = i
                current_pathway = pathway

        mid_i = (start_i + len(unique_indices) - 1) / 2
        label_positions.append((mid_i, current_pathway))

        sns.scatterplot(
            data=merged_df,
            x='cell_pair',
            y='index',
            size='stat ' + self.test,
            hue='scaled_lag',
            # hue_order=sorted(merged_df['cell_reg'].unique()),
            sizes=(20, 200),
            palette=color_map,
            edgecolor=None,
            alpha=0.7,
            legend='brief',
            ax=ax
        )
        # print(ax.get_yticks())

        for b in boundaries:
            ax.axhline(b, color='gray', linestyle='--', linewidth=0.5)

        ax_right = ax.twinx()
        ax_right.set_ylim(ax.get_ylim())
        # Set the ticks at pathway midpoints, with names as labels
        ax_right.set_yticks([pos for pos, name in label_positions])
        ax_right.set_yticklabels([name for pos, name in label_positions])
        ax_right.set_ylabel('pathway')
        # ax_right.invert_yaxis()
        # Hide the secondary axis frame
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['left'].set_visible(False)
        ax_right.spines['bottom'].set_visible(False)
        # ax_right.set_yticklabels(ax_right.get_yticklabels(), rotation=180)

        ax_right.tick_params(axis='y', direction='out', length=4, pad=4)




        handles, labels = ax.get_legend_handles_labels()
        stat_start_idx = next(i for i, l in enumerate(labels) if 'stat' in l.lower())
        size_handles = handles[stat_start_idx+1:]
        size_labels = labels[stat_start_idx+1:]
        ax.legend(size_handles, 
                  size_labels, 
                  title='strength ('+stat_test_map[self.test]+')', 
                      loc='upper left', 
                      bbox_to_anchor=(1.15, 0.85),  
                      frameon=False)

        ax.set_xlabel('cell pair')
        ax.set_ylabel('ligand-receptor-SPG module')
        current_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        if self.cell_name is not None:
            new_labels = [ '-'.join([self.cell_name[int(cell)] for cell in tick.split('-')]) for tick in current_labels]
            ax.set_xticklabels(new_labels, rotation=270)
        else:
            ax.set_xticklabels(current_labels,rotation=270)
        ax.set_xticks(ax.get_xticks())
        # ax.set_yticklabels(ax.get_yticklabels(), rotation=180)


        # plt.colorbar(ax)
        norm = mpl.colors.Normalize(vmin=merged_df['scaled_lag'].min(), 
                                    vmax=merged_df['scaled_lag'].max())
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=color_map)
        sm.set_array([])
        fig = plt.gcf()  # get current figure
        cbar_ax = fig.add_axes([1.15, 0.85, 0.15, 0.01])
        cbar = plt.colorbar(sm, cax=cbar_ax, 
                            orientation='horizontal', 
                            pad=0.2, 
                            shrink=0.3)
        cbar.set_label(f'SPT ({time_unit})')
        cbar.outline.set_visible(False)

        plt.box(False)
        
        return ax,merged_df
    
    def draw_LR_dots(self,ax=None,cell_pairs=None,cell_pair_order=None,pathway_order=None,color_map='Spectral',time_unit='h',time_scale=1,**kwargs):
        self.create_df_CCC(cell_pairs=cell_pairs)
        merged_df=self.CCC_df.copy()
        merged_df['scaled_lag'] = merged_df['lag'] * time_scale
        merged_df['regulation'] = np.where(
            merged_df['stat ' + self.test] > 0,
            'Act.',
            'Inh.'
        )       
        print(merged_df.keys())
        merged_df['stat ' + self.test] = merged_df['stat ' + self.test].abs()
        merged_df = merged_df.groupby(['LR','cell_pair']).agg({
            'stat '+self.test: 'sum',
            'scaled_lag': 'mean',
            'pathway':unique_or_single,
            'signal type':unique_or_single,
        }).reset_index()     

        # merged_df['cell_reg'] = merged_df['cell_pair'] + ',' + merged_df['regulation']
        # merged_df['cell_reg'] = pd.Categorical(merged_df['cell_reg'],categories=sorted(merged_df['cell_reg'].unique()))
        if cell_pair_order is None:
            cell_pair_order=sorted(merged_df['cell_pair'].unique())
        if pathway_order is not None:
            merged_df['pathway'] = pd.Categorical(
                merged_df['pathway'],
                categories=pathway_order,
                ordered=True
            )
        merged_df['cell_pair'] = pd.Categorical(merged_df['cell_pair'],categories=cell_pair_order)
        merged_df['index'] = merged_df['LR'] 
   
        # merged_df['index'] = pd.Categorical(merged_df['index'],categories=sorted(merged_df['index'].unique()))
        merged_df = merged_df.sort_values(by=['pathway', 'index'])
        # self.merged_df=merged_df
        if ax==None:
            ax=plt.gca()
        unique_indices = merged_df['index'].unique()

        # Create mapping: index → pathway
        index_to_pathway = merged_df.drop_duplicates('index').set_index('index')['pathway']

        boundaries = []
        label_positions = []
        current_pathway = None
        start_i = 0

        for i, idx in enumerate(unique_indices):
            pathway = index_to_pathway[idx]
            if pathway != current_pathway:
                if current_pathway is not None:
                    # midpoint between start_i and i-1
                    mid_i = (start_i + i - 1) / 2
                    label_positions.append((mid_i, current_pathway))
                    boundaries.append(i - 0.5)
                start_i = i
                current_pathway = pathway

        mid_i = (start_i + len(unique_indices) - 1) / 2
        label_positions.append((mid_i, current_pathway))

        sns.scatterplot(
            data=merged_df,
            x='cell_pair',
            y='index',
            size='stat ' + self.test,
            hue='scaled_lag',
            # hue_order=sorted(merged_df['cell_reg'].unique()),
            sizes=(20, 200),
            palette=color_map,
            edgecolor=None,
            alpha=0.7,
            legend='brief',
            ax=ax
        )
        # print(ax.get_yticks())

        for b in boundaries:
            ax.axhline(b, color='gray', linestyle='--', linewidth=0.5)

        ax_right = ax.twinx()
        ax_right.set_ylim(ax.get_ylim())
        # Set the ticks at pathway midpoints, with names as labels
        ax_right.set_yticks([pos for pos, name in label_positions])
        ax_right.set_yticklabels([name for pos, name in label_positions])
        ax_right.set_ylabel('pathway')
        # ax_right.invert_yaxis()
        # Hide the secondary axis frame
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['left'].set_visible(False)
        ax_right.spines['bottom'].set_visible(False)

        ax_right.tick_params(axis='y', direction='out', length=4, pad=4)
        ax_right.set_yticklabels(ax_right.get_yticklabels(), rotation=180)




        handles, labels = ax.get_legend_handles_labels()
        stat_start_idx = next(i for i, l in enumerate(labels) if 'stat' in l.lower())
        size_handles = handles[stat_start_idx+1:]
        size_labels = labels[stat_start_idx+1:]
        ax.legend(size_handles, 
                  size_labels, 
                  title='strength ('+stat_test_map[self.test]+')', 
                      loc='upper left', 
                      bbox_to_anchor=(1.15, 0.85),  
                      frameon=False)

        ax.set_xlabel('cell pair')
        ax.set_ylabel('ligand-receptor pair')
        current_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        if self.cell_name is not None:
            new_labels = [ '-'.join([self.cell_name[int(cell)] for cell in tick.split('-')]) for tick in current_labels]
            ax.set_xticklabels(new_labels, rotation=270)
        else:
            ax.set_xticklabels(current_labels, rotation=270)

        ax.set_xticks(ax.get_xticks())
        ax.set_yticklabels(ax.get_yticklabels(), rotation=180)


        # plt.colorbar(ax)
        norm = mpl.colors.Normalize(vmin=merged_df['scaled_lag'].min(), 
                                    vmax=merged_df['scaled_lag'].max())
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=color_map)
        sm.set_array([])
        fig = plt.gcf()  # get current figure
        cbar_ax = fig.add_axes([1.15, 0.85, 0.15, 0.01])
        cbar = plt.colorbar(sm, cax=cbar_ax, 
                            orientation='horizontal', 
                            pad=0.2, 
                            shrink=0.3)
        cbar.set_label(f'SPT ({time_unit})')
        cbar.outline.set_visible(False)


        plt.box(False)
        
        return ax,merged_df
    
    
    def draw_GRN_dots(self,ax=None, cell_list=None,time_list=None,time_unit='h',time_scale=1,color_map='RdBu',**kwargs):
        self.create_df_GRN(cell_list=cell_list,time_list=time_list)
        if ax==None:
            ax=plt.gca()
        merged_df=self.GRN_df
        merged_df['Gene Pair'] = merged_df['source'] + '-' + merged_df['target']
        merged_df['cell_time'] = merged_df['cell_id'].astype(str) + ',' + (merged_df['time']*time_scale).astype(str)+time_unit
        merged_df['Gene Pair'] = pd.Categorical(
            merged_df['Gene Pair'],
            categories=sorted(merged_df['Gene Pair'].unique()),
            ordered=True
        )
        merged_df['regulation'] = merged_df['score'].apply(lambda x: 'Act.' if x > 0 else 'Inh.')
        merged_df['Strength'] = merged_df['score'].abs()

        # vlim = min(merged_df['score'].max(),-merged_df['score'].min())  
        sns.scatterplot(
            data=merged_df,
            x='cell_time',
            y='Gene Pair',
            size='Strength',
            hue='regulation',
            palette={'Act.': 'blue', 'Inh.': 'red'},
            # hue_order=sorted(merged_df['cell_time'].unique()),
            sizes=(20,200),
            # hue_norm=(-vlim, vlim),
            # palette=color_map,
            edgecolor=None,
            alpha=0.7,
            legend='brief',
            ax=ax
        )
        xticklabels = ax.get_xticklabels()
        new_labels = [
            f"{label.get_text().split(',')[0]}\n{label.get_text().split(',')[1]}" 
            for label in xticklabels
        ]
        ax.set_xticklabels(new_labels, rotation=45)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, 
            labels, 
            title="",
            bbox_to_anchor=(1.05, 1),  # x=1.05 pushes it outside to the right
            loc='upper left',          # anchor point inside the legend box
            borderaxespad=0.
        )
        ax.set_xlabel('Cell & Time')
        return ax
    
def get_GRN(J_vx_x,
            time_idx,
            cell_id,
            fate_idx,
            gene_list,
            source_gene=None,
            target_gene=None,
            percentile=95):
    if isinstance(cell_id,int):
        GRN_df = pd.DataFrame(np.mean(J_vx_x[fate_idx[cell_id,:],time_idx,:,:],axis=0), index=gene_list, columns=gene_list)
    elif isinstance(cell_id,list):
        combined_list = fate_idx[cell_id, :].any(axis=0)
        GRN_df = pd.DataFrame(np.mean(J_vx_x[combined_list,time_idx,:,:],axis=0), index=gene_list, columns=gene_list)
    thred=np.percentile(np.abs(GRN_df),q=percentile,axis=(0,1))
    if target_gene is not None and source_gene is not None:
        GRN_df = GRN_df.loc[target_gene, source_gene]
    elif target_gene is not None:
        GRN_df = GRN_df.loc[target_gene, :]
    elif source_gene is not None:
        GRN_df = GRN_df.loc[:, source_gene]
    return GRN_df,thred
# def GRN_HeatMap(J_vx_x,
#                 time,
#                 time_idx,
#                 cell_id,
#                 fate_idx,
#                 gene_list,
#                 source_gene=None,
#                 target_gene=None,
#                 time_scale=1,
#                 time_unit='d',
#                 # clustering=True,
#                 ax=None):

#     GRN_df = get_GRN(J_vx_x,
#                      time_idx,
#                      cell_id,
#                      fate_idx,
#                      gene_list,
#                      source_gene,
#                      target_gene)
#     if ax is None:
#         ax=plt.gca()
#     g = sns.heatmap(GRN_df, 
#                 cmap='vlag', 
#                 #    row_cluster=clustering, 
#                 #    col_cluster=clustering,
#                     xticklabels=True, 
#                     yticklabels=True,
#                     center=0,
#                     cbar_kws={"shrink": 0.4,
#                                 "orientation":'horizontal',
#                                 "aspect": 10, 
#                                 "pad": 0.2
#                                 },
#                     ax=ax)
        
#     ax.set_xlabel("Source genes")
#     ax.set_ylabel("Target genes")
#     colorbar = g.collections[0].colorbar
#     colorbar.set_label("Strength")
#     # plt.savefig(dataset + '/'+'GRN_cell_'+str(fate)+'_t='+str(time_idx)+'.pdf')
#     # plt.show()
#     ax.set_title('fate '+str(cell_id)+ ' at '+str(time[time_idx]*time_scale)+time_unit)

def GRN_Network(J_vx_x,z_original,
                time,
                time_idx,
                cell_id,
                fate_idx,
                gene_list,
                source_gene=None,
                target_gene=None,
                source_gene_order=None,
                target_gene_order=None,
                # edge_thred=0.1,
                percentile=95,
                prun_node=False,
                time_scale=1,
                time_unit='h',
                # clustering=True,
                width_scale=1.,
                nd_size=200,
                fontsize=10,
                ax=None):
    GRN_df,thred = get_GRN(J_vx_x,
                     time_idx,
                     cell_id,
                     fate_idx,
                     gene_list,
                     source_gene,
                     target_gene,
                     percentile=percentile)
    if isinstance(cell_id,int):
        avg_exp=np.mean(z_original[fate_idx[cell_id],time_idx,:],axis=0)
    elif isinstance(cell_id,list):
        combined_list = fate_idx[cell_id, :].any(axis=0)
        avg_exp=np.mean(z_original[combined_list,time_idx,:],axis=0)
    if ax is None:
        ax=plt.gca()
    G = nx.DiGraph()
    target_node_list=sorted(list(GRN_df.index))
    source_node_list=sorted(list(GRN_df.columns))

    node_labels={}
    node_colors={}
    node_size={}


    score_max=np.max(np.abs(GRN_df))
    # print(GRN_df)
    # print(score_max)
    edge_positive={}
    edge_negative={}
    source_node_list_prune=[]
    target_node_list_prune=[]
    for node1 in target_node_list:
        for node2 in source_node_list:
            score=GRN_df[node2][node1]
            node_pair=('source '+node2,'target '+node1)
            # print(score)
            if score>thred:
                edge_positive[node_pair]=score
                if not node1 in target_node_list_prune:
                    target_node_list_prune.append(node1)
                if not node2 in source_node_list_prune:
                    source_node_list_prune.append(node2)
            elif score<-thred:
                edge_negative[node_pair]=-score
                if not node1 in target_node_list_prune:
                    target_node_list_prune.append(node1)
                if not node2 in source_node_list_prune:
                    source_node_list_prune.append(node2)  
    if prun_node:
        source_node_list=source_node_list_prune
        target_node_list=target_node_list_prune
    i=0
    for node in source_node_list:
        node_name='source '+ node
        G.add_node(node_name,layer=1)
        node_labels[node_name]=node
        node_colors[node_name]=palette_short[i]
        node_size[node_name]=avg_exp[gene_list.index(node)]
        i+=1
    for node in target_node_list:
        node_name='target '+node
        G.add_node(node_name,layer=2)
        node_labels[node_name]=node
        if node in source_node_list:
            node_colors['target '+node]=node_colors['source '+node]
        else:
            node_colors['target '+node]=palette_short[i]
            i+=1
        node_size[node_name]=avg_exp[gene_list.index(node)]    
    pos = nx.multipartite_layout(G,
                                 subset_key={'layer1':['source '+node for node in source_node_list],
                                             'layer2':['target '+node for node in target_node_list],
                                             },
                                             align='vertical',)
    print(np.array([pos[node] for node in pos.keys()  if 'source ' in node]).shape)
    print(np.array([pos[node] for node in pos.keys()  if 'target ' in node]).shape)
    array=np.array([pos[node] for node in pos.keys()  if 'source ' in node])
    sorted_indices = np.argsort(array[:, 1])[::-1]
    pos_source=array[sorted_indices,:]
    array=np.array([pos[node] for node in pos.keys()  if 'target ' in node])
    sorted_indices = np.argsort(array[:, 1])[::-1]
    pos_target=array[sorted_indices,:]
    pos2=pos.copy()
    if source_gene_order is not None:
        source_gene_order = source_gene_order +  [item for item in source_node_list if item not in source_gene_order]
        source_gene_order=['source '+node for node in source_gene_order]
        for i,node in enumerate(source_gene_order):
            pos2[node]=pos_source[i,:] 
    if target_gene_order is not None:
        target_gene_order = target_gene_order +  [item for item in target_node_list if item not in target_gene_order]        
        target_gene_order=['target '+node for node in target_gene_order]
        for i,node in enumerate(target_gene_order):
            pos2[node]=pos_target[i,:]    
    pos=pos2
    nx.draw_networkx_nodes(G,pos=pos,
                           node_color=[node_colors.get(node) for node in G.nodes],
                        #    node_size=[node_size.get(node)*nd_size for node in G.nodes()],
                        ax=ax,alpha=0.6,
                        )
    
    # nx.draw_networkx_labels(G,pos=pos,
    #                         labels = node_labels,
    #                         ax=ax)
    new_pos = {
        node: np.array([pos[node][0] - 0.025, pos[node][1]]) if 'source' in node
        else np.array([pos[node][0] + 0.025, pos[node][1]])
        for node in pos
    }
    for node, (x, y) in new_pos.items():
        alignment = 'right' if 'source' in node else 'left'  # adjust threshold as needed
        ax.text(
            x, y, node_labels[node],
            horizontalalignment=alignment,
            verticalalignment='center',
            fontsize=fontsize,
            # fontweight='bold',
            # bbox=dict(facecolor='white', edgecolor='none', pad=0.2),
            # zorder=10
            )
    nx.draw_networkx_edges(G,pos,
                            arrowstyle='-[',
                            edgelist=list(edge_negative.keys()),
                            edge_color=[node_colors.get(u) for (u,v) in list(edge_negative.keys())],
                            ax=ax,
                            width=[width_scale*edge_negative[(u, v)] for (u, v) in list(edge_negative.keys())],
                            alpha=0.6)
    nx.draw_networkx_edges(G,pos,
                            edgelist=list(edge_positive.keys()),
                            edge_color=[node_colors.get(u) for (u,v) in list(edge_positive.keys())],
                            ax=ax,
                            width=[width_scale*edge_positive[(u, v)] for (u, v) in list(edge_positive.keys())],
                            alpha=0.6)    

    plt.box(False)
    return ax,edge_positive,edge_negative

