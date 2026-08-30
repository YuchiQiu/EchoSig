import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, sys
import scanpy as sc
import EchoSig
import yaml
import EchoSig.CCC
# import EchoSig.CCC.causality
import EchoSig.EchoSigTraj
import EchoSig.EchoSigTraj.trainer
import EchoSig.utility
import random
from EchoSig.EchoSigTraj.utility import Sampling
from EchoSig.utility import retrieval_data,read_config_file,cluster_trajectory,get_configs
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
# from EchoSig.CCC.utility import filter_LR,filter_RTF,filter_L_R_TF,compute_CCC_strength,compute_CCC_avg_pathway
from scipy.sparse import issparse
import pickle
import matplotlib as mpl
import matplotlib.colors as mcolors
mpl.rcParams['pdf.fonttype'] = 42
# from statsmodels.tsa.stattools import grangercausalitytests
# import warnings
# warnings.filterwarnings("ignore")


if __name__=="__main__":
    dataset = 'iPSC'
    seed = 9
    species='human'
 
    config,save_CCC_dir,time_scale,time_unit,n_fates,num_sample,time,time_GRN,time_float,lag_list,max_lag, source_id_lst,target_id_lst = get_configs(dataset, seed=seed)


    os.makedirs(save_CCC_dir,exist_ok=True)


    adata,data_train,X,embedding,time_data,model_ae,func,sigma,device,n_genes= retrieval_data(config)
    
    adata.write(save_CCC_dir+'/adata.h5ad')

    time_data=torch.tensor(time_data,dtype=torch.float32).to(device)
    time_data = time_data.reshape(len(time_data),1)   
    embedding=torch.tensor(embedding,dtype=torch.float32).to(device)
    dt=torch.tensor(0.1,dtype=torch.float32).to(device)    

    
    z,z_original,vel,vel_original,g,J_vx_x,dg,time,time_GRN = EchoSig.EchoSigTraj.generate.get_trajectory(num_sample,
                                                                                        time,
                                                                                        time_GRN,
                                                                                        data_train,
                                                                                        model_ae,func,sigma,
                                                                                        device,
                                                                                        time_ascend=True)
    
    time_GRN=time_GRN*time_scale
    time = time*time_scale

    max_exp = np.max(np.max(z_original,axis=0),axis=0)

    z_original_norm = z_original/max_exp




    ################################################################################################
    ###############################cluster module ###########################################
    ################################################################################################

    fate_idx, z_fate, z_original_fate = cluster_trajectory(n_fates=n_fates,z=z,z_original_norm=z_original_norm)


    df_LR_filter,df_RTF_filter,df_LRTF,LR_index,LRTF_index, gene_list=EchoSig.CCC.filter_L_R_TF(adata,species=species,signaling_type=None)
    L_list = list(dict.fromkeys(df_LR_filter.iloc[:, 0].values))
    R_list = list(dict.fromkeys(df_LR_filter.iloc[:, 1].values))
    TF_list = list(dict.fromkeys(df_RTF_filter.iloc[:, 1].values))
    pathway_list =list(dict.fromkeys(df_LR_filter.iloc[:, 2].values))

    np.savez(save_CCC_dir+'traj.npz',
             J_vx_x=J_vx_x,
             g=g,
             dg=dg,
             z=z,
             vel=vel,
             vel_original=vel_original,
             z_original=z_original,
             z_original_norm=z_original_norm,
             cell_name=None,
             fate_idx=fate_idx,
             z_fate = z_fate,
             z_original_fate = z_original_fate,
             gene_list=gene_list,
             L_list=L_list,
             R_list=R_list,
             TF_list=TF_list,
             pathway_list=pathway_list,
             time=time,
             time_GRN=time_GRN,
             time_unit=time_unit,
             time_scale=time_scale,
             )
    


    stat_test_map = {'ftest':'F-test',
                    'chi2':'Chi2 test',
                    }
    for i in range(len(source_id_lst)):
        source_id = source_id_lst[i]
        target_id = target_id_lst[i]        
        save_sub_dir=save_CCC_dir+'/'+str(source_id)+'to'+str(target_id)+'/'
        

        df_CCC = EchoSig.CCC.trajCCC(source_id,target_id,z_original_fate,time, lag_list,max_lag,
                             df_LRTF,gene_list,
                             save_dir=save_sub_dir,
                             curve_thred=0.2, p_thred=0.05, save_fig=False,time_scale=1, # time and time_GRN were scaled in previous lines to 'h'. Here we take `time_scale=1`
                             time_unit=time_unit) 







