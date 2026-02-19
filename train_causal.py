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
# import src.CCC.causality
import EchoSig.TIGON2
import EchoSig.TIGON2.trainer
import EchoSig.utility
import random
from EchoSig.utility import retrieval_data,cluster_trajectory,get_configs
from sklearn.cluster import KMeans



if __name__=="__main__":
    dataset = 'iPSC'
    species='human'
 
    config,_,time_scale,time_unit,n_fates,num_sample,time,time_GRN,time_float,lag_list,max_lag, source_id_lst,target_id_lst = get_configs(dataset)
    save_CCC_dir=config['save_dir']+'/CCC/'
    os.makedirs(save_CCC_dir,exist_ok=True)

    adata,data_train,X,embedding,time_data,model_ae,func,sigma,device,n_genes= retrieval_data(config)
    
    adata.write(save_CCC_dir+'/adata.h5ad')

    time_data=torch.tensor(time_data,dtype=torch.float32).to(device)
    time_data = time_data.reshape(len(time_data),1)   
    embedding=torch.tensor(embedding,dtype=torch.float32).to(device)
    # dt=torch.tensor(0.1,dtype=torch.float32).to(device)    

    
    z,z_original,vel,vel_original,g,J_vx_x,dg,time,time_GRN = EchoSig.TIGON2.generate.get_trajectory(num_sample,
                                                                                        time,
                                                                                        time_GRN,
                                                                                        data_train,
                                                                                        model_ae,func,sigma,
                                                                                        device,
                                                                                        time_ascend=True)
    
    time_GRN=time_GRN*time_scale
    time = time*time_scale

    # normalize reconstructed gene expression to their global maximum. 
    # This will be used as the input to CCC causality module
    max_exp = np.max(np.max(z_original,axis=0),axis=0)
    z_original_norm = z_original/max_exp




    ################################################################################################
    ###############################cluster module ###########################################
    ################################################################################################

    fate_idx, z_fate, z_original_fate = cluster_trajectory(n_fates=n_fates,z=z,z_original_norm=z_original_norm)

   
    
    df_LR_filter,df_RSPG_filter,df_LRSPG,LR_index,LRSPG_index, gene_list=EchoSig.CCC.filter_L_R_SPG(adata,species=species,signaling_type=None)
    L_list = list(dict.fromkeys(df_LR_filter.iloc[:, 0].values))
    R_list = list(dict.fromkeys(df_LR_filter.iloc[:, 1].values))
    SPG_list = list(dict.fromkeys(df_RSPG_filter.iloc[:, 0].values))
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
             fate_idx=fate_idx,
             z_fate = z_fate,
             z_original_fate = z_original_fate,
             gene_list=gene_list,
             L_list=L_list,
             R_list=R_list,
             SPG_list=SPG_list,
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
        print(save_sub_dir)

        df_CCC = EchoSig.CCC.trajCCC(source_id,target_id,z_original_fate,time, lag_list,max_lag,
                             df_LRSPG,gene_list,
                             save_dir=save_sub_dir,
                             curve_thred=0.2,time_scale=1, # time and time_GRN were scaled in previous lines to 'h'. Here we take `time_scale=1`
                             time_unit=time_unit) 







