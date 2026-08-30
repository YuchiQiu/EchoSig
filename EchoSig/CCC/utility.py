import numpy as np
import scanpy as sc
import pandas as pd
import pkgutil
import io
import anndata
import warnings
import os


def _database_bytes(relative_path):
    """Load a bundled database, or one stored under ``ECHOSIG_DB_DIR``."""
    data = pkgutil.get_data("EchoSig.CCC", relative_path)
    if data is not None:
        return data
    database_root = os.environ.get("ECHOSIG_DB_DIR")
    if database_root:
        path = os.path.join(database_root, relative_path)
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                return handle.read()
    raise FileNotFoundError(
        f"Missing EchoSig database file: {relative_path}. "
        "Place it in EchoSig/CCC or set ECHOSIG_DB_DIR."
    )
def split_rows(df, col_indices=[0, 1],):

    df_new = []

    # Iterate through the rows of the DataFrame
    for index, row in df.iterrows():
        # split_rows = [row]

        # Check each specified column
        for col_index in col_indices:
            new_split_rows = []
            row0=row.iloc[0]
            row1=row.iloc[1]
            row0_split=row0.split('_')
            row1_split=row1.split('_')
            for r0 in row0_split:
                for r1 in row1_split:
                    new_row = row.copy()
                    new_row.iloc[0]=r0
                    new_row.iloc[1]=r1
                    df_new.append(new_row)
    df_new = pd.DataFrame(df_new).drop_duplicates().reset_index().drop(columns=['index'])
    
    return df_new
def filter_LR(adata,species= 'mouse',signaling_type="Secreted Signaling",):
    """ Find Ligand-Receptor pairs that appeared in `addata`
    Args:
        adata (anndata)
        species (str, optional): Specifies the species of the dataset. 
            Accepted values are 'human', 'mouse', and 'zebrafish'. Defaults to 'mouse'.        
        signaling_type (str, optional): Specifies the signaling type. 
            Accepted values are None, 'Secreted Signaling', 'Cell-Cell Contact', 'ECM-Receptor'.  
            If set to None, all types of signaling are considered. Defaults to 'Secreted Signaling'.
    Returns:
        _type_: _description_
    """
    gene_list=list(adata.var_names)
    data = _database_bytes("CellChatDB/CellChatDB.ligrec."+species+".csv")
    df_LR = pd.read_csv(io.BytesIO(data), index_col=0)
    if not signaling_type is None:
        df_LR = df_LR[df_LR.iloc[:,3] == signaling_type]
    LR_filtered = pd.DataFrame()
    LR_idx=[]
    for index, row in df_LR.iterrows():
        ligand=row[0].split('_')
        receptor=row[1].split('_')
        lr=ligand+receptor
        if all(elem in gene_list for elem in lr):
            LR_filtered = pd.concat([LR_filtered, row.to_frame().T], ignore_index=True)
            LR_idx.append([[gene_list.index(l) for l in ligand],[gene_list.index(r) for r in receptor]])
    LR_filtered=LR_filtered.copy()
    LR_filtered.columns = ['Ligand','Receptor','Pathway','Signal Type']
    LR_filtered['Interaction Name'] = LR_filtered['Ligand']+'-'+LR_filtered['Receptor']

    return LR_filtered, LR_idx,gene_list
def filter_RTF(adata,species = 'mouse'):
    gene_list=list(adata.var_names)
    data = _database_bytes("exFinderDB/RTF_layer2_"+species+".csv")
    df_RTF = pd.read_csv(io.BytesIO(data),index_col=0)
    RTF_filtered = pd.DataFrame()
    for index, row in df_RTF.iterrows():
        receptor = row['from']
        tf = row['to']
        if all(elem in gene_list for elem in [receptor]+[tf]):
            RTF_filtered = pd.concat([RTF_filtered,row.to_frame().T],ignore_index=True)

    return RTF_filtered
def filter_L_R_TF(adata,species='mouse',
                  signaling_type=None,
                  pathway = None,):
    """ Find Ligand-Receptor-TF graphs that appeared in `addata`
    We don't CONSIDER multiple receptors or ligands in this function. 
    Args:
        adata (anndata)
        species (str, optional): Specifies the species of the dataset. 
            Accepted values are 'human', 'mouse', and 'zebrafish'. Defaults to 'mouse'.        
        signaling_type (str, optional): Specifies the signaling type. 
            Accepted values are None, 'Secreted Signaling', 'Cell-Cell Contact', 'ECM-Receptor'.  
            If set to None, all types of signaling are considered. Defaults to 'Secreted Signaling'.
        pathway (str, optional): Specifies the pathway type. 
            If set to None, all pathways are considered.
            e.g., pathway='FGF8'
    Returns:
        _type_: _description_
    """
    gene_list=list(adata.var_names)
    data = _database_bytes("CellChatDB/CellChatDB.ligrec."+species+".csv")
    df_LR = pd.read_csv(io.BytesIO(data), index_col=0)
    if not signaling_type is None:
        df_LR = df_LR[df_LR.iloc[:,3] == signaling_type]
    if not pathway is None:
        df_LR = df_LR[df_LR.iloc[:,2] == pathway]
    df_LR=split_rows(df_LR)
    mask = (
    df_LR.iloc[:, 0].isin(gene_list) &
    df_LR.iloc[:, 1].isin(gene_list)
    )
    df_LR_filter = df_LR[mask]
    df_LR_filter = df_LR_filter.reset_index(drop=True)

    # LR_filtered = pd.DataFrame()
    # for index, row in df_LR.iterrows():
    #     ligand=row[0].split('_')
    #     receptor=row[1].split('_')
    #     lr=ligand+receptor
    #     if all(elem in gene_list for elem in lr):
    #         LR_filtered = pd.concat([LR_filtered, row.to_frame().T], ignore_index=True)
    # LR_filtered=LR_filtered.copy()
    df_LR_filter.columns = ['Ligand','Receptor','Pathway','Signal Type']
    # LR_filtered['Interaction Name'] = LR_filtered['Ligand']+'-'+LR_filtered['Receptor']

    # receptor_list=list(LR_filtered['Receptor'])
    data = _database_bytes("exFinderDB/layer2_"+species+".csv")
    df_RTF = pd.read_csv(io.BytesIO(data),index_col=0)
    mask = (
    df_RTF.iloc[:, 0].isin(gene_list) &
    df_RTF.iloc[:, 1].isin(gene_list) &
    df_RTF.iloc[:, 0].isin(df_LR_filter['Receptor'].values)
    )
    df_RTF_filter = df_RTF[mask]
    df_RTF_filter = df_RTF_filter.reset_index(drop=True)
    df_LRTF=pd.DataFrame()
    LRTF_index=[]
    for index, row in df_LR_filter.iterrows():
        ligand=row['Ligand']
        receptor=row['Receptor']
        pathway=row['Pathway']
        sig_type=row['Signal Type']
        tfs = list(df_RTF_filter.loc[df_RTF_filter['from']==receptor]['to'])
        for tf in tfs:
            df_LRTF=pd.concat([df_LRTF, 
                               pd.DataFrame([ligand,receptor,tf,pathway,sig_type]).T
                               ], 
                               ignore_index=True)
            LRTF_index.append([gene_list.index(ligand),
                               gene_list.index(receptor),
                               gene_list.index(tf)])
    df_LRTF.columns = ['Ligand','Receptor','TF','Pathway','Signal Type']
    LRTF_index=np.array(LRTF_index)

    LR_index=[]
    for index, row in df_LR_filter.iterrows():
        ligand=row['Ligand']
        receptor=row['Receptor']
        LR_index.append([gene_list.index(ligand),
                         gene_list.index(receptor)])
    LR_index=np.array(LR_index)

    # RTF_filtered = pd.DataFrame()
    # for index, row in df_RTF.iterrows():
    #     receptor = row['from']
    #     tf = row['to']
    #     if receptor in receptor_list:
    #         if all(elem in gene_list for elem in [receptor]+[tf]):
    #             RTF_filtered = pd.concat([RTF_filtered,row.to_frame().T],ignore_index=True)
    return df_LR_filter,df_RTF_filter,df_LRTF,LR_index,LRTF_index, gene_list



def compute_gene_mean(gene,max_exp,method='truncatedMean',trim=0.):
    """
    Compute the Normalized Average Gene Expression using different statistical methods.
    Parameters:
        gene (numpy.ndarray): A 2D array of gene expression data, 
                            where rows are cells and columns are genes.
        max_exp (numpy.ndarray): a 1D array of max of gene expression,
                                its length and order is the same with columns of `gene`
        method (str): The method to calculate the average gene expression.
                    - 'truncatedMean': Computes a trimmed mean, excluding
                        a proportion of values at both ends.
                            If `trim` is 0, it calculates arithmetric mean (Default).
                    - 'triMean': Computes the trimean, a robust measure of 
                        central tendency that combines median and quartiles.
        trim (float): The proportion of data to trim for 'truncatedMean'.
                    Default is 0., meaning no trimming.
    Returns:
        numpy.ndarray: A 1D array where each value represents the average
                    expression of a gene across cells using the specified method.
    """
    from scipy.stats import trim_mean
    if method == 'truncatedMean':
        avg_gene = np.apply_along_axis(lambda x: 
                                        trim_mean(x, 
                                                    proportiontocut=trim),
                                                    axis=0,
                                                    arr=gene
                                                    )
    elif method == 'triMean':
        Q1 = np.percentile(gene, 25, axis=0)
        Q3 = np.percentile(gene, 75, axis=0)
        median = np.percentile(gene, 50, axis=0)
        avg_gene = (Q1 + 2 * median + Q3) / 4

    return avg_gene/max_exp
def compute_geo_avg(exp):
    """Calculate Geometric average of an array
    """
    return np.power(np.prod(exp),1/len(exp))
def compute_CCC_strength(LR_idx,source_exp,target_exp,max_exp,K=0.5):
    source_exp_avg = compute_gene_mean(source_exp,max_exp,method = 'truncatedMean',trim=0.)
    target_exp_avg = compute_gene_mean(target_exp,max_exp,method = 'truncatedMean',trim=0.)
    # CCC_score=compute_CCC_strength(LR_idx,source_exp_avg,target_exp_avg)

    CCC_score =[]
    for l_idx,r_idx in LR_idx:
        L = compute_geo_avg(source_exp_avg[l_idx])
        R = compute_geo_avg(target_exp_avg[r_idx])
        score = (L*R)/(K+L*R)
        CCC_score.append(score)
    CCC_score=np.array(CCC_score)
    # df=LR_df.copy()
    # # df.columns = ['Ligand','Receptor','Pathway','Signal Type']
    # # df['Interaction Name'] = df['Ligand']+'-'+df['Receptor']
    # df['CCC score']=CCC_score

    # # df = pd.DataFrame({'CCC_score': CCC_score, 'pathway': pathway,})
    # pathway_score_df = df.groupby('Pathway')['CCC score'].mean().to_dict()
    # pathway_score = list(pathway_score_df.values())
    # pathway_list = list(pathway_score_df.keys())
    return CCC_score#,pathway_score,pathway_list
def compute_CCC_avg_pathway(CCC_score,pathway,axis=-1):
    CCC_score=np.array(CCC_score)
    CCC_pathway = {}
    for path in list(set(pathway)):
        onehot = pathway==path
        broadcast_shape = [1] * CCC_score.ndim
        broadcast_shape[axis] = -1 
        onehot_reshaped = onehot.reshape(broadcast_shape)
        CCC_pathway[path] = np.sum(CCC_score*onehot_reshaped,axis=axis)
    return CCC_pathway
