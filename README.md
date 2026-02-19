# EchoSig: Dissecting multicellular temporal causal signaling flows 
![overview.png](https://github.com/YuchiQiu/EchoSig/blob/main/overview.png)

**EchoSig** is a deep learning–enabled temporal causal framework for reconstructing multicellular signaling flows from time-series scRNA-seq and spatial transcriptomics data. It builds upon **TIGON+**, a fast and robust dynamical model that infers continuous cellular trajectories and time-resolved gene regulatory networks (GRNs). EchoSig models cell–cell communication (CCC) by estimating signaling strengths and signaling propagation times to capture ligand-associated transmission delays. By integrating intercellular CCC with intracellular GRNs, EchoSig uncovers hierarchical and multiscale causal interaction networks underlying dynamic signaling flows.

# Installation
Use pacakges in `requirement.txt` to install the enviroment. For example, install environment using `pip`
`pip install -r requirements.txt`

# Input and demo data download
Large input data and demo data are available at UIC Indigo server: `https://doi.org/10.25417/uic.31320709`. Please place them to the right path in the directory before running the code. They include
- databases for L-R, and R-SPG pairs in `EchoSig/CCC/`
- demo input data of AnnData for iPSC: `Input/iPSC.h5ad`
- demo output for iPSC analysis `Output/`

# Usage
We provide scripts of workflow for iPSC data. Please follow the scripts below and adapt them as needed for customize dataset. Parameters used for the training are provided by `config/*.yaml` files.  
## STEP 1: Train TIGON+ model
```
dataset=iPSC
python3 train_TIGON.py $dataset
```
## STEP 2: Run causality module for calculating CCC and GRN networks
```
dataset=iPSC
python3 train_causal.py $dataset
```
Parameters for causal inference are given in `EchoSig.utility.get_configs`. For example, `n_fates` determines number of cell fates from `num_sample` trajectories. In `train_causal.py`, `cell_name` is given a dictionary mapping cell fate index to its cell fate annotation. It needs to be defined before running the script.

## STEP 3: Downstream network aggregation and visualization
Refer to `tutorial/iPSC.ipynb` for detailed analysis.

# Output
All outputs are saved under `Output/$dataset/`.
- TIGON+ results are placed in this directory
  - `adata.h5ad` store the AnnData with inferred embedding information.
  - `AE.pth` and `ckpt.pth` store trained model parameters from VAE model and TIGON+ model, respectively.
- Causal model results are saved in `Output/$dataset/CCC/`
  - `traj.npz` store distinct cellular trajectories with their fate, velocity, growth, gene regulation etc. information.
  - subdirectory stores causal CCC results. For example, `1to0/results.csv` provide causal results from cell trajectory `1` to trajectory `0`. The correspondence of cell fate number with its name needs to be defined in `cell_name`. 

# Reference
Yuchi Qiu\*#, Yutong Sha\*, Qing Nie#. Dissecting multicellular temporal causal signaling flows (2026). (submitted)
