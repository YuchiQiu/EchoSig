# EchoSig: Dissecting multicellular temporal causal signaling flows 
![overview.png](https://github.com/YuchiQiu/EchoSig/blob/main/overview.png)

**EchoSig** is a deep learning–enabled temporal causal framework for reconstructing multicellular signaling flows from time-series scRNA-seq and spatial transcriptomics data. It builds upon **TIGON+**, a fast and robust dynamical model that infers continuous cellular trajectories and time-resolved gene regulatory networks (GRNs). EchoSig models cell–cell communication (CCC) by estimating signaling strengths and signaling propagation times to capture ligand-associated transmission delays. By integrating intercellular CCC with intracellular GRNs, EchoSig uncovers hierarchical and multiscale causal interaction networks underlying dynamic signaling flows.

# Installation and input data downloads

# Usage
We provide a work flow for analyzing iPSC data. Please follow the scripts below and adapt them as needed to customize the analysis for your own dataset. Parameters used for the analysis are provided by `.yaml` files in `config/`. 
## STEP 1: Train TIGON+ model
```
python3 train_TIGON.py iPSC
```
## STEP 2: Run causality module for calculating CCC and GRN networks
```
python3 train_causal.py iPSC
```
## STEP 3: Downstream network aggregation and visualization
Refer to `tutorial/iPSC.ipynb` for detailed analysis.

# Reference
Qiu, Y., Sha, Y., & Nie, Q. (2026). Dissecting multicellular temporal causal signaling flows. (submitted)
