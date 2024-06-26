# BMR_Code

## This is the code for our paper "[Leveraging Brain Modularity Prior for Interpretable Representation Learning of fMRI](https://ieeexplore.ieee.org/document/10449452)"



## Abstract  

-  Resting-state functional magnetic resonance imaging (rs-fMRI) can reflect spontaneous neural activities in brain and is widely used for brain disorder analysis.Previous studies propose to extract fMRI representations through diverse machine/deep learning methods for subsequent analysis. But the learned features typically lack biological interpretability, which limits their clinical utility. From the view of graph theory, the brain exhibits a remarkable modular structure in spontaneous brain functional networks, with each module comprised of functionally interconnected brain regions-of-interest (ROIs). However, most existing learning-based methods for fMRI analysis fail to adequately utilize such brain modularity prior. In this paper, we propose a Brain Modularity-constrained dynamic Representation learning (BMR) framework for interpretable fMRI analysis, consisting of three major components: (1) dynamic graph construction, (2) dynamic graph learning via a novel modularity-constrained graph neural network(MGNN), (3) prediction and biomarker detection for interpretable fMRI analysis. Especially, three core neurocognitive modules (i.e., salience network, central executive network, and default mode network) are explicitly incorporated into the MGNN, encouraging the nodes/ROIs within the same module to share similar representations. To further enhance discriminative ability of learned features, we also encourage the MGNN to preserve the network topology of input graphs via a graph topology reconstruction constraint. Experimental results on 534 subjects with rs-fMRI scans from two datasets validate the effectiveness of the proposed method. The identified discriminative brain ROIs and functional connectivities can be regarded as potential fMRI biomarkers to aid in clinical diagnosis.

## Folder Structure

This repository is organized into the following folders:

    - `./main.py`: Contains the main functions for training and testing.
    - `./data_pre.py`: Handles data preparation.
    - `./net`:Includes the models used in this project.

## Dependencies  

The framework needs the following dependencies:

```
torch~=1.13.0
numpy~=1.21.5
torch_scatter~=2.1.0+pt113cu117
scipy~=1.9.3
einops~=0.5.0
```


Many thanks to Dr Byung-Hoon Kim for sharing their project [STAGIN](https://github.com/egyptdj/stagin).

## Citation
Please cite our work if you find this repository helpful:

```bibtex
@ARTICLE{10449452,
  author={Wang, Qianqian and Wang, Wei and Fang, Yuqi and Yap, P.-T. and Zhu, Hongtu and Li, Hong-Jun and Qiao, Lishan and Liu, Mingxia},
  journal={IEEE Transactions on Biomedical Engineering}, 
  title={Leveraging Brain Modularity Prior for Interpretable Representation Learning of fMRI}, 
  year={2024},
  pages={1-11},
  keywords={Functional magnetic resonance imaging;Representation learning;Network topology;Brain modeling;Topology;Autism;Biomedical engineering;Functional MRI;brain modularity;brain disorder;biomarker},
  doi={10.1109/TBME.2024.3370415}}
```
