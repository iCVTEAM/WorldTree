# WorldTree: Towards 4D Dynamic Worlds from Monocular Video using Tree-Chains

[![Paper](https://img.shields.io/badge/Paper-Read-orange)](https://arxiv.org/pdf/2602.11845)

## Abstract

Dynamic reconstruction has achieved remarkable progress, but there remain challenges in monocular input for more practical applications. The prevailing works attempt to construct efficient motion representations, but lack a unified spatiotemporal decomposition framework, suffering from either holistic temporal optimization or coupled hierarchical spatial composition. To this end, we propose WorldTree, a unified framework comprising Temporal Partition Tree (TPT) that enables coarse-to-fine optimization based on the inheritance-based partition tree structure for hierarchical temporal decomposition, and Spatial Ancestral Chains (SAC) that recursively query ancestral hierarchical structure to provide complementary spatial dynamics while specializing motion representations across ancestral nodes. Experimental results on different datasets indicate that our proposed method achieves 8.26% improvement of LPIPS on NVIDIA-LS and 9.09% improvement of mLPIPS on DyCheck compared to the second-best method.

## Installation

Please refer to [MoSca](https://github.com/JiahuiLei/MoSca) for the installation.

The downloaded pre-trained weights should be placed in ```./weights```.

## Data

Download the preprossed data in [worldtree-data](https://huggingface.co/datasets/wangqisen/worldtree-data).

## Running

We provide configures in ```./profile```. We provide scripts in ```./scripts```. Here are the usages with the example of the NVIDIA dataset.

* **Preprocessing**. Using ```./scripts/run_nvidia_ours_prep.py``` with ```./profile/nvidia_divide_final/nvidia_prep.yaml``` for preprocessing the dataset. (Optional: we have provided the preprocessed dataset.)

* **Optimization**. Using ```./scripts/run_nvidia_ours_optim.py``` with ```./profile/nvidia_divide_final/nvidia_fit_final.yaml``` for optimizing and evaluating the root. Using ```./scripts/run_nvidia_ours_optim_dst.py``` with ```./profile/nvidia_divide_final/nvidia_fit_dst_final_w_fachain.yaml``` for optimizing and evaluating the rest of the nodes. (Optional: You can rectify the profile to directly optimize the whole tree. We split the optimization process for better evaluation of different experiments.)

* **Statistics**. Using ```./scripts/run_nvidia_ours_statsonly.py``` with ```./profile/nvidia_divide_final/nvidia_stats.yaml``` for the statistics of the dataset.

## License

We authorize our code under the MIT License. Other code from third-party repositories should adhere to the license of those third-party codes.

## Acknowledgement

This project is primarily built on [MoSca](https://github.com/JiahuiLei/MoSca). We also thank other projects for any assistance they may have provided. Thanks to all the authors for their great contributions.

## Cite

```
@inproceedings{
wang2026worldtree,
title={WorldTree: Towards 4D Dynamic Worlds from Monocular Video using Tree-Chains},
author={Qisen Wang and Yifan Zhao and Jia Li},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=mVo6cyFR6C}
}
```
