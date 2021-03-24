# Invertible Image Denoising
This is the PyTorch implementation of paper: Invertible Denoising Network: A Light Solution for Real Noise Removal (CVPR 2021).

## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb pyyaml`
- TensorBoard: 
  - PyTorch >= 1.1: `pip install tb-nightly future`
  - PyTorch == 1.0: `pip install tensorboardX`
  
## Dataset Preparation
The datasets used in this paper is DND (can be downloaded [here](https://noise.visinf.tu-darmstadt.de/)) and SIDD (can be downloaded [here](https://www.eecs.yorku.ca/~kamel/sidd/)).

## Get Started
Training and testing codes are in ['codes/'](./codes/). Please see ['codes/README.md'](./codes/README.md) for basic usages.

## Invertible Architecture
![Invertible Architecture](./figures/Network_Architecture.jpg)
