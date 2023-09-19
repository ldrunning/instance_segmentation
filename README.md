# instance_segmentation
Introduction
This repository is a PyTorch implementation for instance_segmentation. The code is easy to use for training and testing on various datasets. And multiprocessing training is supported, tested with pytorch 1.6.0.

Usage
Highlight:

Fast multiprocessing training (nn.parallel.DistributedDataParallel) with official nn.SyncBatchNorm.
Better reimplementation results with well designed code structures.
Requirement:

Hardware: 4 GPUs (better with >=11G GPU memory)
Software: PyTorch>=1.6.0, Python3, tensorboardX,
Clone the repository:

git clone https://github.com/ldrunning/instance_segmentation
Train:

Download related datasets and put them under folder specified in config or modify the specified paths.

git clone https://github.com/ldrunning/data

python train_unet.py

Test:

Download trained segmentation models and put them under folder specified in config or modify the specified paths.

python detect_images.py
