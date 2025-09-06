# TextureAutoencoder
A small CUDA optimized script which takes in .ori files containing crystallographic orientation data of polycrystals and outputs a condensed file which encodes texture data for the material

This repo contains two scripts:
1. orientation_autoencoder_cuda_optimized.py
2. orientation_sequence_decoder.py

Let's suppose there I want to model the texture evolution of a material using some kind of neural network, now ODF expansion can lead to expansion terms potentially numbering thousands or more, so the Autoencoder approach is a compact approach to store texture evolution information in a very lightweight format. Let's say in a folder, there are files numbered from step_000.ori, step_001.ori to ... step_050.ori, these are the .ori files for the material for each timestep of its evolution. What script 1 does is encode this whole sequence into a .npy file which contains the whole sequence of texture evolution in latent dimension. Now script 2 decodes a .npy file into a .ori file sequence like step_000.ori, step_001.ori, ..., step_050.ori.

Hyperparameters:
Script 1 contains 6 hyperparameters which can be tuned based on the user's requirements:
1. Grid Size: Default is set at 40, which means it operates on a cubic RVE (Representative Volume Element) of size 40x40x40.
2. Base Channels: The encoder 'neck' size. Larger values mean more features can be accomodated during encoding but also increases computational cost.
3. Latent dimension: The output dimension of the encoded .npy file.
4. Batch size: Can be tweaked based on available GPU memory.
5. Learning rate: Default is 1e-3 but can be tuned based on the user requirements and GPU power available.
6. Number of Epochs: Number of training epochs, more usually reduces the loss in encoding.

The input files for script 1 should be numbered as inside a folder named sim_0, sim_1, etc., (The folder is named as such because the .ori files are produced from Dusseldorf Advanced MAterials Kit [1] simulations):
ori_increment_1.ori, ori_increment_2.ori, ... etc.

The output of script 1 is present in a directory called latent_trajectories. A .pth file is also part of the output which saves the trained model configuration.

# References
1. F. Roters et al., “DAMASK – The Düsseldorf Advanced Material Simulation Kit for modeling multi-physics crystal plasticity, thermal, and damage phenomena from the single crystal up to the component scale,” Comput. Mater. Sci., vol. 158, pp. 420–478, Feb. 2019, doi: 10.1016/j.commatsci.2018.04.030.
