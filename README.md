# TextureAutoencoder
A small CUDA optimized script which takes in .ori files containing crystallographic orientation data of polycrystals and outputs a condensed file which encodes texture data for the material

This repo contains two scripts:
1. orientation_autoencoder_cuda_optimized.py
2. orientation_sequence_decoder.py

Let's suppose there I want to model the texture evolution of a material using some kind of neural network, now ODF expansion can lead to expansion terms potentially numbering thousands or more, so the Autoencoder approach is a compact approach to store texture evolution information in a very lightweight format. Let's say in a folder, there are files numbered from step_000.ori, step_001.ori to ... step_050.ori, these are the .ori files for the material for each timestep of its evolution. What script 1 does is encode this whole sequence into a .npy file which contains the whole sequence of texture evolution in latent dimension. Now script 2 decodes a .npy file into a .ori file sequence like step_000.ori, step_001.ori, ..., step_050.ori.
