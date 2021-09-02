# Checking whether running in google colab or from file
if 'google_colab' not in locals(): google_colab = False

import os
from os.path import join
import sys
import random
import numpy as np
import time
import shutil
import matplotlib.pyplot as plt
import pickle
import random
from functools import reduce
from attrdict import AttrDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data 
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

if not google_colab:
    # Import helper functions and network classes...
    from utils       import (import_data, mk_output_dir, save_generator_output, plot_losses, 
                             save_models, LSLoss, plot_fake_grid, generate_params)
    from DCGAN_Model import  Generator64, Discriminator64, weights_init
    from train_model import  TrainDCGAN
    from analysis    import  ModelAnalysis

# Set random seed for reproducibility in both PyTorch and system Python
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Parameters Dictionary:
# 'input_data'      Reference to the Dataset used
# 'batch_size'      Batch size during training
# 'kernel_size'     Kernel size of the convolutional kernel
# 'G_learning_rate' Size of learning rate in generator training
# 'G_learning_rate' Size of learning rate in discriminator training
# 'beta1'           Beta1 hyperparam for Adam optimizers
# 'beta2'           Beta2 hyperparam for Adam optimizers
# 'leaky_alpha'     Size of α parameter in LeakyReLU activation
# 'real_label'      Image label that the real images will be trained to 
# 'fake_label'      Image label that the fake images will be trained to
# 'gen_updates'     Number of times generator is updated per iteration
# 'loss'            Loss function to be used in training

def main():
    # Certain Parameters are Passed through the Command Line
    try:
        if not google_colab: _ , gpu_num, input_data, num_epochs, param_fn = sys.argv[:5]
    except(ValueError):
        print('''\nError: Incorrect Command Line Arguments...\n
                   Usage: python train.py gpu_num input_data_path num_epochs params.pkl\n''')
        sys.exit()

    # Choosing whether to use a GPU or not - GPU pointer passed through command line
    try:
        gpu_num = [0] if google_colab else list(map(int,gpu_num.strip('[]').split(',')))
    except(ValueError):
        gpu_num = None

    if google_colab: param_fn = 'paramsmodelA.pkl'

    # Create output folder labelled with the time... If the code has been run twice in the same minute reuse old one
    output_dir = mk_output_dir(google_colab)

    # Deciding which dataset tp use, manually add in google colab but command line otherwise
    fn = "drive/MyDrive/NewGANData/12keys/Polyphonic" if google_colab else '../Data/'+ input_data
    print('Dataset: {}'.format(fn.split('/')[-1]))

    params, device = generate_params(param_fn, google_colab, gpu_num, input_data, fn, output_dir)
    dataloader = import_data(fn,params)

    model = TrainDCGAN(dataloader,device,params,output_dir)
    model.train(int(num_epochs))

    return output_dir

if __name__ == '__main__':

    # Useage: python main.py gpu_pointer, input_data_path, num_epochs, params_path

    # Run the training process of the DCGAN and generate outputs.
    output_dir = main()

    # Perform Analysis on the Outputted Files from Training
    analyser = ModelAnalysis(output_dir)
    analyser.run_analysis()






