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
    from utils import (import_data, mk_output_dir, save_generator_output, 
                      plot_losses, save_models, LSLoss, plot_fake_grid, 
                      generate_params, gray2rgb, FIDScore, getFIDModel)
    from DCGAN_Model import Generator64, Discriminator64, weights_init

from pytorch_fid import FrechetInceptionDistance


class TrainDCGAN():

    def __init__(self,dataloader,device,params,output_dir):

        self.dataloader = dataloader
        self.device = device
        self.params = params
        self.output_dir = output_dir
        
        # Create the generator and discriminator and initialize all weights to mean=0.0, stddev=0.2
        self.G = Generator64(params).to(device)
        self.D = Discriminator64(params).to(device)
        self.G.apply(weights_init)
        self.D.apply(weights_init)

        # Create batch of random noise vectors (z) that is used to visualise the progression of the generator
        self.fixed_noise = torch.randn(64, 100, 1, 1, device=device)

        # Choosing loss function
        loss_functions = {'BCE':nn.BCELoss(),'LS':LSLoss}
        self.lossfunction = loss_functions[params.loss]
        print('Loss Function:',self.lossfunction)

        # Setup Adam optimizers for both G and D
        self.D_Adam = optim.Adam(self.D.parameters(), lr=params.D_learning_rate, betas=(params.beta1, params.beta2))
        self.G_Adam = optim.Adam(self.G.parameters(), lr=params.G_learning_rate, betas=(params.beta1, params.beta2))

        # Import pre-trained Frechet Inception Distance Model
        FIDModel = FrechetInceptionDistance.get_inception_model()
        self.FIDModel = FIDModel.to(device)


    def train(self,num_epochs):

        # Setting number of epochs
        if google_colab: num_epochs = int(1000)
        print('Number of Epochs: {}'.format(num_epochs))

        # Lists for storing training data
        img_list, single_img_list, G_losses, D_losses, Dacc_real, Dacc_fake, FIDScores = [],[],[],[],[],[],[]

        ##### Training Loop ######################

        print("\nStarting Training Loop...")
        start_time = time.mktime(time.localtime())

        best_frechet = 9999

        for epoch in range(int(num_epochs)):
             for i, data in enumerate(self.dataloader):

                  ###### Training Discriminator: Maximise log(D(x)) + log(1-D(G(z)))
                  # Resetting discriminator gradients to zero
                  self.D.zero_grad()

                  # Assigning the real images the real labels
                  real_batch = data[0].to(self.device)
                  labels = torch.full((real_batch.shape[0],), self.params.real_label, dtype=torch.float, device=self.device)

                  # Generate labels for real images with discriminator
                  output = self.D(real_batch).view(-1)

                  # Calculate discriminator loss on real images using BCE Loss
                  Dloss_real = self.lossfunction(output, labels)
                  Dacc_real.append((output == labels).float().sum() / output.shape[0])

                  # Back Propagate through Discriminator to Calculate Gradients
                  Dloss_real.backward()
                  D_x = output.mean().item()

                  # Generate Fake Images Using Noise Vector as Input to Generator
                  noise = torch.randn(real_batch.shape[0], 100, 1, 1, device=self.device)
                  fake = self.G(noise)

                  # Use Fake Labels for the Fake Images
                  labels = torch.full((real_batch.shape[0],), self.params.fake_label, dtype=torch.float, device=self.device)

                  # Generate labels for real images with discriminator
                  output = self.D(fake.detach()).view(-1)

                  # Calculate discriminator loss on fake images using BCE Loss
                  Dloss_fake = self.lossfunction(output, labels)
                  Dacc_fake.append((output == labels).float().sum() / output.shape[0])

                  # Back Propagate through Discriminator to Calculate Gradients
                  Dloss_fake.backward()
                  D_G_z_disc = output.mean().item() # D(G(z)) for first pass in loop

                  # Total discriminator loss is sum over real and fake images
                  Dloss = Dloss_real + Dloss_fake

                  # Update Discriminator Neural Weights using Adam Optimiser
                  self.D_Adam.step()

                  ###### Training Generator: minimise log(1-D(G(z)))
                  # Resetting generator gradients to zero 
                  self.G.zero_grad()

                  # Use real labels for the fake images when training generator
                  # Generator tries to fool the discriminator...
                  labels = torch.full((real_batch.shape[0],), self.params.real_label, dtype=torch.float, device=self.device)

                  # Generate labels for fake images again
                  output = self.D(fake).view(-1)
                  D_G_z_gen = output.mean().item() # D(G(z)) for second time

                  # Calculate Generator Loss
                  # How well it can cause discriminator to generate wrong labels
                  Gloss = self.lossfunction(output, labels)

                  # Choosing how many times to backpropagate the generator weights each iteration
                  for g_loop in range(self.params.gen_updates): 
                       # Back Propagate Through Generator to Calculate Gradients
                       Gloss.backward(retain_graph=True)

                  # Update Generator Neural Weights using Adam Optimiser
                  self.G_Adam.step()

                  # Save Losses
                  G_losses.append(Gloss.item())
                  D_losses.append(Dloss.item())

                  # Print Training Performance every 250 iterations
                  if i % 250 == 0:            
                      print('Epoch {:d}/{:d}    L(D)={:.4f}    L(G)={:.4f}    D(x)={:.4f}    D(G(z))={:.4f}'
                           .format(epoch, num_epochs, Dloss.item(), Gloss.item(), D_x, (D_G_z_disc+D_G_z_disc)/2))

                      runtime = time.mktime(time.localtime()) - start_time
                      print('Time Elapsed: {}:{}:{}'.format(int(runtime/3600),int(runtime/60)%60,runtime%60))

                      # Check how the generator is doing by saving G's output on fixed_noise
                      # Use G to Produce Images for Analysis by Epoch
                      if (epoch % (int(num_epochs/100)+1) == 0) or (epoch == num_epochs-1):
                          with torch.no_grad(): fake_batch = self.G(self.fixed_noise).detach().cpu()
                          if i % 2000 == 0:
                              fake_batch[fake_batch>0.75] == 1
                              fake_batch[fake_batch<=0.75] == -1
                              fid_i = FIDScore(real_batch,fake_batch,self.FIDModel,self.device,self.params)
                              FIDScores.append([fid_i,epoch])
                              if fid_i < best_frechet:
                                  print('New Best Model: Epoch {} FID: {}'.format(epoch,fid_i))
                                  save_models(self.G,self.D,self.output_dir,fn=f'BestEpoch{epoch}')
                                  best_frechet = fid_i

                          # Save training data
                          np.save(os.path.join(self.output_dir,'G_losses.npy'),np.array(G_losses))
                          np.save(os.path.join(self.output_dir,'D_losses.npy'),np.array(D_losses))
                          np.save(os.path.join(self.output_dir,'Dacc_real.npy'),np.array(Dacc_real))
                          np.save(os.path.join(self.output_dir,'Dacc_fake.npy'),np.array(Dacc_fake))
                          np.save(os.path.join(self.output_dir,'FIDScores.npy'),np.array(FIDScores))

                          # Save checkpoints of generator and discriminator in case of code crashing
                          save_models(self.G,self.D,self.output_dir,fn='')

                          # Plot Discriminator and Generator Losses Over Time
                          plot_losses(G_losses, D_losses, self.output_dir)

                          # Generate and save images for outputs - single and full batch
                          single_img_list.append(vutils.make_grid(fake_batch[-1], padding=0, normalize=True))
                          img_list.append(vutils.make_grid(fake_batch, padding=2, normalize=True))
                          pfn = open(os.path.join(self.output_dir,"img_list.pkl"), "wb")
                          pickle.dump(img_list, pfn)
                          pfn.close()
                          save_generator_output(single_img_list[-1],self.output_dir,epoch+1)
                          plot_fake_grid(img_list[-1],self.output_dir,epoch+1)


