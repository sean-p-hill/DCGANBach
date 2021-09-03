import os
import numpy as np
from matplotlib import pyplot as plt
import time
from attrdict import AttrDict
import pickle

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

# pip install git+https://github.com/kklemon/pytorch-fid
from pytorch_fid import FrechetInceptionDistance

def gray2rgb(gray,batch=True):
    '''
    Convert images from grayscale to RGB for use with Frechet Inception Distance
    '''
    if batch: return torch.cat([gray for _ in range(3)],1)
    else:     return torch.cat([gray for _ in range(3)],0)

def FIDScore(real_batch,fake_batch,model,device,params):
    '''
    Helper function for evaluating FID score with pytorch_fid
    '''
    print('Calculating Frechet Inception Distance...')

    fid = FrechetInceptionDistance(model, dims=64,batch_size=params.batch_size)
    stats1 = fid.get_activation_statistics([gray2rgb(fake_batch)])
    stats2 = fid.get_activation_statistics([gray2rgb(real_batch)])
    score = fid.calculate_frechet_distance(*stats1, *stats2)

    return score

def mk_output_dir(google_colab):
    """
    Creates an output directory that the results of the network can be saved to
    """
    try: 
        path = 'Output' if not google_colab else "drive/MyDrive/Output"
        output_dir = os.path.join(path,'Output_{}'.format(time.strftime('%d_%m_%H%M')))
        os.mkdir(output_dir)
        print('Output Folder:',output_dir)
    except:
        output_dir = os.path.join('Output',sorted(os.listdir('Output'),key = lambda x:int((x[10:12]+x[7:9]+x[13:17])) if 'Output_' in x else 0)[-1])

    print('Output Folder:',output_dir)

    return output_dir

def save_models(G,D,output_dir,fn=''):
    '''
    Save the models
    '''
    torch.save(G, os.path.join(output_dir,f'generator{fn}.pth'))
    torch.save(D, os.path.join(output_dir,f'discriminator{fn}.pth'))

def import_data(fn,params):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.
    """
    data_transforms = transforms.Compose([transforms.Resize(64),
                                         transforms.Grayscale(num_output_channels=1),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5), (0.5)),])

    try:
        dataset = ImageFolder(root=fn,transform=data_transforms)
        dataloader = DataLoader(dataset, batch_size=params['batch_size'],shuffle=True,num_workers=2)
    except:
    # Pytorch requires images to be inside a subfolder within the directory, so create if not there already  
        for f in os.listdir(fn):
            destination = os.path.join(fn,'Images')
            if 'Images' not in os.listdir(fn): os.makedirs(destination)
            if f.endswith('.png'): shutil.move(os.path.join(fn, f), os.path.join(destination,f))
            
        dataset = ImageFolder(root=fn,transform=data_transforms)
        dataloader = DataLoader(dataset, batch_size=params['batch_size'],shuffle=True,num_workers=2)

    return dataloader

def generate_params(param_fn, google_colab, gpu_num, input_data, fn, output_dir, use_hardcoded=False):
    '''
    Read parameters from .pkl file on disk
    '''
    hardcoded_params = {  'input_data'      : input_data,
                          'batch_size'      : 16, 
                          'kernel_size'     : 4,
                          'G_learning_rate' : 0.004,
                          'D_learning_rate' : 0.0004,  
                          'beta1'           : 0.5, 
                          'beta2'           : 0.9, 
                          'leaky_alpha'     : 0.2,  
                          'real_label'      : 0.9,  
                          'fake_label'      : 0,
                          'gen_updates'     : 1,
                          'loss'            : 'BCE'}

    if not use_hardcoded:
        try:
            param_path = f'drive/MyDrive/Params/{param_fn}' if google_colab else f'Params/{param_fn}'
            with open(param_path, 'rb') as param_file:
                params = pickle.load(param_file)
            print(f'Using parameters from {param_path}')
        except:
            params = hardcoded_params
            print('Using Hardcoded Parameters:')
    else:
        params = hardcoded_params
        print('Using Hardcoded Parameters:')

    params = AttrDict(params)
    params.ngpu = gpu_num
    device = torch.device("cuda:{}".format(gpu_num[0]) if (torch.cuda.is_available() and params.ngpu) else "cpu")
    print('Device:',device)
    if params.ngpu: params['device'] = device

    # Print parameters and save to file as reference for results
    n_files = 3 if 'Monophonic' in fn else 2
    params.input_data = reduce(os.path.join, fn.split('/')[-n_files:]) if google_colab else input_data

    print('\nParameters:')
    for k,v in params.items(): print('{}: {}'.format(k,v))
    pfn = open(os.path.join(output_dir,"params.pkl"), "wb")
    pickle.dump(params, pfn)
    pfn.close()

    return params, device

def plot_real(dataloader,output_dir):

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig(os.path.join(output_dir,'realimages.png'))
    plt.close()

    return real_batch

def generate_images(model_dir, title='Fake Images',show=False, return_g=False,best=False):

    if best:
        model_paths = [f for f in os.listdir(model_dir) if 'generatorBestEpoch' in f]
        if len(model_paths) != 0: 
            path = sorted(model_paths,key = lambda x:int(x.replace('generatorBestEpoch','')[:-4]))[-1]
        else:
            return None, None, None, None
    else:
        path = 'generator.pth'


    generator = torch.load(os.path.join(model_dir,path), map_location=torch.device('cpu'))

    fixed_noise = torch.randn(64, 100, 1, 1)#, device=device)
    print(fixed_noise.shape)
    fake = generator(fixed_noise).detach().cpu()
    grid = torchvision.utils.make_grid(fake, padding=2, pad_value=1)#, normalize=True)
    # Plot the fake images from the last epoch

    # fig = plt.subplot(1,1,1)
    fig = plt.figure()
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(grid,(1,2,0)),cmap='gray')
    fnsuff = 'bestFID' if best else ''
    plt.savefig(os.path.join(model_dir,f'fakeimages_{fnsuff}.png'),dpi=2000)
    plt.close()
    if show: plt.show()
    if return_g: return grid, fake, path, generator
    else:        return grid, fake, path

def save_generator_output(image,output_dir,epoch,fid='',wholebatch=False):

    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if len(image.shape) == 3: ip = np.transpose(np.round(image.numpy()[0,:,:],0),(0,1))
    else:                     ip = np.transpose(np.round(image.numpy(),0),(0,1))
    
    plt.imshow(ip,animated=True,cmap='gray')
    if fid!='': plt.title('Generator Output at Epoch {}'.format(epoch))
    if wholebatch: plt.savefig('{}/Epoch_{}{}.png'.format(output_dir,epoch,fid),dpi=8)
    else:          plt.savefig('Epoch_{}{}'.format(epoch,fid),dpi=8)
    plt.close()

def plot_fake_grid(grid,output_dir,epoch,show=False,title=False):
    # fig = plt.subplot(1,1,1)
    fig = plt.figure()
    plt.axis("off")
    if title: plt.title(title)
    plt.imshow(np.transpose(grid,(1,2,0)))
    plt.savefig(os.path.join(output_dir,f'fakeimages_Epoch_{epoch}.png'),dpi=2000)
    plt.close()
    if show: plt.show()

def plot_losses(G_losses, D_losses, output_dir, epochs=None, title='',show=True):
    fs = 16
    plt.figure(figsize=(10,5))
    plt.title(f"Generator and Discriminator Loss During Training\n{title}",fontsize=fs)
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    xlabel = 'Epochs' if epochs else 'Iterations'
    plt.xlabel(xlabel,fontsize=fs)
    plt.ylabel("Loss",fontsize=fs)
    if epochs: plt.xticks(range(0,len(D_losses)+1,int(len(D_losses)/4)),range(0,epochs+1,int(epochs/4)))
    plt.legend()
    fn = '{}_loss.png'.format(output_dir.split('/')[-1])
    plt.savefig(os.path.join(output_dir,fn))#,dpi=2000)
    plt.close()
    if show: plt.show()

def plot_fid(FIDScores,model,dataset,input_dir):

    plt.plot(((FIDScores[:,1]*1000)/np.max(FIDScores[:,1])).astype(int),FIDScores[:,0],linewidth=2)
    plt.xlabel('Epoch',fontsize=12)
    plt.ylabel('FID Between Real and Fake Images',fontsize=12)
    plt.ylim(0,500)
    plt.xlim(0,1000)
    plt.title(f'Frechet Inception Distance Throughout Training\n{model} Trained on {dataset} Dataset',fontsize=12)
    fn = input_dir.split('/')[-1]
    plt.savefig(os.path.join(input_dir,f'{fn}FIDPlot.png'))
    plt.close()


