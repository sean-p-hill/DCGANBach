import torch.nn as nn

def weights_init(m):
    '''
    Radford et al specify intiailising model weights to normal distribution
    with mean of 0 and stdev of 0.02.
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# 64x64 Images
class Generator64(nn.Module):
    def __init__(self, params):
        super(Generator64, self).__init__()
        self.ngpu = params.ngpu

        layers = []
        
        # 1 x 1 x 100 - Convolutional Transpose Layer & ReLU Activation with Batch Norm
        layers.append(nn.ConvTranspose2d(100, 512, params.kernel_size, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(True))

        # 4 x 4 x 512 - Strided Convolutional Transpose Layer & ReLU Activation with Batch Norm
        layers.append(nn.ConvTranspose2d(512, 256, params.kernel_size, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(True))

        # 8 x 8 x 256 - Strided Convolutional Transpose Layer & ReLU Activation with Batch Norm
        layers.append(nn.ConvTranspose2d(256, 128, params.kernel_size, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(True))
        
        # 16 x 16 x 128 - Strided Convolutional Transpose Layer & ReLU Activation with Batch Norm
        layers.append(nn.ConvTranspose2d( 128, 64, params.kernel_size, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(True))
        
        # 32 x 32 x 64 - Strided Convolutional Transpose Layer & tanh Activation
        layers.append(nn.ConvTranspose2d( 64, 1, params.kernel_size, stride=2, padding=1, bias=False))
        layers.append(nn.Tanh())
        # 64 x 64 x 1 - Output
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)

class Discriminator64(nn.Module):
    def __init__(self, params):
        super(Discriminator64, self).__init__()
        self.ngpu = params.ngpu

        layers = []
            
        # 64 x 64 x 1 - Convolutional Layer & ReLU Activation with Batch Norm4
        layers.append(nn.Conv2d(1, 64, params.kernel_size, stride=2, padding=1, bias=False))
        layers.append(nn.LeakyReLU(params.leaky_alpha, inplace=True))

        # 32 x 32 x 64 - Convolutional Layer & ReLU Activation with Batch Norm
        layers.append(nn.Conv2d(64, 128, params.kernel_size, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.LeakyReLU(params.leaky_alpha, inplace=True))

        # 16 x 16 x 128 - Convolutional Layer & ReLU Activation with Batch Norm
        layers.append(nn.Conv2d(128, 256, params.kernel_size, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.LeakyReLU(params.leaky_alpha, inplace=True))

        # 8 x 8 x 256 - Convolutional Layer & ReLU Activation with Batch Norm
        layers.append(nn.Conv2d(256, 512, params.kernel_size, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.LeakyReLU(params.leaky_alpha, inplace=True))

        # 4 x 4 x 512 - Convolutional Layer & Sigmoid Activation
        layers.append(nn.Conv2d(512, 1, params.kernel_size, stride=1, padding=0, bias=False))

        layers.append(nn.Sigmoid())
        # 1 x 1 - Output 

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)


