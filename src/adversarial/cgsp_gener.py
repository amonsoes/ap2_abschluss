
import os
import numpy as np
import math
import sys
import pandas as pd

import torchvision
import torch
import torch.nn as nn

from src.adversarial.attack_base import Attack



class CGSPAttack(Attack):

    def __init__(self, device, arch, eps, l2_bound=-1, *args, **kwargs):
        super().__init__('CGSP', *args, **kwargs)

        self.label_flag = 'N8'
        self.target_labels = np.array([150, 507, 62, 843, 426, 590, 715, 952])
        self.nz = 16
        self.layer = 1

        if arch == 'res':
            pretrained =  './src/adversarial/resources/cgsp-res-9.pth'
        elif arch == 'deit':
            pretrained =  './src/adversarial/resources/cgsp-deit-9.pth'
        else:
            raise ValueError(f'Architecture {arch} not available currently.')
        
        self.eps = eps
        self.device = device
        self.netG = self.load_from_file(pretrained)

        if l2_bound != -1:
            self.eps = self.get_eps_in_range(l2_bound)

    def forward(self, images, labels):

        batch_size, _, _, _ = images.shape
        images = images.to(self.device)

        target_label = np.random.choice(self.target_labels)
        target_tensor = torch.LongTensor(batch_size).to(self.device)
        target_tensor.fill_(target_label)
        
        target_one_hot = torch.zeros(batch_size, 1000, device=self.device).scatter_(1, target_tensor.unsqueeze(1), 1).to(self.device) # 1000 classes in IgNet
        noise = self.netG(images, target_one_hot, eps=self.eps)
        adv = noise + images

        # Projection, tanh() have been operated in models.
        adv = torch.min(torch.max(adv, images - self.eps), images + self.eps)
        adv = torch.clamp(adv, 0.0, 1.0)
        return adv


    def load_from_file(self, pretrained):
        netG = ConGeneratorResnet(nz=self.nz, layer=self.layer)
        state_dict = torch.load(pretrained, map_location=self.device)
        netG.load_state_dict(state_dict=state_dict)
        if torch.cuda.is_available():
            device_ind = self.device.index
            netG = nn.DataParallel(netG, device_ids = [device_ind, device_ind+1]).to(self.device)
        else:
            netG = nn.DataParallel(netG).to(self.device)
        netG.eval()
        return netG


##### condgenerator.py code #######



# To control feature map in generator
ngf = 64

def snlinear(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(**kwargs), eps=eps)

# This function is based on https://github.com/Muzammal-Naseer/Cross-domain-perturbations
def get_gaussian_kernel(kernel_size=3, pad=2, sigma=1, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, padding=kernel_size-pad,  bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

class ConGeneratorResnet(nn.Module):
    def __init__(self, inception=False, nz =16, layer=1, loc = [1,1,1], data_dim='high'):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        '''
        super(ConGeneratorResnet, self).__init__()
        self.inception = inception
        self.data_dim = data_dim
        self.layer = layer
        self.snlinear = snlinear(in_features=1000, out_features=nz, bias=False)
        if self.layer > 1:
            self.snlinear2 = snlinear(in_features=nz, out_features=nz, bias=False)
        if self.layer > 2:
            self.snlinear3 = snlinear(in_features=nz, out_features=nz, bias=False)
        self.loc = loc
        # Input_size = 3, n, n
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3+nz * self.loc[0], ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf+nz * self.loc[1], ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2+nz * self.loc[2], ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        if self.data_dim == 'high':
            self.resblock3 = ResidualBlock(ngf * 4)
            self.resblock4 = ResidualBlock(ngf * 4)
            self.resblock5 = ResidualBlock(ngf * 4)
            self.resblock6 = ResidualBlock(ngf * 4)
            # self.resblock7 = ResidualBlock(ngf*4)
            # self.resblock8 = ResidualBlock(ngf*4)
            # self.resblock9 = ResidualBlock(ngf*4)

        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)
        self.alf_layer = get_gaussian_kernel(kernel_size=3, pad=2, sigma=1)
            

    def forward(self, input, z_one_hot, eps=0, gap=False):
        z_cond = self.snlinear(z_one_hot)
        if self.layer > 1:
            z_cond = self.snlinear2(z_cond)
        if self.layer > 2:
            z_cond = self.snlinear3(z_cond)
        ## loc 0
        z_img = z_cond.view(z_cond.size(0), z_cond.size(1), 1, 1).expand(
                z_cond.size(0), z_cond.size(1), input.size(2), input.size(3))
        assert self.loc[0] == 1
        x = self.block1(torch.cat((input, z_img), dim=1))
        # loc 1
        z_img = z_cond.view(z_cond.size(0), z_cond.size(1), 1, 1).expand(
                z_cond.size(0), z_cond.size(1), x.size(2), x.size(3))
        if self.loc[1]:
            x = self.block2(torch.cat((x, z_img), dim=1))
        else:
            x = self.block2(x)
        # loc 2
        z_img = z_cond.view(z_cond.size(0), z_cond.size(1), 1, 1).expand(
                z_cond.size(0), z_cond.size(1), x.size(2), x.size(3))
        if self.loc[2]:
            x = self.block3(torch.cat((x, z_img), dim=1))
        else:
            x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        if self.data_dim == 'high':
            x = self.resblock3(x)
            x = self.resblock4(x)
            x = self.resblock5(x)
            x = self.resblock6(x)
            # x = self.resblock7(x)
            # x = self.resblock8(x)
            # x = self.resblock9(x)
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)
        if self.inception:
            x = self.crop(x)
        
        x = torch.tanh(x)
        x = self.alf_layer(x)
        
        return x * eps


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual
    

class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class FocalLoss(nn.Module):
    def __init__(self, gamma = 1, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(reduce=False)

    def forward(self, input, target):
        logp = self.ce(input, target)
        print(logp)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        print((1 - p) ** self.gamma, loss)
        return loss.mean()


def fix_labels(args, test_set):
    val_dict = {}
    with open("val.txt") as file:
        for line in file:
            (key, val) = line.split(' ')
            val_dict[key.split('.')[0]] = int(val.strip())

    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        org_label = val_dict[test_set.samples[i][0].split('/')[-1].split('.')[0]]
        new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples
    return test_set
############################################################

#############################################################
# This will fix labels for NIPS ImageNet
def fix_labels_nips(args, test_set, pytorch=False, target_flag=False):

    '''
    :param pytorch: pytorch models have 1000 labels as compared to tensorflow models with 1001 labels
    '''

    filenames = [i.split('/')[-1] for i, j in test_set.samples]
    # Load provided files and get image labels and names
    image_classes = pd.read_csv(os.path.join(args.data_dir, "images.csv"))
    image_metadata = pd.DataFrame({"ImageId": [f[:-4] for f in filenames]}).merge(image_classes, on="ImageId")
    true_classes = image_metadata["TrueLabel"].tolist()
    target_classes = image_metadata["TargetClass"].tolist()
    val_dict = {}
    for f, i in zip(filenames, range(len(filenames))):
        val_dict[f] = [true_classes[i], target_classes[i]]
    
    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        if target_flag:
            org_label = val_dict[test_set.samples[i][0].split('/')[-1]][1]
        else:
            org_label = val_dict[test_set.samples[i][0].split('/')[-1]][0]
        if pytorch:
            new_data_samples.append((test_set.samples[i][0], org_label-1))
        else:
            new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples
    return test_set
