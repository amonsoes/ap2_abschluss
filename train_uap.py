import torch
import argparse

from torch import nn


import multiprocessing
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torchvision

from torch.utils import model_zoo
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.model.pretrained import DeiT

torch.multiprocessing.set_sharing_strategy('file_system')



IMGNET_MEAN = [0.5, 0.5, 0.5]
IMGNET_STD = [0.5, 0.5, 0.5]
# for DeiT

class UAPTrainer:

    def __init__(self, model, data, epochs, eps):
        self.model = model
        self.data = data
        self.epochs = epochs
        self.eps = eps
        self.device = self.device = next(model.parameters()).device


    def train(self, beta = 12, step_decay = 0.8, y_target = None, loss_fn = None, layer_name = None, uap_init = None):
        '''
        INPUT
        model       self.model
        data      dataloader
        epochs    number of optimization epochs
        eps         maximum perturbation value (L-infinity) norm
        beta        clamping value
        y_target    target class label for Targeted UAP variation
        loss_fn     custom loss function (default is CrossEntropyLoss)
        layer_name  target layer name for layer maximization attack
        uap_init    custom perturbation to start from (default is random vector with pixel values {-self.eps, self.eps})
        
        OUTPUT
        delta.data  adversarial perturbation
        losses      losses per iteration
        '''
        print('\nInitialize Training...\n')
        print(f'PARAMS: beta={beta}, step_decay={step_decay}')
        _, (x_val, y_val) = next(enumerate(self.data))
        batch_size = len(x_val)
        if uap_init is None:
            print('init uap to zero')
            batch_delta = torch.zeros_like(x_val) # initialize as zero vector
        else:
            batch_delta = uap_init.unsqueeze(0).repeat([batch_size, 1, 1, 1])
        delta = batch_delta[0]
        losses = []
        
        # loss function
        if layer_name is None:
            if loss_fn is None: loss_fn = nn.CrossEntropyLoss(reduction = 'none')
            beta = torch.FloatTensor([beta]).to(self.device)
            def clamped_loss(output, target):
                loss = torch.mean(torch.min(loss_fn(output, target), beta))
                return loss
        
        # layer maximization attack
        else:
            def get_norm(self, forward_input, forward_output):
                global main_value
                main_value = torch.norm(forward_output, p = 'fro')
            for name, layer in self.model.named_modules():
                if name == layer_name:
                    handle = layer.register_forward_hook(get_norm)
                    
        batch_delta.requires_grad_()
        for epoch in range(self.epochs):
            print('epoch %i/%i' % (epoch + 1, self.epochs))
            
            # perturbation step size with decay
            eps_step = self.eps * step_decay
            
            for i, (x_val, y_val) in tqdm(enumerate(self.data)):
                if batch_delta.grad is not None:
                    batch_delta.grad.zero_() 
                batch_delta.data = delta.unsqueeze(0).repeat([x_val.shape[0], 1, 1, 1])

                # for targeted UAP, switch output labels to y_target
                if y_target is not None: 
                    y_val = torch.ones(size = y_val.shape, dtype = y_val.dtype) * y_target
                
                perturbed = torch.clamp((x_val + batch_delta).to(self.device), 0, 1)
                outputs = self.model(perturbed)
                
                # loss function value
                if layer_name is None: 
                    loss = clamped_loss(outputs, y_val.to(self.device))
                else: 
                    loss = main_value
                
                if y_target is not None: 
                    loss = -loss # minimize loss for targeted UAP

                losses.append(torch.mean(loss))
                loss.backward()
                
                # batch update
                grad_sign = batch_delta.grad.data.mean(dim = 0).sign()
                delta = delta + grad_sign * eps_step
                delta = torch.clamp(delta, -self.eps, self.eps)
                batch_delta.grad.data.zero_()
        
        if layer_name is not None: 
            handle.remove() # release hook
        
        return delta.data, losses

    def save_uap(self, uap):
        
        save_name = f'sgd-deit-{self.epochs}-{round(self.eps,3)}'
        save_path = f'./src/adversarial/resources/{save_name}.pth'
        torch.save(uap, save_path)
        print(f'successfully saved at: {save_path}')
    


class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super(Normalizer, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        
    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)
    
    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)
    

def normalize_fn(tensor, mean, std):
    """
    Differentiable version of torchvision.functional.normalize
    - default assumes color channel is at dim = 1
    """
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


'''
Load pre-trained ImageNet models

For models pre-trained on Stylized-ImageNet:
[ICLR 2019] ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness
Paper: https://openreview.net/forum?id=Bygh9j09KX
Code: https://github.com/rgeirhos/texture-vs-shape
'''    




# dataloader for ImageNet
def loader_imgnet(dir_data, nb_images = 50000, batch_size = 100, img_size = 224):

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])
    
    val_dataset = ImageFolder(dir_data, val_transform)
    
    # Random subset if not using the full 50,000 validation set
    if nb_images < 50000:
        np.random.seed(0)
        sample_indices = np.random.permutation(range(nb_images))[:nb_images]
        val_dataset = Subset(val_dataset, sample_indices)

    print(f'number of samples: {len(val_dataset)}')
    if len(val_dataset) % batch_size != 0:
        cond = len(val_dataset) % batch_size != 0
        while cond:
            batch_size -= 1
            cond = len(val_dataset) % batch_size != 0
        print(f'new batch size: {batch_size}')
    
    dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = batch_size,                              
        shuffle = True, 
        #num_workers = 2#max(1, multiprocessing.cpu_count() - 1)
    )
    
    return dataloader



# Evaluate model on data with or without UAP
# Assumes data range is bounded by [0, 1]
def evaluate(model, loader, uap = None, n = 5):
    '''
    OUTPUT
    top         top n predicted labels (default n = 5)
    top_probs   top n probabilities (default n = 5)
    top1acc     array of true/false if true label is in top 1 prediction
    top5acc     array of true/false if true label is in top 5 prediction
    outputs     output labels
    labels      true labels
    '''
    probs, labels = [], []
    model.eval()
    
    if uap is not None:
        _, (x_val, y_val) = next(enumerate(loader))
        batch_size = len(x_val)
        uap = uap.unsqueeze(0).repeat([batch_size, 1, 1, 1])
    
    with torch.set_grad_enabled(False):
        for i, (x_val, y_val) in enumerate(loader):
            
            if uap is None:
                out = torch.nn.functional.softmax(model(x_val.cuda()), dim = 1)
            else:
                perturbed = torch.clamp((x_val + uap).cuda(), 0, 1) # clamp to [0, 1]
                out = torch.nn.functional.softmax(model(perturbed), dim = 1)
                
            probs.append(out.cpu().numpy())
            labels.append(y_val)
            
    # Convert batches to single numpy arrays    
    probs = np.stack([p for l in probs for p in l])
    labels = np.array([t for l in labels for t in l])
    
    # Extract top 5 predictions for each example
    top = np.argpartition(-probs, n, axis = 1)[:,:n]
    top_probs = probs[np.arange(probs.shape[0])[:, None], top].astype(np.float16)
    top1acc = top[range(len(top)), np.argmax(top_probs, axis = 1)] == labels
    top5acc = [labels[i] in row for i, row in enumerate(top)]
    outputs = top[range(len(top)), np.argmax(top_probs, axis = 1)]
        
    return top, top_probs, top1acc, top5acc, outputs, labels

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet_path', type=str, help='set path to ImageNet Dataset')
    parser.add_argument('--epochs', type=int, default=10, help='set epochs')
    parser.add_argument('--device', type=str, default='cpu', help='set device')
    parser.add_argument('--nb_images', type=int, default=40000, help='set number of images for training')
    parser.add_argument('--batch_size', type=int, default=10, help='set batch size')
    args = parser.parse_args()


    imgnet_datapath = args.imagenet_path 
    #imgnet_datapath = "./data/imagenet_dummy/" 

    loader = loader_imgnet(imgnet_datapath, nb_images=args.nb_images, batch_size=args.batch_size) # adjust batch size as appropriate


    #model

    model = DeiT.from_pretrained("facebook/deit-base-distilled-patch16-224", torch_dtype=torch.float32)
    input_size = 224
    model.to(args.device)

    print('\nModel Loaded\n')


    nb_epoch = args.epochs
    eps = 10 / 255
    beta = 12
    step_decay = 0.7

    trainer = UAPTrainer(model, loader, nb_epoch, eps)
    uap, losses = trainer.train(beta, step_decay)
    trainer.save_uap(uap)




