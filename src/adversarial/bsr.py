import torch
import numpy as np
import random
import math
import torchvision.transforms as T

from torch import nn
from src.adversarial.attack_base import Attack


class MIFGSM(Attack):
    r"""
    MI-FGSM in the paper 'Boosting Adversarial Attacks with Momentum'
    [https://arxiv.org/abs/1710.06081]

    """

    def __init__(self, model, model_trms, eps=8/255, steps=10, decay=1.0, l2_bound=-1, *args, **kwargs):
        super().__init__("MIFGSM", model, model_trms,  *args, **kwargs)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.supported_mode = ["default", "targeted"]
        self.loss = torch.nn.BCEWithLogitsLoss() if model.model_name in ['clip_det', 'corviresnet'] else torch.nn.CrossEntropyLoss()

        if l2_bound != -1:
            self.eps = self.get_eps_in_range(l2_bound)
        self.alpha = self.eps / steps



    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
            labels = labels.to(torch.float32)

        if self.targeted:
            labels = self.get_target_label(images, labels)
            if isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
                labels = labels.to(torch.float32)

        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            grad = self.get_grad(adv_images, labels)

            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def get_grad(self, adv_images, labels):
        outputs = self.get_logits(adv_images)

        # Calculate loss
        if self.targeted:
            cost = -self.loss(outputs, labels)
        else:
            cost = self.loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, adv_images, retain_graph=False, create_graph=False
        )[0]
        return grad


class BSR(MIFGSM):
    """
    BSR Attack
    'Boosting Adversarial Transferability by Block Shuffle and Rotation'(https://https://arxiv.org/abs/2308.10299)
    
    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_scale (int): the number of shuffled copies in each iteration.
        num_block (int): the number of block in the image.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_scale=10, num_block=3

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/bsr/resnet18 --attack bsr --model=resnet18
        python main.py --input_dir ./path/to/data --output_dir adv_data/bsr/resnet18 --eval
    """
    
    def __init__(self, 
                model,
                model_trms,
                eps=16/255, 
                steps=10,
                decay=1., 
                num_scale=20, 
                num_blocks=3, 
                targeted=False,
                **kwargs):
        super().__init__(model, model_trms, eps, steps, decay, **kwargs)
        self.num_scale = num_scale
        self.num_block = num_blocks

    def get_length(self, length):
        rand = np.random.uniform(2, size=self.num_block)
        rand_norm = np.round(rand/rand.sum()*length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        lengths = self.get_length(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def image_rotation(self, x):
        rotation_transform = T.RandomRotation(degrees=(-24, 24), interpolation=T.InterpolationMode.BILINEAR)
        return  rotation_transform(x)

    def shuffle(self, x):
        dims = [2,3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])
        return torch.cat([torch.cat(self.shuffle_single_dim(self.image_rotation(x_strip), dim=dims[1]), dim=dims[1]) for x_strip in x_strips], dim=dims[0])

    def transform(self, x):
        """
        Scale the input for BSR
        """
        return torch.cat([self.shuffle(x) for _ in range(self.num_scale)])

    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return -self.loss(logits, label.repeat(self.num_scale)) if self.targeted else self.loss(logits, label.repeat(self.num_scale))

    def get_grad(self, adv_images, labels):
        self.model.zero_grad()
        
        diverse_inputs = self.transform(adv_images)
        #labels = labels.expand((1,self.num_scale))

        outputs = self.get_logits(diverse_inputs)

        # Calculate loss
        cost = self.get_loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, adv_images, retain_graph=False, create_graph=False
        )[0]
            
        return grad
        