import torch
import torch.nn as nn
import torchvision.transforms as T
import scipy.ndimage as nd
import torch_dct as dct
import pytorch_colors as colors

from ..attack import Attack


class PGDL2(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 1.0)
        alpha (float): step size. (Default: 0.2)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGDL2(model, eps=1.0, alpha=0.2, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, 
                model,
                surrogate_loss,
                model_trms,
                eps=1.0, 
                alpha=0.2, 
                steps=10, 
                random_start=True, 
                eps_for_division=1e-10
                ):
        super().__init__("PGDL2", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.eps_for_division = eps_for_division
        self.supported_mode = ['default', 'targeted']
        self.loss = surrogate_loss
        self.model_trms = model_trms

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        
        images = images.to(torch.float32) / 255
        adv_images = images.clone().detach()
        
        batch_size = len(images)

        if self.random_start:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_images).normal_()
            # d_flat = delta.view(adv_images.size(0), -1) # possible bug in 'view'?
            d_flat = torch.flatten(delta, start_dim=1)
            n = d_flat.norm(p=2, dim=1).view(adv_images.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*self.eps # <- could add masked computation here
            adv_images = torch.clamp(adv_images + delta, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(self.model_trms(adv_images))

            # Calculate loss
            if self.targeted:
                cost = -self.loss(outputs, target_labels)
            else:
                cost = self.loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            #grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + self.eps_for_division
            grad_norms = torch.norm(torch.flatten(grad, start_dim=1), p=2, dim=1) + self.eps_for_division
            grad = grad / grad_norms.view(batch_size, 1, 1, 1)
            adv_images = adv_images.detach() + self.alpha * grad

            delta = adv_images - images
            #delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            delta_norms = torch.norm(torch.flatten(delta, start_dim=1), p=2, dim=1)
            factor = self.eps / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)

            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
    

class HpfPGDL2(PGDL2):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 1.0)
        alpha (float): step size. (Default: 0.2)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGDL2(model, eps=1.0, alpha=0.2, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, hpf_masker, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.hpf_masker = hpf_masker
            
    def get_alpha_delta(self, mask):
        # mask values are between 0-1
        inv_mask_mean = (1 - mask).mean()
        alpha_delta = 1 + inv_mask_mean
        return alpha_delta.item()
    
    def get_eps_delta(self, mask):
        # yields eps values per pixel depending on the HPF coeffs
        mask_around_0 = mask - 0.5
        scaled_by_alpha = mask_around_0 * self.alpha #
        eps_tensor = torch.full_like(mask, self.eps)
        eps_tensor += scaled_by_alpha
        return eps_tensor

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        # get mask
        attack_mask = self.hpf_masker(images, labels, model=self.model, model_trms=self.model_trms, loss=self.loss)
        
        # adjust eps & alpha according to mask coefficients  
        alpha_delta = self.get_alpha_delta(attack_mask)
        self.adjusted_alpha = self.alpha * alpha_delta
        self.eps_tensor = self.get_eps_delta(attack_mask)

        images = images.to(torch.float32) / 255
        adv_images = images.clone().detach()  
        batch_size = len(images)
        

        if self.random_start:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_images).normal_()
            #d_flat = delta.view(adv_images.size(0),-1)
            d_flat = torch.flatten(delta, start_dim=1)
            n = d_flat.norm(p=2,dim=1).view(adv_images.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*self.eps_tensor*attack_mask
            adv_images = torch.clamp(adv_images + delta, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(self.model_trms(adv_images))

            # Calculate loss
            if self.targeted:
                cost = -self.loss(outputs, target_labels)
            else:
                cost = self.loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            #grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + self.eps_for_division
            grad_norms = torch.norm(torch.flatten(grad, start_dim=1), p=2, dim=1) + self.eps_for_division
            grad = grad / grad_norms.view(batch_size, 1, 1, 1)
            adv_images = adv_images.detach() + self.adjusted_alpha * grad * attack_mask

            delta = adv_images - images
            #delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            delta_norms = torch.norm(torch.flatten(delta, start_dim=1), p=2, dim=1)
            #factor = self.eps / delta_norms
            #factor = torch.min(factor, torch.ones_like(delta_norms))
            #delta = delta * factor.view(-1, 1, 1, 1)
            
            #new for adjusted eps
            factor = self.eps / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)

            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


