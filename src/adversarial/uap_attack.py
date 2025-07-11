import torch
import math

from torch import nn
from src.adversarial.attack_base import Attack
from src.model.pretrained import DeiT

class SGAUAP(Attack):

    def __init__(self, 
                model,
                model_trms,  
                random_uap=False,
                eps=8/255,
                l2_bound=-1,
                *args, 
                **kwargs):
        super().__init__("SGAUAP", model, model_trms, *args, **kwargs)
        self.eps = eps
        if l2_bound != -1:
            self.eps = self.get_eps_in_range(l2_bound)


        if random_uap:
            self.uap = torch.empty((1, 3, 224, 224)).uniform_(-self.eps, self.eps).to(self.device)
        else:  
            self.uap = self.load_uap('./src/adversarial/resources/sga-resnet152-eps.pth')

    def load_uap(self, uap_path):
        uap = torch.load(uap_path).to(self.device)
        return torch.clamp(uap, min=-self.eps, max=self.eps)

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        adv_images = torch.clamp(images + self.uap, min=0, max=1).detach()
        return adv_images
    
    def train_uap(self):
        raise NotImplementedError('Refer to the repository to perform large-scale ImageNet training in order to generate SGA UAPs for surrogates.')


class SGDUAP(Attack):

    """
    https://ojs.aaai.org//index.php/AAAI/article/view/6017
    """
    
    def __init__(self, 
                model,
                model_trms,  
                random_uap=False,
                eps=8/255,
                l2_bound=-1,
                *args, 
                **kwargs):
        super().__init__("SGDUAP", model, model_trms, *args, **kwargs)
        self.eps = eps
        if l2_bound != -1:
            self.eps = self.get_eps_in_range(l2_bound)

        if random_uap:
            self.uap = torch.empty((1, 3, 224, 224)).uniform_(-self.eps, self.eps).to(self.device)
        elif isinstance(model, DeiT):  
            self.uap = self.load_uap('./src/adversarial/resources/sgd-deit-10-0.039.pth')
        else:
            self.uap = self.load_uap('./src/adversarial/resources/sgd-resnet50-eps10.pth')
    
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        adv_images = torch.clamp(images + self.uap, min=0, max=1).detach()
        return adv_images

    def load_uap(self, uap_path):
        uap = torch.load(uap_path).to(self.device)
        return torch.clamp(uap, min=-self.eps, max=self.eps)
    
    def uap_sgd(model, 
                loader, 
                nb_epoch, 
                eps, 
                beta = 12, 
                step_decay=0.8, 
                y_target=None,
                loss_fn = None, 
                layer_name = None, 
                uap_init = None):
        '''
        INPUT
        model       model
        loader      dataloader
        nb_epoch    number of optimization epochs
        eps         maximum perturbation value (L-infinity) norm
        beta        clamping value
        y_target    target class label for Targeted UAP variation
        loss_fn     custom loss function (default is CrossEntropyLoss)
        layer_name  target layer name for layer maximization attack
        uap_init    custom perturbation to start from (default is random vector with pixel values {-eps, eps})
        
        OUTPUT
        delta.data  adversarial perturbation
        losses      losses per iteration
        '''
        _, (x_val, y_val) = next(enumerate(loader))
        batch_size = len(x_val)
        if uap_init is None:
            batch_delta = torch.zeros_like(x_val) # initialize as zero vector
        else:
            batch_delta = uap_init.unsqueeze(0).repeat([batch_size, 1, 1, 1])
        delta = batch_delta[0]
        losses = []
        
        # loss function
        if layer_name is None:
            if loss_fn is None: loss_fn = nn.CrossEntropyLoss(reduction = 'none')
            beta = torch.cuda.FloatTensor([beta])
            def clamped_loss(output, target):
                loss = torch.mean(torch.min(loss_fn(output, target), beta))
                return loss
        
        # layer maximization attack
        else:
            def get_norm(self, forward_input, forward_output):
                global main_value
                main_value = torch.norm(forward_output, p = 'fro')
            for name, layer in model.named_modules():
                if name == layer_name:
                    handle = layer.register_forward_hook(get_norm)
                    
        batch_delta.requires_grad_()
        for epoch in range(nb_epoch):
            print('epoch %i/%i' % (epoch + 1, nb_epoch))
            
            # perturbation step size with decay
            eps_step = eps * step_decay
            
            for i, (x_val, y_val) in enumerate(loader):
                batch_delta.grad.data.zero_()
                batch_delta.data = delta.unsqueeze(0).repeat([x_val.shape[0], 1, 1, 1])

                # for targeted UAP, switch output labels to y_target
                if y_target is not None: y_val = torch.ones(size = y_val.shape, dtype = y_val.dtype) * y_target
                
                perturbed = torch.clamp((x_val + batch_delta).cuda(), 0, 1)
                outputs = model(perturbed)
                
                # loss function value
                if layer_name is None: loss = clamped_loss(outputs, y_val.cuda())
                else: loss = main_value
                
                if y_target is not None: loss = -loss # minimize loss for targeted UAP
                losses.append(torch.mean(loss))
                loss.backward()
                
                # batch update
                grad_sign = batch_delta.grad.data.mean(dim = 0).sign()
                delta = delta + grad_sign * eps_step
                delta = torch.clamp(delta, -eps, eps)
                batch_delta.grad.data.zero_()
        
        if layer_name is not None: handle.remove() # release hook
        
        return delta.data, losses
