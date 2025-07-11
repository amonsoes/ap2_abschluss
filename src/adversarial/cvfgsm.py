import torch
import math
from src.adversarial.attack_base import Attack
from diff_jpeg import diff_jpeg_coding
from src.adversarial.jpeg_ifgm import DiffJPEG


class CVFGSM(Attack):
    r"""
    Adds compression variance to the variance tuning computation of VMIFGSM

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        steps (int): number of iterations. (Default: 10)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        N (int): the number of sampled examples in the neighborhood. (Default: 5)
        beta (float): the upper bound of neighborhood. (Default: 3/2)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.VMIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, N=5, beta=3/2)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, 
                model, 
                surrogate_loss, 
                model_trms, 
                eps=8/255, 
                alpha=2/255, 
                steps=7, 
                decay=1.0, 
                N=5, 
                beta=3/2,
                target_mode='default'):
        
        super().__init__("VMIFGSM", model, model_trms)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = eps / steps
        self.N = N
        self.beta = beta
        self.supported_mode = ['default', 'targeted']
        self.loss = surrogate_loss
        self.model_trms = model_trms
        if target_mode != 'default':
            self.set_target_mode(target_mode)
        
        self.compression_rates = []
        compression = 99
        for i in range(N+1): # in 5-step decrements
            self.compression_rates.append(compression)
            compression -= 5
        # get the mean compression rate 
        self.mean_compression = self.compression_rates.pop(self.get_mean_compression(N))
    
    def get_mean_compression(self, N):
        if N % 2 != 0:
           return math.floor(N / 2)
        else:
            return N // 2

    def set_target_mode(self, mode):
        if mode == 'least_likely':
            self.set_mode_targeted_least_likely()
        elif mode == 'most_likely':
            self.set_mode_targeted_most_likely()
        else:
            print('WARNING: set_target_mode was set to random. If unwanted, change "target_mode" arg to either "least_likely" or "most_likely".')
            self.set_mode_targeted_random()

    def compress(self, img, jpeg_quality):
        img = img * 255
        compressed =  diff_jpeg_coding(image_rgb=img, jpeg_quality=torch.tensor([jpeg_quality]).to(self.device))
        return (compressed / 255).clip(min=0., max=1.)
            
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        v = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            
            adv_images.requires_grad = True
            
            ensemble_grad = torch.zeros_like(adv_images).detach().to(self.device)    
            grad_list = []
            loss_tensor = torch.zeros(self.N)
            
            for  e, comp_rate in enumerate(self.compression_rates):
                #compress = DiffJPEG(height=224, width=224, differentiable=True, quality=comp_rate, device=self.device)
                nc_i = adv_images.detach() 
                nc_i.requires_grad = True
                compressed_n_image = self.compress(nc_i, comp_rate)
                outputs = self.get_logits(self.model_trms(compressed_n_image))


                if self.targeted:
                    cost = -self.loss(outputs, target_labels)
                else:
                    cost = self.loss(outputs, labels)
                loss_tensor[e] = cost

                grad = torch.autograd.grad(cost, nc_i,
                                        retain_graph=False,
                                        create_graph=False)[0]
                
                grad_list.append(grad)
            
            loss_tensor_exp = loss_tensor.exp()
            total_cost_exp = loss_tensor_exp.sum()
            for cost_exp, grad in zip(loss_tensor_exp, grad_list):
                ensemble_grad +=  (1 - (cost_exp / total_cost_exp)) * grad

            # Update adversarial images
            adv_grad = ensemble_grad

            grad = (adv_grad + v) / torch.mean(torch.abs(adv_grad + v), dim=(1,2,3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            # Calculate Gradient Variance
            GV_grad = torch.zeros_like(images).detach().to(self.device)
            
            
            for i in range(self.N):
                #compress = DiffJPEG(height=224, width=224, differentiable=True, quality=self.mean_compression, device=self.device)
                neighbor_images = adv_images.detach() + \
                                  torch.randn_like(images).uniform_(-self.eps*self.beta, self.eps*self.beta)
                neighbor_images.requires_grad = True
                #outputs = self.get_logits(self.model_trms(self.compress(neighbor_images, self.mean_compression)))
                outputs = self.get_logits(self.model_trms(self.compress(neighbor_images, self.compression_rates[i])))
                
                # Calculate loss
                if self.targeted:
                    cost = -self.loss(outputs, target_labels)
                else:
                    cost = self.loss(outputs, labels)
                GV_grad += torch.autograd.grad(cost, neighbor_images,
                                               retain_graph=False, create_graph=False)[0]

            # obtaining the gradient variance
            v = GV_grad / self.N - adv_grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        if self.targeted:
            return adv_images, target_labels
        else:
            return adv_images

    '''def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        v = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            compress = DiffJPEG(height=224, width=224, differentiable=True, quality=self.mean_compression, device=self.device)
            compressed_main_adv_image = compress(adv_images)
            outputs = self.get_logits(self.model_trms(compressed_main_adv_image))

            # Calculate loss
            if self.targeted:
                cost = -self.loss(outputs, target_labels)
            else:
                cost = self.loss(outputs, labels)

            # Update adversarial images
            adv_grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = (adv_grad + v) / torch.mean(torch.abs(adv_grad + v), dim=(1,2,3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            # Calculate Gradient Variance
            GV_grad = torch.zeros_like(images).detach().to(self.device)
            
            for  _ in range(self.N):
                neighbor_images = adv_images.detach() + torch.randn_like(images).uniform_(-self.eps*self.beta, self.eps*self.beta)
                neighbor_images.requires_grad = True
                
                ensemble_grad = torch.zeros_like(adv_images).detach().to(self.device)    
                grad_list = []
                loss_tensor = torch.zeros(self.N)
                
                for  e, comp_rate in enumerate(self.compression_rates):
                    compress = DiffJPEG(height=224, width=224, differentiable=True, quality=comp_rate, device=self.device)
                    nc_i = neighbor_images.detach() 
                    nc_i.requires_grad = True
                
                    compressed_n_image = compress(nc_i)
                    outputs = self.get_logits(self.model_trms(compressed_n_image))

                    if self.targeted:
                        cost = -self.loss(outputs, target_labels)
                    else:
                        cost = self.loss(outputs, labels)
                    loss_tensor[e] = cost

                    grad = torch.autograd.grad(cost, nc_i,
                                            retain_graph=False,
                                            create_graph=False)[0]
                    
                    grad_list.append(grad)
                
                loss_tensor_exp = loss_tensor.exp()
                total_cost_exp = loss_tensor_exp.sum()
                for cost_exp, grad in zip(loss_tensor_exp, grad_list):
                    ensemble_grad +=  (1 - (cost_exp / total_cost_exp)) * grad
                    
                GV_grad += ensemble_grad
            # obtaining the gradient variance
            v = GV_grad / (self.N * len(self.compression_rates)) - adv_grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images'''
    