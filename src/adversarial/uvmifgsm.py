import torch
import math
from src.adversarial.attack_base import Attack

class VMIFGSM(Attack):
    r"""
    VMI-FGSM in the paper 'Enhancing the Transferability of Adversarial Attacks through Variance Tuning
    [https://arxiv.org/abs/2103.15571], Published as a conference paper at CVPR 2021
    Modified from "https://github.com/JHL-HUST/VT"

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

    def __init__(self, model, surrogate_loss, model_trms, eps=8/255, steps=7, decay=1.0, N=5, beta=3/2, l2_bound=-1, *args, **kwargs):
        super().__init__("VMIFGSM", model, model_trms, *args, **kwargs)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.N = N
        self.beta = beta
        self.supported_mode = ['default', 'targeted']
        self.loss = surrogate_loss
        self.model_trms = model_trms
        if l2_bound != -1:
            self.eps = self.get_eps_in_range(l2_bound)
        self.alpha = self.eps / steps


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
            outputs = self.get_logits(self.model_trms(adv_images))

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
            
            for _ in range(self.N):
                neighbor_images = adv_images.detach() + \
                                  torch.randn_like(images).uniform_(-self.eps*self.beta, self.eps*self.beta)
                neighbor_images.requires_grad = True
                outputs = self.get_logits(neighbor_images)
                
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

        return adv_images
    

class UVMIFGSM(VMIFGSM):
    r"""
    Init VMIFGSM with UAP

    """

    def __init__(self, 
                model, 
                surrogate_loss, 
                model_trms,
                uap_path='', 
                init_method='eps', 
                *args, 
                **kwargs):
        super().__init__(model, surrogate_loss, model_trms, *args, **kwargs)
        self.uap = self.load_uap(uap_path)
        if init_method == 'eps':
            self.init_method = self.init_eps
        elif init_method == 'he':
            self.init_method = self.init_he
        elif init_method == 'xavier':
            self.init_method = self.init_xavier
        elif init_method == 'uap':
            self.init_method = self.init_uap
        else:
            raise ValueError('Init method not recognized')

    def load_uap(self, uap_path):
        uap = torch.load(uap_path).to(self.device)
        return uap
    
    def init_eps(self):
        """
        initializes the perturbation based on
        sign(uap)*epsilon
        """
        return self.uap.sign()*self.eps

    def init_he(self):
        """
        initializes the perturbation based on
        uap*he_init_formula
        """
        return torch.clamp(self.uap * (math.sqrt(2/(224*224))), max=self.eps, min=self.eps)

    def init_xavier(self):
        """
        initializes the perturbation based on
        uap*xavier_init_formula
        """
        return torch.clamp(self.uap * (math.sqrt(6)/math.sqrt(224 * 224)), max=self.eps, min=self.eps)

    def init_uap(self):
        """
        initializes the perturbation based on
        uap
        """
        return torch.clamp(self.uap, max=self.eps, min=self.eps)

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
        # initialize vmifgsm with UAP
        adv_images = adv_images + self.init_method()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(self.model_trms(adv_images))

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
            
            for _ in range(self.N):
                neighbor_images = adv_images.detach() + \
                                  torch.randn_like(images).uniform_(-self.eps*self.beta, self.eps*self.beta)
                neighbor_images.requires_grad = True
                outputs = self.get_logits(neighbor_images)
                
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

        return adv_images
    