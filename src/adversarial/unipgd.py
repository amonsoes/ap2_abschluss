import torch
import math
from src.adversarial.attack_base import Attack

class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, surrogate_loss, model_trms, eps=8/255,
                 alpha=2/255, steps=7, random_start=True):
        super().__init__("PGD", model, model_trms)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
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
        
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

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

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class UniPGD(Attack):
    r"""
    This model initializes the perturbation with a prior from a Universal Adversarial Perturbation (UAP)

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        uap: init uap used as an initialization of delta. (Default: 1)
        init_method: choose one of "he"/"xavier"/"eps"/"uap"

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, surrogate_loss, model_trms, eps=8/255,
                 alpha=2/255, steps=7, uap_path='', init_method='uap'):
        super().__init__("PGD", model, model_trms)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.supported_mode = ['default', 'targeted']
        self.loss = surrogate_loss
        self.model_trms = model_trms
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
        
        adv_images = images.clone().detach()

            # Starting at a uniformly random point
        adv_images = adv_images + self.init_method()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

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

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images