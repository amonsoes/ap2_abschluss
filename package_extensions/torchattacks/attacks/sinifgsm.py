import torch
import torch.nn as nn

from ..attack import Attack


class SINIFGSM(Attack):
    r"""
    SI-NI-FGSM in the paper 'NESTEROV ACCELERATED GRADIENT AND SCALEINVARIANCE FOR ADVERSARIAL ATTACKS'
    [https://arxiv.org/abs/1908.06281], Published as a conference paper at ICLR 2020
    Modified from "https://githuba.com/JHL-HUST/SI-NI-FGSM"

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 1.0)
        m (int): number of scale copies. (Default: 5)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.SINIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, m=5)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, surrogate_loss=torch.nn.BCEWithLogitsLoss(), eps=8/255, alpha=2/255, steps=10, decay=1.0, m=5):
        super().__init__("SINIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.m = m
        self.supported_mode = ['default', 'targeted']
        self.loss = surrogate_loss

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        #loss = nn.CrossEntropyLoss()
        loss = nn.BCEWithLogitsLoss()
        
        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            nes_image = adv_images + self.decay*self.alpha*momentum
            # Calculate sum the gradients over the scale copies of the input image
            adv_grad = torch.zeros_like(images).detach().to(self.device)
            for i in torch.arange(self.m):
                nes_images = nes_image / torch.pow(2, i)
                outputs = self.get_logits(nes_images)
                # Calculate loss
                if self.targeted:
                    cost = -self.loss(outputs, target_labels)
                else:
                    cost = self.loss(outputs, labels)
                adv_grad += torch.autograd.grad(cost, adv_images,
                                                retain_graph=False, create_graph=False)[0]
            adv_grad = adv_grad / self.m

            # Update adversarial images
            grad = self.decay*momentum + adv_grad / torch.mean(torch.abs(adv_grad), dim=(1,2,3), keepdim=True)
            momentum = grad
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class VarSINIFGSM(Attack):
    r"""
    SI-NI-FGSM in the paper 'NESTEROV ACCELERATED GRADIENT AND SCALEINVARIANCE FOR ADVERSARIAL ATTACKS'
    [https://arxiv.org/abs/1908.06281], Published as a conference paper at ICLR 2020
    Modified from "https://githuba.com/JHL-HUST/SI-NI-FGSM"

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 1.0)
        m (int): number of scale copies. (Default: 5)
        Q (int): number of randomly sampled images in eps-range of input. (Default: 5)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.SINIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, m=5, Q=5)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, surrogate_loss=torch.nn.BCEWithLogitsLoss(), eps=8/255, alpha=2/255, steps=10, decay=1.0, m=5, Q=5, beta=3/2):
        super().__init__("SINIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.m = m
        self.Q = Q
        self.beta = beta
        self.supported_mode = ['default', 'targeted']
        self.loss = surrogate_loss

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            # Calculate sum the gradients over the scale copies of the input image
            adv_grad = torch.zeros_like(images).detach().to(self.device)
            nes_image = adv_images + self.decay*self.alpha*momentum
            # check if image is already in adversarial subspace
            class_results = self.get_class_result(adv_images)
            if class_results != labels:
                
                for i in torch.arange(self.m):
                    nes_images = nes_image / torch.pow(2, i)
                    adv_grad += self.get_sample_grad(nes_images, labels, adv_images)    
                
                
                for _ in torch.arange(self.Q-1):
                    
                    neighbor_images = adv_images.detach() + \
                                    torch.randn_like(images).uniform_(-self.eps*self.beta, self.eps*self.beta)
                    neighbor_images.requires_grad = True

                    nes_image = neighbor_images + self.decay*self.alpha*momentum
                    
                    for i in torch.arange(self.m):
                        nes_images = nes_image / torch.pow(2, i)
                        adv_grad += self.get_sample_grad(nes_images, labels, neighbor_images)
                adv_grad = adv_grad / (self.m*self.Q)
            
            else:

                for i in torch.arange(self.m):
                    nes_images = nes_image / torch.pow(2, i)
                    adv_grad += self.get_sample_grad(nes_images, labels, adv_images)    
                adv_grad = adv_grad / self.m
              

            # Update adversarial images
            grad = self.decay*momentum + adv_grad / torch.mean(torch.abs(adv_grad), dim=(1,2,3), keepdim=True)
            momentum = grad
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
    
    def get_sample_grad(self, sample, labels, adv_images, target_labels=None):
        outputs = self.get_logits(sample)
        # Calculate loss
        if self.targeted:
            cost = -self.loss(outputs, target_labels)
        else:
            cost = self.loss(outputs, labels)
        return torch.autograd.grad(cost, adv_images,
                                        retain_graph=False, create_graph=False)[0]
        
    def get_class_result(self, inputs):
        out = self.get_logits(inputs)
        if self.loss == torch.nn.BCEWithLogitsLoss():
            class_results = (torch.sigmoid(out) > 0.5).float()
        else:
            class_results = torch.argmax(torch.softmax(out)) 
        # TODO: make task-agnostic (mc or bin)
        return class_results