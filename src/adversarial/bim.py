import torch
import torch.nn as nn

from src.adversarial.attack_base import Attack


class BIM(Attack):
    r"""
    BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)

    .. note:: If steps set to 0, steps will be automatically decided following the paper.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=10)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, model_trms, eps=8 / 255, steps=10, l2_bound=-1, *args, **kwargs):
        super().__init__("BIM", model, model_trms, *args, **kwargs)
        self.eps = eps
        self.steps = steps
        self.loss = torch.nn.CrossEntropyLoss()
        self.supported_mode = ["default", "targeted"]
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
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images + self.alpha * grad.sign()
            delta = adv_images - images
            delta = torch.clamp(delta, min=-self.eps, max=self.eps)
            adv_images = images + delta
            adv_images = torch.clamp(adv_images, min=0.0, max=1.0).detach()

        return adv_images
