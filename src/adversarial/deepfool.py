import torch
import torch.nn as nn

from src.adversarial.attack_base import Attack


class DeepFool(Attack):
    r"""
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]
    Distance Measure : L2
    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (Default: 50)
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, model_trms, steps=1000, overshoot=2.0, l2_bound=-1, kappa=0, *args, **kwargs):
        super().__init__("DeepFool", model, model_trms, *args, **kwargs)
        self.steps = steps
        self.overshoot = overshoot
        self.kappa = kappa
        self.supported_mode = ["default"]
        self.low, self.high = self.get_iq_range_from_l2_bound(l2_bound)
        self.low = self.get_l2_from_bounds(l2_bound)

        if self.low == self.high:
            self.low -= self.low*0.1
            self.high += self.high*0.1


    def forward(self, images, labels):
        r"""
        Overridden.
        """
        adv_images, target_labels = self.forward_return_target_labels(images, labels)
        return adv_images

    def forward_return_target_labels(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        batch_size = len(images)
        correct = torch.tensor([True] * batch_size)
        target_labels = labels.clone().detach().to(self.device)
        curr_steps = 0

        adv_images = []
        for idx in range(batch_size):
            image = images[idx : idx + 1].clone().detach()
            adv_images.append(image)
        l2_dists = torch.zeros_like(labels)
        for i in range(self.steps):
            for idx in range(batch_size):
                early_stop, pre, adv_image = self._forward_indiv(
                    adv_images[idx], labels[idx]
                )
                adv_images[idx] = adv_image
                target_labels[idx] = pre
                if early_stop:
                    correct[idx] = False
            is_adversarial = ~correct
            l2_dists = self.get_l2(images, torch.cat(adv_images).detach())
            print(f'Current Step: {i}, l2_dists:{l2_dists}, adversarial:{is_adversarial}')
            if torch.all(l2_dists >= self.low) and torch.all(l2_dists <= self.high) and all(is_adversarial):
                print('L2 constraint success.')
                break

        adv_images = torch.cat(adv_images).detach()
        return adv_images, target_labels
    
    def get_logits(self, inputs, labels=None, *args, **kwargs):
        inputs = self.model_trms(inputs)
        return super().get_logits(inputs, labels, *args, **kwargs)

    def _forward_indiv(self, image, label):
        is_adv = False
        image.requires_grad = True
        fs = self.get_logits(image)[0]
        _, pre = torch.max(fs, dim=0)
        if pre != label:
            is_adv = True
            return (is_adv, pre, image)

        ws = self._construct_jacobian(fs, image).squeeze()
        image = image.detach()

        f_0 = fs[label]
        w_0 = ws[label]

        wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs[wrong_classes]
        w_k = ws[wrong_classes]

        f_prime = f_k - f_0
        w_prime = w_k - w_0
        value = torch.abs(f_prime) / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        _, hat_L = torch.min(value, 0)

        delta = (
            torch.abs(f_prime[hat_L])
            * w_prime[hat_L]
            / (torch.norm(w_prime[hat_L], p=2) ** 2)
        )

        delta = (delta / (delta.pow(2).sum().sqrt()))
        print(f'delta len:{delta.pow(2).sum().sqrt()}')
        target_label = hat_L if hat_L < label else hat_L + 1

        adv_image = image + (self.high * delta)
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return (is_adv, target_label, adv_image)

    # https://stackoverflow.com/questions/63096122/pytorch-is-it-possible-to-differentiate-a-matrix
    # torch.autograd.functional.jacobian is only for torch >= 1.5.1
    def _construct_jacobian(self, y, x):
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx + 1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)

    def _construct_jacobian(self, y, x):
        return torch.autograd.functional.jacobian(self.get_logits, x, create_graph=False).detach()
