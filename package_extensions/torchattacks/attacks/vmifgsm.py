import torch
import torch.nn as nn
import torchvision.transforms as T
import scipy.ndimage as nd
import torch_dct as dct
import pytorch_colors as colors

from ..attack import Attack

class VarVMIFGSM(Attack):
    r"""

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

    def __init__(self, model, surrogate_loss, surrogate_model_trms, eps=8/255, alpha=2/255, steps=10, decay=1.0, N=5, beta=3/2):
        super().__init__("VMIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.N = N
        self.beta = beta
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

        momentum = torch.zeros_like(images).detach().to(self.device)

        v = torch.zeros_like(images).detach().to(self.device)
        
        images = images / 255

        adv_images = images.clone().detach()
        
        neighbors = torch.zeros_like(adv_images).detach()
        Q = 0

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
            out_adv = self.get_logits(adv_images)
            
            class_results = (torch.sigmoid(out_adv) > 0.5).float() # TODO: make task-agnostic (mc or bin)

            if class_results != labels:
                # add adv_images to 'walking mean'
                neighbors += adv_images
                Q += 1
                for _ in range(self.N-1):
                    
                    neighbor_images = adv_images.detach() + \
                                    torch.randn_like(images).uniform_(-self.eps*self.beta, self.eps*self.beta)

                    # add neighbor to 'walking mean'
                    neighbors += neighbor_images
                    Q += 1
                    
                current_mean = neighbors / Q
                v_Q = current_mean - adv_images
                adv_images = (adv_images + v_Q) / torch.mean(torch.abs(adv_images + v_Q), dim=(1,2,3), keepdim=True)
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()
                
        return adv_images


class MVMIFGSM(Attack):
    r"""

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

    def __init__(self, model, surrogate_loss, eps=8/255, alpha=2/255, steps=7, decay=1.0, N=5, beta=3/2):
        super().__init__("VMIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.N = N
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

        v = torch.zeros_like(images).detach().to(self.device)
        
        #images = images / 255

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)
                
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
            out_adv = self.get_logits(adv_images)
            
            class_results = (torch.sigmoid(out_adv) > 0.5).float() # TODO: make task-agnostic (mc or bin)

            if class_results != labels:

                neighbors = adv_images.expand(self.N, -1, -1, -1).clone().detach()
                i = 1
                for _ in range(self.N-1):
                    
                    neighbor_images = adv_images.detach() + \
                                    torch.randn_like(images).uniform_(-self.eps*self.beta, self.eps*self.beta)

                    # add neighbor to neighbors list
                    neighbors[i] = neighbor_images
                    
                    i += 1
                    
                adv_images = neighbors.mean(0).unsqueeze(0).detach()
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()
                
        return adv_images
    
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

    def __init__(self, model, surrogate_loss, model_trms, eps=8/255, alpha=2/255, steps=7, decay=1.0, N=5, beta=3/2):
        super().__init__("VMIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.N = N
        self.beta = beta
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

        momentum = torch.zeros_like(images).detach().to(self.device)

        v = torch.zeros_like(images).detach().to(self.device)
        
        images = images / 255

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
    
class YCBCRTransform:
    
    def __init__(self):
        print('Warning: input tensors must be in the format RGB')
    
    def __call__(self, tensor):
        ycbcr = self.get_channels(tensor)
        return ycbcr
    
    @staticmethod
    def normalize(tensor):
        tensor / 255
        return tensor

    @staticmethod
    def to_int(tensor):
        tensor += 0.5
        tensor *= 255
        return tensor
        
    def get_channels(self, tensor):
        return colors.rgb_to_ycbcr(tensor)
    
    def inverse(self, tensor):
        return colors.ycbcr_to_rgb(tensor)
    
    
class Patchify:
    
    def __init__(self, img_size, patch_size, n_channels):
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2

        assert (img_size // patch_size) * patch_size == img_size
        
    
    def __call__(self, x):
        p = x.unfold(1, 8, 8).unfold(2, 8, 8).unfold(3, 8, 8) # x.size() -> (batch, model_dim, n_patches, n_patches)
        self.unfold_shape = p.size()
        p = p.contiguous().view(-1,8,8)
        return p
    
    def inverse(self, p):
        if not hasattr(self, 'unfold_shape'):
            raise AttributeError('Patchify needs to be applied to a tensor in ordfer to revert the process.')
        x = p.view(self.unfold_shape)
        output_h = self.unfold_shape[1] * self.unfold_shape[4]
        output_w = self.unfold_shape[2] * self.unfold_shape[5]
        x = x.permute(0,1,4,2,5,3).contiguous()
        x = x.view(3, output_h, output_w)
        return x 

class DCT:
    
    def __init__(self, img_size=224, patch_size=8, n_channels=3, diagonal=0):
        """
        diagonal parameter will decide how much cosine components will be taken into consideration
        while calculating fgsm patches.
        """
        print('DCT class transforms on 3d tensors')
        self.patchify = Patchify(img_size=img_size, patch_size=patch_size, n_channels=n_channels)
        self.normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.mask = torch.flip(torch.triu(torch.ones((8,8)), diagonal=diagonal), dims=[0])
        
    def __call__(self, tensor):
        p, fgsm_coeffs = self.patched_dct(tensor)
        dct_coeffs = self.patchify.inverse(p)
        fgsm_coeffs = self.patchify.inverse(fgsm_coeffs)
        fgsm_coeffs = fgsm_coeffs / fgsm_coeffs.max()
        return dct_coeffs, fgsm_coeffs
    
    def patched_dct(self, tensor):
        p = self.patchify(tensor)
        fgsm_coeff_tensor = torch.zeros(p.shape, dtype=torch.float32)
        for e, patch in enumerate(p):
            
            dct_coeffs = dct.dct_2d(patch, norm='ortho')
            dct_coeffs[0][0] = 0.0
            fgsm_coeffs = self.calculate_fgsm_coeffs(dct_coeffs)
            fgsm_coeff_tensor[e] = fgsm_coeffs
            p[e] = dct_coeffs
        return p, fgsm_coeff_tensor
    
    def calculate_fgsm_coeffs(self, patch):
        sum_patch = sum(patch[self.mask == 1].abs())
        return torch.full((8,8), fill_value=sum_patch)

class LaplacianOfGaussian:
    
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):
        return torch.tensor(nd.gaussian_laplace(img, self.sigma))

    
class HpfVMIFGSM(VMIFGSM):
    
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

        momentum = torch.zeros_like(images).detach().to(self.device)

        v = torch.zeros_like(images).detach().to(self.device)

        # =====inserted===

        attack_mask = self.hpf_masker(images, labels, self.model, self.model_trms, self.loss)
        
        alpha_delta = self.get_alpha_delta(attack_mask)
        self.adjusted_alpha = self.alpha * alpha_delta
        self.eps_tensor = self.get_eps_delta(attack_mask)
        max_adjusted_eps = self.eps_tensor.max()
        # =======

        images = images.to(torch.float32) / 255
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
            
            attack_mask = self.hpf_masker.update_saliency_mask(adv_grad.clone().detach())

            grad = (adv_grad + v) / torch.mean(torch.abs(adv_grad + v), dim=(1,2,3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            # Calculate Gradient Variance
            GV_grad = torch.zeros_like(images).detach().to(self.device)
            for _ in range(self.N):
                neighbor_images = adv_images.detach() + \
                                  torch.randn_like(images).uniform_(-max_adjusted_eps*self.beta, max_adjusted_eps*self.beta)
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

            adv_images = adv_images.detach() + self.adjusted_alpha*attack_mask*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps_tensor, max=self.eps_tensor)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
