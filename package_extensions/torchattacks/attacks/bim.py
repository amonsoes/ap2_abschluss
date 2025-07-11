import torch
import torch.nn as nn
from torch import Tensor
import scipy.ndimage as nd
import torch_dct as dct
import torchvision.transforms as T
import torchvision.transforms.functional as F
import pytorch_colors as colors

from ..attack import Attack

class GradGaussianBlur(T.GaussianBlur):
    r"""
    Gradient guided Gaussian Blur. All other args except for grad will be passed
    to T.GaussianBlur
    
    Arguments:
        grad (torch.Tensor) : gradient of Loss with respect to image x
    
    """
    def __init__(self, kernel_size=15, sigma=2.5):
        super().__init__(kernel_size=kernel_size, sigma=sigma)

    def forward(self, delta: Tensor, grad: Tensor) -> Tensor:
        delta_sign = delta.sign()
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        blurred =  F.gaussian_blur(delta, self.kernel_size, [sigma, sigma])
        blurred_delta_sign = blurred.sign()
        remove_delta = blurred_delta_sign != delta_sign
        blurred[remove_delta] = delta[remove_delta]
        return blurred


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
        self.n = 0
        self.tile_mean = 0
        
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
        
        self.tile_mean += fgsm_coeff_tensor.mean()
        self.n += 1
        return p, fgsm_coeff_tensor
    
    def calculate_fgsm_coeffs(self, patch):
        masked_patch = patch[self.mask == 1].abs()
        sum_patch = sum(masked_patch)
        return torch.full((8,8), fill_value=sum_patch.item())

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
    def __init__(self, model, surrogate_loss,  model_trms, eps=8/255, alpha=2/255, steps=7):
        super().__init__("BIM", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps*255 + 4, 1.25*eps*255))
        else:
            self.steps = steps
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
        
        images = images / 255
        adv_images = images.clone().detach()
        

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(self.model_trms(adv_images))

            # Calculate loss
            if self.targeted:
                cost = -self.loss(outputs, target_labels)
            else:
                cost = self.loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False,
                                       create_graph=False)[0]

            adv_images = adv_images + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            #self.observer(delta.squeeze().detach())
            adv_images = torch.clamp(images + delta, min=0., max=1.).detach()
            
        return adv_images
    
class GradGaussianBIM(Attack):
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
    def __init__(self, model, surrogate_loss,  model_trms, eps=8/255, alpha=2/255, steps=7, gaussian=False, gauss_kernel=15, gauss_sigma=2.5):
        super().__init__("BIM", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps*255 + 4, 1.25*eps*255))
        else:
            self.steps = steps
        self.supported_mode = ['default', 'targeted']
        if gaussian:
            self.gaussian_blur = GradGaussianBlur(kernel_size=gauss_kernel, sigma=gauss_sigma)
        else:
            self.gaussian_blur = None
        self.observer = Observer(measure_freq=True, img_size=224, cutoff_diagonal=-3)
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
        
        images = images / 255
        adv_images = images.clone().detach()
        

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(self.model_trms(adv_images))

            # Calculate loss
            if self.targeted:
                cost = -self.loss(outputs, target_labels)
            else:
                cost = self.loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False,
                                       create_graph=False)[0]

            adv_images = adv_images + self.alpha*grad.sign()
            if i == self.steps - 1:
                if self.gaussian_blur != None:
                    delta = adv_images - images
                    delta = self.gaussian_blur(delta, grad)
                    delta = torch.clamp(delta, min=-self.eps, max=self.eps)
                else:
                    delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            else:
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            self.observer(delta.squeeze().detach())
            adv_images = torch.clamp(images + delta, min=0., max=1.).detach()
            
        return adv_images


class HpfBIM(BIM):
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
        self.eps_tensor = self.get_eps_delta(attack_mask)
        alpha_delta = self.get_alpha_delta(attack_mask)
        self.adjusted_alpha = self.alpha * alpha_delta
        
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
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False,
                                       create_graph=False)[0]
            
            attack_mask = self.hpf_masker.update_saliency_mask(grad.clone().detach())
            
            adv_images = adv_images + self.adjusted_alpha*attack_mask*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images
    

class YcbcrHpfBIM(Attack):
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
    def __init__(self, model, diagonal=-4, log_sigma=3.0, dct_gauss_ksize=13, log_mu=0.6, eps=8/255, alpha=2/255, steps=10):
        super().__init__("YcbcrHpfBIM", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps*255 + 4, 1.25*eps*255))
        else:
            self.steps = steps
        self.supported_mode = ['default', 'targeted']
        
        # added attributes
        self.dct = DCT(img_size=224, patch_size=8, n_channels=3, diagonal=diagonal)
        self.log = LaplacianOfGaussian(log_sigma)
        self.gaussian = T.GaussianBlur(kernel_size=dct_gauss_ksize)
        self.to_grey = T.Grayscale()
        self.log_mu = log_mu
        self.dct_mu = 1 - self.log_mu
        self.to_ycbcr = YCBCRTransform()
    
    def normalize(self, imgs):
        imgs = imgs.to(torch.float32) - 128
        imgs = imgs / 128
        return imgs
    
    def normalize_2(self, imgs):
        imgs = imgs.to(torch.float32) / 255
        return imgs

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.BCEWithLogitsLoss()

        ori_images = images.clone().detach()
        ycbcr_images = self.to_ycbcr(images)        
        norm_images = self.normalize(ori_images)
        
        
        img_shape = ori_images.shape
        log_masks = torch.ones(img_shape)
        dct_coeff_masks = torch.ones(img_shape)
        
        for e, (img, norm_img) in enumerate(zip(ori_images, norm_images)):
            log_mask = self.log(img.cpu())
            if log_mask.min().item() < 0:
                log_mask = log_mask + (-1*log_mask.min().item())
            log_mask = log_mask / log_mask.max().item()
            log_masks[e] = log_mask
            dct_coeffs, fgsm_coeffs = self.dct(norm_img)
            dct_coeff_masks[e] = fgsm_coeffs
        
        log_masks = log_masks.to(self.device)
        dct_coeff_masks = dct_coeff_masks.to(self.device)
        
        images = self.normalize_2(images)
        ori_images = self.normalize_2(ori_images)
        hpf_mask = (log_masks*self.log_mu)+(dct_coeff_masks*self.dct_mu)

        for _ in range(self.steps):
            images.requires_grad = True
            outputs = self.get_logits(images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False,
                                       create_graph=False)[0]
            #print(f"{images.device},{log_mask.device},{dct_coeff_masks.device}")
            adv_images = images + self.alpha*grad.sign()
            
            # squeeze values into range 0 < (eps-x < x < eps+x) < 1
            a = torch.clamp(ori_images - self.eps, min=0)
            b = (adv_images >= a).float()*adv_images + (adv_images < a).float()*a
            c = (b > ori_images+self.eps).float()*(ori_images+self.eps) + (b <= ori_images + self.eps).float()*b
            images = torch.clamp(c, max=1).detach()

        ycbcr_adv_images = self.to_ycbcr(images)
        
        ycbcr_difference = ycbcr_images - ycbcr_adv_images
        ycbcr_difference[0][0] *= hpf_mask[0][0]
        ycbcr_images_adv = ycbcr_images + ycbcr_difference
        
        images = self.to_ycbcr.inverse(ycbcr_images_adv)
        images = torch.clamp(images, min=0, max=1)

        return images
    
class Observer:
    
    """This class measures certain things during transforms
    """
    
    def __init__(self, measure_freq, img_size, cutoff_diagonal):
        self.measure_freq = measure_freq
        if self.measure_freq:
            self.dct = DCT(img_size=img_size, patch_size=8, n_channels=3, diagonal=cutoff_diagonal)
    
    def __call__(self, img):
        if self.measure_freq:
            dct_coeffs, fgsm_coeffs = self.dct(img)
            