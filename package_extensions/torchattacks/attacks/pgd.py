import torch
import torch.nn as nn
import torchvision.transforms as T
import scipy.ndimage as nd
import torch_dct as dct
import pytorch_colors as colors

from ..attack import Attack


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
        super().__init__("PGD", model)
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

        images = images / 255
        
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
        """target = torch.zeros(tensor.shape)        
        target[0] =  16 + (tensor[0]*65.738 + tensor[1]*129.057 + tensor[2]*25.064)/256 
        target[1] = 128 - (tensor[0]*39.945 + tensor[1]*175.494 + tensor[2]*112.439)/256
        target[2] = 128 + (tensor[0]*112.439 + tensor[1]*94.154 + tensor[2]*18.285)/256
        return target"""

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

    
class HpfPGD(PGD):
    
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
        
        attack_mask = self.hpf_masker(images, labels, self.model, self.model_trms, self.loss)
        
        alpha_delta = self.get_alpha_delta(attack_mask)
        self.adjusted_alpha = self.alpha * alpha_delta
        self.eps_tensor = self.get_eps_delta(attack_mask)
        max_adjusted_eps = self.eps_tensor.max()
        
        images = images / 255
        adv_images = images.clone().detach()
        
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-max_adjusted_eps, max_adjusted_eps)*attack_mask
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -self.loss(outputs, target_labels)
            else:
                cost = self.loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            
            attack_mask = self.hpf_masker.update_saliency_mask(grad.clone().detach())
            
            adv_images = adv_images.detach() + self.adjusted_alpha*attack_mask*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps_tensor, max=self.eps_tensor)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class YcbcrHpfPGD(PGD):
    
    def __init__(self, model, diagonal=-4, log_sigma=3.0, dct_gauss_ksize=13, log_mu=0.6, lf_boosting=0.0, mf_boosting=0.0, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.dct = DCT(img_size=224, patch_size=8, n_channels=3, diagonal=diagonal)
        self.log = LaplacianOfGaussian(log_sigma)
        self.gaussian = T.GaussianBlur(kernel_size=dct_gauss_ksize)
        self.to_grey = T.Grayscale()
        self.log_mu = log_mu
        self.dct_mu = 1 - self.log_mu
        self.lf_boosting = lf_boosting
        self.mf_boosting = mf_boosting
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
        
        ycbcr_images = self.to_ycbcr(images)
        norm_images = self.normalize(images)
        
        img_shape = images.shape
        log_masks = torch.ones(img_shape)
        dct_coeff_masks = torch.ones(img_shape)
        
        for e, (img, norm_img) in enumerate(zip(images, norm_images)):
            log_mask = self.log(img.cpu())
            if log_mask.min().item() < 0:
                log_mask = log_mask + (-1*log_mask.min().item())
            log_mask = log_mask / log_mask.max().item()
            log_masks[e] = log_mask
            dct_coeffs, fgsm_coeffs = self.dct(norm_img)
            dct_coeff_masks[e] = fgsm_coeffs
        
        log_masks = log_masks.to(self.device)
        dct_coeff_masks = dct_coeff_masks.to(self.device)
        
        hpf_mask = ((log_masks*self.log_mu)+(dct_coeff_masks*self.dct_mu))
        
        if self.lf_boosting > 0.0:
            inv_mask = 1 - hpf_mask
            hpf_mask = (self.lf_boosting*inv_mask) + hpf_mask
        
        if self.mf_boosting > 0.0:
            mf_mask = torch.clone(hpf_mask)
            mf_mask[mf_mask < 0.15] = 1 # remove low_freqs in inv_mask
            mf_mask[mf_mask > 0.75] = 1 # remove high frequencies
            inv_mask = 1 - mf_mask
            hpf_mask = (self.mf_boosting*inv_mask) + hpf_mask
        
        images = self.normalize_2(images)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            random_start = torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            random_start[0][0] *= hpf_mask[0][0]
            adv_images = adv_images + random_start
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        ycbcr_adv_images = self.to_ycbcr(adv_images)
        
        ycbcr_difference = ycbcr_images - ycbcr_adv_images
        ycbcr_difference[0][0] *= hpf_mask[0][0]
        ycbcr_images_adv = ycbcr_images + ycbcr_difference
        
        adv_images = self.to_ycbcr.inverse(ycbcr_images_adv)
        adv_images = torch.clamp(adv_images, min=0, max=1)

        return adv_images
    