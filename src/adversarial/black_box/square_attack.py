from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from torch import Tensor as t
import torch
import torchvision.transforms as T

from src.adversarial.black_box.score_black_box_attack import ScoreBlackBoxAttack
from src.adversarial.black_box.utils import LaplacianOfGaussian, DCT, normalize_for_dct


class SquareAttack(ScoreBlackBoxAttack):
    """
    Square Attack
    """

    def __init__(self, 
                p_init,
                model,
                device,
                model_trms,
                name='Square',
                *args,
                **kwargs):
        """
        :param max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        :param epsilon: radius of lp-ball of perturbation
        :param p: specifies lp-norm  of perturbation
        :param lb: data lower bound
        :param ub: data upper bound
        """
        super().__init__(max_extra_queries=np.inf,
                        name = "Square",
                        *args,
                        **kwargs)
                         
        self.best_loss = None
        self.i = 0
        self.p_init = p_init
        self.model = model
        self.device = device
        self.model_trms = model_trms
        self.loss = torch.nn.CrossEntropyLoss()
    
    def __call__(self, imgs, y):
        imgs = imgs.unsqueeze(0)
        y = torch.tensor([y])
        x_adv, logs_dict = self.run(imgs, y, loss_fct=self.get_loss, early_stop_extra_fct=self.early_stopping_crit)
        x_adv.squeeze(0)
        return x_adv
    
    def get_loss(self, img, y):
        x_hat = self.model(img)
        loss = self.loss(x_hat, y)
        return loss

    def early_stopping_crit(self, img, y):
        x_hat = self.model(img)
        pred = x_hat.argmax()
        if pred.item() == y.item():
            return torch.Tensor([0]).bool().to(self.device)
        else:
            return torch.Tensor([1]).byte().to(self.device)

    def p_selection(self, p_init, it, n_iters):
        """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
        it = int(it / n_iters * 10000)

        if 10 < it <= 50:
            p = p_init / 2
        elif 50 < it <= 200:
            p = p_init / 4
        elif 200 < it <= 500:
            p = p_init / 8
        elif 500 < it <= 1000:
            p = p_init / 16
        elif 1000 < it <= 2000:
            p = p_init / 32
        elif 2000 < it <= 4000:
            p = p_init / 64
        elif 4000 < it <= 6000:
            p = p_init / 128
        elif 6000 < it <= 8000:
            p = p_init / 256
        elif 8000 < it <= 10000:
            p = p_init / 512
        else:
            p = p_init

        return p

    def pseudo_gaussian_pert_rectangles(self, x, y):
        delta = torch.zeros([x, y])
        x_c, y_c = x // 2 + 1, y // 2 + 1

        counter2 = [x_c - 1, y_c - 1]
        for counter in range(0, max(x_c, y_c)):
            delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
                max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

            counter2[0] -= 1
            counter2[1] -= 1

        delta /= torch.sqrt(torch.sum(delta ** 2, dim=1, keepdim=True))
        return delta

    def meta_pseudo_gaussian_pert(self, s):
        delta = torch.zeros([s, s])
        n_subsquares = 2
        if n_subsquares == 2:
            delta[:s // 2] = self.pseudo_gaussian_pert_rectangles(s // 2, s)
            delta[s // 2:] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
            delta /= torch.sqrt(torch.sum(delta ** 2, dim=1, keepdim=True))
            if np.random.rand(1) > 0.5: delta = torch.transpose(delta, 0, 1)

        elif n_subsquares == 4:
            delta[:s // 2, :s // 2] = self.pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
            delta[s // 2:, :s // 2] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
            delta[:s // 2, s // 2:] = self.pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
            delta[s // 2:, s // 2:] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
            delta /= torch.sqrt(torch.sum(delta ** 2, dim=1, keepdim=True))

        return delta

    def get_square_and_compute_delta(self, xs, n_features, p, c, h, w, deltas):
        """computes squares and calculates perturbation delta. this should be overwritten
            by HPF variant
            
        Args:
            xs: input image
            n_features: number of features c x h x w
            p: p as defined by square attack paper
            c: channels dim
            h: height dim
            w: width dim
            deltas: init delta or delta from last iter

        Returns:
           deltas : new delta 
        """
        for i_img in range(xs.shape[0]):
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_window = self.x[i_img, :, center_h:center_h+s, center_w:center_w+s]
            x_best_window = xs[i_img, :, center_h:center_h+s, center_w:center_w+s]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while torch.sum(torch.abs(torch.clamp(x_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], self.lb, self.ub) - x_best_window) < 10**-7) == c*s*s:
                deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = t(np.random.choice([-self.epsilon, self.epsilon], size=[c, 1, 1]))
        return deltas

    def _perturb(self, xs_t, y, loss_fct):

        #xs = xs_t.permute(0,3,1,2)
        xs = xs_t
        c, h, w = xs.shape[1:]
        n_features = c*h*w
        n_queries = torch.zeros(xs.shape[0])

        if self.p == 'inf':
            if self.is_new_batch:
                self.x = xs.clone()
                init_delta = t(np.random.choice([-self.epsilon, self.epsilon], size=[xs.shape[0], c, 1, w]))
                xs = torch.clamp(xs + init_delta, self.lb, self.ub)
                self.best_loss = loss_fct(xs, y)
                n_queries += torch.ones(xs.shape[0])
                self.i = 0

            deltas = xs - self.x
            p = self.p_selection(self.p_init, self.i, 10000)
            deltas = self.get_square_and_compute_delta(xs, n_features, p, c, h, w, deltas)
            x_new = torch.clamp(self.x + deltas, self.lb, self.ub)

        elif self.p == '2':
            if self.is_new_batch:
                self.x = xs.clone()
                delta_init = torch.zeros(xs.shape)
                s = h // 5
                sp_init = (h - s * 5) // 2
                center_h = sp_init + 0
                for _ in range(h // s):
                    center_w = sp_init + 0
                    for _ in range(w // s):
                        delta_init[:, :, center_h:center_h + s, center_w:center_w + s] += self.meta_pseudo_gaussian_pert(s).reshape(
                            [1, 1, s, s]) * t(np.random.choice([-1, 1], size=[xs.shape[0], c, 1, 1]))
                        center_w += s
                    center_h += s
                xs = torch.clamp(xs + delta_init / torch.sqrt(torch.sum(delta_init ** 2, dim=(1, 2, 3), keepdim=True)) * (self.epsilon), self.lb, self.ub) 
                self.best_loss = loss_fct(xs.permute(0,2,3,1), y)
                n_queries += torch.ones(xs.shape[0])
                self.i = 0

            deltas = xs - self.x
            p = self.p_selection(self.p_init, self.i, 10000)
            s = max(int(round(np.sqrt(p * n_features / c))), 3)
            if s % 2 == 0:
                s += 1

            s2 = s + 0
            ### window_1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)
            new_deltas_mask = torch.zeros(xs.shape)
            new_deltas_mask[:, :, center_h:center_h + s, center_w:center_w + s] = 1.0

            ### window_2
            center_h_2 = np.random.randint(0, h - s2)
            center_w_2 = np.random.randint(0, w - s2)
            new_deltas_mask_2 = torch.zeros(xs.shape)
            new_deltas_mask_2[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 1.0

            ### compute total norm available
            curr_norms_window = torch.sqrt(
                torch.sum(((xs - self.x) * new_deltas_mask) ** 2, dim=(2, 3), keepdim=True))
            curr_norms_image = torch.sqrt(torch.sum((xs - self.x) ** 2, dim=(1, 2, 3), keepdim=True))
            mask_2 = torch.max(new_deltas_mask, new_deltas_mask_2)
            norms_windows = torch.sqrt(torch.sum((deltas * mask_2) ** 2, dim=(2, 3), keepdim=True))

            ### create the updates
            new_deltas = torch.ones([self.x.shape[0], c, s, s])
            new_deltas = new_deltas * self.meta_pseudo_gaussian_pert(s).reshape([1, 1, s, s])
            new_deltas *= t(np.random.choice([-1, 1], size=[self.x.shape[0], c, 1, 1]))
            old_deltas = deltas[:, :, center_h:center_h + s, center_w:center_w + s] / (1e-10 + curr_norms_window)
            new_deltas += old_deltas
            new_deltas = new_deltas / torch.sqrt(torch.sum(new_deltas ** 2, dim=(2, 3), keepdim=True)) * (
                torch.max((self.epsilon) ** 2 - curr_norms_image ** 2, torch.zeros_like(curr_norms_image)) / c + norms_windows ** 2) ** 0.5
            deltas[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 0.0  # set window_2 to 0
            deltas[:, :, center_h:center_h + s, center_w:center_w + s] = new_deltas + 0  # update window_1

            x_new = self.x + deltas / torch.sqrt(torch.sum(deltas ** 2, dim=(1, 2, 3), keepdim=True)) * (self.epsilon)
            x_new = torch.clamp(x_new, self.lb, self.ub).permute(0,2,3,1)


        new_loss = loss_fct(x_new, y)
        n_queries += torch.ones(xs.shape[0])
        idx_improved = new_loss > self.best_loss
        self.best_loss = idx_improved * new_loss + ~idx_improved * self.best_loss
        idx_improved = torch.reshape(idx_improved, [-1, *[1]*len(x_new.shape[:-1])])
        x_new = idx_improved * x_new + ~idx_improved * xs
        self.i += 1
        
        return x_new, n_queries

    def _config(self):
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "max_extra_queries": "inf" if np.isinf(self.max_extra_queries) else self.max_extra_queries,
            "max_loss_queries": "inf" if np.isinf(self.max_loss_queries) else self.max_loss_queries,
            "attack_name": self.__class__.__name__
        }

'''

This file is copied from the following source:
link: https://github.com/max-andr/square-attack/blob/master/attack.py

The original license is placed at the end of this file.

@article{ACFH2020square,
  title={Square Attack: a query-efficient black-box adversarial attack via random search},
  author={Andriushchenko, Maksym and Croce, Francesco and Flammarion, Nicolas and Hein, Matthias},
  conference={ECCV},
  year={2020}
}

basic structure for main:
    1. config args and prior setup
    2. define funtions that find the fraction of pixels changed on every iteration (p), define 
       pseudo gaussian perturbation and meta gaussian perturbation, and insert perturbation using square attack.
    3. return results
    
'''



"""
Implements Square attacks
"""
      
'''

Copyright (c) 2019, Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion, Matthias Hein
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''

class HPFSquareAttack(SquareAttack):
    
    def __init__(self, 
                diagonal=-4, 
                log_sigma=3.0, 
                dct_gauss_ksize=13, 
                log_mu=0.6, 
                lf_boosting=0.0, 
                mf_boosting=0.0, 
                input_size=224, 
                *args, 
                **kwargs):
        super().__init__(*args, **kwargs)
        self.dct = DCT(img_size=224, patch_size=8, n_channels=3, diagonal=diagonal)
        self.log = LaplacianOfGaussian(log_sigma)
        self.gaussian = T.GaussianBlur(kernel_size=dct_gauss_ksize)
        self.to_grey = T.Grayscale()
        self.log_mu = log_mu
        self.dct_mu = 1 - self.log_mu
        self.hpf_mask_tau = 0.7
        self.saliency_mask_tau = 1 - self.hpf_mask_tau
        self.lf_boosting = 0.5
        self.mf_boosting = mf_boosting
        self.resize = T.Resize(input_size) # for image sizes that cannot be split into 8x8
        self.hpf_mask = None # will hold the HPF coeffs. these scale the perturbation squares 

    def __call__(self, imgs, y):
        imgs = imgs.unsqueeze(0)
        self.hpf_mask = self.get_hpf_mask(images=imgs)
        y = torch.tensor([y])
        x_adv, logs_dict = self.run(imgs, y, loss_fct=self.get_loss, early_stop_extra_fct=self.early_stopping_crit)
        x_adv.squeeze(0)
        return x_adv

    def get_hpf_mask(self, images):
        images = images * 255
        norm_images = normalize_for_dct(images)
        
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
        hpf_mask = hpf_mask / hpf_mask.max()
        
        if self.lf_boosting > 0.0:
            inv_mask = 1 - hpf_mask
            hpf_mask = (self.lf_boosting*inv_mask) + hpf_mask
        
        if self.mf_boosting > 0.0:
            mf_mask = torch.clone(hpf_mask)
            mf_mask[mf_mask < 0.15] = 1 # remove low_freqs in inv_mask
            mf_mask[mf_mask > 0.75] = 1 # remove high frequencies
            inv_mask = 1 - mf_mask
            hpf_mask = (self.mf_boosting*inv_mask) + hpf_mask

        """alpha_delta = self.get_alpha_delta(hpf_mask)
        self.adjusted_alpha = self.alpha * alpha_delta
        self.eps_tensor = self.get_eps_delta(hpf_mask)
        max_adjusted_eps = self.eps_tensor.max()"""
        
        return hpf_mask

    def get_square_and_compute_delta(self, xs, n_features, p, c, h, w, deltas):
        """computes squares and calculates perturbation delta. this should be overwritten
            by HPF variant
            
        Args:
            xs: input image
            n_features: number of features c x h x w
            p: p as defined by square attack paper
            c: channels dim
            h: height dim
            w: width dim
            deltas: init delta or delta from last iter

        Returns:
           deltas : new delta 
        """
        for i_img in range(xs.shape[0]):
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_window = self.x[i_img, :, center_h:center_h+s, center_w:center_w+s]
            x_best_window = xs[i_img, :, center_h:center_h+s, center_w:center_w+s]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while torch.sum(torch.abs(torch.clamp(x_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], self.lb, self.ub) - x_best_window) < 10**-7) == c*s*s:
                deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = t(np.random.choice([-self.epsilon, self.epsilon], size=[c, 1, 1]))
        return deltas*self.hpf_mask
