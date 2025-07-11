from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import Tensor as t

import numpy as np
import torch
import scipy.ndimage as nd
import torch_dct as dct
import torchvision.transforms as T

from src.adversarial.black_box.score_black_box_attack import ScoreBlackBoxAttack
from src.adversarial.black_box.compute import lp_step
from src.adversarial.black_box.utils import LaplacianOfGaussian, DCT


class NESAttack(ScoreBlackBoxAttack):
    """
    NES Attack
    """

    def __init__(self,
                fd_eta, 
                nes_lr, 
                q,
                model,
                device,
                model_trms,
                *args,
                **kwargs):
        """
        :param max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        :param epsilon: radius of lp-ball of perturbation
        :param p: specifies lp-norm  of perturbation
        :param fd_eta: forward difference step
        :param lr: learning rate of NES step
        :param q: number of noise samples per NES step
        :param lb: data lower bound
        :param ub: data upper bound
        """
        super().__init__(max_extra_queries=np.inf,
                         *args,
                         **kwargs)
        self.q = q
        self.fd_eta = fd_eta
        self.lr = nes_lr
        self.loss = torch.nn.CrossEntropyLoss()
        self.model = model
        self.device = device
        self.model_trms = model_trms
    
    def __call__(self, img, target):
        logs_dict = self.run(img, loss_fct=self.loss, )
        return logs_dict, target

    def _perturb(self, xs_t, loss_fct):
        _shape = list(xs_t.shape)
        dim = np.prod(_shape[1:])
        num_axes = len(_shape[1:])
        gs_t = torch.zeros_like(xs_t)
        for _ in range(self.q):
            # exp_noise = torch.randn_like(xs_t) / (dim ** 0.5)
            exp_noise = torch.randn_like(xs_t)
            fxs_t = xs_t + self.fd_eta * exp_noise
            bxs_t = xs_t - self.fd_eta * exp_noise
            est_deriv = (loss_fct(fxs_t) - loss_fct(bxs_t)) / (4. * self.fd_eta)
            gs_t += t(est_deriv.reshape(-1, *[1] * num_axes)) * exp_noise
        # perform the step
        new_xs = lp_step(xs_t, gs_t, self.lr, self.p)
        return new_xs, 2 * self.q * torch.ones(_shape[0])

    def _config(self):
        return {
            "name": self.name,
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "max_extra_queries": "inf" if np.isinf(self.max_extra_queries) else self.max_extra_queries,
            "max_loss_queries": "inf" if np.isinf(self.max_loss_queries) else self.max_loss_queries,
            "lr": self.lr,
            "q": self.q,
            "fd_eta": self.fd_eta,
            "attack_name": self.__class__.__name__
        }
    
'''

Original License
   
MIT License

Copyright (c) 2019 Abdullah Al-Dujaili

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

class HpfNESAttack(NESAttack):
    
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
        self.lf_boosting = lf_boosting
        self.mf_boosting = mf_boosting
        self.resize = T.Resize(input_size) # for image sizes that cannot be split into 8x8