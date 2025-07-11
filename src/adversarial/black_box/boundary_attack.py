from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import scipy.ndimage as nd
import torch_dct as dct
import torchvision.transforms as T

from torch import Tensor as t
from src.adversarial.black_box.decision_black_box_attack import DecisionBlackBoxAttack
from src.adversarial.black_box.utils import LaplacianOfGaussian, DCT

class BoundaryAttack(DecisionBlackBoxAttack):
    """
    Boundary Attack
    """
    def __init__(self,
                steps, 
                spherical_step, 
                source_step, 
                source_step_convergence, 
                step_adaptation, 
                update_stats_every_k,
                targeted=False,
                *args,
                **kwargs):
        """
        :param spherical step: step taken on sphere around the source img
        :param source step: step taken towards the source img
        :param source_step_convergence: threshold float after which the attack converged
        :param step_adaptation: change of step size of spherical and source step
        :param update_stats_every_k: candidates will be checked every k steps
        """
        super().__init__(*args, **kwargs)
        self.steps = steps
        self.spherical_step = spherical_step
        self.source_step = source_step
        self.source_step_convergence = source_step_convergence
        self.step_adaptation = step_adaptation
        self.update_stats_every_k = update_stats_every_k
        self.query = 0
        self.targeted = targeted

    def __call__(self, target_sample, y):
        """
        performs boundary attack
        :param target_sample: original image
        :param y: target label 
        """
        
        self.query = 0

        # project to bxcxhxw
        target_sample = target_sample.unsqueeze(0)
        # randomly initialize and reduce l-norm in the direction of source
        best_advs = self.initialize(target_sample, y)
        init_adv = best_advs
        # broadcast number of spherical and source steps to all imgs in batch
        shape = list(target_sample.shape)
        N = shape[0]
        ndim = target_sample.ndim
        spherical_steps = torch.ones(N) * self.spherical_step
        source_steps = torch.ones(N) * self.source_step

        stats_spherical_adversarial = ArrayQueue(maxlen=100, N=N)
        stats_step_adversarial = ArrayQueue(maxlen=30, N=N)

        # loop to start algorithm
        for step in range(1, self.steps + 1):
            # check convergence criterion
            converged = source_steps < self.source_step_convergence
            if converged.all():
                break  # pragma: no cover
            converged = self.atleast_kd(converged, ndim)

            #get directions towards source
            unnormalized_source_directions = target_sample - best_advs
            source_norms = torch.norm(self.flatten(unnormalized_source_directions), dim = -1, p = 2)
            source_directions = unnormalized_source_directions / self.atleast_kd(
                source_norms, ndim
            )

            # only check spherical candidates every k steps
            check_spherical_and_update_stats = step % self.update_stats_every_k == 0

            candidates, spherical_candidates = self.draw_proposals(
                target_sample,
                best_advs,
                unnormalized_source_directions,
                source_directions,
                source_norms,
                spherical_steps,
                source_steps,
            )


            is_adv = self.is_adversarial(candidates, y)
            self.query += N

            if check_spherical_and_update_stats:
                spherical_is_adv = self.is_adversarial(spherical_candidates, y)
                self.query += N
                stats_spherical_adversarial.append(spherical_is_adv)
                stats_step_adversarial.append(is_adv)
            else:
                spherical_is_adv = None

            # in theory, we are closer per construction
            # but limited numerical precision might break this
            distances = torch.norm(self.flatten(target_sample - candidates), dim=-1, p=2)
            closer = distances < source_norms
            is_best_adv = is_adv & closer
            is_best_adv = self.atleast_kd(is_best_adv, ndim)

            cond = (~converged)&(is_best_adv)
            best_advs = torch.where(cond, candidates, best_advs)

            if self.query > self.max_queries:
                break

            diff = self.distance(best_advs, target_sample)
            if diff <= self.epsilon:
                print("{} steps".format(self.query))
                print("Mean Squared Error: {}".format(diff))
                break
            if is_best_adv:
                print("Mean Squared Error: {}".format(diff))
                print("Calls: {}".format(self.query))


            if check_spherical_and_update_stats:
                print('\nadjusting spatial and sperical step size...\n')
                full = stats_spherical_adversarial.isfull().to(self.device)
                if full.any():
                    probs = stats_spherical_adversarial.mean().to(self.device)
                    cond1 = (probs > 0.5) & full
                    spherical_steps = torch.where(
                        cond1, spherical_steps * self.step_adaptation, spherical_steps
                    )
                    source_steps = torch.where(
                        cond1, source_steps * self.step_adaptation, source_steps
                    )
                    cond2 = (probs < 0.2) & full
                    spherical_steps = torch.where(
                        cond2, spherical_steps / self.step_adaptation, spherical_steps
                    )
                    source_steps = torch.where(
                        cond2, source_steps / self.step_adaptation, source_steps
                    )
                    stats_spherical_adversarial.clear(cond1 | cond2)


                full = stats_step_adversarial.isfull().to(self.device)
                if full.any():
                    probs = stats_step_adversarial.mean().to(self.device)
                    cond1 = (probs > 0.25) & full
                    source_steps = torch.where(
                        cond1, source_steps * self.step_adaptation, source_steps
                    )
                    cond2 = (probs < 0.1) & full
                    source_steps = torch.where(
                        cond2, source_steps / self.step_adaptation, source_steps
                    )
                    stats_step_adversarial.clear(cond1 | cond2)
                print(f'new source step:{source_steps}')
                print(f'new spherical step:{spherical_steps}')
        if torch.all(best_advs == init_adv):
            print('WARNING: attack found no better adv than rand adv.')
        return best_advs.squeeze(0), self.query, diff
    
    def draw_noise_for_init(self, input_xi):
        random_noise = t(np.random.uniform(self.lb, self.ub, size = input_xi.shape))
        return random_noise

    def draw_noise(self, D):
        random_noise = torch.normal(mean=torch.ones(D, 1))
        return random_noise

    def _config(self):
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "attack_name": self.__class__.__name__
        }

    def atleast_kd(self, x, k):
        shape = x.shape + (1,) * (k - x.ndim)
        return x.reshape(shape)
    
    def flatten(self, x, keep = 1):
        return x.flatten(start_dim=keep)

    def draw_proposals(
        self,
        originals, 
        perturbed,
        unnormalized_source_directions,
        source_directions,
        source_norms,
        spherical_steps,
        source_steps,
    ):
        """_summary_

        Args:
            originals: original images x
            perturbed: perturbed images x_hat
            unnormalized_source_directions: unnorm. x - x_hat that show in the direction of the original image 
            source_directions: norm. x - x_hat that show in the direction of the original image
            source_norms: 
            spherical_steps: nr of steps along the sphere in the form (batch_s,). entries are the nr of steps 
            source_steps: _nr of steps into the direction fo the original image in the form (batch_s,). entries are the nr of steps

        Returns:
            _type_: _description_
        """
        # remember the actual shape
        shape = originals.shape
        assert perturbed.shape == shape
        assert unnormalized_source_directions.shape == shape
        assert source_directions.shape == shape

        # flatten everything to (batch, size)
        originals = self.flatten(originals)
        perturbed = self.flatten(perturbed)
        unnormalized_source_directions = self.flatten(unnormalized_source_directions)
        source_directions = self.flatten(source_directions)
        N, D = originals.shape

        assert source_norms.shape == (N,)
        assert spherical_steps.shape == (N,)
        assert source_steps.shape == (N,)

        # draw from an iid Gaussian (we can share this across the whole batch)
        eta = self.draw_noise(D)

        # make orthogonal (source_directions are normalized)
        eta = eta.T - torch.matmul(source_directions, eta) * source_directions
        assert eta.shape == (N, D)

        # rescale
        norms = torch.norm(eta, dim=-1, p=2)
        assert norms.shape == (N,)
        eta = eta * self.atleast_kd(spherical_steps * source_norms / norms, eta.ndim)

        # project on the sphere using Pythagoras
        distances = self.atleast_kd((spherical_steps ** 2 + 1).sqrt(), eta.ndim)
        directions = eta - unnormalized_source_directions
        spherical_candidates = originals + directions / distances

        # clip
        min_, max_ = self.lb, self.ub
        spherical_candidates = spherical_candidates.clamp(min_, max_)

        # step towards the original inputs
        new_source_directions = originals - spherical_candidates
        assert new_source_directions.ndim == 2
        new_source_directions_norms = torch.norm(self.flatten(new_source_directions), dim=-1, p=2)

        # length if spherical_candidates would be exactly on the sphere
        lengths = source_steps * source_norms
        
        # length including correction for numerical deviation from sphere
        lengths = lengths + new_source_directions_norms - source_norms

        # make sure the step size is positive
        lengths = torch.max(lengths, torch.zeros_like(lengths))

        # normalize the length
        lengths = lengths / new_source_directions_norms
        lengths = self.atleast_kd(lengths, new_source_directions.ndim)

        candidates = spherical_candidates + lengths * new_source_directions

        # clip
        candidates = candidates.clamp(min_, max_)

        # restore shape
        candidates = candidates.reshape(shape)
        spherical_candidates = spherical_candidates.reshape(shape)
        return candidates, spherical_candidates
    
    def initialize(self, input_xi, label_or_target):
        """ 
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        success = 0
        num_evals = 0
        # Find a misclassified random noise.
        while True:
                random_noise = self.draw_noise_for_init(input_xi)
                success = self.is_adversarial(random_noise, label_or_target)[0]
                if success:
                        break
                if self.query > self.max_queries:
                        break
                assert num_evals < 1e4,"Initialization failed! "
                "Use a misclassified image as `target_image`" 

        # Binary search to minimize l2 distance to original image.
        low = 0.0
        high = 1.0
        while high - low > 0.001:
                mid = (high + low) / 2.0
                blended = (1 - mid) * input_xi + mid * random_noise 
                success = self.is_adversarial(blended, label_or_target)
                if success:
                        high = mid
                else:
                        low = mid
                if self.query > self.max_queries:
                        break

        initialization = (1 - high) * input_xi + high * random_noise 

        return initialization

    def _perturb(self, xs_t, ys_t):
        adv, q = self.boundary_attack(xs_t, ys_t)
        return adv, q

class ArrayQueue:
    def __init__(self, maxlen: int, N: int):
        # we use NaN as an indicator for missing data
        self.data = np.full((maxlen, N), np.nan)
        self.next = 0
        # used to infer the correct framework because this class uses NumPy
        self.tensor = None

    @property
    def maxlen(self) -> int:
        return int(self.data.shape[0])

    @property
    def N(self) -> int:
        return int(self.data.shape[1])

    def append(self, x) -> None:
        if self.tensor is None:
            self.tensor = x
        x = x.cpu().numpy()
        assert x.shape == (self.N,)
        self.data[self.next] = x
        self.next = (self.next + 1) % self.maxlen

    def clear(self, dims) -> None:
        if self.tensor is None:
            self.tensor = dims  # pragma: no cover
        dims = dims.cpu().numpy()
        assert dims.shape == (self.N,)
        assert dims.dtype == np.bool
        self.data[:, dims] = np.nan

    def mean(self):
        assert self.tensor is not None
        result = np.nanmean(self.data, axis=0)
        return torch.from_numpy(result)

    def isfull(self):
        assert self.tensor is not None
        result = ~np.isnan(self.data).any(axis=0)
        return torch.from_numpy(result)
'''

Original License

MIT License

Copyright (c) 2020 Jonas Rauber et al.

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


class HPFBoundaryAttack(BoundaryAttack):
    
    # overwritten FN for HPF adjustment: draw_noise_for_init, draw_noise
    
    def __init__(self, hpf_masker, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hpf_masker = hpf_masker

    def draw_noise_for_init(self, input_xi):
        self.hpf_mask = self.hpf_masker(input_xi)
        self.hpf_mask = self.hpf_mask.to('cpu')
        self.boolean_mask = (self.hpf_mask < 0.6)
        random_noise = t(np.random.uniform(self.lb, self.ub, size = input_xi.shape))*self.hpf_mask
        #random_noise[self.boolean_mask] = 0
        return random_noise

    def draw_noise(self, D):
        #TODO: normal needs to be drawn from original shape and then reshaped as passed back
        random_noise = torch.normal(mean=torch.ones_like(self.hpf_mask))
        random_noise = random_noise * self.hpf_mask
        random_noise = random_noise.reshape(D, 1)
        return random_noise

    def initialize(self, input_xi, label_or_target):
        """ 
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        success = 0
        num_evals = 0
        # Find a misclassified random noise.
        while True:
                random_noise = self.draw_noise_for_init(input_xi)
                success = self.is_adversarial(random_noise, label_or_target)[0]
                if success:
                        break
                if self.query > self.max_queries:
                        break
                assert num_evals < 1e4,"Initialization failed! "
                "Use a misclassified image as `target_image`" 

        # Binary search to minimize l2 distance to original image.
        low = 0.0
        high = 1.0
        while high - low > 0.001:
                mid = (high + low) / 2.0
                blended = (1 - mid) * input_xi + mid * random_noise 
                success = self.is_adversarial(blended, label_or_target)
                if success:
                        high = mid
                else:
                        low = mid
                if self.query > self.max_queries:
                        break

        initialization = (1 - high) * input_xi + high * random_noise 

        return initialization


    def initialize(self, input_xi, label_or_target):
        """ 
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        success = 0
        num_evals = 0
        # Find a misclassified random noise.
        while True:
                random_noise = self.draw_noise_for_init(input_xi)
                success = self.is_adversarial(random_noise, label_or_target)[0]
                if success:
                        break
                if self.query > self.max_queries:
                        break
                assert num_evals < 1e4,"Initialization failed! "
                "Use a misclassified image as `target_image`" 

        # Binary search to minimize l2 distance to original image.
        #random_noise = torch.where(random_noise != 0, random_noise, input_xi)
        low = 0.0
        high = 1.0
        while high - low > 0.001:
                mid = (high + low) / 2.0
                blended = (1 - (mid*(1-self.hpf_mask))) * input_xi + (mid*self.hpf_mask) * random_noise
                
                #blended = (1 - mid) * input_xi + mid * (blended * self.hpf_mask)
                success = self.is_adversarial(blended, label_or_target)
                if success:
                        high = mid
                else:
                        low = mid
                if self.query > self.max_queries:
                        break

        initialization = (1 - (high*(1-self.hpf_mask))) * input_xi + (high*self.hpf_mask) * random_noise 

        return initialization
    