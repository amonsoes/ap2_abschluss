
import numpy as np
import torch
import math
import random

from torchattacks.attacks.deepfool import DeepFool
from src.adversarial.pgd import PGD
from src.adversarial.attack_base import Attack

class SparseFool(Attack):
    r"""
    Attack in the paper 'SparseFool: a few pixels make a big difference'
    [https://arxiv.org/abs/1811.02248]

    Modified from "https://github.com/LTS4/SparseFool/"

    Distance Measure : L0

    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (Default: 10)
        lam (float): parameter for scaling DeepFool noise. (Default: 3)
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.SparseFool(model, steps=10, lam=3, overshoot=0.02)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, model_trms, steps=10, lam=3, overshoot=0.02, *args, **kwargs):
        super().__init__("SparseFool", model, model_trms, *args, **kwargs)
        self.steps = steps
        self.lam = lam
        self.overshoot = overshoot
        self.deepfool = DeepFool(model)
        self.supported_mode = ["default"]



    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        batch_size = len(images)
        correct = torch.tensor([True] * batch_size)
        curr_steps = 0

        adv_images = []
        for idx in range(batch_size):
            image = images[idx : idx + 1].clone().detach()
            adv_images.append(image)

        while (True in correct) and (curr_steps < self.steps):
            for idx in range(batch_size):
                image = images[idx : idx + 1]
                label = labels[idx : idx + 1]
                adv_image = adv_images[idx]

                fs = self.get_logits(self.model_trms(adv_image))[0]
                _, pre = torch.max(fs, dim=0)
                if pre != label:
                    correct[idx] = False
                    continue

                adv_image, target_label = self.deepfool.forward_return_target_labels(
                    adv_image, label
                )
                adv_image = image + self.lam * (adv_image - image)

                adv_image.requires_grad = True
                fs = self.get_logits(adv_image)[0]
                _, pre = torch.max(fs, dim=0)

                if pre == label:
                    pre = target_label

                cost = fs[pre] - fs[label]
                grad = torch.autograd.grad(
                    cost, adv_image, retain_graph=False, create_graph=False
                )[0]
                grad = grad / grad.norm()

                adv_image = self._linear_solver(image, grad, adv_image)
                adv_image = image + (1 + self.overshoot) * (adv_image - image)
                adv_images[idx] = torch.clamp(adv_image, min=0, max=1).detach()

            curr_steps += 1

        adv_images = torch.cat(adv_images).detach()

        return adv_images



    def _linear_solver(self, x_0, coord_vec, boundary_point):
        input_shape = x_0.size()

        plane_normal = coord_vec.clone().detach().view(-1)
        plane_point = boundary_point.clone().detach().view(-1)

        x_i = x_0.clone().detach()

        f_k = torch.dot(plane_normal, x_0.view(-1) - plane_point)
        sign_true = f_k.sign().item()

        beta = 0.001 * sign_true
        current_sign = sign_true

        while current_sign == sign_true and coord_vec.nonzero().size()[0] > 0:

            f_k = torch.dot(plane_normal, x_i.view(-1) - plane_point) + beta

            pert = f_k.abs() / coord_vec.abs().max()

            mask = torch.zeros_like(coord_vec)
            mask[
                np.unravel_index(torch.argmax(coord_vec.abs()).cpu(), input_shape)
            ] = 1.0  # nopep8

            r_i = torch.clamp(pert, min=1e-4) * mask * coord_vec.sign()

            x_i = x_i + r_i
            x_i = torch.clamp(x_i, min=0, max=1)

            f_k = torch.dot(plane_normal, x_i.view(-1) - plane_point)
            current_sign = f_k.sign().item()

            coord_vec[r_i != 0] = 0

        return x_i



class SparseSigmaAttack(PGD):
    """
    https://arxiv.org/pdf/1909.05040

    x_test in the format bs (batch size) x heigth x width x channels
    y_test in the format bs

    This is the L_0 + L_inf attack from the paper, with the following mods:
    - instead of the L2-bound perturbation of alpha, we make the step-perturbation L_inf-bound
    - instead of perturbing at most k pixels per channel, we ALWAYS perturb k pixel per channel
    - we compute the epsilon bound by L2/math.sqrt(k*3), diving the L2 distance on the L_inf bound k pixek perturbations
    """

    def __init__(self, sparse_attack_kappa=-1, l2_bound=-1, k=400, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kappa = sparse_attack_kappa
        #in args eps is originally cast to float
        self.l2_bound = l2_bound
        self.epsilon = 0.8
        self.alpha = 120000.0/255.0/2.0
        self.k = k
        self.random_start=False
        if l2_bound != -1:
            self.epsilon = self.get_eps_in_range(l2_bound, k)
            self.alpha = (self.epsilon / self.steps)
            self.l2_bound = l2_bound


    def get_eps_in_range(self, l2_bound, k, method='high'):
        low, high = self.get_iq_range_from_l2_bound(l2_bound)
        if method == 'random':
            l2_bound = random.uniform(low, high)
        elif method == 'high':
            l2_bound = high
        return math.sqrt((l2_bound**2)/k)



    """def forward_2(self, x_nat, y_nat):
        ''' PGD attack wrt L0-norm + sigma-map constraints
        
            it returns adversarial examples (if found) adv for the images x_nat, with correct labels y_nat,
            such that:
                - each image of the batch adv differs from the corresponding one of
                x_nat in at most k pixels
                - (1 - kappa*sigma)*x_nat <= adv <= (1 + kappa*sigma)*x_nat
            
            it returns also a vector of flags where 1 means no adversarial example found
            (in this case the original image is returned in adv) '''
        
        self.sigma = self.sigma_map(x_nat)
        self.sigma = self.sigma.numpy()
        x_nat = x_nat.permute(0, 2, 3, 1).numpy()
        y_nat = y_nat.numpy()
        if self.random_start:
            x2 = x_nat + np.random.uniform(-self.kappa, self.kappa, x_nat.shape)
            x2 = np.clip(x2, 0, 1)
        else:
            x2 = np.copy(x_nat)
            
        adv_not_found = np.ones(y_nat.shape)
        adv = np.zeros(x_nat.shape)
        
        
        for i in range(self.steps):
            if i > 0:
                #pred, grad = sess.run([attack.model.correct_prediction, attack.model.grad], feed_dict={attack.model.x_input: x2, attack.model.y_input: y_nat})
                pred, grad = self.get_pred_and_grad(torch.tensor(x2), torch.tensor(y_nat))
                pred, grad = pred.numpy(), grad.numpy()
                adv_not_found = np.minimum(adv_not_found, pred.astype(int))
                adv[np.logical_not(pred)] = np.copy(x2[np.logical_not(pred)])
                
                grad /= (1e-10 + np.sum(np.abs(grad), axis=(1,2,3), keepdims=True))
                x2 = np.add(x2, (np.random.random_sample(grad.shape)-0.5)*1e-12 + self.alpha * grad, casting='unsafe')
            
            x2 = self.project_L0_sigma(x2, self.eps, self.sigma, self.kappa, x_nat)
            
        return torch.from_numpy(adv).permute(0, 3, 1, 2)"""

    def get_pred_and_grad(self, images, labels):
        output, grad =  super().get_pred_and_grad(images, labels)
        pred = (output.detach().cpu().max(dim=-1)[1] == labels)
        return pred, grad

    def sigma_map(self, x):
        x = x.detach().numpy()
        sh = [4]
        sh.extend(x.shape)
        t = np.zeros(sh)
        t[0,:,:-1] = x[:,1:]
        t[0,:,-1] = x[:,-1]
        t[1,:,1:] = x[:,:-1]
        t[1,:,0] = x[:,0]
        t[2,:,:,:-1] = x[:,:,1:]
        t[2,:,:,-1] = x[:,:,-1]
        t[3,:,:,1:] = x[:,:,:-1]
        t[3,:,:,0] = x[:,:,0]

        mean1 = (t[0] + x + t[1])/3
        sd1 = np.sqrt(((t[0]-mean1)**2 + (x-mean1)**2 + (t[1]-mean1)**2)/3)

        mean2 = (t[2] + x + t[3])/3
        sd2 = np.sqrt(((t[2]-mean2)**2 + (x-mean2)**2 + (t[3]-mean2)**2)/3)

        sd = np.minimum(sd1, sd2)
        sd = np.sqrt(sd)
        
        return torch.tensor(sd)

    def project_L0_sigma(self, y, eps, sigma, kappa, x_nat):
        ''' projection of the batch y to a batch x such that:
                - 0 <= x <= 1
                - each image of the batch x differs from the corresponding one of
                x_nat in at most eps pixels
                - (1 - kappa*sigma)*x_nat <= x <= (1 + kappa*sigma)*x_nat '''
        
        x = np.copy(y)
        p1 = 1.0/np.maximum(1e-12, sigma)*(x_nat > 0).astype(float) + 1e12*(x_nat == 0).astype(float)
        p2 = 1.0/np.maximum(1e-12, sigma)*(1.0/np.maximum(1e-12, x_nat) - 1)*(x_nat > 0).astype(float) + 1e12*(x_nat == 0).astype(float) + 1e12*(sigma == 0).astype(float)
        lmbd_l = np.maximum(-kappa, np.amax(-p1, axis=-1, keepdims=True))
        lmbd_u = np.minimum(kappa, np.amin(p2, axis=-1, keepdims=True))
        
        lmbd_unconstr = np.sum((y - x_nat)*sigma*x_nat, axis=-1, keepdims=True)/np.maximum(1e-12, np.sum((sigma*x_nat)**2, axis=-1, keepdims=True))
        lmbd = np.maximum(lmbd_l, np.minimum(lmbd_unconstr, lmbd_u))
        
        p12 = np.sum((y - x_nat)**2, axis=-1, keepdims=True)
        p22 = np.sum((y - (1 + lmbd*sigma)*x_nat)**2, axis=-1, keepdims=True)
        p3 = np.sort(np.reshape(p12 - p22, [x.shape[0],-1]))[:,-eps]
        
        x  = x_nat + lmbd*sigma*x_nat*((p12 - p22) >= p3.reshape([-1, 1, 1, 1]))
        
        return x
    

    """def forward(self, x_nat, y_nat):
        ''' PGD attack wrt L0-norm + box constraints
        
            it returns adversarial examples (if found) adv for the images x_nat, with correct labels y_nat,
            such that:
                - each image of the batch adv differs from the corresponding one of
                x_nat in at most k pixels
                - lb <= adv - x_nat <= ub
            
            it returns also a vector of flags where 1 means no adversarial example found
            (in this case the original image is returned in adv) 
        '''
        x_nat = x_nat.permute(0, 2, 3, 1).numpy()
        y_nat = y_nat.numpy()
        lb = np.maximum(-self.epsilon, -x_nat)
        ub = np.minimum(self.epsilon, 1.0 - x_nat)
        if self.random_start:
            x2 = x_nat + np.random.uniform(lb, ub, x_nat.shape)
            x2 = np.clip(x2, 0, 1)
        else:
            x2 = np.copy(x_nat)
            
        adv_not_found = np.ones(y_nat.shape)
        adv = np.zeros(x_nat.shape)

        for i in range(self.steps):
            if i > 0:
                #pred, grad = sess.run([attack.model.correct_prediction, attack.model.grad], feed_dict={attack.model.x_input: x2, attack.model.y_input: y_nat})
                pred, grad = self.get_pred_and_grad(x2, y_nat)
                adv_not_found = np.minimum(adv_not_found, pred.astype(int))
                adv[np.logical_not(pred)] = np.copy(x2[np.logical_not(pred)])
                
                #grad /= (1e-10 + np.sum(np.abs(grad), axis=(1,2,3), keepdims=True))
                #x2 = np.add(x2, (np.random.random_sample(grad.shape)-0.5)*1e-12 + self.alpha * grad, casting='unsafe')

                x2 = np.add(x2, self.alpha * np.sign(grad), casting='unsafe')
            
            x2 = x_nat + self.project_L0_box(x2 - x_nat, lb, ub)
            x2 = np.clip(x2, 0, 1)
            
        return torch.from_numpy(x2).permute(0, 3, 1, 2)"""


    def forward(self, x_nat, y_nat):
        ''' PGD attack wrt L0-norm + box constraints
        
            it returns adversarial examples (if found) adv for the images x_nat, with correct labels y_nat,
            such that:
                - each image of the batch adv differs from the corresponding one of
                x_nat in at most k pixels
                - lb <= adv - x_nat <= ub
            
            it returns also a vector of flags where 1 means no adversarial example found
            (in this case the original image is returned in adv) 
        '''

        batch_size, c, h, w = x_nat.shape
        x2 = x_nat.clone().detach()
            
        adv_not_found = torch.ones_like(y_nat)
        adv = torch.zeros_like(x_nat)

        pred, grad = self.get_pred_and_grad(x2, y_nat)

        importance = grad.abs()

        # Flatten spatial dimensions for sorting
        importance_flat = importance.view(batch_size,-1)  # Shape: (batch, ch* width * height)
        values, topk_indices = torch.topk(importance_flat, k=self.k, dim=-1)
        
        # Create a zero mask of the same shape
        mask = torch.zeros_like(importance_flat, dtype=torch.bool)

        # Scatter 1s into the mask at the top-k locations
        mask.scatter_(dim=-1, index=topk_indices, value=True)

        # Reshape back to original (batch, width, height)
        important_pixels = mask.view(batch_size,c, w, h)

        for i in range(self.steps):

            pred, grad = self.get_pred_and_grad(x2, y_nat)

            #project to L0+L_inf box
            step = self.alpha*grad.sign()

            step[~important_pixels] = 0

            x2 = x2 + step
            delta = x_nat - x2
            delta = torch.clamp(delta, min=-self.epsilon, max=self.epsilon)
            x2 = torch.clamp(x_nat + delta, min=0.0, max=1.0)
            
        return x2

    def project_L0_box(self, delta, lb, ub):
        ''' projection of the batch delta to a batch x such that:
                - each image of the batch x has at most k pixels with non-zero channels
                - lb <= x <= ub '''

        x = np.copy(delta)
        
        # Ensure initial box constraints
        x = np.clip(x, lb, ub)

        # Compute pixel-wise importance: sum over color channels
        importance = np.sum(np.abs(x), axis=3)  # Shape: (batch, width, height)

        # Flatten spatial dimensions for sorting
        importance_flat = importance.reshape(importance.shape[0], -1)  # Shape: (batch, width * height)

        # Get the indices of the k most important perturbed pixels
        top_k_indices = np.argsort(-importance_flat, axis=-1)[:, :self.k]  # Sort descending

        # Create mask: Only allow perturbations at the selected top-k pixel locations
        mask = np.zeros_like(importance_flat, dtype=bool)
        batch_indices = np.arange(importance_flat.shape[0])[:, None]  # Expand for batch indexing
        mask[batch_indices, top_k_indices] = True  # Select top-k indices

        # Reshape mask back to image dimensions and expand for color channels
        mask = mask.reshape(importance.shape + (1,))  # Shape: (batch, width, height, 1)

        # Set perturbation values at selected locations
        x = np.where(mask, np.sign(delta) * self.epsilon, 0)  # Keep sign, set to max allowed perturbation

        # Enforce box constraints again
        x = np.clip(x, lb, ub)
            
        #x = np.copy(delta)
        #p1 = np.sum(x**2, axis=-1)
        #p2 = np.minimum(np.minimum(ub - x, x - lb), 0)
        #p2 = np.sum(p2**2, axis=-1)
        #p3 = np.sort(np.reshape(p1-p2, [p2.shape[0],-1]))[:,-self.k]
        #x = x*(np.logical_and(lb <= x, x <= ub)) + lb*(lb > x) + ub*(x > ub)
        #x *= np.expand_dims((p1 - p2) >= p3.reshape([-1, 1, 1]), -1)
        
        #x = np.clip(x, a_min=-self.epsilon, a_max=self.epsilon)
        return x

    def project_L0_box(self, grad):
        ''' projection of the batch delta to a batch x such that:
                - each image of the batch x has at most k pixels with non-zero channels
                - lb <= x <= ub '''
        batch_size, c, h, w = grad.shape
        delta = self.alpha*grad.sign()

        
        # Ensure initial box constraints
        delta = torch.clamp(delta, max=self.epsilon, min=-self.epsilon)

        # Compute pixel-wise importance: sum over color channels
        importance = grad.abs().sum(dim=1)  # batch x ch x h x w -> batch x 1 x h x w 

        # Flatten spatial dimensions for sorting
        importance_flat = importance.reshape(importance.shape[0], -1)  # Shape: (batch, width * height)
        values, topk_indices = torch.topk(importance.view(batch_size,-1), k=self.k, dim=-1)
        

        # Create a zero mask of the same shape
        mask = torch.zeros_like(importance_flat, dtype=torch.bool)

        # Scatter 1s into the mask at the top-k locations
        mask.scatter_(dim=-1, index=topk_indices, value=True)

        # Reshape back to original (batch, width, height)
        mask = mask.view(batch_size, w, h)
        mask = mask.unsqueeze(1).expand(-1, 3, -1, -1)
        delta[~mask] = 0
        return delta


    def project_L0_box(self, grad):
        ''' projection of the batch delta to a batch x such that:
                - each image of the batch x has at most k pixels with non-zero channels
                - lb <= x <= ub '''

    def get_pred_and_grad(self, x_nat, y_nat):
        x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
        x.requires_grad_()
        y = torch.from_numpy(y_nat)

        with torch.enable_grad():
            output = self.model(self.model_trms(x.to(self.device)))
            loss = self.loss(output, y.to(self.device))

        grad = torch.autograd.grad(loss, x)[0]
        grad = grad.detach().permute(0, 2, 3, 1).numpy()

        pred = (output.detach().cpu().max(dim=-1)[1] == y).numpy()
        return pred, grad

    def get_pred_and_grad(self, x_nat, y_nat):
        x_nat.requires_grad_()

        with torch.enable_grad():
            output = self.model(self.model_trms(x_nat.to(self.device)))
            loss = self.loss(output, y_nat.to(self.device))

        grad = torch.autograd.grad(loss, x_nat)[0]

        pred = (output.detach().cpu().max(dim=-1)[1] == y_nat).numpy()
        return pred, grad
