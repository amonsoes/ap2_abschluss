import sys
import os
import numpy as np
import torch
import torchvision.transforms as T

from torch import nn
from torch.nn import functional as F



class PriorRGFAttack(object):


    def __init__(self, 
                model,
                surrogate_model,
                model_trms,
                surrogate_model_trms,
                max_queries,
                samples_per_draw,
                method,
                p,
                dataprior,
                input_size,
                eps,
                sigma,
                learning_rate,
                num_classes,
                device):
        """PriorRGF as in https://arxiv.org/pdf/1906.06919.pdf
        Approximates true grad by means of a prior (surrogate model gradient)

        Args:
            model (_type_): model to attack. In paper InceptionV3.
            surrogate_model (_type_): model that yields the prior. In paper Resnet152.
            max_queries (_type_): number of iterations of grad approximation and attack.
            samples_per_draw (_type_): Number of samples (rand vecs) to estimate the gradient.
            method (_type_): Methods used in the attack. uniform: RGF, biased: P-RGF (\lambda^*), fixed_biased: P-RGF (\lambda=0.5)
            p (_type_): norm used in the attack. One of "l2", "linf".
            dataprior (_type_): Whether to use data prior in the attack.
            input_size (_type_): input size p -> p x p.
            eps (_type_): eps ball around data point.
            sigma (_type_): sampling variance for random vecs.
            learning_rate (_type_): adjustment rate for attack as in alpha.
            num_classes (_type_): number of classes in dataset.
            device (_type_): device used for computation.
        """
        self.device = device
        self.learning_rate = learning_rate
        self.eps = eps
        self.sigma = sigma
        self.max_queries = max_queries
        self.samples_per_draw = samples_per_draw
        self.method = method
        self.norm = p
        self.dataprior = dataprior
        self.image_height, self.image_width = input_size, input_size
        self.resize = T.Resize(input_size)
        self.in_channels = 3
        self.num_classes = num_classes
        self.model = model
        self.surrogate_model = surrogate_model
        self.model_trms = model_trms
        self.surrogate_model_trms = surrogate_model_trms
        self.model.to(self.device).eval()
        self.surrogate_model.to(self.device).eval()
        self.targeted = None # only support untargeted attack now
        self.target_type = 'random'
        self.clip_min = 0.0
        self.clip_max = 1.0

    def __call__(self, images, true_labels):
        """Algorithm 1 in paper

        Args:
            img (_type_): input image in [0, 1]

        """
        target_labels = None
        images = images.to(self.device)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        images = images.unsqueeze(0)
        true_labels = torch.Tensor([true_labels]).to(torch.long).to(self.device)
        adv_images = images.clone().detach()
        adv_images.requires_grad = True
        assert images.size(0) == 1
        logits_real_images = self.model(self.model_trms(images.to(self.device)))
        l = self.xent_loss(logits_real_images, true_labels, target_labels)
        lr = float(self.learning_rate)
        total_q = 0
        ite = 0
        while total_q <= self.max_queries: # only to max nr of queries
            total_q += 1
            sigma = self.set_up_sigma(self.sigma, ite, adv_images, true_labels, target_labels, l)            
            prior, alpha = self.get_prior_and_alpha(images, adv_images, true_labels, sigma, ite)
            q = self.samples_per_draw
            lmda, return_prior = self.get_lambda(alpha, q)
            grad = self.get_grad_for_attack(adv_images, true_labels, target_labels, prior, lmda, q, return_prior, sigma, l)
            adv_images = self.attack(images, adv_images, grad, lr)
            adv_labels = self.get_pred(self.model, self.model_trms(adv_images))
            logits_ = self.model(self.model_trms(adv_images))
            l = self.xent_loss(logits_, true_labels.to(self.device), target_labels)
            print('queries:', total_q, 'loss:', l, 'learning rate:', lr, 'sigma:', sigma, 'prediction:', adv_labels,
                    'distortion:', torch.max(torch.abs(adv_images - images)).item(), torch.norm((adv_images - images)).item())
            ite += 1
        return adv_images, total_q, 0.0
    
    def get_prior_and_alpha(self, images, adv_images, true_labels, sigma, ite):
        """
        This computes the prior aka transfer grad from the surrogate model
        """
        alpha = None # in case method == 'uniform' or 'fixed_bias'
        target_labels = None
        images.requires_grad = True
        logits_real_images = self.model(self.model_trms(images))
        l = self.xent_loss(logits_real_images, true_labels, target_labels)
        if self.method != "uniform":
            prior = torch.squeeze(self.get_grad(self.surrogate_model, self.surrogate_model_trms(adv_images), true_labels, target_labels))  # C,H,W
            prior = self.resize(prior) # get back to image size after trm from surrogate model
            # Find the cosine value below
            # alpha = torch.sum(true * prior) / torch.clamp(torch.sqrt(torch.sum(true * true) * torch.sum(prior * prior)), min=1e-12)  # 这个alpha仅仅用来看看梯度对不对，后续会更新
            # log.info("alpha = {:.3}".format(alpha))
            prior = prior / torch.clamp(torch.sqrt(torch.mean(torch.mul(prior, prior))),min=1e-12)
        if self.method == "biased":
            start_iter = 3  # is start_iter=3 do math when gradient norm
            if ite % 10 == 0 or ite == start_iter:
                # Estimate norm of true gradient
                s = 10
                # pert shape = 10,C,H,W
                pert = torch.randn(size=(s, adv_images.size(1), adv_images.size(2), adv_images.size(3)))
                for i in range(s):
                    pert[i] = pert[i] / torch.clamp(torch.sqrt(torch.mean(torch.mul(pert[i], pert[i]))), min=1e-12)
                pert = pert.to(self.device)
                # pert = (10,C,H,W), adv_images = (1,C,H,W)
                eval_points =  adv_images + sigma * pert # broadcast, because tensor shape doesn't match exactly
                # eval_points shape = (10,C,H,W) reshape to (10*1, C, H, W)
                eval_points = eval_points.view(-1, adv_images.size(1), adv_images.size(2), adv_images.size(3))
                target_labels_s = None
                if target_labels is not None:
                    target_labels_s = target_labels.repeat(s)
                losses = self.xent_loss(self.model(self.model_trms(eval_points)), true_labels.repeat(s), target_labels_s)  # shape = (10*B,)
                norm_square = torch.mean(((losses - l) / sigma) ** 2) # scalar
            while True:
                logits_for_prior_loss = self.model(self.model_trms(adv_images + sigma* prior)) # prior may be C,H,W
                prior_loss = self.xent_loss(logits_for_prior_loss, true_labels, target_labels)  # shape = (batch_size,)
                diff_prior = (prior_loss - l)[0].item()
                if diff_prior == 0:
                    sigma *= 2
                    print("sigma={:.4f}, multiply sigma by 2".format(sigma))
                else:
                    break
                    # alpha is the cosine sim between true and estimated gradient (*< *)?
            est_alpha = diff_prior / sigma / torch.clamp(torch.sqrt(torch.sum(torch.mul(prior,prior)) * norm_square), min=1e-12)
            est_alpha = est_alpha.item()
            # log.info("Estimated alpha = {:.3f}".format(est_alpha))
            alpha = est_alpha   # alpha describes whether the gradient of the surrogate model is useful 
            # the larger λ, the large the belief in prior
            if alpha < 0:  # If the angle is greater than 90 degrees, cos becomes negative.
                prior = -prior  # v = -v , negative the transfer gradient,
                alpha = -alpha
        return prior, alpha
    
    def get_lambda(self, alpha, q):
        n = self.image_height * self.image_width * self.in_channels # n num of features
        d = 50 * 50 * self.in_channels # What?
        gamma = 3.5
        A_square = d / n * gamma
        return_prior = False
        
        # calculate best lambda
        if self.method == 'biased':
            if self.dataprior:
                best_lambda = A_square * (A_square - alpha ** 2 * (d + 2 * q - 2)) / (
                        A_square ** 2 + alpha ** 4 * d ** 2 - 2 * A_square * alpha ** 2 * (q + d * q - 1))
            else:
                best_lambda = (1 - alpha ** 2) * (1 - alpha ** 2 * (n + 2 * q - 2)) / (
                        alpha ** 4 * n * (n + 2 * q - 2) - 2 * alpha ** 2 * n * q + 1)
            print("best_lambda = {:.4f}".format(best_lambda))
            if best_lambda < 1 and best_lambda > 0:
                lmda = best_lambda
            else:
                if alpha ** 2 * (n + 2 * q - 2) < 1:
                    lmda = 0
                else:
                    lmda = 1
            if abs(alpha) >= 1:
                lmda = 1
            print("lambda = {:.3f}".format(lmda))
            if lmda == 1:
                return_prior = True   # lmda =1, we trust this prior as true gradient
        elif self.method == "fixed_biased":
            lmda = 0.5
        return lmda, return_prior
            
    def define_target(self, true_labels, logits):
        if self.targeted:
            if self.target_type == 'random':
                target_labels = torch.randint(low=0, high=self.num_classes,
                                                size=true_labels.size()).long().cuda()
                invalid_target_index = target_labels.eq(true_labels)
                while invalid_target_index.sum().item() > 0:
                    target_labels[invalid_target_index] = torch.randint(low=0, high=logits.shape[1],
                                size=target_labels[invalid_target_index].shape).long().cuda()     
                    invalid_target_index = target_labels.eq(true_labels)
            elif self.target_type == 'least_likely':
                target_labels = logits.argmin(dim=1)
            elif self.target_type == "increment":
                target_labels = torch.fmod(true_labels + 1, self.num_classes)
            else:
                raise NotImplementedError('Unknown target_type: {}'.format(self.target_type))
        else:
            target_labels = None
        return target_labels
    
    def set_up_sigma(self, sigma, ite, adv_images, true_labels, target_labels, l):
        if ite % 2 == 0 and sigma != self.sigma:
            print("checking if sigma could be set to be 1e-4") # why?
            rand = torch.randn_like(adv_images)
            rand = torch.div(rand, torch.clamp(torch.sqrt(torch.mean(torch.mul(rand, rand))), min=1e-12))
            logits_1 = self.model(self.model_trms(adv_images + self.sigma * rand))
            rand_loss = self.xent_loss(logits_1, true_labels, target_labels)  # shape = (batch_size,)
            total_q += 1
            rand =  torch.randn_like(adv_images)
            rand = torch.div(rand, torch.clamp(torch.sqrt(torch.mean(torch.mul(rand, rand))), min=1e-12))
            logits_2 = self.model(self.model_trms(adv_images + self.sigma * rand))
            rand_loss2= self.xent_loss(logits_2, true_labels, target_labels) # shape = (batch_size,)
            total_q += 1
            if (rand_loss - l)[0].item() != 0 and (rand_loss2 - l)[0].item() != 0:
                sigma = self.sigma
                print("set sigma back to 1e-4, sigma={:.4f}".format(sigma))
        return sigma
    
    def get_grad_for_attack(self, adv_images, true_labels, target_labels, prior, lmda, q, return_prior, sigma, l):
        if not return_prior:
            if self.dataprior:
                upsample = nn.UpsamplingNearest2d(size=(adv_images.size(-2), adv_images.size(-1)))  # H, W of original image
                pert = torch.randn(size=(q, self.in_channels, 50, 50))
                pert = upsample(pert)
            else:
                pert = torch.randn(size=(q, adv_images.size(-3), adv_images.size(-2), adv_images.size(-1)))  # q,C,H,W
            pert = pert.to(self.device)
            # line 7 - 10, Alg 1, sample uniform vectors and add the to grad g with sample variance 
            for i in range(q):
                if self.method == 'biased' or self.method == 'fixed_biased':
                    angle_prior = torch.sum(pert[i] * prior) / \
                                    torch.clamp(torch.sqrt(torch.sum(pert[i] * pert[i]) * torch.sum(prior * prior)),min=1e-12)  # C,H,W x B,C,H,W
                    pert[i] = pert[i] - angle_prior * prior  # prior = B,C,H,W so pert[i] = B,C,H,W  # FIXME 这里不支持batch模式
                    pert[i] = pert[i] / torch.clamp(torch.sqrt(torch.mean(torch.mul(pert[i], pert[i]))), min=1e-12)
                    # pert[i] line 9 rightmost term in equation (?)
                    pert[i] = np.sqrt(1-lmda) * pert[i] + np.sqrt(lmda) * prior  # paper's Algorithm 1: line 9
                else:
                    pert[i] = pert[i] / torch.clamp(torch.sqrt(torch.mean(torch.mul(pert[i], pert[i]))),min=1e-12)
            ########
            while True:
                eval_points = adv_images + sigma * pert  # (1,C,H,W)  pert=(q,C,H,W)
                logits_ = self.model(self.model_trms(eval_points))
                target_labels_q = None
                if target_labels is not None:
                    target_labels_q = target_labels.repeat(q)
                losses = self.xent_loss(logits_, true_labels.repeat(q), target_labels_q)  # shape = (q,)
                grad = (losses - l).view(-1, 1, 1, 1) * pert  # (q,1,1,1) * (q,C,H,W)
                grad = torch.mean(grad, dim=0, keepdim=True)  # 1,C,H,W
                norm_grad = torch.sqrt(torch.mean(torch.mul(grad,grad)))
                if norm_grad.item() == 0:
                    sigma *= 5
                    print("estimated grad == 0, multiply sigma by 5. Now sigma={:.4f}".format(sigma))
                else:
                    break
            grad = grad / torch.clamp(torch.sqrt(torch.mean(torch.mul(grad,grad))), min=1e-12) # could be line 12

            """def print_loss(model, direction):
                length = [1e-4, 1e-3]
                les = []
                for ss in length:
                    logits_p = model(adv_images + ss * direction)
                    loss_p = self.xent_loss(logits_p, true_labels, target_labels)
                    les.append((loss_p - l)[0].item())
                log.info("losses: ".format(les))

            if args.show_loss:
                if args.method == 'biased' or args.method == 'fixed_biased':
                    show_input = adv_images + lr * prior
                    logits_show = self.model(show_input)
                    lprior = self.xent_loss(logits_show, true_labels, target_labels) - l
                    print_loss(self.model, prior)
                    show_input_2 = adv_images + lr * grad
                    logits_show2 = self.model(show_input_2)
                    lgrad = self.xent_loss(logits_show2, true_labels, target_labels) - l
                    print_loss(self.model, grad)
                    log.info(lprior, lgrad)"""
        else:
            grad = prior
        return grad
    
    def attack(self, images, adv_images, grad, lr):
        if self.norm == "l2":
            # Bandits version
            adv_images = adv_images + lr * grad / torch.clamp(torch.sqrt(torch.mean(torch.mul(grad,grad))),min=1e-12)
            adv_images = self.l2_proj_step(images, self.eps, adv_images)
            # Below is the original author's L2 norm projection-based update
            # adv_images = adv_images + lr * grad / torch.clamp(torch.sqrt(torch.mean(torch.mul(grad,grad))),min=1e-12)
            # norm = torch.clamp(torch.norm(adv_images - images),min=1e-12).item()
            # factor = min(1, eps / norm)
            # adv_images = images + (adv_images - images) * factor
        else:
            if grad.dim() == 3:
                grad = grad.unsqueeze(0)
            adv_images = adv_images + lr * torch.sign(grad)
            adv_images = torch.min(torch.max(adv_images, images - self.eps), images + self.eps)
        adv_images = torch.clamp(adv_images, self.clip_min, self.clip_max)
        return adv_images


    # helper functions used in class
    
    def xent_loss(self, logit, true_labels, target_labels=None):
        if self.targeted:
            return -F.cross_entropy(logit, target_labels, reduction='none')
        else:
            return F.cross_entropy(logit, true_labels, reduction='none')

    def get_grad(self, model, x, true_labels, target_labels):
        with torch.enable_grad():
            x.requires_grad_()
            logits = model(x)
            loss = self.xent_loss(logits, true_labels, target_labels).mean()
            gradient = torch.autograd.grad(loss, x)[0]
        return gradient

    def get_pred(self, model, x):
        with torch.no_grad():
            logits = model(x)
        return logits.max(1)[1]

    def norm_calc(self, t, p=2):
        assert len(t.shape) == 4
        if p == 2:
            norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        elif p == 1:
            norm_vec = t.abs().sum(dim=[1, 2, 3]).view(-1, 1, 1, 1)
        else:
            raise NotImplementedError('Unknown norm p={}'.format(p))
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    def l2_proj_step(self, image, epsilon, adv_image):
        delta = adv_image - image
        out_of_bounds_mask = (self.norm_calc(delta) > epsilon).float()
        return out_of_bounds_mask * (image + epsilon * delta / self.norm_calc(delta)) + (1 - out_of_bounds_mask) * adv_image

class HPFPriorRGFAttack(PriorRGFAttack):
    
    def __init__(self, hpf_masker, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hpf_masker = hpf_masker

    def get_alpha_delta(self, mask):
        # mask values are between 0-1
        inv_mask_mean = (1 - mask).mean()
        alpha_delta = 1 + inv_mask_mean
        return alpha_delta.item()
    
    def get_eps_delta(self, mask):
        # yields eps values per pixel depending on the HPF coeffs
        mask_around_0 = mask - 0.5
        scaled_by_alpha = mask_around_0 * self.learning_rate #
        eps_tensor = torch.full_like(mask, self.eps)
        eps_tensor += scaled_by_alpha
        return eps_tensor

    def __call__(self, images, true_labels):
        """Algorithm 1 in paper

        Args:
            img (_type_): input image in [0, 1]

        """
        target_labels = None
        images = images.to(self.device)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        images = images.unsqueeze(0)
        
        attack_mask = self.hpf_masker.compute_for_hpf_mask(images)
        
        true_labels = torch.Tensor([true_labels]).to(torch.long).to(self.device)
        adv_images = images.clone().detach()
        adv_images.requires_grad = True
        assert images.size(0) == 1
        logits_real_images = self.model(self.model_trms(images))
        l = self.xent_loss(logits_real_images, true_labels, target_labels)
        lr = float(self.learning_rate)
        total_q = 0
        ite = 0
        while total_q <= self.max_queries: # only to max nr of queries
            total_q += 1
            sigma = self.set_up_sigma(self.sigma, ite, adv_images, true_labels, target_labels, l)            
            prior, alpha = self.get_prior_and_alpha(images, adv_images, true_labels, sigma, ite)
            q = self.samples_per_draw
            lmda, return_prior = self.get_lambda(alpha, q)
            grad = self.get_grad_for_attack(adv_images, true_labels, target_labels, prior, lmda, q, return_prior, sigma, l)
            adv_images = self.attack(images, adv_images, grad, lr, ite)
            adv_labels = self.get_pred(self.model, self.model_trms(adv_images))
            logits_ = self.model(self.model_trms(adv_images))
            l = self.xent_loss(logits_, true_labels, target_labels)
            print('queries:', total_q, 'loss:', l, 'learning rate:', lr, 'sigma:', sigma, 'prediction:', adv_labels,
                    'distortion:', torch.max(torch.abs(adv_images - images)).item(), torch.norm((adv_images - images)).item())
            ite += 1
        return adv_images, total_q, 0.0

    def attack(self, images, adv_images, grad, lr, ite):
        attack_mask = self.hpf_masker.update_saliency_mask(grad)
        if ite == 0:
            self.eps_tensor = self.get_eps_delta(attack_mask)
            alpha_delta = self.get_alpha_delta(attack_mask)
            self.adjusted_lr = self.learning_rate * alpha_delta
        if self.norm == "l2":
            # Bandits version
            adv_images = adv_images + self.adjusted_lr * (attack_mask * grad) / torch.clamp(torch.sqrt(torch.mean(torch.mul(grad,grad))),min=1e-12)
            adv_images = self.l2_proj_step(images, self.eps, adv_images)
            # Below is the original author's L2 norm projection-based update
            # adv_images = adv_images + lr * grad / torch.clamp(torch.sqrt(torch.mean(torch.mul(grad,grad))),min=1e-12)
            # norm = torch.clamp(torch.norm(adv_images - images),min=1e-12).item()
            # factor = min(1, eps / norm)
            # adv_images = images + (adv_images - images) * factor
        else:
            if grad.dim() == 3:
                grad = grad.unsqueeze(0)
            adv_images = adv_images + self.adjusted_lr * attack_mask * torch.sign(grad)
            adv_images = torch.min(torch.max(adv_images, images - self.eps_tensor), images + self.eps_tensor)
        adv_images = torch.clamp(adv_images, self.clip_min, self.clip_max)
        return adv_images

def get_expr_dir_name(dataset, method, surrogate_arch, norm, targeted, target_type, args):
    from datetime import datetime
    # dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'P-RGF_{}_attack_on_defensive_model_{}_surrogate_arch_{}_{}_{}'.format(method, dataset, surrogate_arch, norm, target_str)
    else:
        dirname = 'P-RGF_{}_attack_{}_surrogate_arch_{}_{}_{}'.format(method, dataset,surrogate_arch,norm,target_str)
    return dirname

def set_log_file(fname):
    # set log file
    # simple tricks for duplicating logging destination in the logging module such as:
    # logging.getLogger().addHandler(logging.FileHandler(filename))
    # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything
    import subprocess
    # sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', 0)
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        print('{:s}: {}'.format(prefix, args.__getattribute__(key)))

if __name__ == "__main__":
    pass
