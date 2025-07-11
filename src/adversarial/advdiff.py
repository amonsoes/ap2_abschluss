import sys
import os 
import random
import argparse

sys.path.append(".")
sys.path.append('./taming-transformers')

import torch
from omegaconf import OmegaConf

from torchvision.utils import save_image
from torch.backends import cudnn

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

from src.adversarial.attack_base import Attack
    
parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', type=int, default=6)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--scale', type=float, default=3.0)
parser.add_argument('--ddim-steps', type=int, default=200)
parser.add_argument('--ddim-eta', type=float, default=0.0)
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--s', type=float, default=1.0)
parser.add_argument('--a', type=float, default=0.5)
parser.add_argument('--save-dir', type=str, default='advdiff/')
args = parser.parse_args()


class AdvDiffusion(Attack):

    def __init__(self, ddim_steps, scale, ddim_eta, surrogate_model, batch_size, *args, **kwargs):
        super().__init__('AdvDiffusion', *args, **kwargs)
        self.ddim_steps = args.ddim_steps
        self.ddim_eta = args.ddim_eta
        self.scale = args.scale
        self.model = self.get_model()
        self.sampler = DDIMSampler(self.model, vic_model=self.surrogate_model)
        self.classes =  np.arange(1000)
        self.n_samples_per_class = batch_size
    
    def forward(self, images, labels):
        all_samples = list()
        all_labels = list()

        with torch.no_grad():
            with self.model.ema_scope():
                uc = self.model.get_learned_conditioning(
                    {self.model.cond_stage_key: torch.tensor(self.n_samples_per_class*[1000]).to(self.model.device)}
                    )

                for class_label in self.classes:
                    print(f"rendering {self.n_samples_per_class} examples of class '{class_label}' in {self.ddim_steps} steps and using s={self.scale:.2f}.")
                    xc = torch.tensor(self.n_samples_per_class*[class_label])
                    c = self.model.get_learned_conditioning({self.model.cond_stage_key: xc.to(self.model.device)})

                    samples_ddim, _ = self.sampler.sample(S=self.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=self.n_samples_per_class,
                                                    shape=[3, 64, 64],
                                                    verbose=False,
                                                    unconditional_guidance_scale=self.scale,
                                                    unconditional_conditioning=uc, 
                                                    eta=self.ddim_eta,
                                                    label=xc.to(self.model.device),
                                                    K=args.K,s=args.s,a=args.a)

                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                                min=0.0, max=1.0)
                    all_samples.append(x_samples_ddim)
                    all_labels.append(xc)

        adv_images = torch.cat(all_samples, 0)
        labels = torch.cat(all_labels, 0)

        return adv_images 

    def get_model(self):
        config = OmegaConf.load("src/adversarial/resources/cin256-v2.yaml")  
        model = load_model_from_config(config, "src/adversarial/resources/model.ckpt")
        return model


#### ldm.modules.diffusionmodules.util.py code ####

def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out

def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


#### ddim_adv.py code #####


import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import torchvision.transforms as T

from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

from torchvision.utils import save_image, make_grid

#weights = ResNet50_Weights.DEFAULT
#preprocess = weights.transforms()

preprocess = T.Compose([
    T.Resize((256,256)),
    T.CenterCrop((224,224)),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_target_label(logits, label, device): # seond-like label for attack
    
    rates, indices = logits.sort(1, descending=True) 
    rates, indices = rates.squeeze(0), indices.squeeze(0)  
    
    tar_label = torch.zeros_like(label).to(device)
    
    for i in range(label.shape[0]):
        if label[i] == indices[i][0]:  # classify is correct
            tar_label[i] = indices[i][1]
        else:
            tar_label[i] = indices[i][0]
    
    return tar_label


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", vic_model=None, **kwargs):
        
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        
        self.vic_model = vic_model

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               label=None,K=10,s=2,a=1,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,label=label,
                                                    K=K,s=s,a=a
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,label=None,K=10,s=0.75,a=0.5):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]


        pri_img = img.detach().requires_grad_(True)
        
        for k in range(K):
            
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod
            
            img = pri_img.detach().requires_grad_(True)
            
            print(f"Running Adversarial Sampling at {k} step")
            
            print(f"Running DDIM Sampling with {total_steps} timesteps")
        
            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                if mask is not None:
                    assert x0 is not None
                    img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                    img = img_orig * mask + (1. - mask) * img
                outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                          quantize_denoised=quantize_denoised, temperature=temperature,
                                          noise_dropout=noise_dropout, score_corrector=score_corrector,
                                          corrector_kwargs=corrector_kwargs,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
                img, pred_x0 = outs
                
                    
                '''
                if index % 20 == 0:
                    x_samples_ddim = self.model.decode_first_stage(img)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                                 min=0.0, max=1.0)
                    save_image(x_samples_ddim, f"img/Diff_{index}.png", nrow=1, normalize=True)
                '''
                    
                if(index > total_steps * 0 and index <= total_steps * 0.2):

                    sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
                    with torch.enable_grad():
                        img_n = img.detach().requires_grad_(True)
                        img_transformed = self.model.differentiable_decode_first_stage(img_n) # image transformation from latent code
                        img_transformed = torch.clamp((img_transformed+1.0)/2.0, 
                         min=0.0, max=1.0)
                        img_transformed = preprocess(img_transformed).to(device) # image transformation to model input
                        logits = self.vic_model(img_transformed)
                        log_probs = F.log_softmax(logits, dim=-1)
                        tar_label = get_target_label(logits, label, device)
                        selected = log_probs[range(len(logits)), tar_label]
                        gradient = torch.autograd.grad(selected.sum(), img_n)[0] # adversarial guidance
                    #gradient = torch.clamp(gradient, min=-0.3, max=0.3) # possible gradiant clamp
                    #img = img + s * gradient.float() * sqrt_one_minus_at * 10
                    img = img + s * gradient.float()
                    
                if callback: callback(i)
                if img_callback: img_callback(pred_x0, i)

                if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates['x_inter'].append(img)
                    intermediates['pred_x0'].append(pred_x0)
                    
            x_samples_ddim = self.model.decode_first_stage(img)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                             min=0.0, max=1.0)
            with torch.enable_grad(): # adversarial prior
                img_n = img.detach().requires_grad_(True)
                img_transformed = self.model.differentiable_decode_first_stage(img_n)
                img_transformed = torch.clamp((img_transformed+1.0)/2.0, 
                 min=0.0, max=1.0)
                img_transformed = preprocess(img_transformed).to(device)
                logits = self.vic_model(img_transformed)
                log_probs = F.log_softmax(logits, dim=-1)
                tar_label = get_target_label(logits, label, device)
                selected = log_probs[range(len(logits)), tar_label]
                gradient = torch.autograd.grad(selected.sum(), img_n)[0]
            
            img_transformed = preprocess(x_samples_ddim).to(device)
            logits = self.vic_model(img_transformed)
            log_probs = F.log_softmax(logits, dim=-1)
            pred = torch.argmax(log_probs, dim=1)  # [B]
            success_num = (pred == label).sum().item()
            print(pred)
            print(f"Success {b-success_num} / {b}")
            if b-success_num > 0 : # early exit
                break
            #gradient = torch.clamp(gradient, min=-0.3, max=0.3)
            pri_img = pri_img + a * gradient.float()
                
        return img[pred != label], intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0


##### advdiff.py code #####

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def main():
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    
    model = get_model()
    vic_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(model.device)
    vic_model.eval()
    sampler = DDIMSampler(model, vic_model=vic_model)

    classes =  np.arange(1000)
    n_samples_per_class = args.batch_size

    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    ddim_steps = args.ddim_steps
    ddim_eta = args.ddim_eta
    scale = args.scale   # for unconditional guidance


    all_samples = list()
    all_labels = list()

    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
                )

            for class_label in classes:
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class*[class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=n_samples_per_class,
                                                 shape=[3, 64, 64],
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc, 
                                                 eta=ddim_eta,
                                                 label=xc.to(model.device),
                                                 K=args.K,s=args.s,a=args.a)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                             min=0.0, max=1.0)
                all_samples.append(x_samples_ddim)
                all_labels.append(xc)

    img = torch.cat(all_samples, 0)
    labels = torch.cat(all_labels, 0)

    save_img = img.permute(0,2,3,1)

    np.savez(os.path.join(args.save_dir, 'AdvDiff.npz'), save_img.detach().cpu().numpy(), labels.detach().cpu().numpy())


#### utils code #####

import importlib

import torch
import numpy as np
from collections import abc
from einops import rearrange
from functools import partial

import multiprocessing as mp
from threading import Thread
from queue import Queue

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def _do_parallel_data_prefetch(func, Q, data, idx, idx_to_fn=False):
    # create dummy dataset instance

    # run prefetching
    if idx_to_fn:
        res = func(data, worker_id=idx)
    else:
        res = func(data)
    Q.put([idx, res])
    Q.put("Done")


def parallel_data_prefetch(
        func: callable, data, n_proc, target_data_type="ndarray", cpu_intensive=True, use_worker_id=False
):
    # if target_data_type not in ["ndarray", "list"]:
    #     raise ValueError(
    #         "Data, which is passed to parallel_data_prefetch has to be either of type list or ndarray."
    #     )
    if isinstance(data, np.ndarray) and target_data_type == "list":
        raise ValueError("list expected but function got ndarray.")
    elif isinstance(data, abc.Iterable):
        if isinstance(data, dict):
            print(
                f'WARNING:"data" argument passed to parallel_data_prefetch is a dict: Using only its values and disregarding keys.'
            )
            data = list(data.values())
        if target_data_type == "ndarray":
            data = np.asarray(data)
        else:
            data = list(data)
    else:
        raise TypeError(
            f"The data, that shall be processed parallel has to be either an np.ndarray or an Iterable, but is actually {type(data)}."
        )

    if cpu_intensive:
        Q = mp.Queue(1000)
        proc = mp.Process
    else:
        Q = Queue(1000)
        proc = Thread
    # spawn processes
    if target_data_type == "ndarray":
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(np.array_split(data, n_proc))
        ]
    else:
        step = (
            int(len(data) / n_proc + 1)
            if len(data) % n_proc != 0
            else int(len(data) / n_proc)
        )
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(
                [data[i: i + step] for i in range(0, len(data), step)]
            )
        ]
    processes = []
    for i in range(n_proc):
        p = proc(target=_do_parallel_data_prefetch, args=arguments[i])
        processes += [p]

    # start processes
    print(f"Start prefetching...")
    import time

    start = time.time()
    gather_res = [[] for _ in range(n_proc)]
    try:
        for p in processes:
            p.start()

        k = 0
        while k < n_proc:
            # get result
            res = Q.get()
            if res == "Done":
                k += 1
            else:
                gather_res[res[0]] = res[1]

    except Exception as e:
        print("Exception: ", e)
        for p in processes:
            p.terminate()

        raise e
    finally:
        for p in processes:
            p.join()
        print(f"Prefetching complete. [{time.time() - start} sec.]")

    if target_data_type == 'ndarray':
        if not isinstance(gather_res[0], np.ndarray):
            return np.concatenate([np.asarray(r) for r in gather_res], axis=0)

        # order outputs
        return np.concatenate(gather_res, axis=0)
    elif target_data_type == 'list':
        out = []
        for r in gather_res:
            out.extend(r)
        return out
    else:
        return gather_res

            
if __name__ == '__main__':
    main()
