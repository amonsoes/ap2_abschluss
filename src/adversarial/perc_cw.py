
import math

from typing import Tuple, Optional
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np

from src.adversarial.attack_base import Attack
from src.datasets.data_transforms.img_transform import CustomCompose

class CIEDE2000Loss:
    
    def __init__(self, device, batch_size):
        self.device = device
        self.batch_size = batch_size
        
    
    def __call__(self, ori, ref):
        rgb2lab_ref = self.rgb2lab_diff(ref)
        rgb2lab_input = self.rgb2lab_diff(ori)
        ciede2000_score = self.ciede2000_diff(rgb2lab_ref,rgb2lab_input)
        return torch.norm(ciede2000_score.view(self.batch_size, -1), dim=1)

    def ciede2000_diff(self, lab1, lab2):
        '''
        CIEDE2000 metric to claculate the color distance map for a batch of image tensors defined in CIELAB space
        
        '''
        
        lab1=lab1.to(self.device)    
        lab2=lab2.to(self.device)
        
        L1 = lab1[:,0,:,:]
        A1 = lab1[:,1,:,:]
        B1 = lab1[:,2,:,:]
        L2 = lab2[:,0,:,:]
        A2 = lab2[:,1,:,:]
        B2 = lab2[:,2,:,:]   
        kL = 1
        kC = 1
        kH = 1
        
        mask_value_0_input1=((A1==0)*(B1==0)).float()
        mask_value_0_input2=((A2==0)*(B2==0)).float()
        mask_value_0_input1_no=1-mask_value_0_input1
        mask_value_0_input2_no=1-mask_value_0_input2
        B1=B1+0.0001*mask_value_0_input1
        B2=B2+0.0001*mask_value_0_input2 
        
        C1 = torch.sqrt((A1 ** 2.) + (B1 ** 2.))
        C2 = torch.sqrt((A2 ** 2.) + (B2 ** 2.))   
    
        aC1C2 = (C1 + C2) / 2.
        G = 0.5 * (1. - torch.sqrt((aC1C2 ** 7.) / ((aC1C2 ** 7.) + (25 ** 7.))))
        a1P = (1. + G) * A1
        a2P = (1. + G) * A2
        c1P = torch.sqrt((a1P ** 2.) + (B1 ** 2.))
        c2P = torch.sqrt((a2P ** 2.) + (B2 ** 2.))


        h1P = hpf_diff(B1, a1P)
        h2P = hpf_diff(B2, a2P)
        h1P=h1P*mask_value_0_input1_no
        h2P=h2P*mask_value_0_input2_no 
        
        dLP = L2 - L1
        dCP = c2P - c1P
        dhP = dhpf_diff(C1, C2, h1P, h2P)
        dHP = 2. * torch.sqrt(c1P * c2P) * torch.sin(radians(dhP) / 2.)
        mask_0_no=1-torch.max(mask_value_0_input1,mask_value_0_input2)
        dHP=dHP*mask_0_no

        aL = (L1 + L2) / 2.
        aCP = (c1P + c2P) / 2.
        aHP = ahpf_diff(C1, C2, h1P, h2P)
        T = 1. - 0.17 * torch.cos(radians(aHP - 39)) + 0.24 * torch.cos(radians(2. * aHP)) + 0.32 * torch.cos(radians(3. * aHP + 6.)) - 0.2 * torch.cos(radians(4. * aHP - 63.))
        dRO = 30. * torch.exp(-1. * (((aHP - 275.) / 25.) ** 2.))
        rC = torch.sqrt((aCP ** 7.) / ((aCP ** 7.) + (25. ** 7.)))    
        sL = 1. + ((0.015 * ((aL - 50.) ** 2.)) / torch.sqrt(20. + ((aL - 50.) ** 2.)))
        
        sC = 1. + 0.045 * aCP
        sH = 1. + 0.015 * aCP * T
        rT = -2. * rC * torch.sin(radians(2. * dRO))

    #     res_square=((dLP / (sL * kL)) ** 2.) + ((dCP / (sC * kC)) ** 2.) + ((dHP / (sH * kH)) ** 2.) + rT * (dCP / (sC * kC)) * (dHP / (sH * kH))

        res_square=((dLP / (sL * kL)) ** 2.) + ((dCP / (sC * kC)) ** 2.)*mask_0_no + ((dHP / (sH * kH)) ** 2.)*mask_0_no + rT * (dCP / (sC * kC)) * (dHP / (sH * kH))*mask_0_no
        mask_0=(res_square<=0).float()
        mask_0_no=1-mask_0
        res_square=res_square+0.0001*mask_0    
        res=torch.sqrt(res_square)
        res=res*mask_0_no


        return res

    def rgb2lab_diff(self, rgb_image):
        '''
        Function to convert a batch of image tensors from RGB space to CIELAB space.    
        parameters: xn, yn, zn are the CIE XYZ tristimulus values of the reference white point. 
        Here use the standard Illuminant D65 with normalization Y = 100.
        '''
        rgb_image=rgb_image.to(self.device)
        res = torch.zeros_like(rgb_image)
        xyz_image = rgb2xyz(rgb_image,self.device)
        
        xn = 95.0489
        yn = 100
        zn = 108.8840
        
        x = xyz_image[:,0, :, :]
        y = xyz_image[:,1, :, :]
        z = xyz_image[:,2, :, :]

        L = 116*xyz_lab(y/yn,self.device) - 16
        a = 500*(xyz_lab(x/xn,self.device) - xyz_lab(y/yn,self.device))
        b = 200*(xyz_lab(y/yn,self.device) - xyz_lab(z/zn,self.device))
        res[:, 0, :, :] = L
        res[:, 1, :, :] = a
        res[:, 2, :, :] = b
    
        return res


def rgb2xyz(rgb_image,device):
    mt = torch.tensor([[0.4124, 0.3576, 0.1805], 
                   [0.2126, 0.7152, 0.0722],
                   [0.0193, 0.1192, 0.9504]]).to(device)
    mask1=(rgb_image > 0.0405).float()
    mask1_no=1-mask1
    temp_img = mask1* (((rgb_image + 0.055 ) / 1.055 ) ** 2.4)
    temp_img = temp_img+mask1_no * (rgb_image / 12.92)    
    temp_img = 100 * temp_img

    res = torch.matmul(mt, temp_img.permute(1, 0, 2,3).contiguous().view(3, -1)).view(3, rgb_image.size(0),rgb_image.size(2), rgb_image.size(3)).permute(1, 0, 2,3)
    return res

def xyz_lab(xyz_image,device):
    mask_value_0=(xyz_image==0).float().to(device)
    mask_value_0_no=1-mask_value_0
    xyz_image=xyz_image+0.0001*mask_value_0
    mask1= (xyz_image > 0.008856).float()     
    mask1_no= 1-mask1
    res = mask1 * (xyz_image) ** (1 /3)
    res = res+mask1_no * ((7.787 * xyz_image) + (16/ 116))
    res=res*mask_value_0_no
    return res    

def rgb2lab_diff(rgb_image,device):
    '''
    Function to convert a batch of image tensors from RGB space to CIELAB space.    
    parameters: xn, yn, zn are the CIE XYZ tristimulus values of the reference white point. 
    Here use the standard Illuminant D65 with normalization Y = 100.
    '''
    rgb_image=rgb_image.to(device)
    res = torch.zeros_like(rgb_image)
    xyz_image = rgb2xyz(rgb_image,device)
    
    xn = 95.0489
    yn = 100
    zn = 108.8840
    
    x = xyz_image[:,0, :, :]
    y = xyz_image[:,1, :, :]
    z = xyz_image[:,2, :, :]

    L = 116*xyz_lab(y/yn,device) - 16
    a = 500*(xyz_lab(x/xn,device) - xyz_lab(y/yn,device))
    b = 200*(xyz_lab(y/yn,device) - xyz_lab(z/zn,device))
    res[:, 0, :, :] = L
    res[:, 1, :, :] = a
    res[:, 2, :, :] = b
  
    return res


def degrees(n): return n * (180. / np.pi)
def radians(n): return n * (np.pi / 180.)
def hpf_diff(x, y):
    mask1=((x == 0) * (y == 0)).float()
    mask1_no = 1-mask1

    tmphp = degrees(torch.atan2(x*mask1_no, y*mask1_no))
    tmphp1 = tmphp * (tmphp >= 0).float()
    tmphp2 = (360+tmphp)* (tmphp < 0).float()

    return tmphp1+tmphp2

def dhpf_diff(c1, c2, h1p, h2p):

    mask1  = ((c1 * c2) == 0).float()
    mask1_no  = 1-mask1
    res1=(h2p - h1p)*mask1_no*(torch.abs(h2p - h1p) <= 180).float()
    res2 = ((h2p - h1p)- 360) * ((h2p - h1p) > 180).float()*mask1_no
    res3 = ((h2p - h1p)+360) * ((h2p - h1p) < -180).float()*mask1_no

    return res1+res2+res3

def ahpf_diff(c1, c2, h1p, h2p):

    mask1=((c1 * c2) == 0).float()
    mask1_no=1-mask1
    mask2=(torch.abs(h2p - h1p) <= 180).float()
    mask2_no=1-mask2
    mask3=(torch.abs(h2p + h1p) < 360).float()
    mask3_no=1-mask3

    res1 = (h1p + h2p) *mask1_no * mask2
    res2 = (h1p + h2p + 360.) * mask1_no * mask2_no * mask3 
    res3 = (h1p + h2p - 360.) * mask1_no * mask2_no * mask3_no
    res = (res1+res2+res3)+(res1+res2+res3)*mask1
    return res*0.5

def ciede2000_diff(lab1, lab2,device):
    '''
    CIEDE2000 metric to claculate the color distance map for a batch of image tensors defined in CIELAB space
    
    '''
    
    lab1=lab1.to(device)    
    lab2=lab2.to(device)
       
    L1 = lab1[:,0,:,:]
    A1 = lab1[:,1,:,:]
    B1 = lab1[:,2,:,:]
    L2 = lab2[:,0,:,:]
    A2 = lab2[:,1,:,:]
    B2 = lab2[:,2,:,:]   
    kL = 1
    kC = 1
    kH = 1
    
    mask_value_0_input1=((A1==0)*(B1==0)).float()
    mask_value_0_input2=((A2==0)*(B2==0)).float()
    mask_value_0_input1_no=1-mask_value_0_input1
    mask_value_0_input2_no=1-mask_value_0_input2
    B1=B1+0.0001*mask_value_0_input1
    B2=B2+0.0001*mask_value_0_input2 
    
    C1 = torch.sqrt((A1 ** 2.) + (B1 ** 2.))
    C2 = torch.sqrt((A2 ** 2.) + (B2 ** 2.))   
   
    aC1C2 = (C1 + C2) / 2.
    G = 0.5 * (1. - torch.sqrt((aC1C2 ** 7.) / ((aC1C2 ** 7.) + (25 ** 7.))))
    a1P = (1. + G) * A1
    a2P = (1. + G) * A2
    c1P = torch.sqrt((a1P ** 2.) + (B1 ** 2.))
    c2P = torch.sqrt((a2P ** 2.) + (B2 ** 2.))


    h1P = hpf_diff(B1, a1P)
    h2P = hpf_diff(B2, a2P)
    h1P=h1P*mask_value_0_input1_no
    h2P=h2P*mask_value_0_input2_no 
    
    dLP = L2 - L1
    dCP = c2P - c1P
    dhP = dhpf_diff(C1, C2, h1P, h2P)
    dHP = 2. * torch.sqrt(c1P * c2P) * torch.sin(radians(dhP) / 2.)
    mask_0_no=1-torch.max(mask_value_0_input1,mask_value_0_input2)
    dHP=dHP*mask_0_no

    aL = (L1 + L2) / 2.
    aCP = (c1P + c2P) / 2.
    aHP = ahpf_diff(C1, C2, h1P, h2P)
    T = 1. - 0.17 * torch.cos(radians(aHP - 39)) + 0.24 * torch.cos(radians(2. * aHP)) + 0.32 * torch.cos(radians(3. * aHP + 6.)) - 0.2 * torch.cos(radians(4. * aHP - 63.))
    dRO = 30. * torch.exp(-1. * (((aHP - 275.) / 25.) ** 2.))
    rC = torch.sqrt((aCP ** 7.) / ((aCP ** 7.) + (25. ** 7.)))    
    sL = 1. + ((0.015 * ((aL - 50.) ** 2.)) / torch.sqrt(20. + ((aL - 50.) ** 2.)))
    
    sC = 1. + 0.045 * aCP
    sH = 1. + 0.015 * aCP * T
    rT = -2. * rC * torch.sin(radians(2. * dRO))

#     res_square=((dLP / (sL * kL)) ** 2.) + ((dCP / (sC * kC)) ** 2.) + ((dHP / (sH * kH)) ** 2.) + rT * (dCP / (sC * kC)) * (dHP / (sH * kH))

    res_square=((dLP / (sL * kL)) ** 2.) + ((dCP / (sC * kC)) ** 2.)*mask_0_no + ((dHP / (sH * kH)) ** 2.)*mask_0_no + rT * (dCP / (sC * kC)) * (dHP / (sH * kH))*mask_0_no
    mask_0=(res_square<=0).float()
    mask_0_no=1-mask_0
    res_square=res_square+0.0001*mask_0    
    res=torch.sqrt(res_square)
    res=res*mask_0_no


    return res

def quantization(x):
   """quantize the continus image tensors into 255 levels (8 bit encoding)"""
   x_quan=torch.round(x*255)/255
   return x_quan


class PerC_AL(Attack):
    """
    PerC_AL: Alternating Loss of Classification and Color Differences to achieve imperceptibile perturbations with few iterations.

    Parameters
    ----------
    max_iterations : int
        Number of iterations for the optimization.
    alpha_l_init: float
        step size for updating perturbations with respect to classification loss
    alpha_c_init: float
        step size for updating perturbations with respect to perceptual color differences
    kappa : float, optional
        kappa of the adversary for Carlini's loss, in term of distance between logits.
        Note that this approach only supports kappa setting in an untargeted case
    device : torch.device, optional
        Device on which to perform the adversary.

    """

    def __init__(self,
             model: torch.nn.Module,
             model_trms: CustomCompose,
             steps: int = 1000,
             alpha_l: float = 1.,
             #for relatively easy untargeted case, alpha_c_init is adjusted to a smaller value (e.g., 0.1 is used in the paper) 
             alpha_c: float = 0.1,
             kappa: float = 0.,
             batch_size: int = 32, 
             l2_bound=-1,
             *args,
             **kwargs
                ) -> None:
        super().__init__('PercAL', model, model_trms, *args, **kwargs)
        self.max_iterations = steps
        self.alpha_l_init = alpha_l
        self.alpha_c_init = alpha_c
        self.kappa = kappa
        self.original_kappa = kappa
        self.batch_size = batch_size
        self.loss = nn.BCEWithLogitsLoss(reduction='sum') if model.model_name in ['clip_det', 'corviresnet'] else nn.CrossEntropyLoss(reduction='sum')
        self.l2_bound = 0 if l2_bound == -1 else l2_bound
        self.low, self.high = self.get_iq_range_from_l2_bound(l2_bound)
        self.low = self.get_l2_from_bounds(l2_bound)

        if self.low == self.high:
            self.low -= self.low*alpha_c
            self.high += self.high*alpha_c
        

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Performs the adversary of the model given the inputs and labels.

        Parameters
        inputs : torch.Tensor
            Batch of image examples in the range of [0,1].
        labels : torch.Tensor
            Original labels if untargeted, else labels of targets.

        Returns
        -------
        torch.Tensor
            Batch of image samples modified to be adversarial
        """
        self.kappa = self.original_kappa
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        if self.targeted:
            labels = self.get_target_label(inputs, labels)

        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')

        # set for schedule and targeting
        alpha_l_min=self.alpha_l_init/100
        alpha_c_min=self.alpha_c_init/10
        multiplier = -1 if self.targeted else 1

        # init attack params
        X_adv_round_best=inputs.clone()
        batch_size=inputs.shape[0]
        delta=torch.zeros_like(inputs, requires_grad=True)
        inputs_LAB=rgb2lab_diff(inputs,self.device)
        mask_isadv= torch.zeros(batch_size,dtype=torch.uint8).to(self.device)
        color_l2_delta_bound_best=(torch.ones(batch_size)*100000).to(self.device)

        # set cases according to target mode
        if (self.targeted==False) and self.kappa!=0:
            labels_onehot = torch.zeros(labels.size(0), 1000, device=self.device)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))
        
        # start optimization
        for i in range(self.max_iterations):

            # cosine annealing for alpha_l and alpha_c 
            alpha_c=alpha_c_min+0.5*(self.alpha_c_init-alpha_c_min)*(1+math.cos(i/self.max_iterations*math.pi))
            alpha_l=alpha_l_min+0.5*(self.alpha_l_init-alpha_l_min)*(1+math.cos(i/self.max_iterations*math.pi))

            # get cross-entropy loss for adv sample, scale CE-grad to unit length, update delta by adversarial gradient
            loss = multiplier * (self.loss(self.model(self.model_trms(inputs+delta)), labels) + self.kappa)
            loss.backward()
            grad_a=delta.grad.clone()
            delta.grad.zero_()
            delta.data[~mask_isadv]=delta.data[~mask_isadv]+alpha_l*(grad_a.permute(1,2,3,0)/torch.norm(grad_a.view(batch_size,-1),dim=1)).permute(3,0,1,2)[~mask_isadv]  
            
            # compute CIEDE2000 difference and get fidelity gradients, scale, update delta by color gradient
            d_map=ciede2000_diff(inputs_LAB,rgb2lab_diff(inputs+delta,self.device),self.device).unsqueeze(1)
            color_dis=torch.norm(d_map.view(batch_size,-1),dim=1)
            color_loss=color_dis.sum()
            color_loss
            color_loss.backward()
            grad_color=delta.grad.clone()
            delta.grad.zero_()
            delta.data[mask_isadv]=delta.data[mask_isadv]-alpha_c* (grad_color.permute(1,2,3,0)/torch.norm(grad_color.view(batch_size,-1),dim=1)).permute(3,0,1,2)[mask_isadv]
            delta.data=(inputs+delta.data).clamp(0,1)-inputs

            # quantize image (not included in any backward comps) & check if samples are adversarial
            X_adv_round=quantization(inputs+delta.data)
            mask_isadv = self.check_if_adv(X_adv_round, labels)

            # update adversarial image if: (1) color dist is less (2) images are adversarial
            mask_best=(color_dis.data<color_l2_delta_bound_best)
            mask=mask_best * mask_isadv
            color_l2_delta_bound_best[mask]=color_dis.data[mask]
            X_adv_round_best[mask]=X_adv_round[mask]

            # if in l2_bound range and adversarial, quit optimization

            if self.l2_bound != -1:
                _ ,_ ,l2_distances = self.get_l2_loss(inputs+delta, inputs, None)
                print(f'L2 distances: {l2_distances}, adv loss: {loss} is adversarial: {mask_isadv}')
                if torch.all(l2_distances > self.high):
                    self.kappa=0
                if torch.all(l2_distances >= self.low) and torch.all(l2_distances <= self.high) and torch.all(mask_isadv):
                    break

        return X_adv_round_best
    
    def check_if_adv(self, X_adv_round, labels):
            outputs = self.model(self.model_trms(X_adv_round))
            one_hot_labels = torch.eye(len(outputs[0]), device=self.device)[labels].to(self.device)

            i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the largest logit
            #outputs[outputs.argmax(dim=1)] = -1e10

            if self.targeted:
                j = torch.masked_select(outputs, one_hot_labels.bool()) # get the target logit
                adv_loss = (i - j) + self.kappa
                is_adv = adv_loss <= 0.0
                #return torch.clamp((i-j), min=-self.kappa)
            else:
                label_mask = torch.full_like(labels.view(-1,1), -1e+03).to(torch.float32)
                outputs.scatter_(1, labels.view(-1,1), label_mask)
                j, _ = torch.max(outputs, dim=1)
                adv_loss = (i - j) + self.kappa
                is_adv = adv_loss <= 0.0
            return is_adv

    
    def get_iq_range_from_l2_bound(self, l2_bound):
        if l2_bound <= 4.0:
            low, high = 0.0, 4.0
        elif l2_bound <= 8.0:
            low, high = 4.0, 8.0
        elif l2_bound <= 15.0:
            low, high = 8.0, 15.0
        elif l2_bound <= 23.0:
            low, high = 15.0, 23.0
        elif l2_bound <= 38.0:
            low, high = 23.0, 38.0
        else:
            low, high = 38.0, 134.0
        return low, high

    def get_l2_loss(self, adv_images, images, attack_mask):
        current_iq_loss = (adv_images - images).pow(2).sum(dim=(1,2,3)).sqrt()
        real_losses = current_iq_loss
        current_iq_loss = torch.clamp(current_iq_loss - self.l2_bound, min=0)
        iq_loss = current_iq_loss.sum()
        return iq_loss, current_iq_loss, real_losses

class PerC_CW:
    """
	PerC_CW: C&W with a new substitute penaly term concerning perceptual color differences.
    Modified from https://github.com/jeromerony/fast_adversarial/blob/master/fast_adv/adversarys/carlini.py

    Parameters
    ----------
    image_constraints : tuple
        Bounds of the images.
    num_classes : int
        Number of classes of the model to adversary.
    kappa : float, optional
        kappa of the adversary for Carlini's loss, in term of distance between logits.
    attack_lr : float
        Learning rate for the optimization.
    n_starts : int
        Number of search steps to find the best scale constant for Carlini's loss.
    steps : int
        Maximum number of iterations during a single search step.
    c : float
        constant of the adversary.
    device : torch.device, optional
        Device to use for the adversary.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 model_trms: torchvision.transforms.Compose,
                 image_constraints: Tuple[float, float] = [0,1],
                 num_classes: int = 1000,
                 kappa: float = 0,
                 attack_lr: float = 0.01,
                 n_starts: int = 1,
                 steps: int = 1000,
                 abort_early: bool = False,
                 c: float = 10,
                 target_mode: str = 'most_likely',
                 verbose_cw: bool = False,
                 write_protocol: bool = True,
                 protocol_file: str = './results.txt',
                 eps: float = 0.08
                ) -> None:
        self.model = model
        self.model_trms = model_trms
        self.kappa = kappa
        self.attack_lr = attack_lr
        self.n_starts = n_starts
        self.steps = steps
        self.abort_early = abort_early
        self.c = c
        self.num_classes = num_classes
        self.repeat = self.n_starts >= 10
        self.boxmin = image_constraints[0]
        self.boxmax = image_constraints[1]
        self.boxmul = (self.boxmax - self.boxmin) / 2
        self.boxplus = (self.boxmin + self.boxmax) / 2
        self.set_target_mode(target_mode)
        self.protocol_file = protocol_file
        self.write_protocol = write_protocol
        self.device = next(model.parameters()).device
        self.self.targeted = True

    @staticmethod
    def _arctanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x *= (1. - eps)
        return (torch.log((1 + x) / (1 - x))) * 0.5

    def _step(self,step_n, model: nn.Module, optimizer: optim.Optimizer, inputs: torch.Tensor, tinputs: torch.Tensor,
              modifier: torch.Tensor, labels: torch.Tensor, labels_infhot: torch.Tensor, targeted: bool,
              const: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = inputs.shape[0]
        adv_input = torch.tanh(tinputs + modifier) * self.boxmul + self.boxplus
        # calculate the global perceptual color differences (L2 norm of the color distance map)
        l2 = torch.norm(ciede2000_diff(rgb2lab_diff(inputs,self.device),rgb2lab_diff(adv_input,self.device),self.device).view(batch_size, -1),dim=1)
        logits = model(self.model_trms(adv_input))

        # confusing naming but 'real' refers to the target_labels if self.targeted == True
        real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other = (logits - labels_infhot).max(1)[0]
        if self.targeted:
            # if self.targeted, optimize for making the other class most likely
            logit_dists = torch.clamp(other - real + self.kappa, min=0)
        else:
            # if non-self.targeted, optimize for making this class least likely.
            logit_dists = torch.clamp(real - other + self.kappa, min=0)

        logit_loss = (const * logit_dists).sum()
        l2_loss = l2.sum()
        loss = logit_loss + l2_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'\n{step_n} - iq_loss: {l2_loss.item()}, f_loss: {logit_dists.item()} cost: {loss.item()}')

        return adv_input.detach(), logits.detach(), l2.detach(), logit_dists.detach(), loss.detach()

    def __call__(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Performs the adversary of the model given the inputs and labels.

        Parameters
        ----------
        inputs : torch.Tensor
            Batch of image examples.
        labels : torch.Tensor
            Original labels

        Returns
        -------
        torch.Tensor
            Batch of image samples modified to be adversarial
        """
        
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        batch_size = inputs.shape[0]
        tinputs = self._arctanh((inputs - self.boxplus) / self.boxmul)

        # set the lower and upper bounds accordingly
        lower_bound = torch.zeros(batch_size, device=self.device)
        CONST = torch.full((batch_size,), self.c, device=self.device)
        upper_bound = torch.full((batch_size,), 1e10, device=self.device)

        o_best_l2 = torch.full((batch_size,), 1e10, device=self.device)
        o_best_score = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)
        o_best_adversary = inputs.clone()

        target_label = self.get_target(inputs, labels)
        
        # setup the target variable, we need it to be in one-hot form for the loss function
        labels_onehot = torch.zeros(target_label.size(0), self.num_classes, device=self.device)
        labels_onehot.scatter_(1, target_label.unsqueeze(1), 1)
        labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, target_label.unsqueeze(1), float('inf'))

        for outer_step in range(self.n_starts):
            

            # setup the modifier variable, this is the variable we are optimizing over
            modifier = torch.zeros_like(inputs, requires_grad=True)

            # setup the optimizer
            optimizer = optim.Adam([modifier], lr=self.attack_lr)
            best_l2 = torch.full((batch_size,), 1e10, device=self.device)
            best_score = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)

            # The last iteration (if we run many steps) repeat the search once.
            """if self.repeat and outer_step == (self.n_starts - 1):
                CONST = upper_bound"""

            prev = float('inf')
            for iteration in range(self.steps):
                # perform the adversary
                adv, logits, l2, logit_dists, loss = self._step(iteration, self.model, optimizer, inputs, tinputs, modifier,
                                                                target_label, labels_infhot, self.self.targeted, CONST)

                # check if we should abort search if we're getting nowhere.
                if self.abort_early and iteration % (self.steps // 10) == 0:
                    if logit_dists > prev:
                        break
                    prev = logit_dists

                # adjust the best result found so far
                predicted_classes = (self.model(self.model_trms(adv)) - labels_onehot * self.kappa).argmax(1) if self.self.targeted else \
                    (self.model(self.model_trms(adv)) + labels_onehot * self.kappa).argmax(1)

                is_adv = (predicted_classes == target_label) if self.self.targeted else (predicted_classes != labels)
                is_smaller = l2 < best_l2
                o_is_smaller = l2 < o_best_l2
                is_both = is_adv * is_smaller
                o_is_both = is_adv * o_is_smaller
                

                best_l2[is_both] = l2[is_both]
                best_score[is_both] = predicted_classes[is_both]
                o_best_l2[o_is_both] = l2[o_is_both]
                o_best_score[o_is_both] = predicted_classes[o_is_both]
                o_best_adversary[o_is_both] = adv[o_is_both]
                
            # adjust the constant as needed
            adv_found = (best_score == target_label) if self.self.targeted else ((best_score != labels) * (best_score != -1))
            upper_bound[adv_found] = torch.min(upper_bound[adv_found], CONST[adv_found])
            adv_not_found = ~adv_found
            lower_bound[adv_not_found] = torch.max(lower_bound[adv_not_found], CONST[adv_not_found])
            is_smaller = upper_bound < 1e9
            CONST[is_smaller] = (lower_bound[is_smaller] + upper_bound[is_smaller]) / 2
            CONST[(~is_smaller) * adv_not_found] *= 10

        # return the best solution found
        return (o_best_adversary, target_label)
    
    def set_target_mode(self, mode):
        if mode == 'least_likely':
            self.get_target = self.get_least_likely_label
        elif mode == 'most_likely':
            self.get_target = self.get_most_likely_label
        else:
            print('WARNING: set_target_mode was set to random. If unwanted, change "target_mode" arg to either "least_likely" or "most_likely".')
            self.get_target = self.get_random_target_label
    
    @torch.no_grad()
    def get_most_likely_label(self, inputs, labels=None):
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        outputs[:, labels] = -1e+03
        #_, target_labels = torch.max(torch.cat((outputs[:,:labels], outputs[:,labels+1:]), dim=1), dim=1)
        _, target_labels = torch.max(outputs, dim=1)
        return target_labels.long().to(self.device)

    @torch.no_grad()
    def get_least_likely_label(self, inputs, labels=None):
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            _, t = torch.kthvalue(outputs[counter][l], self._kth_min)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    @torch.no_grad()
    def get_random_target_label(self, inputs, labels=None):
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            t = (len(l)*torch.rand([1])).long().to(self.device)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    @torch.no_grad()
    def get_output_with_eval_nograd(self, inputs):
        given_training = self.model.training
        if given_training:
            self.model.eval()
        outputs = self.get_logits(self.model_trms(inputs))
        if given_training:
            self.model.train()
        return outputs
    
    def get_logits(self, inputs, labels=None, *args, **kwargs):
        logits = self.model(inputs)
        return logits
