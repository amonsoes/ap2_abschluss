import torch
import torch.nn as nn
import torch.optim as optim
import torch_dct as dct
import csv

from torchvision import transforms as T
from src.adversarial.attack_base import Attack
from diff_jpeg import diff_jpeg_coding


class Patchify:
    
    def __init__(self, img_size, patch_size, n_channels):
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2

        #assert (img_size // patch_size) * patch_size == img_size
        
    def __call__(self, x):
        p = x.unfold(1, 1, 1).unfold(2, 8, 8).unfold(3, 8, 8) # x.size() -> (batch, model_dim, n_patches, n_patches)
        self.unfold_shape = p.size()
        p = p.contiguous().view(p.size(0),-1,1,8,8)
        return p
    
    def inverse(self, p):
        if not hasattr(self, 'unfold_shape'):
            raise AttributeError('Patchify needs to be applied to a tensor in ordfer to revert the process.')
        x = p.view(self.unfold_shape)
        output_c = self.unfold_shape[1] * self.unfold_shape[4]
        output_h = self.unfold_shape[2] * self.unfold_shape[5]
        output_w = self.unfold_shape[3] * self.unfold_shape[6]
        x = x.permute(0,1,4,2,5,3,6).contiguous()
        x = x.view(1, output_c, output_h, output_w)
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
        self.n = 0
        self.tile_mean = 0
        
    def normalize_tensor_for_dct(self, tensor):
        tensor -= 0.5
        tensor /= 0.5
        return tensor

    def shift_tensor_for_image(self, tensor):
        tensor /= 2
        tensor += 0.5
        return tensor
        
    def __call__(self, tensor):
        tensor = self.normalize_tensor_for_dct(tensor)
        dct_patches = self.patched_dct(tensor)
        return dct_patches

    def inverse(self, dct_patches):
        image_tensor = torch.zeros_like(dct_patches)
        for e, patch in enumerate(dct_patches):
            image_patch = dct.idct_2d(patch, norm='ortho')
            image_tensor[e] = image_patch
        image_tensor = self.patchify.inverse(image_tensor)
        image_tensor = self.shift_tensor_for_image(image_tensor)
        return image_tensor
    
    def patched_dct(self, tensor):
        p = self.patchify(tensor)
        for e, patch in enumerate(p):
            dct_coeffs = dct.dct_2d(patch, norm='ortho')
            p[e] = dct_coeffs
        self.n += 1
        return p
    
    def calculate_fgsm_coeffs(self, patch):
        masked_patch = patch[self.mask == 1].abs()
        sum_patch = sum(masked_patch)
        return torch.full((8,8), fill_value=sum_patch.item())

class DCTFull:
    
    def __init__(self, img_size=224, n_channels=3, diagonal=0):
        """
        diagonal parameter will decide how much cosine components will be taken into consideration
        while calculating fgsm patches.
        """
        print('DCT class transforms on 3d tensors')
        self.normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.n = 0
        self.tile_mean = 0
        
    def __call__(self, tensor):
        dct_tensor = self.normal_dct(tensor)
        return dct_tensor
    
    def normalize_tensor_for_dct(self, tensor):
        tensor -= 0.5
        tensor /= 0.5
        return tensor

    def shift_tensor_for_image(self, tensor):
        tensor /= 2
        tensor += 0.5
        return tensor
    
    def inverse(self, dct_tensor):
        dct_tensor = dct_tensor.squeeze(0)
        image_tensor = torch.zeros_like(dct_tensor)
        for e, channel in enumerate(dct_tensor):
            image_pixels = dct.idct_2d(channel, norm='ortho')
            image_tensor[e] = image_pixels
        image_tensor = self.shift_tensor_for_image(image_tensor)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor

    def normal_dct(self, tensor):
        tensor = tensor.squeeze(0)
        tensor = self.normalize_tensor_for_dct(tensor)
        dct_tensor = torch.zeros_like(tensor)
        for e, channel in enumerate(tensor): # check if batch_size is first dim
            dct_coeffs = dct.dct_2d(channel, norm='ortho')
            dct_tensor[e] = dct_coeffs
        dct_tensor = dct_tensor.unsqueeze(0)
        return dct_tensor
    
    def calculate_fgsm_coeffs(self, patch):
        masked_patch = patch[self.mask == 1].abs()
        sum_patch = sum(masked_patch)
        return torch.full((8,8), fill_value=sum_patch.item())

class DCTCW(Attack):
    r"""
    Optimize over DCT coefficients of entire image. The optimization should reduce HF
    details until the image is adversarial
    This should prevent the JPEG algorithm from removing the attack info.
    """
    def __init__(self, 
                model, 
                model_trms, 
                c=1, 
                kappa=0, 
                steps=10000, 
                attack_lr=0.01, 
                write_protocol=False,
                protocol_file=False,
                n_starts=1,
                verbose_cw=False,
                target_mode='random',
                eps=0.04,
                dct_type='patched',
                N=1,
                use_ensembling=False):
        super().__init__("CW", model, model_trms)
        self.c = c
        self.original_c = self.c
        self.kappa = kappa
        self.steps = steps
        self.lr = attack_lr
        self.supported_mode = ['default', 'targeted']
        self.loss = nn.MSELoss(reduction='none') # will be used to get L2 dist in loss fn
        self.flatten = nn.Flatten()
        self.write_protocol = write_protocol
        self.protocol_file = protocol_file
        if self.protocol_file:
            self.protocol_dir = "/".join(self.protocol_file.split('/')[:-2]) + '/'
        self.eps = 0.001
        self.n_samples = n_starts
        self.use_attack_mask = False
        self.n = 0 # used for calculation cumulative avg of adjustment runtimes
        self.ca_runtime = 0 # cumulative average at n
        self.verbose_cw = verbose_cw
        self.forward_fn = self.forward if N <= 1 else self.forward_ensembling
        self.set_target_mode(target_mode)
        self.eps = eps # this is needed for a maximum perturbation estimation
        if dct_type == 'full':
            self.dct = DCTFull()
            self.lf_grad_fn = self.lf_grad_full
        elif dct_type == 'patched':
            self.dct = DCT()
            self.lf_grad_fn = self.lf_grad_patched
        self.phi_tensor = self.get_phi_tensor(224)
        self.beta = 1000
        self.N = N
        self.compression_rates = []
        for _ in range(N): # in 5-step decrements
            compression -= 5
            self.compression_rates.append(compression)
        self.main_comp_rate = self.compression_rates.pop(-1)
    
    def set_target_mode(self, mode):
        if mode == 'least_likely':
            self.set_mode_targeted_least_likely()
        elif mode == 'most_likely':
            self.set_mode_targeted_most_likely()
        else:
            print('WARNING: set_target_mode was set to random. If unwanted, change "target_mode" arg to either "least_likely" or "most_likely".')
            self.set_mode_targeted_random()

    def get_phi_tensor(self, image_size):
        """
        Get phi tensor that defines penalty magnitude
        for constrained optimization
        """
        phi_tensor = torch.ones((image_size, image_size))
        phi_values = torch.linspace(start=0.0, end=1.0, steps=image_size)
        for i in range(len(phi_values)):
            for p in range(i+1):
                #print(i-p, p)
                phi_tensor[i-p, p] = phi_values[i]
        phi_tensor = torch.flip(phi_tensor, dims=[0,1])
        return phi_tensor
        
    def forward(self, images, labels):
        best_adv_images_from_starts = self.forward_fn(images, labels)
        return best_adv_images_from_starts

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        print(f'\nCalculating adversarial example for images...\n\n')
        #images = images / 255
        self.c = self.original_c
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        # define momentum and gradient variant as tensors with 0

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        
        #implement outer step like in perccw

        # set the lower and upper bounds accordingly
        lower_bound = torch.zeros(1, device=self.device)
        CONST = torch.full((1,), self.c, device=self.device)
        upper_bound = torch.full((1,), 1e10, device=self.device)
        
        # set markers for inter-c-run comparison
        best_adv_images_from_starts = images.clone().detach()
        best_cost_from_starts = 1e10
        adv_found = torch.tensor([0])
    

        for outer_step in range(self.n_samples):    
            # search to get optimal c
            print(f'step in binary search for constant c NR:{outer_step}, c:{self.c}\n')
        
            # w = torch.zeros_like(images).detach() # Requires 2x times    
            w = self.dct(images.clone().detach()).clone().detach()
            
            #w_hat = w[:,:50,:50].clone().detach()
            w.requires_grad = True

            best_adv_images = images.clone().detach()
            #best_iq = 1e10*torch.ones((len(images))).to(self.device)
            best_cost = 1e10*torch.ones((len(images))).to(self.device)
            dim = len(images.shape)

            self.optimizer = optim.Adam([w], lr=self.lr)
            self.optimizer.zero_grad()

            for step in range(self.steps):
                    
                adv_images = self.dct.inverse(w)

                # Calculate image quality loss
                iq_loss, current_iq_loss = self.get_iq_loss(adv_images, images, None)
                
                # Calculate adversarial loss including robustness loss
                self.optimizer.zero_grad()
                adv_loss, outputs = self.get_adversarial_loss(adv_images, labels, target_labels)

                reg_penalization = self.get_reg_penalization(adv_images)
                cost = iq_loss + self.c*adv_loss + self.beta*reg_penalization
                
                # Update adversarial images
                self.optimizer.zero_grad()
                cost.backward()
                
                w.grad = self.lf_grad(w.grad)
                
                self.optimizer.step()

                _, pre = torch.max(outputs.detach(), 1)
                is_adversarial = (pre == target_labels).float()            
                is_lower_in_cost = (best_cost > cost.detach())
                
                mask =  is_adversarial * is_lower_in_cost
                #mask = mask.view([-1]+[1]*(dim-1))

                best_cost = mask*cost.detach() + (1-mask)*best_cost
                best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images
                # either the current adv_images is the new best_adv_images or the old one depending on mask
                
                print(f'\n{step} - iq_loss: {current_iq_loss.item()}, f_loss: {adv_loss.item()} cost: {cost.item()}')  

            # set the best output of run as best adv if (1) an adv was found (2) the cost is lower than the one from the last starts
            adv_found_in_run = torch.any(best_adv_images != images) # could only be different if one output was adv during optim
            is_lower_than_all_starts = (best_cost_from_starts > best_cost).float()
            mask = adv_found_in_run * is_lower_than_all_starts
            best_adv_images_from_starts = mask*best_adv_images + (1-mask)*best_adv_images_from_starts
            best_cost_from_starts = mask*best_cost + (1-mask)*best_cost_from_starts

            # adjust the constant as needed
            adv_found = adv_found_in_run
            upper_bound[adv_found] = torch.min(upper_bound[adv_found], CONST[adv_found])
            adv_not_found = ~adv_found
            lower_bound[adv_not_found] = torch.max(lower_bound[adv_not_found], CONST[adv_not_found])
            is_smaller = upper_bound < 1e9
            CONST[is_smaller] = (lower_bound[is_smaller] + upper_bound[is_smaller]) / 2
            CONST[(~is_smaller) * adv_not_found] *= 10
            self.c = CONST.item()
        
            # from perccw.py
            ######################   
                            
        if self.protocol_file:
            self.write_to_protocol_dir(iq_loss, adv_loss, cost)
            self.write_runtime_to_protocol_dir()
        best_adv_images_from_starts = torch.clip(best_adv_images_from_starts, min=0.0, max=1.0)
        return (best_adv_images_from_starts, target_labels)

    def compress(self, images, compression_rate):
        img = images * 255
        compressed = diff_jpeg_coding(image_rgb=img, jpeg_quality=torch.tensor([compression_rate]).to(self.device))
        compressed_img = (compressed / 255).clip(min=0., max=1.)
        return compressed_img
        


    def forward_ensembling(self, images, labels):
        r"""
        Overridden.
        """
        print(f'\nCalculating adversarial example for images...\n\n')
        #images = images / 255
        self.c = self.original_c
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        # define momentum and gradient variant as tensors with 0

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        
        #implement outer step like in perccw

        # set the lower and upper bounds accordingly
        lower_bound = torch.zeros(1, device=self.device)
        CONST = torch.full((1,), self.c, device=self.device)
        upper_bound = torch.full((1,), 1e10, device=self.device)
        
        # set markers for inter-c-run comparison
        best_adv_images_from_starts = images.clone().detach()
        best_cost_from_starts = 1e10
        adv_found = torch.tensor([0])
    

        for outer_step in range(self.n_samples):    
            # search to get optimal c
            print(f'step in binary search for constant c NR:{outer_step}, c:{self.c}\n')
        
            # w = torch.zeros_like(images).detach() # Requires 2x times    
            w = self.dct(images.clone().detach()).clone().detach()
            
            # w_hat = w[:,:50,:50].clone().detach()
            w.requires_grad = True

            best_adv_images = images.clone().detach()
            best_cost = 1e10*torch.ones((len(images))).to(self.device)
            dim = len(images.shape)

            self.optimizer = optim.Adam([w], lr=self.lr)
            self.optimizer.zero_grad()

            for step in range(self.steps):
                    
                adv_images = self.dct.inverse(w)

                # Calculate image quality loss including domain regularization
                iq_loss, current_iq_loss = self.get_iq_loss(adv_images, images, None)
                reg_penalization = self.get_reg_penalization(adv_images)            
                
                # Calculate adversarial loss including robustness loss with ensembling
                ensemble_grad = torch.zeros_like(adv_images).detach().to(self.device)    
                grad_list = []
                cost_tensor = torch.zeros(self.N_comp_rates)
                for e, compr_rate in enumerate(self.compression_rates):
                    adv_images_i = adv_images.clone().detach()
                    adv_images_i.requires_grad = True
                    comp_images = self.compress(adv_images_i, compr_rate)
                    adv_loss, outputs = self.get_adversarial_loss(comp_images, labels, target_labels)
                    cost = iq_loss + self.c*adv_loss + self.beta*reg_penalization
                    cost_tensor[e] = cost
                    
                    grad = torch.autograd.grad(cost, adv_images_i,
                                            retain_graph=False,
                                            create_graph=False)[0]
                
                    grad_list.append(grad) 

                
                # make backward call
                self.optimizer.zero_grad()
                cost.backward()
                
                # restrict gradient and update
                w.grad = self.lf_grad(w.grad)
                self.optimizer.step()
                
                # evaluate current state of adv images
                _, pre = torch.max(outputs.detach(), 1)
                is_adversarial = (pre == target_labels).float()            
                is_lower_in_cost = (best_cost > cost.detach())
                
                mask =  is_adversarial * is_lower_in_cost
                #mask = mask.view([-1]+[1]*(dim-1))

                best_cost = mask*cost.detach() + (1-mask)*best_cost
                best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images
                # either the current adv_images is the new best_adv_images or the old one depending on mask
                
                print(f'\n{step} - iq_loss: {current_iq_loss.item()}, f_loss: {adv_loss.item()} cost: {cost.item()}')  

            # set the best output of run as best adv if (1) an adv was found (2) the cost is lower than the one from the last starts
            adv_found_in_run = torch.any(best_adv_images != images) # could only be different if one output was adv during optim
            is_lower_than_all_starts = (best_cost_from_starts > best_cost).float()
            mask = adv_found_in_run * is_lower_than_all_starts
            best_adv_images_from_starts = mask*best_adv_images + (1-mask)*best_adv_images_from_starts
            best_cost_from_starts = mask*best_cost + (1-mask)*best_cost_from_starts

            # adjust the constant as needed
            adv_found = adv_found_in_run
            upper_bound[adv_found] = torch.min(upper_bound[adv_found], CONST[adv_found])
            adv_not_found = ~adv_found
            lower_bound[adv_not_found] = torch.max(lower_bound[adv_not_found], CONST[adv_not_found])
            is_smaller = upper_bound < 1e9
            CONST[is_smaller] = (lower_bound[is_smaller] + upper_bound[is_smaller]) / 2
            CONST[(~is_smaller) * adv_not_found] *= 10
            self.c = CONST.item()
        
            # from perccw.py
            ######################   
                            
        if self.protocol_file:
            self.write_to_protocol_dir(iq_loss, adv_loss, cost)
            self.write_runtime_to_protocol_dir()
        best_adv_images_from_starts = torch.clip(best_adv_images_from_starts, min=0.0, max=1.0)
        return (best_adv_images_from_starts, target_labels)

    def lf_grad(self, grad):
        grad = self.lf_grad_fn(grad)
        return grad
    
    def lf_grad_full(self, grad):
        w_hat_grad = grad
        grad_new = torch.zeros_like(w_hat_grad)
        grad_new[:,:50,:50] = w_hat_grad[:,:50,:50]
        return grad_new

    def lf_grad_patched(self, grad):
        w_hat_grad = grad
        grad_new = torch.zeros_like(w_hat_grad)
        grad_new[:,:,:,:3,:3] = w_hat_grad[:,:,:,:3,:3]
        return grad_new
    
    def get_reg_penalization(self, adv_images):
        # 1 x where x > 1.0
        # 2 x where x < 0.0
        # return 1 + 2
        vals_above = (adv_images[adv_images > 1.0] - 1).sum()
        vals_below = (adv_images[adv_images < 0.0].abs()).sum()
        return vals_above + vals_below

    def get_adversarial_loss(self, adv_images, labels, target_labels):
        outputs = self.get_outputs(adv_images)
        if self.targeted:
            f_loss = self.f(outputs, target_labels).sum()
        else:
            f_loss = self.f(outputs, labels).sum()
        return f_loss, outputs

    def get_outputs(self, adv_images):
        return self.get_logits(self.model_trms(adv_images))

    def get_rand_points_around_sample(self, sample_list):
        sample = sample_list[0]
        for _ in range(self.n_samples-1):
            neighbor_image = torch.clamp((sample.detach() + torch.randn_like(sample).uniform_(-self.eps, self.eps)), min=0.0, max=1.0).to(self.device)
            neighbor_image.requires_grad = True
            sample_list.append(neighbor_image)
        return sample_list
    
    def write_to_protocol_dir(self, iq_loss, f_loss, cost):
            with open(self.protocol_file, 'a') as report_file:
                report_obj = csv.writer(report_file)
                report_obj.writerow(['1', str(iq_loss.item()), str(f_loss.item()), str(cost.item())])
    
    def write_runtime_to_protocol_dir(self):
        with open(self.protocol_dir + 'runtimes.csv', 'a') as runtimes_file:
            runtimes_obj = csv.writer(runtimes_file)
            runtimes_obj.writerow([self.ca_runtime])
    
    # this should be overwritten by subclass custom CW
    def get_iq_loss(self, adv_images, images, attack_mask):
        current_iq_loss = self.loss(self.flatten(adv_images), self.flatten(images)).sum(dim=1)
        iq_loss = current_iq_loss.sum()
        return iq_loss, current_iq_loss

    def tanh_space(self, x):
        return 1/2*(torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x*2-1)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]), device=self.device)[labels].to(self.device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the second largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool()) # get the largest logit

        if self.targeted:
            return torch.clamp(( i - j + self.kappa), min=0) # to mimic perccw's inclusion of kappa
            #return torch.clamp((i-j), min=-self.kappa)
        else:
            return torch.clamp((j-i), min=-self.kappa)
        
    def bin_search_c(self, images, labels, low=1e-01, high=5, steps=30, iters=20):
        print(f'\ninitialize binary search for c in range {low} to {high} in steps {steps}.\n')
        # get random starting point c
        c_values = torch.linspace(low, high, steps)
        best_c = c_values[0]
        best_loss = 9e+04
        for i in range(iters):
            print(f'\n...initializing iteration {i+1}...\n')
            self.c = c_values[i]
            best_adv_img = self(images, labels)
            if self.loss < best_loss:
                # self.loss will change in the forward fn to the one obtained in current iter
                best_c = c_values[i]
                best_loss = self.loss
        return best_c, best_loss
    
    def update_ca_runtime(self, new_value):
        new_n = self.n + 1
        ca_runtime_new = (new_value + (self.n * self.ca_runtime)) / new_n
        self.n = new_n
        self.ca_runtime = ca_runtime_new
        