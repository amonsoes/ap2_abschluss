
import random
import numpy as np
import torch


from scipy.fftpack import dct, idct
from src.adversarial.attack_base import Attack



class PPBA(Attack):

    def __init__(self,
                model, 
                model_trms, 
                low_dim=1500, 
                freq_dim=28, 
                stride=7,
                mom=1,
                order='strided',
                r=2352,
                n_queries=2000,
                rho=0.05,
                l2_bound=-1,
                kappa=0,
                *args, 
                **kwargs):
        super().__init__('PPBA', model, model_trms, *args, **kwargs)
        self.low_dim=low_dim
        self.freq_dim=freq_dim 
        self.stride=stride
        self.mom=mom
        self.order=order
        self.r=r
        self.max_iter=n_queries
        self.rho=rho
        self.total_queries = []
        self.total_success = []
        self.total_l2 = 0
        self.kappa = kappa
        self.l2_bound = l2_bound
        self.l2_bound = 0 if l2_bound == -1 else l2_bound
        self.low, self.high = self.get_iq_range_from_l2_bound(l2_bound)
        self.low = self.get_l2_from_bounds(l2_bound)

        if self.low == self.high:
            self.low -= self.low*0.1
            self.high += self.high*0.1

        image_size = 299 if self.model.model_name == 'inception' else 224
        self.image_size = image_size

        if self.model.model_name == "inception":
            self.freq_dim = 38
            self.stride = 9

        if self.order == "strided":
            Random_Matrix = np.zeros((self.low_dim, 3 * image_size * image_size))
            indices = block_order(image_size, 3, initial_size=self.freq_dim, stride=self.stride)
        else:
            Random_Matrix = np.zeros((self.low_dim, 3 * self.freq_dim * self.freq_dim))
            indices = random.sample(range(self.r), self.low_dim)
        for i in range(self.low_dim):
            Random_Matrix[i][indices[i]] = 1

        if self.order == "strided":
            Random_Matrix = torch.from_numpy(Random_Matrix).view(-1, 3, image_size, image_size)
        else:
            Random_Matrix = torch.from_numpy(Random_Matrix).view(-1, 3, self.freq_dim, self.freq_dim)


        self.Random_Matrix = (
            block_idct(
                self.expand_vector(
                    Random_Matrix, size=(image_size if self.order == "strided" else self.freq_dim),
                ),
                block_size=image_size,
            )
            .view(self.low_dim, -1)
            .to(self.device)
        )

    def forward(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        
        n_samples = x.shape[0]
        N_query = 0
        z = np.zeros((1, self.low_dim))
        prev_f, l2_scores = self.get_loss(z, x, y, N_query)
        prev_f = prev_f[0]
        is_success = 0 if prev_f > 0 else 1
        effective_number = [
            np.ones((1, self.low_dim)),
            np.ones((1, self.low_dim)),
            np.ones((1, self.low_dim)),
        ]
        ineffective_number = [
            np.ones((1, self.low_dim)),
            np.ones((1, self.low_dim)),
            np.ones((1, self.low_dim)),
        ]

        for k in range(self.max_iter):
            print(f'iter:{k}, current_loss:{prev_f}, current_l2:{l2_scores}')
            u = np.zeros((n_samples, self.low_dim))
            r = np.random.uniform(size=(n_samples, self.low_dim))
            effective_probability = [
                effective_number[i] / (effective_number[i] + ineffective_number[i])
                for i in range(len(effective_number))
            ]
            probability = PPBA.get_probability(effective_probability)

            # u[r < probability[0] + probability[1]] = 0
            u[r < probability[0]] = -1
            u[r >= probability[0] + probability[1]] = 1

            uz = z + self.rho * u

            uz_l2 = np.linalg.norm(uz, axis=1)
            uz = uz * np.minimum(1, self.l2_bound / uz_l2).reshape(-1, 1)

            fu, l2_scores = self.get_loss(uz, x, y, N_query)
            if fu.min() < prev_f:
                worked_u = u[fu < prev_f] # TODO: some issue here
                effective_probability[0] = effective_probability[0] * self.mom + (
                    worked_u == -1
                ).sum(0)
                effective_probability[1] = effective_probability[1] * self.mom + (
                    worked_u == 0
                ).sum(0)
                effective_probability[2] = effective_probability[2] * self.mom + (
                    worked_u == 1
                ).sum(0)
                not_worked_u = u[fu >= prev_f]
                ineffective_number[0] = ineffective_number[0] * self.mom + (
                    not_worked_u == -1
                ).sum(0)
                ineffective_number[1] = ineffective_number[1] * self.mom + (
                    not_worked_u == 0
                ).sum(0)
                ineffective_number[2] = ineffective_number[2] * self.mom + (
                    not_worked_u == 1
                ).sum(0)
                z = uz[np.argmin(fu)]
                prev_f = fu.max()
            else:
                ineffective_number[0] += (u == -1).sum(0)
                ineffective_number[1] += (u == 0).sum(0)
                ineffective_number[2] += (u == 1).sum(0)

            if prev_f <= 0:
                is_success = 1

            if self.l2_bound != 0:
                if torch.all(l2_scores > self.low):
                    self.kappa = 0
                is_success = (prev_f - self.kappa) <= 0.0
                print(f'samples are adversarial:{is_success}')
                if torch.all(l2_scores >= self.low) and torch.all(l2_scores <= self.high) and is_success:
                    break
        
        current_q = k
        N_query = 0

        z = torch.from_numpy(z).float().to(self.device)
        perturbation = (z @ self.Random_Matrix).view(1, 3, self.image_size, self.image_size)
        new_image = (x + perturbation).clamp(0, 1)
        if current_q > self.max_iter:
            is_success = 0
        print(current_q, is_success, perturbation.view(1, -1).norm(2, 1).item())

        self.total_queries.append(current_q)
        self.total_success.append(is_success)
        self.total_l2 += perturbation.view(1, -1).norm(2, 1).item()
        return torch.clamp(new_image, min=0.0, max=1.0), [current_q]

    def get_l2(self, adv_images, images):
        l2_scores = (adv_images - images).pow(2).sum(dim=(1,2,3)).sqrt()
        return l2_scores

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

    def get_random_l2_from_bounds(self, l2_bound):
        low, high = self.get_iq_range_from_l2_bound(l2_bound)
        rand_l2_bound = random.uniform(low, high)
        return rand_l2_bound

    def cw_loss(self, x, y, N_query, targeted=False):
        N_query += x.shape[0]
        outputs = self.model(self.model_trms(x))
        one_hot_labels = torch.eye(len(outputs[0])).to(self.device)[y]

        i, ind_i = torch.max((1 - one_hot_labels) * outputs, dim=1) # max logit that is not the actual label
        j, ind_j = torch.max((one_hot_labels) * outputs, dim=1) # either gt or target logit

        if targeted:
            return torch.clamp((i - j) + self.kappa, min=0)

        else:
            return torch.clamp((j - i) + self.kappa, min=0)

    def get_loss(self, zs, x, y, N_query):
        z = torch.from_numpy(zs).float().to(self.device).view(-1, self.low_dim)
        perturbation = (z @ self.Random_Matrix).view(z.shape[0], 3, self.image_size, self.image_size)
        new_image = (x + perturbation).clamp(0, 1)
        l2 = self.get_l2(new_image, x)
        loss = self.cw_loss(new_image, y, N_query, targeted=self.targeted)
        loss = loss.cpu().numpy()
        return loss, l2
    
    def check_if_adv(self, x, y, N_query, targeted):
        N_query += x.shape[0]
        outputs = self.model(self.model_trms(x))
        one_hot_labels = torch.eye(len(outputs[0]))[y].to(self.device)

        i, ind_i = torch.max((1 - one_hot_labels) * outputs, dim=1) # max logit that is not the actual label
        j, ind_j = torch.max((one_hot_labels) * outputs, dim=1) # max logit

        if targeted:
            return torch.clamp(i - j, min=0)

        else:
            return torch.clamp(j - i, min=0)

    def get_probability(success_probability):
        probability = [v / sum(success_probability) for v in success_probability]
        return probability

    def expand_vector(self, x, size):
        batch_size = x.size(0)
        x = x.view(-1, 3, size, size)
        z = torch.zeros(batch_size, 3, self.image_size, self.image_size)
        z[:, :, :size, :size] = x
        return z


########### utils #############

# defines a block order, starting with top-left (initial_size x initial_size) submatrix
# expanding by stride rows and columns whenever exhausted
# randomized within the block and across channels
# e.g. (initial_size=2, stride=1)
# [1, 3, 6]
# [2, 4, 9]
# [5, 7, 8]

def block_order(image_size, channels, initial_size=1, stride=1):
    order = torch.zeros(channels, image_size, image_size)
    total_elems = channels * initial_size * initial_size
    perm = torch.randperm(total_elems)
    order[:, :initial_size, :initial_size] = perm.view(
        channels, initial_size, initial_size
    )
    for i in range(initial_size, image_size, stride):
        num_elems = channels * (2 * stride * i + stride * stride)
        perm = torch.randperm(num_elems) + total_elems
        num_first = channels * stride * (stride + i)
        order[:, : (i + stride), i : (i + stride)] = perm[:num_first].view(
            channels, -1, stride
        )
        order[:, i : (i + stride), :i] = perm[num_first:].view(channels, stride, -1)
        total_elems += num_elems
    return order.view(1, -1).squeeze().long().sort()[1]



# applies DCT to each block of size block_size
def block_dct(x, block_size=8, masked=False, ratio=0.5):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
    mask[:, :, : int(block_size * ratio), : int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = x[
                :,
                :,
                (i * block_size) : ((i + 1) * block_size),
                (j * block_size) : ((j + 1) * block_size),
            ].numpy()
            submat_dct = dct(dct(submat, axis=2, norm="ortho"), axis=3, norm="ortho")
            if masked:
                submat_dct = submat_dct * mask
            submat_dct = torch.from_numpy(submat_dct)
            z[
                :,
                :,
                (i * block_size) : ((i + 1) * block_size),
                (j * block_size) : ((j + 1) * block_size),
            ] = submat_dct
    return z


# applies IDCT to each block of size block_size
def block_idct(x, block_size=8, masked=False, ratio=0.5):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
    if type(ratio) != float:
        for i in range(x.size(0)):
            mask[i, :, : int(block_size * ratio[i]), : int(block_size * ratio[i])] = 1
    else:
        mask[:, :, : int(block_size * ratio), : int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = x[
                :,
                :,
                (i * block_size) : ((i + 1) * block_size),
                (j * block_size) : ((j + 1) * block_size),
            ].numpy()
            if masked:
                submat = submat * mask
            z[
                :,
                :,
                (i * block_size) : ((i + 1) * block_size),
                (j * block_size) : ((j + 1) * block_size),
            ] = torch.from_numpy(
                idct(idct(submat, axis=3, norm="ortho"), axis=2, norm="ortho")
            )
    return z