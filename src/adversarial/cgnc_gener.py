
import torch
import torch
import torch.nn as nn
import math
import numpy as np

from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from src.adversarial.attack_base import Attack

class CGNCAttack(Attack):

    def __init__(self, device, arch, eps, l2_bound=-1, *args, **kwargs):
        """
        let 'arch' be res for ResNet152 or 'inc' for Inception
        """
        super().__init__('CGNC', *args, **kwargs)
        if arch == 'inc':
            pretrained = './src/adversarial/resources/CGNC_gen_res152.pth'
            uses_inception = True
        else:
            pretrained =  './src/adversarial/resources/CGNC_gen_res152.pth'
            uses_inception = False
        self.eps = eps
        self.device = device
        self.netG = self.load_from_file(pretrained, uses_inception=uses_inception)
        self.text_cond_dict = torch.load('./src/adversarial/resources/text_feature.pth', map_location=self.device)
        self.label_flag = 'N8' # params from original repo

        if l2_bound != -1:
            self.eps = self.get_eps_in_range(l2_bound)



    def load_from_file(self, pretrained, uses_inception):

        netG = CrossAttenGenerator(inception=uses_inception, nz=16) # params from original repo
        state_dict = torch.load(pretrained, map_location=self.device)
        netG.load_state_dict(state_dict=state_dict)
        netG.to(self.device)
        netG.eval()
        return netG

    def forward(self, images, labels):

        batch_size, _, _, _ = images.shape
        images = images.to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        class_ids = self.get_classes(self.label_flag) # set predefined set of target labels
        id = self.get_labels_from_cond_dict(class_ids, batch_size)
        cond = torch.tile(self.text_cond_dict[id], (batch_size, 1)).to(torch.float).to(self.device)
        noise = self.netG(images, cond, eps=self.eps)
        adv = noise + images

        # Projection, tanh() have been operated in models.
        adv = torch.min(torch.max(adv, images - self.eps), images + self.eps)
        adv = torch.clamp(adv, 0.0, 1.0)
        return adv

    def get_labels_from_cond_dict(self, class_ids, batch_size):
        """if batch_size == len(class_ids):
            labels = torch.tensor(class_ids)
        else:
            labels = torch.tensor(np.random.choice(class_ids, size=batch_size))"""
        
        return np.random.choice(class_ids, size=1).item()


    def get_classes(self, label_flag):
        if label_flag == 'N8':
            label_set = np.array([150, 426, 843, 715, 952, 507, 590, 62])
        elif label_flag == 'C20':
            label_set = np.array([4, 65, 70, 160, 249, 285, 334, 366, 394, 396, 458, 580, 593, 681, 815, 822, 849,
                                875, 964, 986])
        elif label_flag == 'C50':
            label_set = np.array([9, 71, 74, 86, 102, 141, 150, 181, 188, 223, 245, 275, 308, 332, 343, 352, 386,
                                405, 426, 430, 432, 450, 476, 501, 510, 521, 529, 546, 554, 567, 588, 597, 640,
                                643, 688, 712, 715, 729, 817, 830, 853, 876, 878, 883, 894, 906, 917, 919, 940,
                                988])
        elif label_flag == 'C100':
            label_set = np.array([6, 8, 31, 41, 43, 47, 48, 50, 56, 57, 66, 89, 93, 107, 121, 124, 130, 156, 159,
                                168, 170, 172, 178, 180, 202, 206, 214, 219, 220, 230, 248, 252, 269, 304, 323,
                                325, 339, 351, 353, 356, 368, 374, 379, 387, 395, 401, 435, 449, 453, 464, 472,
                                496, 504, 505, 509, 512, 527, 530, 542, 575, 577, 604, 636, 638, 647, 682, 683,
                                687, 704, 711, 713, 730, 733, 739, 746, 747, 763, 766, 774, 778, 783, 799, 809,
                                832, 843, 845, 846, 891, 895, 907, 930, 937, 946, 950, 961, 963, 972, 977, 984,
                                998])
        elif label_flag == 'C200':
            label_set = np.array([7, 12, 13, 14, 16, 22, 25, 36, 49, 58, 75, 84, 88, 104, 105, 112, 113, 114, 115,
                                117, 120, 134, 140, 143, 144, 155, 158, 165, 173, 182, 183, 194, 196, 200, 204,
                                207, 212, 218, 225, 231, 242, 244, 250, 261, 262, 266, 270, 277, 282, 288, 292,
                                297, 301, 310, 316, 320, 321, 327, 330, 348, 357, 359, 361, 365, 371, 375, 381,
                                382, 389, 407, 409, 411, 412, 413, 414, 418, 422, 436, 437, 445, 446, 448, 456,
                                461, 468, 470, 471, 474, 475, 480, 484, 486, 489, 491, 495, 500, 502, 506, 511,
                                514, 515, 526, 531, 535, 544, 547, 549, 561, 562, 566, 582, 591, 598, 603, 605,
                                610, 611, 612, 613, 616, 618, 619, 621, 627, 635, 641, 648, 653, 654, 656, 657,
                                658, 661, 662, 672, 673, 680, 686, 689, 691, 693, 697, 700, 705, 706, 707, 716,
                                725, 735, 743, 750, 752, 760, 768, 772, 776, 781, 790, 791, 796, 798, 800, 802,
                                811, 819, 823, 824, 828, 833, 834, 836, 848, 855, 874, 890, 893, 898, 903, 922,
                                923, 928, 931, 935, 936, 939, 943, 944, 945, 948, 955, 960, 967, 969, 970, 971,
                                980, 983, 990, 992, 999])
        else:
            raise ValueError
        return label_set
# To control feature map in generator
ngf = 64


def snconv2d(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(**kwargs), eps=eps)


def snlinear(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(**kwargs), eps=eps)


def get_gaussian_kernel(kernel_size=3, pad=2, sigma=1, channels=3):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                      kernel_size=kernel_size, groups=channels, padding=kernel_size - pad, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class CrossAttenGenerator(nn.Module):
    def __init__(self, inception=False, device='cuda', num_head=1, nz=16, loc=[1, 1, 1], context_dim=512):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 to 3x299x299.
        '''
        super(CrossAttenGenerator, self).__init__()
        self.inception = inception
        self.device = device
        self.snlinear = snlinear(in_features=512, out_features=nz, bias=False)
        self.loc = loc

        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3 + nz * self.loc[0], ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf + nz * self.loc[1], ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)

        )
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2 + nz * self.loc[2], ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        self.cross_att2 = SpatialTransformer(ngf * 4, num_head, ngf * 4 // num_head, depth=1, context_dim=context_dim)
        self.resblock3 = ResidualBlock(ngf * 4)
        self.resblock4 = ResidualBlock(ngf * 4)
        self.cross_att4 = SpatialTransformer(ngf * 4, num_head, ngf * 4 // num_head, depth=1, context_dim=context_dim)
        self.resblock5 = ResidualBlock(ngf * 4)
        self.resblock6 = ResidualBlock(ngf * 4)

        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )
        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)
        self.alf_layer = get_gaussian_kernel(kernel_size=3, pad=2, sigma=1)

    def forward(self, input, cond, eps=16):
        text_cond = cond.unsqueeze(1).to(torch.float)
        z_cond = self.snlinear(cond.float())

        # loc 0
        z_img = z_cond[:, :, None, None].expand(z_cond.size(0), z_cond.size(1), input.size(2), input.size(3))
        x = self.block1(torch.cat((input, z_img), dim=1))

        # loc 1
        z_img = z_cond[:, :, None, None].expand(z_cond.size(0), z_cond.size(1), x.size(2), x.size(3))
        x = self.block2(torch.cat((x, z_img), dim=1)) if self.loc[1] else self.block2(x)

        # loc 2
        z_img = z_cond[:, :, None, None].expand(z_cond.size(0), z_cond.size(1), x.size(2), x.size(3))
        x = self.block3(torch.cat((x, z_img), dim=1)) if self.loc[2] else self.block3(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.cross_att2(x, text_cond)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.cross_att4(x, text_cond)
        x = self.resblock5(x)
        x = self.resblock6(x)

        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)
        if self.inception:
            x = self.crop(x)
        x = torch.tanh(x)
        x = self.alf_layer(x)

        return x * eps


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual


########################### attention code ##########################


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
    
    'Cancel the checkpointing operation.'
    # def forward(self, x, context=None):
    #     return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in