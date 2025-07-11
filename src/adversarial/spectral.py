import torch
import pickle
from tqdm import tqdm
import torch_dct as dct
import torchvision.transforms as T

class Fourier:
    
    def __init__(self, greyscale_processing):
        # n_channels defines the number of output channels after the adversarial attack
        self.greyscale_processing = greyscale_processing
    
    def to_magn_phase(self, img):
        f = torch.fft.fftn(img)
        fshift = torch.fft.fftshift(f)
        angle = fshift.angle()
        magn = fshift.abs()
        return magn, angle
    
    @staticmethod
    def to_img(magn, phase):
        f_img = magn*torch.exp(phase*1j)
        i_fshift = torch.fft.ifftshift(f_img)
        img = torch.fft.ifftn(i_fshift)
        img = torch.real(img)
        return img
    

class SpectrumNorm:
    
    def __init__(self, num_features, path_power_dict, path_delta, greyscale_processing, dataset_type, is_adv=True, img_size=224):
        self.greyscale_processing = greyscale_processing
        self.n_channels = 1 if greyscale_processing else 3
        if is_adv:
            self.process = self.process_adv
        else:
            self.process = self.process_training
        self.spectrum_difference = SpectrumDifference(greyscale_processing=greyscale_processing,
                                                    path=path_delta, 
                                                    dataset_type=dataset_type)
        self.fourier = Fourier(greyscale_processing=False)
        self.power_corrector = PowerDictCorrection(greyscale_processing=greyscale_processing,
                                                   path=path_power_dict,
                                                   num_features=num_features, 
                                                   img_size=img_size,
                                                   dataset_type=dataset_type)

    def __call__(self, img):
        img = self.process(img)
        return img

    def process_adv(self, img):
        magn, phase = self.fourier.to_magn_phase(img)
        magn = self.spectrum_difference(magn)
        closest_d = self.power_corrector.get_closest_feature(magn)
        magn = self.power_corrector(magn, closest_d)
        img = self.fourier.to_img(magn, phase)
        img = img.int()
        return img
    
    def process_training(self, magn):
        magn = self.spectrum_difference(magn)
        closest_d = self.power_corrector.get_closest_feature(magn)
        magn = self.power_corrector(magn, closest_d)
        return magn

class SpectrumDifference:

    def __init__(self, greyscale_processing, dataset_type, path=''):
        self.dataset_type = dataset_type
        self.greyscale_processing = greyscale_processing
        if path:
            self.delta = self.load_delta(path)
        self.fourier = Fourier(greyscale_processing)
        
    def __call__(self, magn):
        return magn - self.delta

    def load_delta(self, path):
        if self.greyscale_processing:
            path += 'delta_grey'
        else:
            path += 'delta'
        path = path + '_' + self.dataset_type + '.pickle'
        with open(path, 'rb') as f:
            delta = pickle.load(f)
        return delta
    
    def get_delta(self, datasets, save_path='./src/adversarial/adv_resources/'):
        print('\ncreating delta...\n')
        save_path += 'delta'
        save_path = save_path + '_grey' if self.greyscale_processing else save_path + ''
        save_path = save_path + '_' + self.dataset_type + '.pickle'
        s_mean_real = torch.zeros((1, 224, 224)) if self.greyscale_processing else torch.zeros((3, 224, 224))
        s_mean_syn = torch.zeros((1, 224, 224)) if self.greyscale_processing else torch.zeros((3, 224, 224))
        n_real, n_syn = 0, 0
        for data in datasets:
            for img, y in tqdm(data):
                
                magn, _ = self.fourier.to_magn_phase(img)
                
                magn_real = magn[y==1]
                n_real += len(magn_real)
                magn_syn = magn[y==0]
                n_syn += len(magn_syn)
                
                magn_real = torch.sum(magn_real, axis=0)
                magn_syn = torch.sum(magn_syn, axis=0)
                
                s_mean_real += magn_real
                s_mean_syn += magn_syn
                
        s_mean_real = s_mean_real / n_real
        s_mean_syn = s_mean_syn / n_syn
        delta = s_mean_syn - s_mean_real
        with open(save_path, 'wb') as f:
            pickle.dump(delta, f)
        print(f'\n\nsaved delta at{save_path}\n')
        print('\ndone\n')
        return delta

    def get_delta(self, datasets, save_path='./src/adversarial/adv_resources/'):
        print('\ncreating delta...\n')
        save_path += 'delta'
        save_path = save_path + '_grey' if self.greyscale_processing else save_path + ''
        save_path = save_path + '_' + self.dataset_type + '.pickle'
        s_mean_real = torch.zeros((1, 224, 224)) if self.greyscale_processing else torch.zeros((3, 224, 224))
        s_mean_syn = torch.zeros((1, 224, 224)) if self.greyscale_processing else torch.zeros((3, 224, 224))
        n_real, n_syn = 0, 0
        for data in datasets:
            for batch, y in tqdm(data):
                for img, label in zip(batch, y):
                    magn, _ = self.fourier.to_magn_phase(img)
                    if label == 0:
                        n_real += 1
                        s_mean_real += magn
                    else:
                        n_syn += 1
                        s_mean_syn += magn
        s_mean_real = s_mean_real / n_real
        s_mean_syn = s_mean_syn / n_syn
        delta = s_mean_syn - s_mean_real
        with open(save_path, 'wb') as f:
            pickle.dump(delta, f)
        print(f'\n\nsaved delta at{save_path}\n')
        print('\ndone\n')
        return delta    


class PowerDictCorrection:

    def __init__(self, greyscale_processing, dataset_type, path='', num_features=40, img_size=224):
        self.dataset_type = dataset_type
        self.greyscale_processing = greyscale_processing
        self.img_size = img_size
        self.hms = img_size / 2
        self.index_mat = self.build_index_mat(1, img_size) if self.greyscale_processing else self.build_index_mat(3, img_size)
        self.fourier = Fourier(greyscale_processing)
        if path:
            self.power_dict = self.load_power_dict(path)
            self.power_dict_trunc = self.power_dict[:, :self.img_size//4]
    
    """def __call__(self, magn, closest_d, e):
        power_fimg = self.spectral_power_dist(magn)
        for p, i in enumerate(self.index_mat):
            for k, j in enumerate(i):
                magn[p , k] = magn[p, k]*(closest_d[e, j]/power_fimg[j])
        return magn
    
    def get_closest_feature(self, magn):
        power_fimg = torch.zeros(1 ,(self.img_size)) if self.greyscale_processing else torch.zeros(3 ,(self.img_size))
        for e, channel in enumerate(magn):
            power_fimg[e] = self.spectral_power_dist(channel)
        power_fimg = power_fimg[:, :self.img_size//4]
        distances = [self.compute_feature_sim(d_trunc, power_fimg, e) for e, d_trunc in enumerate(self.power_dict_trunc)]
        index = sorted(distances, key=lambda x: x[1])[0][0]
        closest_d = self.power_dict[index]
        return closest_d
    
    def spectral_power_dist(self, channel):
        feature_l = torch.zeros((self.img_size))
        for i in range(self.img_size):
            feature_l[i] += torch.sum(channel[self.index_mat == i])
        return feature_l / feature_l[0]
        
    """

    def __call__(self, magn, closest_d):
        power_fimg = self.spectral_power_dist(magn)
        for e, chan in enumerate(self.index_mat):
            for p, i in enumerate(chan):
                for k, j in enumerate(i):
                    magn[e, p , k] = magn[e, p, k]*(closest_d[j]/power_fimg[j])
        return magn

    def spectral_power_dist(self, magn):
        feature_l = torch.zeros((self.img_size))
        for i in range(self.img_size):
            feature_l[i] += torch.sum(magn[self.index_mat == i])
        return feature_l / feature_l[0]    
    
    def get_closest_feature(self, magn):
        power_fimg = self.spectral_power_dist(magn)
        power_fimg = power_fimg[:self.img_size//4]
        distances = [self.compute_feature_sim(d_trunc, power_fimg, e) for e, d_trunc in enumerate(self.power_dict_trunc)]
        index = sorted(distances, key=lambda x: x[1])[0][0]
        closest_d = self.power_dict[index]
        return closest_d
    
    def compute_feature_sim(self, d_trunc, power_fimg_trunc, e):
        return e, torch.sum((power_fimg_trunc-d_trunc)**2)
        
    def build_index_mat(self, channels, img_size):
        m_i = torch.tensor([[i for _ in range(img_size)]for i in range(img_size)])
        m_j = torch.tensor([[x for x in range(img_size)] for _ in range(img_size)])
        index_mat_channel = torch.floor(torch.sqrt((m_i-self.hms)**2+(m_j - self.hms)**2)).long()
        return index_mat_channel.repeat(channels, 1, 1)
    
    def create_power_dict(self, data_test, num_features, save_path='./src/adversarial/adv_resources/'):
        print('\ncreating power dictionary...\n')
        save_path += 'power_dict'
        save_path = save_path + '_grey' if self.greyscale_processing else save_path + ''
        save_path = save_path + '_' + self.dataset_type + '.pickle'
        ls = torch.ones((num_features, self.img_size))
        data_iter = iter(data_test)
        added = 0
        while added < num_features-1:
            imgs, y = next(data_iter)
            imgs_real = imgs[y == 1]
            for img in imgs_real:
                if added >= num_features:
                    break
                magn, _ = self.fourier.to_magn_phase(img)
                feature_l = self.spectral_power_dist(magn)
                ls[added] = feature_l
                added += 1
        with open(save_path, 'wb') as f:
            pickle.dump(ls, f)
        print(f'\n\nsaved dictionary at{save_path}\n')
        print('\ndone\n')
        return torch.tensor(ls)

    def load_power_dict(self, path):
        if self.greyscale_processing:
                path += 'power_dict_grey'
        else:
            path += 'power_dict'
        path = path + '_' + self.dataset_type + '.pickle'
            
        with open(path, 'rb') as f:
            power_dict = pickle.load(f)
        return power_dict
    

class Patchify:
    
    def __init__(self, img_size, patch_size, n_channels):
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2

        assert (img_size // patch_size) * patch_size == img_size
        
    
    def __call__(self, x):
        p = x.unfold(1, 8, 8).unfold(2, 8, 8).unfold(3, 8, 8) # x.size() -> (batch, model_dim, n_patches, n_patches)
        self.unfold_shape = p.size()
        p = p.contiguous().view(-1,8,8)
        return p
    
    def inverse(self, p):
        if not hasattr(self, 'unfold_shape'):
            raise AttributeError('Patchify needs to be applied to a tensor in ordfer to revert the process.')
        x = p.view(self.unfold_shape)
        output_h = self.unfold_shape[1] * self.unfold_shape[4]
        output_w = self.unfold_shape[2] * self.unfold_shape[5]
        x = x.permute(0,1,4,2,5,3).contiguous()
        x = x.view(3, output_h, output_w)
        return x 

class DCT:
    
    def __init__(self, img_size=224, patch_size=8, n_channels=3, diagonal=0):
        """
        diagonal parameter will decide how much cosine components will be taken into consideration
        while calculating fgsm patches.
        """
        print('DCT class transforms on 3d tensors')
        self.patchify = Patchify(img_size=img_size, patch_size=patch_size, n_channels=n_channels)
        self.mask = torch.flip(torch.triu(torch.ones((8,8)), diagonal=diagonal), dims=[0])
        
    def __call__(self, tensor):
        p, fgsm_coeffs = self.patched_dct(tensor)
        dct_coeffs = self.patchify.inverse(p)
        fgsm_coeffs = self.patchify.inverse(fgsm_coeffs)
        fgsm_coeffs = fgsm_coeffs / fgsm_coeffs.max()
        return dct_coeffs, fgsm_coeffs
    
    def patched_dct(self, tensor):
        p = self.patchify(tensor)
        fgsm_coeff_tensor = torch.zeros(p.shape, dtype=torch.float32)
        for e, patch in enumerate(p):
            
            dct_coeffs = dct.dct_2d(patch, norm='ortho')
            dct_coeffs[0][0] = 0.0
            fgsm_coeffs = self.calculate_fgsm_coeffs(dct_coeffs)
            fgsm_coeff_tensor[e] = fgsm_coeffs
            p[e] = dct_coeffs
        return p, fgsm_coeff_tensor
    
    def calculate_fgsm_coeffs(self, patch):
        sum_patch = sum(patch[self.mask == 1].abs())
        return torch.full((8,8), fill_value=sum_patch)


if __name__ == '__main__':
    pass