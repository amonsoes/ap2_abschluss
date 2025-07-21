import csv

from torch.utils.data import DataLoader, Subset, random_split
from src.datasets.subsets import Nips17Subset, CustomCIFAR10, CustomCIFAR100, SurveySubset, C25Subset
from src.datasets.data_transforms.img_transform import PreTransforms, PostTransforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100

class Data:
    
    def __init__(self, dataset_name, *args, **kwargs):
        self.dataset = self.loader(dataset_name, *args, **kwargs)
    
    def loader(self, dataset_name, *args, **kwargs):
        if dataset_name == 'nips17':
            dataset = Nips17ImgNetData(*args, **kwargs)
        elif dataset_name == 'c25':
            dataset = Cozzolino25(*args, **kwargs)
        elif dataset_name == 'mnist':
            dataset = MNISTDataset(*args, **kwargs)
        elif dataset_name == 'cifar10':
            dataset = CIFAR10Dataset(*args, **kwargs)
        elif dataset_name == 'survey':
            dataset = SurveyDataset(*args, **kwargs)
        else:
            raise ValueError('Dataset not recognized')
        return dataset

class BaseDataset:
    
    def __init__(self,
                dataset_name,
                model,
                device,
                batch_size,
                transform,
                adversarial_opt,
                adversarial_training_opt,
                jpeg_compression,
                jpeg_compression_rate,
                target_transform=None,
                input_size=224):


        self.transform_type = transform
        self.transforms = PreTransforms(device=device,
                                    target_transform=target_transform, 
                                    input_size=input_size, 
                                    dataset_type=dataset_name,
                                    model=model,
                                    adversarial_opt=adversarial_opt)
        self.post_transforms = PostTransforms(transform,
                                            adversarial_opt=adversarial_opt,
                                            input_size=input_size,
                                            jpeg_compression=jpeg_compression,
                                            jpeg_compression_rate=jpeg_compression_rate,
                                            dataset_type=dataset_name,
                                            model=model,
                                            device=device,
                                            val_base_trm_size=self.transforms.val_base_trm_size)
        self.adversarial_opt = adversarial_opt
        self.adversarial_training_opt = adversarial_training_opt
        self.device = device
        self.batch_size = batch_size
        self.x, self.y = input_size, input_size

class Cozzolino25(BaseDataset):

    def __init__(self, n_datapoints, *args,**kwargs):
        super().__init__('c25', *args, **kwargs)
        self.dataset_type = 'c25'

        self.test_data = self.get_data(transform_val=self.transforms.transform_val, 
                                    target_transform=self.transforms.target_transform)
        if n_datapoints == -1:
            self.test = self.train = self.validation =  DataLoader(self.test_data, batch_size=self.batch_size)
        else:
            self.test_data, _ = random_split(self.test_data, [n_datapoints, len(self.test_data)-n_datapoints])
            self.test = self.train = self.validation =  DataLoader(self.test_data, batch_size=self.batch_size)
    
    def get_data(self, transform_val, target_transform):
        path = './data/c25/'
        labels = path + 'list.csv'

        test_data = C25Subset(label_path=labels, 
                                        root=path, 
                                        transform=transform_val, 
                                        target_transform=target_transform, 
                                        adversarial=self.adversarial_opt.adversarial, 
                                        is_test_data=True)
        
        return test_data


class Nips17ImgNetData(BaseDataset):

    def __init__(self, n_datapoints, *args,**kwargs):
        super().__init__('nips17', *args, **kwargs)
        
        self.categories = self.get_categories()
        self.dataset_type = 'nips17'

        self.test_data = self.get_data(transform_val=self.transforms.transform_val, 
                                    target_transform=self.transforms.target_transform)
        if n_datapoints == -1:
            self.test = self.train = self.validation =  DataLoader(self.test_data, batch_size=self.batch_size)
        else:
            self.test_data, _ = random_split(self.test_data, [n_datapoints, len(self.test_data)-n_datapoints])
            self.test = self.train = self.validation =  DataLoader(self.test_data, batch_size=self.batch_size)

    def get_data(self, transform_val, target_transform):
        path_test = './data/nips17/'
        path_labels = path_test + 'images.csv'
        path_images = path_test + 'images/'
        test = Nips17Subset(label_path=path_labels, 
                            img_path=path_images, 
                            transform=transform_val, 
                            target_transform=target_transform, 
                            adversarial=self.adversarial_opt.adversarial, 
                            is_test_data=True)
        return test
        
    def get_categories(self):
        categories = {}
        path = './data/nips17/categories.csv'
        with open(path, 'r') as cats:
            filereader = csv.reader(cats)
            next(filereader)
            for ind, cat in filereader:
                categories[int(ind) - 1] = cat
        return categories

class SurveyDataset(Nips17ImgNetData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dist_dict = {
            'percal':{'4.0':[12,13,14,15],
                      '8.0':[16,17,18,19],
                      '15.0':[8,9,10,11],
                      '23.0':[0,1,2,3],
                      '38.0':[4,5,6,7]},
            'square':{'4.0':[20,21,22,23],
                      '8.0':[24,25,26,27],
                      '15.0':[28,29,30,31],
                      '23.0':[32,33,34,35],
                      '38.0':[36,37,38,39]},
            'ppba':{'4.0':[40,41,42,43],
                      '8.0':[44,45,46,47],
                      '15.0':[48,49,50,51],
                      '23.0':[52,53,54,55],
                      '38.0':[56,57,58,59]},
            'rcw':{'4.0':[76,77,78,79],
                      '8.0':[64,65,66,67],
                      '15.0':[68,69,70,71],
                      '23.0':[72,73,74,75],
                      '38.0':[60,61,62,63]},
            'cgnc':{'4.0':[80,81,82,83],
                      '8.0':[84,85,86,87],
                      '15.0':[88,89,90,91],
                      '23.0':[92,93,94,95],
                      '38.0':[96,97,98,99]},
            'cgsp':{'4.0':[100,101,102,103],
                      '8.0':[104,105,106,107],
                      '15.0':[108,109,110,111],
                      '23.0':[112,113,114,115],
                      '38.0':[116,117,118,119]},
            'sparsesigmaattack':{'4.0':[120,121,122,123],
                      '8.0':[124,125,126,127],
                      '15.0':[128,129,130,131],
                      '23.0':[132,133,134,135],
                      '38.0':[136,137,138,139]},
            'hpf_fgsm':{'4.0':[140,141,142,143],
                      '8.0':[144,145,146,147],
                      '15.0':[148,149,150,151],
                      '23.0':[152,153,154,155],
                      '38.0':[156,157,158,159]},
            'dim':{'4.0':[160,161,162,163],
                      '8.0':[164,165,166,167],
                      '15.0':[168,169,170,171],
                      '23.0':[172,173,174,175],
                      '38.0':[176,177,178,179]},
            'bsr':{'4.0':[180,181,182,183],
                      '8.0':[184,185,186,187],
                      '15.0':[188,189,190,191],
                      '23.0':[192,193,194,195],
                      '38.0':[196,197,198,199]},
            'vmifgsm':{'4.0':[200,201,202,203],
                      '8.0':[204,205,206,207],
                      '15.0':[208,209,210,211],
                      '23.0':[212,213,214,215],
                      '38.0':[216,217,218,219]},
            'pgdl2':{'4.0':[220,221,222,223],
                      '8.0':[224,225,226,227],
                      '15.0':[228,229,230,231],
                      '23.0':[232,233,234,235],
                      '38.0':[236,237,238,239]},
            'sgd_uap':{'4.0':[240,241,242,243],
                      '8.0':[244,245,246,247],
                      '15.0':[248,249,250,251],
                      '23.0':[252,253,254,255],
                      '38.0':[256,257,258,259]},
            'sga_uap':{'4.0':[260,261,262,263],
                      '8.0':[264,265,266,267],
                      '15.0':[268,269,270,271],
                      '23.0':[272,273,274,275],
                      '38.0':[276,277,278,279]},
        }

    def get_data(self, transform_val, target_transform):
        path_test = './data/survey/'
        path_labels = path_test + 'images.csv'
        path_images = path_test + 'images/'
        test = SurveySubset(label_path=path_labels, 
                            img_path=path_images, 
                            transform=transform_val, 
                            target_transform=target_transform, 
                            adversarial=self.adversarial_opt.adversarial, 
                            is_test_data=True)
        return test
        
    def get_categories(self):
        categories = {}
        path = './data/survey/categories.csv'
        with open(path, 'r') as cats:
            filereader = csv.reader(cats)
            next(filereader)
            for ind, cat in filereader:
                categories[int(ind) - 1] = cat
        return categories



class MNISTDataset(BaseDataset):
    
    def __init__(self,
                n_datapoints,
                *args,
                **kwargs):
        super().__init__('mnist', *args, **kwargs)
        
        self.train_val_data, self.test_data =  self.get_data()
        self.dataset_type = 'mnist'
        
        if self.adversarial_training_opt.adversarial_training:
            self.train = DataLoader(self.train_val_data, batch_size=self.batch_size, shuffle=True)
        else:
            self.train_data, self.val_data = self.train_val_data.split_random(self.train_val_data, lengths=[0.8, 0.2])
            self.train_data = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
            self.validation = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        if n_datapoints == -1:
            self.test = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        else:
            self.test_data, _ = random_split(self.test_data, [n_datapoints, len(self.test_data)-n_datapoints])
            self.test = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
    
    def get_data(self):
        train_val_data = MNIST(root='./data', train=True, download=True, transform=self.transforms.transform_train)
        test_data = MNIST(root='./data', train=False, download=True, transform=self.transforms.transform_val)
        return train_val_data, test_data
    

class CIFAR10Dataset(BaseDataset):
    
    def __init__(self,
                 n_datapoints,
                *args,
                **kwargs):
        super().__init__('cifar10', *args, **kwargs)
        
        self.train_val_data, self.test_data =  self.get_data()
        self.dataset_type = 'cifar10'

        if self.adversarial_training_opt.adversarial_training:
            self.train = DataLoader(self.train_val_data, batch_size=self.batch_size, shuffle=True)
        else:
            train_size = int(len(self.train_val_data) * 0.8) # 80% training data
            valid_size = len(self.train_val_data) - train_size # 20% validation data
            self.train_data, self.val_data = random_split(self.train_val_data, [train_size, valid_size])
            
            self.train_data = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
            self.validation = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        if n_datapoints == -1:
            self.test = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        else:
            self.test_data, _ = random_split(self.test_data, [n_datapoints, len(self.test_data)-n_datapoints])
            self.test = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def get_data(self):
        train_val_data = CIFAR10(root='./data', train=True, download=True, transform=self.transforms.transform_train)
        test_data = CustomCIFAR10(root='./data', train=False, download=True, transform=self.transforms.transform_val, adversarial=self.adversarial_opt.adversarial, is_test_data=True)
        return train_val_data, test_data


class CIFAR100Dataset(BaseDataset):
    
    def __init__(self,
                *args,
                **kwargs):
        super().__init__('cifar100', *args, **kwargs)
        
        self.train_val_data, self.test_data =  self.get_data()
        self.dataset_type = 'cifar100'

        if self.adversarial_training_opt.adversarial_training:
            self.train = DataLoader(self.train_val_data, batch_size=self.batch_size, shuffle=True)
        else:
            self.train_data, self.val_data = random_split(self.train_val_data, [0.8, 0.2])
            # self.train_data, self.val_data = self.train_val_data.split_random(self.train_val_data, lengths=[0.8, 0.2])
            self.train = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
            self.validation = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        self.test = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def get_data(self):
        train_val_data = CIFAR100(root='./data', train=True, download=True, transform=self.transforms.transform_train)
        test_data = CustomCIFAR100(adversarial=self.adversarial_opt.adversarial, is_test_data=True, root='./data', train=False, download=True, transform=self.transforms.transform_val)
        return train_val_data, test_data
        
        
if __name__ == '__main__':
    pass
