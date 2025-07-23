import os
import pandas as pd
import torch
import torchvision.transforms as T


from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.io import read_image
from PIL import Image


class Nips17Subset(Dataset):
    
    def __init__(self, img_path, label_path, transform, target_transform, adversarial, is_test_data=False):
        super().__init__()
        self.labels = pd.read_csv(label_path)
        self.img_dir = img_path
        if adversarial:
            self.getitem_func = self.getitem_adversarial
        elif is_test_data:
            self.getitem_func = self.getitem_withpath
        else:
            self.getitem_func = self.getitem
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tup = self.getitem_func(idx)
        return tup

    def getitem_adversarial(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_path + '.png')
        label = self.labels.iloc[idx, 6] - 1
        image = self.transform(image)
        label = self.target_transform(label)
        return image, label, img_path

    def getitem_withpath(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_path + '.png')
        label = self.labels.iloc[idx, 6] - 1
        image = self.transform(image)
        label = self.target_transform(label)
        return image, label, img_path
    
    def getitem(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_path + '.png')
        label = self.labels.iloc[idx, 6] - 1
        image = self.transform(image, label)
        label = self.target_transform(label)
        return image, label
    
class C25Subset(Dataset):

    # only for Synthbuster data

    def __init__(self, root, label_path, transform, target_transform, adversarial, is_test_data=False):
        super().__init__()
        self.labels = pd.read_csv(label_path)
        self.img_dir = root
        if adversarial:
            self.getitem_func = self.getitem_adversarial
        elif is_test_data:
            self.getitem_func = self.getitem_withpath
        else:
            self.getitem_func = self.getitem
        self.transform = transform
        self.nex_lst = []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tup = self.getitem_func(idx)
        return tup

    def getitem_adversarial(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.labels.iloc[idx, 1]
        image = self.transform(image)
        image = self.check_input(image)
        label = self.target_transform(label)
        return image, label, img_path

    def getitem_withpath(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.labels.iloc[idx, 1]
        image = self.transform(image)
        image = self.check_input(image)
        label = self.target_transform(label)
        return image, label, img_path
    
    def getitem(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.labels.iloc[idx, 1]
        image = self.transform(image, label)
        image = self.check_input(image)
        label = self.target_transform(label)
        return image, label
    
    def check_input(self, img):
        # check if image is 3 x w x h
        c, _, _ = img.shape
        if c == 1:
            # project greyscale to color by just copying the c dimension
            img = img.repeat(3,1,1)
            print('project greyscale to 3 channels')
        return img
    
    def target_transform(self, label):
        if label.startswith('real'):
            return 0
        else:
            return 1



class SurveySubset(Dataset):
    
    def __init__(self, img_path, label_path, transform, target_transform, adversarial, is_test_data=False):
        super().__init__()
        self.labels = pd.read_csv(label_path)
        self.img_dir = img_path
        if adversarial:
            self.getitem_func = self.getitem_adversarial
        elif is_test_data:
            self.getitem_func = self.getitem_withpath
        else:
            self.getitem_func = self.getitem
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tup = self.getitem_func(idx)
        return tup

    def getitem_adversarial(self, idx):
        img_path = os.path.join(self.img_dir, str(idx))
        image = Image.open(img_path + '.png')
        label = self.labels.iloc[idx, -1]
        image = self.transform(image)
        label = self.target_transform(label)
        return image, label, img_path

    def getitem_withpath(self, idx):
        img_path = os.path.join(self.img_dir, str(idx))
        image = Image.open(img_path + '.png')
        label = self.labels.iloc[idx, -1]
        image = self.transform(image)
        label = self.target_transform(label)
        return image, label, img_path
    
    def getitem(self, idx):
        img_path = os.path.join(self.img_dir, str(idx))
        image = Image.open(img_path + '.png')
        label = self.labels.iloc[idx, -1]
        image = self.transform(image, label)
        label = self.target_transform(label)
        return image, label

class CustomCIFAR10(CIFAR10):
    
    #helps us to overwrite CIFAR10 obj for our purposes
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, 'nopathgiven' # to make this work with our version

class CustomCIFAR10(CIFAR10):
    
    #helps us to overwrite CIFAR10 obj for our purposes
    
    def __init__(self, adversarial, is_test_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pil_to_tensor = T.PILToTensor()
        if adversarial:
            self.getitem_func = self.getitem_adversarial
        else:
            self.getitem_func = self.getitem

    def __getitem__(self, index: int):
        img, target, path = self.getitem_func(index)
        return img, target, path
    
    def getitem_adversarial(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.pil_to_tensor(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, 'nopathgiven' # to make this work with our version

    def getitem(self, index: int):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img = self.pil_to_tensor(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, 'nopathgiven' # to make this work with our version

class CustomCIFAR100(CIFAR100):
    
    #helps us to overwrite CIFAR10 obj for our purposes
    
    def __init__(self, adversarial, is_test_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pil_to_tensor = T.PILToTensor()
        if adversarial:
            self.getitem_func = self.getitem_adversarial
        else:
            self.getitem_func = self.getitem

    def __getitem__(self, index: int):
        img, target, path = self.getitem_func(index)
        return img, target, path
    
    def getitem_adversarial(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.pil_to_tensor(img)

        if self.transform is not None:
            img = self.transform(img, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, 'nopathgiven' # to make this work with our version

    def getitem(self, index: int):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img = self.pil_to_tensor(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, 'nopathgiven' # to make this work with our version
        
    

    
    
    


if __name__ == '__main__':
    pass