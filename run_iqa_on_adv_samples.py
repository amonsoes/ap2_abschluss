import argparse

import os
import pandas as pd


from torch.utils.data import Dataset
from PIL import Image

from src.datasets.data import BaseDataset

class AdvSubset(Dataset):
    
    def __init__(self, img_path, frqa_path, transform=None, target_transform=None):
        super().__init__()
        self.frqa = pd.read_csv(frqa_path)
        self.img_dir = img_path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_path + '.png')
        label = self.labels.iloc[idx, 6] - 1
        image = self.transform(image, label)
        label = self.target_transform(label)
        return image, label

class AdvData(BaseDataset):

    def __init__(self, 
                n_datapoints, 
                l2_bound, 
                attack_type, 
                transform_val=None, 
                target_transform=None,
                *args,
                **kwargs):
        super().__init__('adv_data', *args, **kwargs)
        self.n_datapoints = n_datapoints
        self.l2_bound = l2_bound
        self.attack_type = attack_type
        self.transform = transform_val
        self.target_transform = target_transform

    def get_data(self):
        path_test = f'./data/perc_eval/{self.l2_bound}/{self.attak_type}/'
        path_frqa = path_test + 'frqa.csv'
        test = AdvSubset(frqa_path=path_frqa, 
                            img_path=path_test, 
                            transform=self.transform, 
                            target_transform=self.target_transform, 
                            adversarial=self.adversarial_opt.adversarial, 
                            is_test_data=True)
        return test

if __name__ == '__main__':
    pass