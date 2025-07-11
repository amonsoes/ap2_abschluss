import json
import random
import torchvision.transforms as T
import pandas as pd
import csv
import os
import torch
import numpy as np

from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_distances
from torchvision.utils import save_image




class DatasetConstructor:

    def __init__(self):
        self.n = 0
        self.to_grey = T.Grayscale()
        nodes = self.get_node_level_and_children(72)
        self.classes = self.get_classes_from_nodes(nodes)
        self.data = ImgNetSubset()
        self.radial_dir = './data/radial_profiles/'
        self.clr_dir = './data/color_profiles/'
        self.survey_dir = './data/survey/'
        self.clear_directories()
        self.cls_file = self.survey_dir + 'cls_file.csv'
        with open(self.cls_file, 'w') as f:
            csv_file = csv.writer(f)
            csv_file.writerow(['filename', 'class'])

        self.gather_pool_of_images(self.classes)


    ########## semantically diverse functions

    def clear_directories(self):
        os.system(f'rm -rf {self.radial_dir}/*')
        os.system(f'rm -rf {self.clr_dir}/*')
        os.system(f'rm -rf {self.survey_dir}/*')

    def get_node_level_and_children(self, num_classes):

        with open('./hierarchy.json', 'r') as f:
            classes_json = json.loads(f.read())
            nodes = DatasetConstructor.recursive_depth([], [classes_json], num_classes)
            return nodes

    @staticmethod
    def recursive_depth(leafs, classes_json, num_classes, leaf_incl_prob=0.95):

        call_stack = []
        for node in classes_json:
            children = node['children']

            for child in children:
                if not child.get('children', 0):
                    choice = random.choices([0,1], weights=[10*leaf_incl_prob, 10*(1-leaf_incl_prob)])
                    if choice == [1]:
                        leafs.append(child)
                else:
                    call_stack.append(child)
        
        if len(call_stack) + len(leafs) >= num_classes:
            leafs.extend(call_stack)
            return leafs
        else:
            leaf_incl_prob -= 0.01
            leafs = DatasetConstructor.recursive_depth(leafs, classes_json=call_stack, num_classes=num_classes, leaf_incl_prob=leaf_incl_prob)
            return leafs
        
    def get_classes_from_nodes(self, nodes):
        leafs = []
        for node in nodes:
            rand_leaf = self.rand_walk_node_recursively(node)
            leafs.append(rand_leaf)
        return leafs

    def rand_walk_node_recursively(self, node):
        if not node.get('children', 0):
            return node
        else:
            rand_sub_node = random.choice(node['children'])
            sub_node = self.rand_walk_node_recursively(rand_sub_node)
        return sub_node


    ####### frequency analysis

    def compute_rad_profile(self, img):
        spectrum = self.magn_spectrum(img)
        rprofile = self.radial_profile(spectrum)
        return rprofile

    def magn_spectrum(self, img):
        img = self.to_grey(img)
        f = torch.fft.fft2(img)
        fshift = torch.fft.fftshift(f)
        magnitude_spectrum = torch.abs(fshift)
        return magnitude_spectrum

    @staticmethod
    def radial_profile(spectrum):
        b, h, w = spectrum.shape
        y, x = torch.tensor(np.indices((h, w)))
        center = (int(h/2), int(w/2))
        r = torch.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.to(torch.int32)

        tbin = torch.bincount(torch.ravel(r), torch.ravel(spectrum))
        nr = torch.bincount(torch.ravel(r))
        radialprofile = tbin / torch.maximum(nr, torch.ones_like(nr))
        return radialprofile
    
    def compute_frequency_dissimilarity(self, rprofile):
            rprofiles = [rprofile.numpy()]
            for rfile in os.listdir(self.radial_dir):
                rprofile_other = torch.load(self.radial_dir + rfile).numpy()
                rprofiles.append(rprofile_other)
            rprofile_mat = np.stack(rprofiles, axis=0)
            distance_matrix = cosine_distances(rprofile_mat)
            return distance_matrix

    def frequency_dissimilarity(self, images):
        rprofiles = []
        for image in images:
            rprofile = self.compute_rad_profile(image)
            rprofiles.append((image, rprofile))
        if os.listdir(self.radial_dir):
            dissimilarities = []
            for image, rprofile in rprofiles:
                dissim = self.compute_frequency_dissimilarity(rprofile)
                dissimilarities.append((dissim[0].sum(), rprofile, image)) # the first vector corresponds to the image
            max_dissim_rprofile = max(dissimilarities, key=lambda x: x[0])
            torch.save(max_dissim_rprofile[1], self.radial_dir + str(self.n))
            max_dissim_image = max_dissim_rprofile[-1]
            max_dissim_score = max_dissim_rprofile[0]
        else:
            max_dissim_image, rand_rprofile = random.choice(rprofiles)
            torch.save(rand_rprofile, self.radial_dir + str(self.n))
            max_dissim_score = 0.0
        self.n += 1
        return max_dissim_image, max_dissim_score


    ############# color & luminance differences

    def compute_int_clr_dissimilarity(self, int_clr_profile):
            int_clr_profiles = [int_clr_profile.numpy()]
            for int_clr_file in os.listdir(self.clr_dir):
                int_clr_profile_other = torch.load(self.clr_dir + int_clr_file).numpy()
                int_clr_profiles.append(int_clr_profile_other)
            int_clr_profile_mat = np.stack(int_clr_profiles, axis=0)
            distance_matrix = cosine_distances(int_clr_profile_mat)
            return distance_matrix

    def clr_int_dissimilarity(self, ycbcr_images, images):
        int_clr_profiles = []
        for ycbcr_img, img in zip(ycbcr_images, images):
            int_clr_profile = ycbcr_img.sum(dim=(1,2))
            int_clr_profiles.append((img, int_clr_profile))
        if os.listdir(self.clr_dir):
            dissimilarities = []
            for img, int_clr_profile in int_clr_profiles:
                dissim = self.compute_int_clr_dissimilarity(int_clr_profile)
                dissimilarities.append((dissim[0].sum(), int_clr_profile, img))
            max_dissim_clr_profile = max(dissimilarities, key=lambda x: x[0])
            torch.save(max_dissim_clr_profile[1], self.clr_dir + str(self.n))
            max_dissim_image = max_dissim_clr_profile[-1]
            max_dissim_score = max_dissim_clr_profile[0]
        else:
            max_dissim_image, rand_rprofile = random.choice(int_clr_profiles)
            torch.save(rand_rprofile, self.clr_dir + str(self.n))
            max_dissim_score = 0.0
        self.n += 1
        return max_dissim_image, max_dissim_score


    ############# 
    
    def gather_pool_of_images(self, classes):
        imgs_dict = {str(i):[] for i in range(0, 1000)}
        for e,cls in enumerate(classes):
            images, ycbcr_images = self.data.get_images_for_class(cls)
            if e % 2 == 0:
                max_dissim_img, dissim = self.frequency_dissimilarity(images)
            else:
                max_dissim_img, dissim = self.clr_int_dissimilarity(ycbcr_images, images)
            save_image(max_dissim_img, f'{self.survey_dir}{e}.png')
            self.save_cls(filename=e, cls=cls['index'])

        return imgs_dict
    
    def save_cls(self, filename, cls):
        with open(self.cls_file, 'a') as f:
            csv_file = csv.writer(f)
            csv_file.writerow([filename, cls])



class ImgNetSubset(Dataset):
    
    def __init__(self):
        super().__init__()
        self.dir = './data/ext_nips17/'
        self.wnid_to_class = self.get_wn2id()
        self.class_to_wnid = {v:k for k,v in self.wnid_to_class.items()}
        self.file_to_wnid = self.get_image_ids()
        self.file_to_class = {k:self.wnid_to_class[v] for k,v in self.file_to_wnid.items()}
        self.ind_to_file = {str(e):file for e, file in enumerate(self.file_to_wnid.keys())}
        self.transform = T.Compose([T.Resize(256,  interpolation=InterpolationMode.BILINEAR),
                                    T.CenterCrop(256),
                                    T.ToTensor()])
        self.transform_ycbcr = T.Compose([T.Resize(256,  interpolation=InterpolationMode.BILINEAR),
                                    T.CenterCrop(256),
                                    T.ToTensor()])
        

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):

        filename = self.ind_to_file[str(idx)]
        wn_id = self.file_to_wnid[filename]
        image = Image.open(self.dir + '/' + wn_id + '/' +   filename)
        image = self.transform(image)
        label = self.wnid_to_class[wn_id]
        return image, label, filename

    def get_images_for_class(self, cls):
        cls = str(cls['index'])
        wn_id = self.class_to_wnid[cls]
        images = []
        images_ycbcr = []
        filenames = [k for k,v in self.file_to_class.items() if v == cls]
        for fn in filenames:
            image = Image.open(self.dir + '/' + wn_id + '/' + fn)
            image_ycbcr = self.transform_ycbcr(image.convert('YCbCr'))
            image = self.transform(image)
            images.append(image)
            images_ycbcr.append(image_ycbcr)
        return images, images_ycbcr


    def get_wn2id(self):
        with open('./data/ext_nips17/imagenet_class_index.json', 'r') as f:
            classes_json = json.loads(f.read())
        wnid_classes = {wn_id[0] : cls for cls, wn_id  in classes_json.items()}
        return wnid_classes

    def get_image_ids(self):
        file_2_wnid = {}
        for root, dirs, files in os.walk(self.dir, topdown=False):
            for file in files:
                if file.endswith('.JPEG'):
                    file_2_wnid[file] = root.split('/')[-1]
        return file_2_wnid



    



if __name__ == '__main__':

    dataset_constructor = DatasetConstructor()



    
