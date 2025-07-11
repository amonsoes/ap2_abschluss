"""
Step 0: Set up Experiment Object that keeps track of average l2, cos, pearsoncc
Step 1: Load dataset
Step 2: For a specific file hash in the real images, get all the corresponding adversarial samples for all attack classes from the 15.0 distortion category
Step 3: If not all are present as adversarial samples, pass this iteration
Step 4: Compute delta by subtracting the original image from the adversarial sample
Step 5: Normalize all distortions to unit length
Step 6: for all attacks used by Fezza19, compute the l2, cos, pearson to all the other attacks, add values to average l2, cos, pearsoncc
Step 7: Repeat for all of our attacks, make sure you store the avgl2, avg cos, pearsoncc in another object obviously

! You'll need to (1) normalize the cosine similarity and (2) subtract it from 1 (so 1- cos(x,y)), since
0 is complete negative correlation and  1 is complete correlation (0.5 is orthogonality)
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ot
import argparse


from sklearn.decomposition import PCA
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from matplotlib.patches import Arc, Wedge
from src.datasets.subsets import Nips17Subset

G1_COLOR = 'm' # magenta
G2_COLOR = 'g' # green


class Nips17Subset(Dataset):
    
    def __init__(self):
        super().__init__()
        path_test = './data/nips17/'
        path_labels = path_test + 'images.csv'
        path_images = path_test + 'images/'
        self.labels = pd.read_csv(path_labels)
        self.img_dir = path_images
        self.transform = T.Compose([T.Resize(256,  interpolation=InterpolationMode.BILINEAR),
                                    T.CenterCrop(224),
                                    T.ToTensor()])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_path + '.png')
        image = self.transform(image)
        label = self.labels.iloc[idx, 6] - 1
        return image, label, img_path.split('/')[-1]


class Stats:

    def __init__(self, data, attacks, attacks2, processing_type):
        """
        pass attacks as list with the names corresponding to the naming in the dataset like this ['pgdl2',...,'vmifgsm']
        """
        len_attacks = len(attacks)
        self.len_attacks = len(attacks)

        self.data = data
        self.adv_data_path = './data/perc_eval_copy/8.0'

        if processing_type == 'conv-attn':
            
            #len_attacks = int(len_attacks//2)
            #self.len_attacks = len_attacks

            self.total_l2 = torch.zeros((len_attacks, len_attacks),dtype=torch.float32)
            self.total_cos = torch.zeros((len_attacks, len_attacks),dtype=torch.float32)
            self.total_pearson = torch.zeros((len_attacks, len_attacks),dtype=torch.float32)
            self.total_msssim = torch.zeros((len_attacks, len_attacks),dtype=torch.float32)

        else:
            len_attacks_2 = len(attacks2)
            self.len_attacks_2 = len(attacks2)

            self.total_l2 = torch.zeros((len_attacks, len_attacks),dtype=torch.float32)
            self.total_cos = torch.zeros((len_attacks, len_attacks),dtype=torch.float32)
            self.total_pearson = torch.zeros((len_attacks, len_attacks),dtype=torch.float32)
            self.total_msssim = torch.zeros((len_attacks, len_attacks),dtype=torch.float32)

            self.total_l2_2 = torch.zeros((len_attacks_2, len_attacks_2),dtype=torch.float32)
            self.total_cos_2 = torch.zeros((len_attacks_2, len_attacks_2),dtype=torch.float32)
            self.total_pearson_2 = torch.zeros((len_attacks_2, len_attacks_2),dtype=torch.float32)
            self.total_msssim_2 = torch.zeros((len_attacks_2, len_attacks_2),dtype=torch.float32)

        self.cos = nn.CosineSimilarity(dim=(1,2,3), eps=1e-6)

        self.attacks = attacks
        self.attacks2 = attacks2

        self.attacks_to_id = {attack:e for e, attack in enumerate(attacks)}
        self.attacks2_to_id = {attack:e for e, attack in enumerate(attacks2)}

        self.n = 0
        self.to_tensor = T.ToTensor()
        self.msssim = SSIM(data_range=1.0).to('cpu')
        self.processing_type = processing_type
    
    def process_data(self):
        for image, label, filename in tqdm(self.data):
            if self.processing_type == 'conv':
                self.process_image_attacks1(filename, image)
                self.process_image_attacks2(filename, image)
            else:
                self.process_image_attacks1_convattn(filename, image)
            self.n += 1


    def process_image_attacks1(self, filename, image):
        for attack in self.attacks:
            id = self.attacks_to_id[attack]
            adv_image1 = self.load_adv_sample(attack, filename)
            pert1_int = image - adv_image1
            pert1 = self.to_unit_length(pert1_int)
            for sub_attack in self.attacks:
                if sub_attack != attack:
                    # don't compute for identical attacks
                    sub_id = self.attacks_to_id[sub_attack]
                    adv_image2 = self.load_adv_sample(sub_attack, filename)
                    pert2_int = image - adv_image2
                    pert2 = self.to_unit_length(pert2_int)

                    # compute l2, cos, pearson
                    cos_result = self.norm_cosine_sim(pert1, pert2)
                    l2_result = self.l2(pert1, pert2)
                    pearson_result = self.norm_pearson_corr(pert1, pert2)
                    msssim_result = self.ms_ssim(pert1, pert2)

                    # fill matrices with result
                    self.total_cos[id, sub_id] += cos_result
                    self.total_l2[id, sub_id] += l2_result
                    self.total_pearson[id, sub_id] += pearson_result
                    self.total_msssim[id, sub_id] += msssim_result

    def process_image_attacks1_convattn(self, filename, image):
        for attack in self.attacks:
            id = self.attacks_to_id[attack]
            adv_image1 = self.load_adv_sample(attack, filename)
            pert1_int = image - adv_image1
            pert1 = self.to_unit_length(pert1_int)
            sub_attack = attack + '_res'

            adv_image2 = self.load_adv_sample(sub_attack, filename)
            pert2_int = image - adv_image2
            pert2 = self.to_unit_length(pert2_int)

            # compute l2, cos, pearson
            cos_result = self.norm_cosine_sim(pert1, pert2)
            l2_result = self.l2(pert1, pert2)
            pearson_result = self.norm_pearson_corr(pert1, pert2)
            msssim_result = self.ms_ssim(pert1, pert2)

            # fill matrices with result
            self.total_cos[id, id] += cos_result
            self.total_l2[id, id] += l2_result
            self.total_pearson[id, id] += pearson_result
            self.total_msssim[id, id] += msssim_result

    def check_if_original(self, attack, filename):
        pass

    def process_image_attacks2(self, filename, image):
        for attack in self.attacks2:
            id = self.attacks2_to_id[attack]
            adv_image1 = self.load_adv_sample(attack, filename)
            pert1 = self.to_unit_length(image - adv_image1)
            for sub_attack in self.attacks2:
                if sub_attack != attack:
                    # don't compute for identical attacks
                    sub_id = self.attacks2_to_id[sub_attack]
                    adv_image2 = self.load_adv_sample(sub_attack, filename)
                    pert2 = self.to_unit_length(image - adv_image2)

                    # compute l2, cos, pearson
                    cos_result = self.norm_cosine_sim(pert1, pert2)
                    l2_result = self.l2(pert1, pert2)
                    pearson_result = self.norm_pearson_corr(pert1, pert2)
                    msssim_result = self.ms_ssim(pert1, pert2)

                    # fill matrices with result
                    self.total_cos_2[id, sub_id] += cos_result
                    self.total_l2_2[id, sub_id] += l2_result
                    self.total_pearson_2[id, sub_id] += pearson_result
                    self.total_msssim_2[id, sub_id] += msssim_result

    
    def compare_on_sample(self):
        pearson_hm = PearsonHeatmap(self.attacks, self.attacks)
        pca_plot = PCAPlot(self.attacks, self.attacks)
        l2_plot = L2BarChart(self.attacks, self.attacks)
        for image, label, filename in tqdm(self.data):
            pearson_hm.process_data(image, filename)
            pca_plot.process_data(image, filename)
            l2_plot.process_data(image, filename)
            self.n += 1

    def get_results(self):
        if self.processing_type == 'conv':
            avg_l2, avg_cos, avg_pearson = self.get_results_conv()
        else:
            avg_l2, avg_cos, avg_pearson = self.get_results_conv_attn()
        return avg_l2, avg_cos, avg_pearson

    def get_results_conv(self):
        avg_l2 = self.total_l2 / self.n
        avg_cos = self.total_cos / self.n
        avg_pearson = self.total_pearson / self.n
        avg_msssim = self.total_msssim / self.n

        avg_l2_2 = self.total_l2_2 / self.n
        avg_cos_2 = self.total_cos_2 / self.n
        avg_pearson_2 = self.total_pearson_2 / self.n
        avg_msssim_2 = self.total_msssim_2 / self.n

        n_denom_1 = self.len_attacks**2 - self.len_attacks
        n_denom_2 = self.len_attacks_2**2 - self.len_attacks_2
        
        print('\nAttack Group 2:\n')
        print('Average l2 distances:')
        print(avg_l2)
        print('Average Inverted Cosine Similarities:')
        print(avg_cos)
        print('Average Inverted Pearson Correlation Coefficient:')
        print(avg_pearson)
        print('Average Inverted MSSSIM:')
        print(avg_msssim)

        print('\nAttack Group 2:\n')
        print('Average l2 distances:')
        print(avg_l2_2)
        print('Average Inverted Cosine Similarities:')
        print(avg_cos_2)
        print('Average Inverted Pearson Correlation Coefficient:')
        print(avg_pearson_2)
        print('Average Inverted MSSSIM:')
        print(avg_msssim_2)

        # average overall-results


        print('\nAttack Group 1:\n')
        print('Overall Average l2 distances:')
        print(avg_l2.sum() / n_denom_1)
        print('Overall Average Inverted Cosine Similarities:')
        print(avg_cos.sum() / n_denom_1)
        print('Overall Average Inverted Pearson Correlation Coefficient:')
        print(avg_pearson.sum()  / n_denom_1)
        print('Overall Average MSSSIM:')
        print(avg_msssim.sum()  / n_denom_1)

        print('\nAttack Group 2:\n')
        print('Overall Average l2 distances:')
        print(avg_l2_2.sum() / n_denom_2)
        print('Overall Average Inverted Cosine Similarities:')
        print(avg_cos_2.sum()/ n_denom_2)
        print('Overall Average Inverted Pearson Correlation Coefficient:')
        print(avg_pearson_2.sum() /n_denom_2)
        print('Overall Average MSSSIM:')
        print(avg_msssim_2.sum()/ n_denom_2)

        self.heatmap(avg_l2, avg_l2_2, 'Average L2')
        self.heatmap(avg_cos, avg_cos_2, 'Average Inverted Normalized Cosine Similarity')
        self.heatmap(avg_pearson, avg_pearson_2, 'Average Inverted Normalized PCC')
        self.heatmap(avg_msssim, avg_msssim_2, 'Average Inverted SSIM')

        return avg_l2, avg_cos, avg_pearson


    def get_results_conv_attn(self):
        avg_l2 = self.total_l2 / self.n
        avg_cos = self.total_cos / self.n
        avg_pearson = self.total_pearson / self.n
        avg_msssim = self.total_msssim / self.n


        n_denom_1 = self.len_attacks**2 - self.len_attacks
        
        print('\nAttack Group 2:\n')
        print('Average l2 distances:')
        print(avg_l2)
        print('Average Inverted Cosine Similarities:')
        print(avg_cos)
        print('Average Inverted Pearson Correlation Coefficient:')
        print(avg_pearson)
        print('Average Inverted MSSSIM:')
        print(avg_msssim)

        # average overall-results


        print('\nAttack Group 1:\n')
        print('Overall Average l2 distances:')
        print(avg_l2.sum() / n_denom_1)
        print('Overall Average Inverted Cosine Similarities:')
        print(avg_cos.sum() / n_denom_1)
        print('Overall Average Inverted Pearson Correlation Coefficient:')
        print(avg_pearson.sum()  / n_denom_1)
        print('Overall Average MSSSIM:')
        print(avg_msssim.sum()  / n_denom_1)


        return avg_l2, avg_cos, avg_pearson

    def heatmap(self, matrix1, matrix2, analysis_type):

        fig, (ax1, ax2) = plt.subplots(1, 2)
        pearson_dataframe = pd.DataFrame(matrix1.numpy(), index=self.attacks, columns=self.attacks)
        pearson_dataframe2 = pd.DataFrame(matrix2.numpy(), index=self.attacks2, columns=self.attacks2)

        sns.heatmap(pearson_dataframe, cmap="coolwarm", annot=True, linecolor='white', linewidths=0.1, ax=ax1)
        sns.heatmap(pearson_dataframe2, cmap="coolwarm", annot=True, linecolor='white', linewidths=0.1, ax=ax2)

        ax1.set_title('Group 1')
        ax2.set_title('Group 2')
        plt.gcf().subplots_adjust(bottom=0.15)
        fig.suptitle(f"{analysis_type} of Adversarial Perturbations", weight='bold')
        plt.show()
        

    def load_adv_sample(self, attack, filename):
        img_path = f'{self.adv_data_path}/{attack}/{filename}.png'
        image = Image.open(img_path)
        image = self.to_tensor(image)
        return image
    
    def to_unit_length(self, x):
        return x / (x.norm(p=2) + 1e-10)

    def norm_cosine_sim(self, x, y):
        """
        deltas are centered around 0.
        To get cos values between 0,1, we normalize.

        1 -> same directionality
        0.5 -> orthoginality
        0 -> negative directionality

        invert for this computation
        """
        x = x.flatten()
        y = y.flatten()
        return 1 - ((1 + F.cosine_similarity(x, y, dim=0) ) / 2).item()
    
    def ms_ssim(self, x, y):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        ms_ssim =  1 - self.msssim(x, y)
        return ms_ssim

    
    def norm_pearson_corr(self, x, y):
        """
        Compute the Pearson correlation coefficient between two images.
        
        Args:
            img1, img2: torch.Tensor of shape (C, H, W) or (H, W).
            
        Returns:
            Pearson correlation coefficient (scalar tensor).

        1 -> perfect correlation
        0.5 -> not correlated
        0 -> perfect negative correlation

        invert for this computation
        """
        x = x.flatten()
        y = y.flatten()
        
        mean1, mean2 = x.mean(), y.mean()
        std1, std2 = x.std(), y.std()
        covariance = ((x - mean1) * (y - mean2)).mean()      
        pearson_corr = covariance / (std1 * std2 + 1e-8)  # Add small epsilon to avoid division by zero
        
        return  1 - ((1 + pearson_corr.item()) / 2)

    def l2(self, x, y):
        #distance = (x - y).pow(2).sum(dim=(1,2,3)).sqrt()
        distance = (x - y).pow(2).sum().sqrt()
        return distance.item()


    def wasserstein_distance_sinkhorn(self, img1, img2, epsilon=0.01, niter=50):
        """
        Computes the Sinkhorn approximation of the Wasserstein distance between two images.
        img1, img2: PyTorch tensors or NumPy arrays of shape (H, W) or (C, H, W).
        epsilon: Entropic regularization (higher = smoother, lower = closer to true Wasserstein).
        niter: Number of Sinkhorn iterations.
        """
        # Flatten images to 2D (pixels as samples)
        img1, img2 = img1.reshape(-1), img2.reshape(-1)

        # Create uniform weights (assuming equal mass)
        a, b = np.ones(len(img1)) / len(img1), np.ones(len(img2)) / len(img2)

        # Compute the cost matrix (L2 distance between pixel values)
        M = np.abs(np.subtract.outer(img1, img2)) ** 2  # Squared Euclidean

        # Compute Sinkhorn distance
        wasserstein_dist = ot.sinkhorn2(a, b, M, reg=epsilon, numItermax=niter)
        
        return np.sqrt(wasserstein_dist)  # Square root to get the proper metric form
    

#
#
###### graphical analysis code #######
#
#

class Visualization:

    def __init__(self, attacks):
        self.to_tensor = T.ToTensor()
        self.attacks = attacks
        self.adv_data_path = './data/perc_eval_copy/8.0'

    def load_adv_sample(self, attack, filename):
        img_path = f'{self.adv_data_path}/{attack}/{filename}.png'
        image = Image.open(img_path)
        image = self.to_tensor(image)
        return image
    
    def get_perturbations(self, image, filename, attacks):

        perturbations = []

        for attack in attacks:
            adv_image = self.load_adv_sample(attack, filename)
            perturbation = adv_image - image
            perturbation = self.pert_to_unit_length(perturbation)
            perturbations.append(perturbation.flatten().numpy())

        return perturbations
    
    def pert_to_unit_length(self, x):
        norms = x.pow(2).sum().sqrt()
        return x / (norms + 1e-10)


class PCAPlot(Visualization):

    def __init__(self, attacks, attacks2=False):
        super().__init__(attacks)
        self.pca = PCA(n_components=2)
        self.attacks2 = attacks2

    def process_data(self, image, filename):

        if self.attacks2:

            # Convert to numpy array (shape: [num_samples, 150528])
            X = np.array(self.get_perturbations(image, filename, self.attacks))
            # Apply PCA to reduce to 2D
            X_pca = self.pca.fit_transform(X)
            X_pca = self.to_unit_length(X_pca)

            # repeat for other attack group
            Y = np.array(self.get_perturbations(image, filename, self.attacks2))
            Y_pca = self.pca.fit_transform(Y)
            Y_pca = self.to_unit_length(Y_pca)

            # plot
            self.scatterplot(X_pca, Y_pca)
            self.arrows(X_pca)
            self.arrows(Y_pca)
        else:
            # Convert to numpy array (shape: [num_samples, 150528])
            X = np.array(self.get_perturbations(image, filename, self.attacks))
            # Apply PCA to reduce to 2D
            X_pca = self.pca.fit_transform(X)
            X_pca = self.to_unit_length(X_pca)

            # plot
            self.scatterplot(X_pca)
            self.arrows(X_pca)

    def scatterplot(self, X_pca, Y_pca=False):
        # Scatter plot in 2D
        plt.figure(figsize=(6,6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, color=G1_COLOR, label='Group 1')
        if isinstance(Y_pca, np.ndarray):
            plt.scatter(Y_pca[:, 0], Y_pca[:, 1], alpha=0.7, color=G2_COLOR, label='Group 2')
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA Projection of Perturbations", weight='bold')

        ax = plt.gca()
        ax.set_facecolor('#eeeeee')
        plt.grid(color='white')
        plt.legend()
        plt.show()

    def get_maximum_cos(self, X_pca):
        v1, v2 = None, None
        max_dissim = 1 # cos is in range [1,-1] where -1 means completely dissimilar
        for pc_pert in X_pca:
            for pc_sub_pert in X_pca:
                cos_theta = np.dot(pc_pert, pc_sub_pert) / (np.linalg.norm(pc_pert) * np.linalg.norm(pc_sub_pert))
                if cos_theta <= max_dissim:
                    v1 = pc_pert
                    v2 = pc_sub_pert
                    max_dissim = cos_theta
        theta = np.arccos(np.clip(max_dissim, -1.0, 1.0))  # Clip for numerical stability
        theta_deg = np.degrees(theta)
        return v1, v2, theta_deg

    def arrows(self, X_pca, plot_cos=True):

        if plot_cos:
            # plot the biggest angle in X_pca:
            max_v1, max_v2, theta_deg = self.get_maximum_cos(X_pca)
            A = np.array([max_v1, max_v2])
    
        
        # Quiver plot in 2D
        plt.figure(figsize=(6,6))
        # plt the rest of the vectors
        plt.quiver([0]*len(X_pca), [0]*len(X_pca), X_pca[:, 0], X_pca[:, 1], width=0.003, scale_units='xy', scale=1)
        # plt vectors that have biggest cos angle
        plt.quiver([0]*len(A), [0]*len(A), A[:, 0], A[:, 1], width=0.003, color=[G1_COLOR, G2_COLOR],scale_units='xy', scale=1)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA Projection of Perturbation Directions", weight='bold')

        ax = plt.gca()
        ax.set_facecolor('#eeeeee')

        limits = X_pca.max()
        # make angle
        arc_radius = limits * 0.5  # Adjust for visibility
        arc = Arc((0, 0), 2 * arc_radius, 2 * arc_radius, angle=0, theta1=0, theta2=theta_deg, color='yellow')
        wedge = Wedge((0, 0), arc_radius, 0, theta_deg, color='yellow', alpha=0.3)

        # Add text label θ
        # Compute text position (polar to Cartesian)
        theta_mid = np.radians(theta_deg / 2)  # Midpoint of the angle
        text_x = arc_radius * 0.7 * np.cos(theta_mid)  # Scale down for better position
        text_y = arc_radius * 0.7 * np.sin(theta_mid)

        ax.text(text_x, text_y, r'$\theta$', fontsize=14, fontweight='bold')

        # Set equal axis scaling
        ax.set_xlim(-limits *1.1, limits*1.1)
        ax.set_ylim(-limits*1.1, limits*1.1)
        ax.set_aspect('equal')

        # Add grid and legend
        ax.add_patch(arc)
        ax.add_patch(wedge)
        plt.grid(color='white')
        plt.show()

    def to_unit_length(self, x):
        if len(x.shape) > 1:
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1  
        else:
            norms = np.linalg.norm(x, keepdims=True)
            norms[norms < 1e-10] = 1  
        return x / (norms + 1e-10)

class PearsonHeatmap(Visualization):

    def __init__(self, attacks, attacks2=False):
        super().__init__(attacks)
        self.attacks_to_id = {attack:e for e,attack in enumerate(attacks)}
        self.attacks2 = attacks2
        if attacks2:
            self.attacks2_to_id = {attack:e for e,attack in enumerate(attacks2)}

    def process_data(self, image, filename):

        if self.attacks2:

            pearson_matrix = self.get_pearson_matrix(image, filename, self.attacks)
            pearson_matrix2 = self.get_pearson_matrix(image, filename, self.attacks2)

            # Plot heatmap of correlations
            fig, (ax1, ax2) = plt.subplots(1, 2)
            pearson_dataframe = pd.DataFrame(pearson_matrix.numpy(), index=self.attacks, columns=self.attacks)
            pearson_dataframe2 = pd.DataFrame(pearson_matrix2.numpy(), index=self.attacks2, columns=self.attacks2)

            sns.heatmap(pearson_dataframe, cmap="coolwarm", annot=True, linecolor='white', linewidths=0.1, ax=ax1)
            sns.heatmap(pearson_dataframe2, cmap="coolwarm", annot=True, linecolor='white', linewidths=0.1, ax=ax2)

            ax1.set_title('Group 1')
            ax2.set_title('Group 2')
            fig.suptitle("Pearson Correlation of Adversarial Perturbations", weight='bold')
            plt.show()
        
        else:
        
            pearson_matrix = self.get_pearson_matrix(image, filename, self.attacks)

            # Plot heatmap of correlations
            plt.figure(figsize=(8,6))
            pearson_dataframe = pd.DataFrame(pearson_matrix.numpy(), index=self.attacks, columns=self.attacks)
            sns.heatmap(pearson_dataframe, cmap="coolwarm", annot=True, linecolor='white', linewidths=0.1)
            plt.title("Pearson Correlation of Adversarial Perturbations", weight='bold')
            plt.show()

    def get_pearson_matrix(self, image, filename, attacks):
        
        n_attacks = len(attacks)
        pearson_matrix = torch.zeros((n_attacks,n_attacks))
        adv_samples = {attack: self.load_adv_sample(attack, filename) for attack in attacks}

        for attack, adv_sample in adv_samples.items():
            attack_id = self.attacks_to_id[attack]
            pert = self.pert_to_unit_length(adv_sample - image)
            for sub_attack, sub_adv_sample in adv_samples.items():
                sub_pert = self.pert_to_unit_length(sub_adv_sample - image)
                sub_attack_id = self.attacks_to_id[sub_attack]
                pearson_result = self.norm_pearson_corr(pert, sub_pert)
                pearson_matrix[attack_id, sub_attack_id] = pearson_result

        return pearson_matrix


    def norm_pearson_corr(self, x, y):
        """
        Compute the Pearson correlation coefficient between two images.
        
        Args:
            img1, img2: torch.Tensor of shape (C, H, W) or (H, W).
            
        Returns:
            Pearson correlation coefficient (scalar tensor).

        1 -> perfect correlation
        0.5 -> not correlated
        0 -> perfect negative correlation

        invert for this computation
        """
        x = x.flatten()
        y = y.flatten()
        
        mean1, mean2 = x.mean(), y.mean()
        std1, std2 = x.std(), y.std()
        covariance = ((x - mean1) * (y - mean2)).mean()      
        pearson_corr = covariance / (std1 * std2 + 1e-8)  # Add small epsilon to avoid division by zero
        return pearson_corr
    

class L2BarChart(Visualization):
    
    def __init__(self, attacks, attacks2):
        super().__init__(attacks)
        self.attacks2 = attacks2
        
    def process_data(self, image, filename):
        
        len_attacks_1 = len(self.attacks)
        len_attacks_2 = len(self.attacks2)
        l2_mat_1 = torch.zeros((len_attacks_1,len_attacks_1))
        l2_mat_2 = torch.zeros((len_attacks_2,len_attacks_2))

        perturbations1 = self.get_perturbations(image, filename, self.attacks)
        perturbations2 = self.get_perturbations(image, filename, self.attacks2)

        for e, pert in enumerate(perturbations1):
            for k, sub_pert in enumerate(perturbations1):
                l2_distance = (pert - sub_pert).pow(2).sum().sqrt()
                l2_mat_1[e, k] = l2_distance
        avg_l2_1 = l2_mat_1.sum() / (len_attacks_1*len_attacks_1)


        for e, pert in enumerate(perturbations2):
            for k, sub_pert in enumerate(perturbations2):
                l2_distance = (pert - sub_pert).pow(2).sum().sqrt()
                l2_mat_2[e, k] = l2_distance
        avg_l2_2 = l2_mat_2.sum() / (len_attacks_2*len_attacks_2)
        

        # visualize

        plt.bar(['G1', 'G2'], [avg_l2_1.item(), avg_l2_2.item()], color=[G1_COLOR, G2_COLOR])
        plt.xlabel("Attack Groups")
        plt.ylabel("L2")
        plt.title("Avg L2 of Distortions in a Group", weight='bold')

        ax = plt.gca()
        ax.set_facecolor('#eeeeee')
        plt.show()
        
    def get_perturbations(self, image, filename, attacks):

        perturbations = torch.zeros((len(attacks),*image.shape))

        for e, attack in enumerate(attacks):
            adv_image = self.load_adv_sample(attack, filename)
            perturbation = adv_image - image
            perturbation = self.pert_to_unit_length(perturbation)
            perturbations[e] = perturbation

        return perturbations


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--processing_type', type=str, default='conv-attn', help='set to conv for comparison against fezza19, set to conv-attn for convolution attention comparison')
    args = parser.parse_args()

    data = Nips17Subset()

    if args.processing_type == 'conv': 
    
        attacks=[
            'pgdl2',
            'vmifgsm',
            'hpf_fgsm',
            'sparsesigmaattack',
            'rcw',
            'percal',
            'sgd_uap',
            'sga_uap',
            'dim',
            'bsr',
            'cgnc',
            'cgsp',
            'square_attack',
            'ppba'
        ]

        attacks2=[
            'fgsm',
            'pgd',
            'bim',
            'mifgsm',
            'cw',
            'deepfool'
        ]
    
    elif args.processing_type == 'conv-attn':

        attacks=[
            'pgdl2',
            'hpf_fgsm',
            'percal',
            'sgd_uap',
            'dim',
            'square_attack',
        ]

        attacks2=[
            'pgdl2',
            'hpf_fgsm',
            'percal',
            'sgd_uap',
            'dim',
            'square_attack',
        ]


    stats = Stats(data, attacks, attacks2, args.processing_type)
    stats.process_data()
    stats.get_results()


    print('done')


    