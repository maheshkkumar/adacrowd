"""Script to load WorldExpo'10 dataset for the adacrowd model
"""

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_img(path):
    img = Image.open(path)
    if img.mode == 'L':
        img = img.convert('RGB')
    return img


class WE(Dataset):
    def __init__(
            self,
            data_path,
            main_transform=None,
            img_transform=None,
            gt_transform=None,
            k_shot=1,
            num_scenes=None):

        self.img_path = data_path + '/images'
        self.gt_path = data_path + '/csvs'

        self.data_files = []
        folders = os.listdir(self.img_path)
        folders = folders if num_scenes is None else folders[:num_scenes]
        print(
            "Num scenes: {}, loaded scenes: {}".format(
                num_scenes,
                len(folders)))
        for fol in folders:
            img_fold = os.path.join(self.img_path, fol)
            images = [os.path.join(img_fold, img) for img in os.listdir(
                img_fold) if img.endswith('.jpg')]
            np.random.shuffle(images)
            gui_imgs = images[:k_shot]
            crowd_imgs = images[k_shot:]

            combined_imgs = [[gui_imgs, crowd_img] for crowd_img in crowd_imgs]
            self.data_files += combined_imgs

        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

    def __getitem__(self, index):

        fname = self.data_files[index]

        # reading the crowd image, ground truth density map, list of unlabeled
        # images
        img, den, gui_imgs = self.read_image_and_gt(fname)

        # applying transformation to the crowd images and it's ground-truth
        if self.main_transform is not None:
            img, den = self.main_transform(img, den)

        # applying image transformation to crowd images
        if self.img_transform is not None:
            img = self.img_transform(img)

            # applying transformation consisting of ToTensor() and Meanstd()
            gui_imgs = [self.img_transform(gui_img) for gui_img in gui_imgs]

        # applying transformation to the crowd density maps
        if self.gt_transform is not None:
            den = self.gt_transform(den).unsqueeze(0)

            # creating a tensor out of the list of unlabeled image tensors
        gui_imgs = torch.cat(gui_imgs, dim=0)

        # returning crowd image, density map, tensor containing unlabeled
        # images and the list of unlabeled image names
        return img, den, gui_imgs

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, fname):
        """
        Method to read images and ground-truths
        """

        gui_imgs = fname[0]
        crowd_img = fname[1]

        crowd_im = read_img(os.path.join(self.img_path, crowd_img))
        gui_imgs = [read_img(os.path.join(self.img_path, gui_img))
                   for gui_img in gui_imgs]
        gt_path = crowd_img.replace('images', 'csvs').replace('.jpg', '.csv')

        den = pd.read_csv(gt_path, sep=',', header=None).values

        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)
        return crowd_im, den, gui_imgs

    def get_num_samples(self):
        return self.num_samples
