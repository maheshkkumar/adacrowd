"""Script to load FDST dataset for the baseline model
"""

import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data


class City(data.Dataset):
    def __init__(
            self,
            data_path,
            main_transform=None,
            img_transform=None,
            gt_transform=None):

        self.data_files = []
        for fol in os.listdir(data_path):
            img_fold = os.path.join(data_path, fol, 'images')
            images = [os.path.join(img_fold, img) for img in os.listdir(
                img_fold) if img.endswith('.jpg')]
            self.data_files += images

        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        print("Total images: {}".format(len(self.data_files)))

    def __getitem__(self, index):
        fname = self.data_files[index]

        img, den = self.read_image_and_gt(fname)
        if self.main_transform is not None:
            img, den = self.main_transform(img, den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, fname):
        img = Image.open(fname)
        if img.mode == 'L':
            img = img.convert('RGB')

        gt_path = fname.replace('images', 'gt').replace('.jpg', '.csv')
        den = pd.read_csv(gt_path, sep=',', header=None).values

        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)
        return img, den

    def get_num_samples(self):
        return self.num_samples
