"""Script to load FDST dataset for the baseline model
"""

import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data


class FDST(data.Dataset):
    def __init__(
            self,
            data_path,
            main_transform=None,
            img_transform=None,
            gt_transform=None,
            scene_folder=None):

        self.data_files = []
        data_path = os.path.join(data_path, 'images')
        if scene_folder is None:
            # Loop through all the subfolder and create a single training image
            # list
            scene_folders = [os.path.join(data_path, sf)
                             for sf in os.listdir(data_path)]
            for sf in scene_folders:
                view_folders = [os.path.join(sf, tf) for tf in os.listdir(sf)]
                for tf in view_folders:
                    images = [
                        os.path.join(
                            tf,
                            img) for img in os.listdir(tf)if img.endswith('.jpg')]
                    self.data_files += images
        else:
            # Loop through all the subfolders and collect validation images
            scene_folder = os.path.join(data_path, scene_folder)
            view_folders = [os.path.join(scene_folder, vf)
                            for vf in os.listdir(scene_folder)]
            for vf in view_folders:
                images = [os.path.join(vf, img) for img in os.listdir(
                    vf) if img.endswith('.jpg')]
                self.data_files += images

        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

    def __getitem__(self, index):
        fname = self.data_files[index]

        img, den = self.read_image_and_gt(fname)
        if self.main_transform is not None:
            img = self.main_transform(img)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, fname):
        img = Image.open(os.path.join(fname))
        if img.mode == 'L':
            img = img.convert('RGB')

        gt_path = fname.replace('images', 'csvs').replace('.jpg', '.csv')
        den = pd.read_csv(gt_path, sep=',', header=None).values

        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)
        return img, den

    def get_num_samples(self):
        return self.num_samples
