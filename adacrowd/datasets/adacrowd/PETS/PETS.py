import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data


def read_img(path):
    img = Image.open(path)

    if img.mode == 'L':
        img = img.convert('RGB')
    return img


class PETS(data.Dataset):
    def __init__(
            self,
            data_path,
            main_transform=None,
            img_transform=None,
            gt_transform=None,
            k_shot=1,
            val_folders=None):

        self.data_files = []

        # Loop through all the subfolders and collect th images incase of
        # validation
        images = []
        print(data_path, val_folders)
        for vf in val_folders:
            val_path = os.path.join(data_path, vf)
            images = [
                os.path.join(
                    val_path,
                    img) for img in os.listdir(val_path) if (
                    img.endswith('.jpg'))]

        np.random.shuffle(images)
        gui_imgs = images[:k_shot]

        if 'train' in data_path:
            self.data_files = gui_imgs
            print("Number of train images: {}".format(len(self.data_files)))
        else:
            self.data_files = images
            print("Number of test images: {}".format(len(self.data_files)))

        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

    def __getitem__(self, index):

        fname = self.data_files[index]

        # reading the crowd image, ground truth density map, list of unlabeled
        # images
        img, den = self.read_image_and_gt(fname)

        # applying main transformation (consists of image resizing)
        if self.main_transform is not None:
            img = self.main_transform(img)

        # applying image transformation (consists of ToTensor() and
        # Normalization of channels)
        if self.img_transform is not None:
            img = self.img_transform(img)

        # applying transformation to the crowd density maps
        if self.gt_transform is not None:
            den = self.gt_transform(den).unsqueeze(0)

        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, fname):
        """
        Method to read images and ground-truths
        """
        crowd_img = read_img(fname)
        den = None

        gt_path = fname.replace('images', 'csvs').replace('.jpg', '.csv')
        den = pd.read_csv(gt_path, sep=',', header=None).values
        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)

        return crowd_img, den

    def get_num_samples(self):
        return self.num_samples
