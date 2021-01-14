import os

from torch.utils.data import DataLoader

from datasets.misc.utils import data_transforms

from .PETS import PETS
from .setting import cfg_data


def loading_data(scene_folder=None):
    train_main_transform, val_main_transform, img_transform, gt_transform, restore_transform = data_transforms(
        cfg_data=cfg_data)

    # validation folders
    VAL_FOLDERS = [os.path.join(vf, scene_folder)
                   for vf in cfg_data.VAL_FOLDER]

    # train loader
    train_set = PETS(
        os.path.join(
            cfg_data.DATA_PATH,
            'train'),
        main_transform=train_main_transform,
        img_transform=img_transform,
        gt_transform=gt_transform)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg_data.TRAIN_BATCH_SIZE,
        num_workers=1,
        shuffle=True,
        drop_last=True)

    # validation loader
    val_path = os.path.join(cfg_data.DATA_PATH, 'test')
    val_set = PETS(
        val_path,
        main_transform=val_main_transform,
        img_transform=img_transform,
        gt_transform=gt_transform,
        val_folders=VAL_FOLDERS)
    val_loader = DataLoader(
        val_set,
        batch_size=cfg_data.VAL_BATCH_SIZE,
        num_workers=1,
        shuffle=False,
        drop_last=True)

    return train_loader, val_loader, restore_transform
