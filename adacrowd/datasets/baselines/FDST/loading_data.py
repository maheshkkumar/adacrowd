import os

from torch.utils.data import DataLoader

from datasets.misc.utils import data_transforms

from .FDST import FDST
from .setting import cfg_data


def loading_data(scene_folder=None):

    train_main_transform, val_main_transform, img_transform, gt_transform, restore_transform = data_transforms(
        cfg_data=cfg_data)

    # train loader
    train_set = FDST(
        os.path.join(
            cfg_data.DATA_PATH,
            'train'),
        main_transform=train_main_transform,
        img_transform=img_transform,
        gt_transform=gt_transform,
        scene_folder=scene_folder)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg_data.TRAIN_BATCH_SIZE,
        num_workers=10,
        shuffle=True,
        drop_last=True)

    # validation loader
    val_path = os.path.join(cfg_data.DATA_PATH, 'test')
    val_set = FDST(
        val_path,
        main_transform=val_main_transform,
        img_transform=img_transform,
        gt_transform=gt_transform,
        scene_folder=scene_folder)
    val_loader = DataLoader(
        val_set,
        batch_size=cfg_data.VAL_BATCH_SIZE,
        num_workers=10,
        shuffle=False,
        drop_last=True)

    return train_loader, val_loader, restore_transform
