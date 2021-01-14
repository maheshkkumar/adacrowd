from torch.utils.data import DataLoader

from datasets.misc.utils import data_transforms

from .setting import cfg_data
from .WE import WE


def loading_data():

    train_main_transform, val_main_transform, img_transform, gt_transform, restore_transform = data_transforms(
        cfg_data=cfg_data)

    train_set = WE(
        cfg_data.DATA_PATH + '/train',
        main_transform=train_main_transform,
        img_transform=img_transform,
        gt_transform=gt_transform)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg_data.TRAIN_BATCH_SIZE,
        num_workers=10,
        shuffle=True,
        drop_last=True)

    val_set = WE(
        cfg_data.DATA_PATH + '/test',
        main_transform=val_main_transform,
        img_transform=img_transform,
        gt_transform=gt_transform)
    val_loader = DataLoader(
        val_set,
        batch_size=cfg_data.VAL_BATCH_SIZE,
        num_workers=10,
        shuffle=False,
        drop_last=True)

    return train_loader, val_loader, restore_transform
