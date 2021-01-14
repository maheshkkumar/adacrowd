import os

from torch.utils.data import DataLoader

from datasets.misc.utils import data_transforms

from .FDST import FDST
from .setting import cfg_data


def loading_data(k_shot=1, scene_folder=None):

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
        k_shot=k_shot,
        scene_folder=scene_folder)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg_data.TRAIN_BATCH_SIZE,
        num_workers=10,
        shuffle=True,
        drop_last=True)

    # test
    test_path = os.path.join(cfg_data.DATA_PATH, 'test')
    test_set = FDST(
        test_path,
        main_transform=val_main_transform,
        img_transform=img_transform,
        gt_transform=gt_transform,
        scene_folder=scene_folder)
    test_loader = DataLoader(
        test_set,
        batch_size=cfg_data.VAL_BATCH_SIZE,
        num_workers=10,
        shuffle=False,
        drop_last=True)

    return train_loader, test_loader, restore_transform
