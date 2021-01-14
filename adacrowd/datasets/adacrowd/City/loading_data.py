from torch.utils.data import DataLoader

from datasets.misc.utils import data_transforms

from .City import City
from .setting import cfg_data


def loading_data(k_shot=1, num_scenes=None):
    train_main_transform, val_main_transform, img_transform, gt_transform, restore_transform = data_transforms(
        cfg_data=cfg_data)

    train_set = City(
        cfg_data.DATA_PATH + '/train',
        main_transform=train_main_transform,
        img_transform=img_transform,
        gt_transform=gt_transform,
        k_shot=k_shot,
        num_scenes=num_scenes)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg_data.TRAIN_BATCH_SIZE,
        num_workers=10,
        shuffle=True,
        drop_last=True)

    val_set = City(
        cfg_data.DATA_PATH + '/test',
        main_transform=val_main_transform,
        img_transform=img_transform,
        gt_transform=gt_transform,
        k_shot=k_shot)
    val_loader = DataLoader(
        val_set,
        batch_size=cfg_data.VAL_BATCH_SIZE,
        num_workers=10,
        shuffle=False,
        drop_last=True)

    return train_loader, val_loader, restore_transform
