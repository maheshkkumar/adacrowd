from torch.utils.data import DataLoader

from datasets.misc.utils import data_transforms

from .Mall import Mall
from .setting import cfg_data


def loading_data():
    _, _, img_transform, gt_transform, restore_transform = data_transforms(
        cfg_data=cfg_data)

    # train loader
    train_set = Mall(
        cfg_data.DATA_PATH + '/train',
        'train',
        main_transform=None,
        img_transform=img_transform,
        gt_transform=gt_transform)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg_data.TRAIN_BATCH_SIZE,
        num_workers=0,
        shuffle=True,
        drop_last=True)

    # test loader
    test_set = Mall(
        cfg_data.DATA_PATH + '/test',
        'test',
        main_transform=None,
        img_transform=img_transform,
        gt_transform=gt_transform)
    test_loader = DataLoader(
        test_set,
        batch_size=cfg_data.VAL_BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        drop_last=False)

    return train_loader, test_loader, restore_transform
