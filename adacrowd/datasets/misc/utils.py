import torchvision.transforms as standard_transforms

import misc.transforms as own_transforms


def data_transforms(cfg_data):

    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA

    # train and val main data transformations
    if cfg_data.DATASET == 'City':
        train_main_transform = own_transforms.Compose([
            own_transforms.RandomHorizontallyFlip()
        ])
        val_main_transform = None
    elif cfg_data.DATASET in ['FDST', 'PETS']:
        train_main_transform = standard_transforms.Compose([
            own_transforms.FreeScale(cfg_data.TRAIN_SIZE),
        ])
        val_main_transform = standard_transforms.Compose([
            own_transforms.FreeScale(cfg_data.TRAIN_SIZE),
        ])
    else:
        train_main_transform = own_transforms.Compose([
            own_transforms.RandomCrop(cfg_data.TRAIN_SIZE),
            own_transforms.RandomHorizontallyFlip()
        ])

        val_main_transform = None

    # image and gt transformations
    if cfg_data.DATASET == 'FDST':
        gt_transform = standard_transforms.Compose([
            own_transforms.GTScaleDown(cfg_data.TRAIN_DOWNRATE),
            own_transforms.LabelNormalize(log_para)
        ])
    else:
        gt_transform = standard_transforms.Compose([
            own_transforms.LabelNormalize(log_para)
        ])

    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    return train_main_transform, val_main_transform, img_transform, gt_transform, restore_transform
