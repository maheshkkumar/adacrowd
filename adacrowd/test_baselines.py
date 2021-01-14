import argparse
import json
import os

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from misc.utils import AverageMeter
from models.cc_baselines import CrowdCounter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 3035
torch.backends.cudnn.benchmark = True

# seeding the random function
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

SINGLE_FOLDER_DATASETS = ["WE", "City"]


def load_dataloader(data_mode):
    if data_mode == 'WE':
        from datasets.baselines.WE.loading_data import loading_data
        from datasets.baselines.WE.setting import cfg_data
    elif data_mode == 'Mall':
        from datasets.baselines.Mall.loading_data import loading_data
        from datasets.baselines.Mall.setting import cfg_data
    elif data_mode == 'PETS':
        from datasets.baselines.PETS.loading_data import loading_data
        from datasets.baselines.PETS.setting import cfg_data
    elif data_mode == 'FDST':
        from datasets.baselines.FDST.loading_data import loading_data
        from datasets.baselines.FDST.setting import cfg_data
    elif data_mode == 'City':
        from datasets.baselines.City.loading_data import loading_data
        from datasets.baselines.City.setting import cfg_data
    return loading_data, cfg_data


def test(dataset, args, scene_folder=None):

    maes = AverageMeter()
    mses = AverageMeter()

    model_path = args.model_path
    model = CrowdCounter([args.gpu_id], args.model_name,
                         num_norm=args.num_norm).to(device)
    checkpoint = torch.load(model_path, map_location='cuda:0')
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    if dataset in ['PETS', 'FDST']:
        file_name = "scene_{}_stats.json".format(scene_folder)
    else:
        file_name = "stats.json"

    model_name = args.model_path.split(os.sep)[-1].split('.pth')[0]
    output_folder = os.path.join(
        args.result_folder,
        args.model_name,
        dataset,
        model_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    load_data, _ = load_dataloader(dataset)
    if dataset in ['PETS', 'FDST']:
        _, val_loader, _ = load_data(scene_folder=scene_folder)
    else:
        _, val_loader, _ = load_data()

    for _, data in enumerate(tqdm(val_loader, total=len(val_loader))):
        img, gt, _ = data
        img = Variable(img).to(device)
        gt = Variable(gt).to(device)
        pred_map = model.test_forward(img)

        pred_img = pred_map.data.cpu().numpy().squeeze()
        gt_img = gt.data.cpu().numpy().squeeze()

        pred_count = np.sum(pred_img) / 100.
        gt_count = np.sum(gt_img) / 100.

        difference = gt_count - pred_count
        maes.update(abs(difference))
        mses.update(difference ** 2)

    mae = maes.avg
    mse = np.sqrt(mses.avg)
    print(
        "Model: {}, Dataset: {}, MAE: {}, MSE: {}".format(
            args.model_name,
            dataset,
            mae,
            mse))
    RESULT = {'mae': mae, 'mse': mse}

    output_path = os.path.join(output_folder, file_name)
    with open(output_path, 'w') as fp:
        json.dump(RESULT, fp)


if __name__ == '__main__':
    FLAGS = argparse.ArgumentParser()
    FLAGS.add_argument(
        '-mn',
        '--model_name',
        help="Name of the model to be evaluated",
        required=True,
        type=str)
    FLAGS.add_argument(
        '-mp',
        '--model_path',
        help="Path of the pre-trained model",
        required=True,
        type=str)
    FLAGS.add_argument(
        '-data',
        '--dataset',
        help="Dataset to be evaluated",
        default='WE',
        type=str,
        choices=[
            'WE',
            'UCSD',
            'Mall',
            'PETS',
            'City'])
    FLAGS.add_argument(
        '-norm',
        '--num_norm',
        help="Number of normalization layers",
        required=True,
        type=int)
    FLAGS.add_argument('-gpu', '--gpu_id', help="GPU ID", default=0, type=int)
    FLAGS.add_argument(
        '-r',
        '--result_folder',
        help="Path of the results folder",
        type=str,
        required=True)

    args = FLAGS.parse_args()
    dataset = args.dataset
    if dataset in ['PETS', 'FDST']:
        # TODO: Add the list of scenes/sub-folders in PETS or FDST folder to
        # evalute every scene/sub-folder
        scene_folders = []
        for sf in scene_folders:
            test(dataset=dataset, args=args, scene_folder=sf)
    else:
        test(dataset=dataset, args=args)
