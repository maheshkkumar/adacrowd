import argparse
import json
import os

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from misc.utils import AverageMeter
from models.adacrowd.blocks import assign_adaptive_params
from models.cc_adacrowd import CrowdCounterAdaCrowd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 3035
torch.backends.cudnn.benchmark = True

if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

SINGLE_FOLDER_DATASETS = ["WE", "City"]


def load_dataloader(data_mode):
    if data_mode == 'WE':
        from datasets.adacrowd.WE.loading_data import loading_data
        from datasets.adacrowd.WE.setting import cfg_data
    elif data_mode == 'Mall':
        from datasets.adacrowd.Mall.loading_data import loading_data
        from datasets.adacrowd.Mall.setting import cfg_data
    elif data_mode == 'PETS':
        from datasets.adacrowd.PETS.loading_data import loading_data
        from datasets.adacrowd.PETS.setting import cfg_data
    elif data_mode == 'FDST':
        from datasets.adacrowd.FDST.loading_data import loading_data
        from datasets.adacrowd.FDST.setting import cfg_data
    elif data_mode == 'City':
        from datasets.adacrowd.City.loading_data import loading_data
        from datasets.adacrowd.City.setting import cfg_data
    return loading_data, cfg_data


def test(dataset, args, scene_folder=None):
    maes = AverageMeter()
    mses = AverageMeter()

    model_path = args.model_path
    model = CrowdCounterAdaCrowd([args.gpu_id],
                                 args.model_name,
                                 num_gbnnorm=args.num_norm).to(device)
    checkpoint = torch.load(model_path, map_location='cuda:0')
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # loading the training and testing dataloader
    load_data, data_args = load_dataloader(dataset)

    RESULT = {}

    for trail in range(1, args.trails + 1):
        if dataset == ['PETS', 'FDST']:
            train_loader, val_loader, _ = load_data(
                k_shot=args.k_shot, scene_folder=scene_folder)
        elif dataset in ['WE', 'City']:
            train_loader, val_loader, _ = load_data(
                k_shot=args.k_shot)
        else:
            train_loader, val_loader, _ = load_data(k_shot=args.k_shot)

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

        output_path = os.path.join(output_folder, file_name)

        with torch.no_grad():

            # Compute the accuracy of the model with the updated model
            for didx, data in enumerate(
                    tqdm(val_loader, total=len(val_loader))):

                # clearing the cache to have memory to test all the images
                torch.cuda.empty_cache()

                # loading the data based on the type of folder of the dataset
                if dataset in SINGLE_FOLDER_DATASETS:
                    crow_imgs, gt, gui_imgs = data
                else:
                    crow_imgs, gt = data

                # converting the data to torch variable
                crow_imgs = Variable(crow_imgs).to(device)
                gt = Variable(gt).to(device)

                # computing the mean latent representation for the unlabeled
                # images
                if dataset in SINGLE_FOLDER_DATASETS:
                    mean_latent = model.compute_k_mean(
                        gui_imgs, dataset=dataset)
                else:
                    # Iterate through train images to compute the mean and std
                    # for the decoder
                    mean_latent = model.compute_k_mean(train_loader)

                # incorporating the mean latent values of the target dataset
                # (mean and std) to the decoder of the source model
                assign_adaptive_params(mean_latent, model.CCN.crowd_decoder)

                # forward pass to generate the crowd images latent
                # representation
                crow_img = model.CCN.crowd_encoder(crow_imgs)

                # generate the density map for the extracted crowd image
                # output
                pred_map = model.CCN.crowd_decoder(crow_img)

                # calculate the predicted crowd count to determine the
                # performance of the model
                pred_img = pred_map.data.cpu().numpy()
                gt_img = gt.data.cpu().numpy()

                pred_count = np.sum(pred_img) / 100.
                gt_count = np.sum(gt_img) / 100.

                if dataset in ['Mall']:
                    gt = gt.unsqueeze(0)

                difference = gt_count - pred_count
                maes.update(abs(difference))
                mses.update(difference * difference)

            mae = maes.avg
            mse = np.sqrt(mses.avg)

            # saving the results
            RESULT[str(trail).zfill(2)] = {'mae': mae, 'mse': mse}

            if dataset in ['PETS', 'FDST']:
                print(
                    "Dataset: {}, Scene: {}, Trail: {}, MAE: {}, MSE: {}".format(
                        dataset, scene_folder, trail, mae, mse))
            else:
                print(
                    "Dataset: {}, Trail: {}, MAE: {}, MSE: {}".format(
                        dataset, trail, mae, mse))
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
        '-k',
        '--k_shot',
        help="Number of K images used for computing the affine transformation parameters",
        required=True,
        type=int)
    FLAGS.add_argument(
        '-norm',
        '--num_norm',
        help="Number of normalization layers",
        required=True,
        type=int)
    FLAGS.add_argument('-gpu', '--gpu_id', help="GPU ID", default=3, type=int)
    FLAGS.add_argument(
        '-r',
        '--result_folder',
        help="Path of the results folder",
        type=str,
        required=True)
    FLAGS.add_argument(
        '-t',
        '--trails',
        help="Number of random trails to calculate mean and std scores",
        required=True,
        type=int)

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
