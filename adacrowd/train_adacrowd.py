import os

import numpy as np
import torch

from config.adacrowd import cfg
from datasets.adacrowd.WE.loading_data import loading_data
from datasets.adacrowd.WE.setting import cfg_data
from trainer_adacrowd import Trainer_AdaCrowd

seed = cfg.SEED
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

gpus = cfg.GPU_ID
if len(gpus) == 1:
    torch.cuda.set_device(gpus[0])

torch.backends.cudnn.benchmark = True

data_mode = cfg.DATASET


net = cfg.NET
print("Net: {}".format(net))
assert net in ['CSRNet_GBN', 'Res101_GBN',
               'Res101_SFCN_GBN'], "Invalid network"

pwd = os.path.split(os.path.realpath(__file__))[0]
cc_trainer = Trainer_AdaCrowd(loading_data, cfg_data, pwd)
cc_trainer.forward()
