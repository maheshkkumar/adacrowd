import time

from easydict import EasyDict as edict

# initialization
__C = edict()
cfg = __C

# seed value
__C.SEED = 3035

# datasets: WE, Mall, PETS, FDST, City(CityUHK-X)
__C.DATASET = 'WE'

# networks: CSRNet_GBN, Res101_GBN, Res101_SFCN_GBN
__C.NET = 'Res101_SFCN_GBN'

# resume training
__C.RESUME = False
__C.RESUME_PATH = None

# gpu id
__C.GPU_ID = [1]

# learning rate settings
__C.LR = 1e-5

# lr scheduler settings
__C.LR_DECAY = 0.995  # decay rate
__C.LR_DECAY_START = -1
__C.NUM_EPOCH_LR_DECAY = 1  # decay frequency
__C.MAX_EPOCH = 110

# training settings
__C.GRAD_CLIP = 1
__C.NUM_GBNNORM = 6
__C.K_SHOT = 1  # number of unlabeled images: 1/5
__C.NUM_SCENES = 103  # 103 for WorldExpo'10

# print settings
__C.PRINT_FREQ = 10

# experiment settings
now = time.strftime("%m-%d_%H-%M", time.localtime())
__C.EXP_DETAIL = 'num_gbnnorm_{}_k_shot_{}_scenes_{}'.format(
    __C.NUM_GBNNORM, __C.K_SHOT, __C.NUM_SCENES)
__C.EXP_NAME = now \
    + '_' + __C.DATASET \
    + '_' + __C.NET \
    + '_' + str(__C.LR) \
    + '_' + __C.EXP_DETAIL
#  \

# experiments & logging
__C.EXP_PATH = './experiments'

# validation settings
__C.VAL_DENSE_START = 50
__C.VAL_FREQ = 2

# visualization settings
__C.VISIBLE_NUM_IMGS = 1

# model path for testing
__C.MODEL_PATH = None
