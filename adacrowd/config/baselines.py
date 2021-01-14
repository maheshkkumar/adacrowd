import time

from easydict import EasyDict as edict

# initalization
__C = edict()
cfg = __C

# seed value
__C.SEED = 3035

# datasets: WE, Mall, PETS, FDST, City(CityUHK-X)
__C.DATASET = 'WE'

# networks: CSRNet_BN, Res101_BN, Res101_SFCN_BN
__C.NET = 'CSRNet_BN'

# resume training
__C.RESUME = False
__C.RESUME_PATH = None

# gpu id
__C.GPU_ID = [0]

# learning rate settings
__C.LR = 1e-5

# lr scheduler settings
__C.LR_DECAY = 0.995
__C.LR_DECAY_START = -1
__C.NUM_EPOCH_LR_DECAY = 1

# training settings
__C.MAX_EPOCH = 110
__C.NUM_NORM = 6

# print settings
__C.PRINT_FREQ = 5

# experiment details
now = time.strftime("%m-%d_%H-%M", time.localtime())
experiment_description = 'num_norm={}'.format(
    __C.NUM_NORM)

__C.EXP_NAME = now \
    + '_' + __C.DATASET \
    + '_' + __C.NET \
    + '_' + str(__C.LR) \
    + '_' + experiment_description

# experiments & logging
__C.EXP_PATH = './experiments'

# validation settings
__C.VAL_DENSE_START = 50
__C.VAL_FREQ = 5

# visualization settings
__C.VISIBLE_NUM_IMGS = 1

# model path for testing
__C.MODEL_PATH = None
