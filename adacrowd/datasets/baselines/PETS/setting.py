from easydict import EasyDict as edict

# initalization
__C_PETS = edict()
cfg_data = __C_PETS
__C_PETS.DATASET = 'PETS'

# dataset parameters
__C_PETS.STD_SIZE = (576, 768)
__C_PETS.TRAIN_SIZE = (288, 384)
__C_PETS.DATA_PATH = ''
__C_PETS.MEAN_STD = ([0.543214261532, 0.577665150166, 0.553619801998], [
                     0.299925357103, 0.279885113239, 0.298922419548])

# standard data parameters
__C_PETS.LABEL_FACTOR = 1
__C_PETS.LOG_PARA = 100.

# training parameters
__C_PETS.TRAIN_BATCH_SIZE = 1
__C_PETS.TRAIN_DOWNRATE = 2

# validation parameters
__C_PETS.VAL_BATCH_SIZE = 1
__C_PETS.VAL_FOLDER = [
    'images/S1_L1/Time_13-57',
    'images/S1_L1/Time_13-59',
    'images/S1_L2/Time_14-06',
    'images/S1_L2/Time_14-31']
