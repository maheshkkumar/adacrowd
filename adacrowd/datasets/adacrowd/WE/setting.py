from easydict import EasyDict as edict

# initalization
__C_WE = edict()
cfg_data = __C_WE
__C_WE.DATASET = 'WE'

# dataset parameters
__C_WE.STD_SIZE = (576, 720)
__C_WE.TRAIN_SIZE = (512, 672)
__C_WE.DATA_PATH = ''
__C_WE.MEAN_STD = ([0.504379212856, 0.510956227779, 0.505369007587], [
                   0.22513884306, 0.225588873029, 0.22579960525])

# standard data parameters
__C_WE.LABEL_FACTOR = 1
__C_WE.LOG_PARA = 100.

# training parameters
__C_WE.TRAIN_BATCH_SIZE = 1

# validation parameters
__C_WE.VAL_BATCH_SIZE = 1
__C_WE.VAL_FOLDER = ['104207', '200608', '200702', '202201', '500717']
