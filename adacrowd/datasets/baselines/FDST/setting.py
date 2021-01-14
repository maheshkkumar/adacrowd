from easydict import EasyDict as edict

# initalization
__C_FDST = edict()
cfg_data = __C_FDST
__C_FDST.DATASET = 'FDST'

# dataset parameters
__C_FDST.STD_SIZE = (1080, 1920)
__C_FDST.TRAIN_SIZE = (360, 640)
__C_FDST.DATA_PATH = ''
__C_FDST.MEAN_STD = ([0.484614104033, 0.455819487572, 0.432390660048], [
                     0.23891659081, 0.229008644819, 0.226914435625])

# standard data parameters
__C_FDST.LABEL_FACTOR = 1
__C_FDST.LOG_PARA = 100.

# training parameters
__C_FDST.TRAIN_BATCH_SIZE = 1
__C_FDST.TRAIN_DOWNRATE = 3

# validation parameters
__C_FDST.VAL_BATCH_SIZE = 1
