from easydict import EasyDict as edict

# initalization
__C_City = edict()
cfg_data = __C_City
__C_City.DATASET = 'City'

# dataset parameters
__C_City.STD_SIZE = (384, 512)
__C_City.TRAIN_SIZE = (384, 512)
__C_City.DATA_PATH = ''
__C_City.MEAN_STD = ([0.49051812291145325, 0.48466143012046814, 0.4433270990848541], [
    0.21193557977676392, 0.2080429345369339, 0.2046535313129425])

# standard data parameters
__C_City.LABEL_FACTOR = 1
__C_City.LOG_PARA = 100.

# training parameters
__C_City.TRAIN_BATCH_SIZE = 1  # imgs

# validation parameters
__C_City.VAL_BATCH_SIZE = 1

# TODO: list all the validation scene folders
__C_City.VAL_FOLDER = []
