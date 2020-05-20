from easydict import EasyDict

cfgs = EasyDict()
#************model set
cfgs.MODEL_NAME = 'deeplabv3plus'
cfgs.MODEL_BACKBONE = 'res101_atrous'
cfgs.MODEL_ASPP_OUTDIM = 256
cfgs.MODEL_SHORTCUT_DIM = 48
cfgs.MODEL_SHORTCUT_KERNEL = 1
cfgs.MODEL_OUTPUT_STRIDE = 16
cfgs.TRAIN_BN_MOM = 0
#********
cfgs.crop_height = 480
cfgs.crop_width = 480
cfgs.num_classes = 22
#**
cfgs.data_dir = '/mnt/data/LXY.data/voc2010'
cfgs.LabelFile = '../datasets/voc2010v2.csv'
cfgs.save_model_path = '/mnt/data/LXY.data/models/imgseg'
cfgs.train_file = '../datasets/voctrain.txt'
cfgs.val_file = '../datasets/vocval.txt'