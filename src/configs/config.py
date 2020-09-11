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
cfgs.num_classes = 24
#**
cfgs.data_dir = '/data/detect/VOC/VOCdevkit/VOC2010' #'/mnt/data/LXY.data/voc2010'
cfgs.LabelFile = '../datasets/voc2010v3.csv'
cfgs.save_model_path = '/data/models/img_seg' #'/mnt/data/LXY.data/models/imgseg'
cfgs.train_file = '../datasets/voctrain.txt'
cfgs.val_file = '../datasets/vocval.txt'
#
cfg = EasyDict()
# cfg.NAME = "SemSegFPNHead"
# cfg.IN_FEATURES = ["p2", "p3", "p4", "p5"]
# # Label in the semantic segmentation ground truth that is ignored, i.e., no loss is calculated for
# # the correposnding pixel.
# cfg.IGNORE_VALUE = 255
# # Number of classes in the semantic segmentation head
# cfg.NUM_CLASSES = 24
# # Number of channels in the 3x3 convs inside semantic-FPN heads.
# cfg.CONVS_DIM = 128
# # Outputs from semantic-FPN heads are up-scaled to the COMMON_STRIDE stride.
# cfg.COMMON_STRIDE = 4
# # Normalization method for the convolution layers. Options: "" (no norm), "GN".
# cfg.NORM = "GN"
# cfg.LOSS_WEIGHT = 1.0
cfg.NAME = "StandardPointHead"
cfg.NUM_CLASSES = 24
# Names of the input feature maps to be used by a mask point head.
cfg.IN_FEATURES = ("r2")
# Number of points sampled during training for a mask point head.
cfg.TRAIN_NUM_POINTS = 2304 #14 * 14
# Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
# original paper.
cfg.OVERSAMPLE_RATIO = 3
# Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
# the original paper.
cfg.IMPORTANCE_SAMPLE_RATIO = 0.75
# Number of subdivision steps during inference.
cfg.SUBDIVISION_STEPS = 2
# Maximum number of points selected at each subdivision step (N).
cfg.SUBDIVISION_NUM_POINTS = 8096 #28 * 28
cfg.FC_DIM = 256
cfg.NUM_FC = 3
cfg.CLS_AGNOSTIC_MASK = False
# If True, then coarse prediction features are used as inout for each layer in PointRend's MLP.
cfg.COARSE_PRED_EACH_LAYER = True
cfg.HEAD_CHANNELS = 256