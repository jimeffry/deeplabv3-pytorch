# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from backbone import build_backbone
from aspp import ASPP

class DeeplabV3plus(nn.Module):
    def __init__(self, cfg):
        super(DeeplabV3plus, self).__init__()
        self.backbone = None        
        self.backbone_layers = None
        input_channel = 2048        
        self.aspp = ASPP(dim_in=input_channel, 
                dim_out=cfg.MODEL_ASPP_OUTDIM, 
                rate=16//cfg.MODEL_OUTPUT_STRIDE,
                bn_mom = cfg.TRAIN_BN_MOM)
        self.dropout1 = nn.Dropout(0.5)
        # self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4) 
        # self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE//4)
        self.upsample4 = nn.Upsample(scale_factor=4)
        self.upsample_sub = nn.Upsample(scale_factor=cfg.MODEL_OUTPUT_STRIDE//4)

        indim = 256
        self.shortcut_conv = nn.Sequential(
                nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1, padding=cfg.MODEL_SHORTCUT_KERNEL//2,bias=True),
                # SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
                nn.BatchNorm2d(cfg.MODEL_SHORTCUT_DIM),
                nn.ReLU(inplace=True),        
        )        
        self.cat_conv = nn.Sequential(
                nn.Conv2d(cfg.MODEL_ASPP_OUTDIM+cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
                # SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
                nn.BatchNorm2d(cfg.MODEL_ASPP_OUTDIM),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
                # SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
                nn.BatchNorm2d(cfg.MODEL_ASPP_OUTDIM),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.num_classes, 1, 1, padding=0)
        self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
        # self.backbone_layers = self.backbone.get_layers()
        # print(len(self.backbone_layers))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        # print('backout',layers[0].shape,layers[1].shape,layers[2].shape,layers[3].shape)
        feature_aspp = self.aspp(layers[-1])
        # print('asspp',feature_aspp.shape)
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp,feature_shallow],1)
        # print('feature',feature_cat.size())
        result = self.cat_conv(feature_cat) 
        # print('result',result.size())
        result = self.cls_conv(result)
        result = self.upsample4(result)
        #for test
        if not self.training:
            result = result.permute(0,2,3,1)
            result = F.softmax(result, dim=3)
            result = torch.argmax(result, dim=3)
            output = result.new_tensor(result,dtype=torch.int32)
        return output

