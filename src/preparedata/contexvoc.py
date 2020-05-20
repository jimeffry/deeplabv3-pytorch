# -*- encoding: utf-8 -*-
import os.path as osp
import os
import sys
from PIL import Image
import numpy as np
import json
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import random

from dataEnhance import *
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs


class ContextVoc(Dataset):
    def __init__(self,filein,cropsize=(640, 480),mode='train', *args, **kwargs):
        super(ContextVoc, self).__init__(*args, **kwargs)
        self.mode = mode
        self._annopath = osp.join('%s', 'Seglabels22', '%s.png')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.rootpath = cfgs.data_dir
        self.ids = list()
        self.loadtxt(filein)
        self.total_num = self.__len__()
        self.shulf_num = list(range(self.total_num))
        random.shuffle(self.shulf_num)
        self.enhancedata(cropsize)

    def loadtxt(self,fpath):
        fr = open(fpath,'r')
        fcnts = fr.readlines()
        for tmp in fcnts:
            self.ids.append((self.rootpath,tmp.strip()))

    def __len__(self):
        return len(self.ids)

    def enhancedata(self,cropsize):
        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness = 0.5,
                contrast = 0.5,
                saturation = 0.5),
            HorizontalFlip(),
            RandomScale((1.0, 1.25, 1.5)),
            RandomCrop(cropsize)
            ])

    def pull_item(self, index):
        idx = self.shulf_num[index]
        img_id = self.ids[idx]
        img_path = self._imgpath % img_id
        label_path = self._annopath % img_id
        img = Image.open(img_path)
        label = Image.open(label_path)
        label = label.convert('L')
        if self.mode == 'train':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)
        # label = self.convert_dice_labels(label)
        return img, torch.from_numpy(label)

    def __getitem__(self, idx):
        img,gt = self.pull_item(idx)
        return img, gt

    def convert_dice_labels(self,label):
        # return semantic_map -> [H, W, class_num]
        semantic_map = []
        for index in range(cfgs.num_classes):
            class_map = np.equal(label, index)
            # class_map = np.all(equality, axis=-1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1).astype(np.float)
        semantic_map = np.transpose(semantic_map,(2,0,1))
        return semantic_map



if __name__ == "__main__":
    from tqdm import tqdm
    ds = ContextVoc(filein='../datasets/vocval.txt', mode='train')
    uni = []
    for im, lb in tqdm(ds):
    # for ids,(im,lb) in enumerate(dataloader_train):
        # lb_uni = np.unique(lb).tolist()
        # uni.extend(lb_uni)
        print(lb.size())
    print(uni)
    print(set(uni))



