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

from dataEnhance import *
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs


class CityScapes(Dataset):
    def __init__(self, rootpth, cropsize=(640, 480), mode='train', *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode

        with open(cfgs.LabelFile, 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = dict()
        for el in labels_info:
            if el['trainId']==255: 
                tid =0 
            else:
                tid = el['trainId'] +1
            self.lb_map[el['id']] = tid

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(rootpth, 'leftImg8bit', mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(rootpth, 'gtFine', mode)
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelIds' in el]
            names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

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
            RandomScale((0.75, 1.0, 1.25, 1.5)),
            RandomCrop(cropsize)
            ])


    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth)
        label = Image.open(lbpth)
        if self.mode == 'train':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64) #[np.newaxis, :]
        label = self.convert_labels(label)
        return img, label

    def __len__(self):
        return self.len

    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label

    # def convert_dice_labels(label, label_info):
    #     # return semantic_map -> [H, W, class_num]
    #     semantic_map = []
    #     void = np.zeros(label.shape[:2])
    #     for index, info in enumerate(label_info):
    #         color = label_info[info][:3]
    #         class_11 = label_info[info][3]
    #         if class_11 == 1:
    #             # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
    #             equality = np.equal(label, color)
    #             class_map = np.all(equality, axis=-1)
    #             # semantic_map[class_map] = index
    #             semantic_map.append(class_map)
    #         else:
    #             equality = np.equal(label, color)
    #             class_map = np.all(equality, axis=-1)
    #             void[class_map] = 1
    #     semantic_map.append(void)
    #     semantic_map = np.stack(semantic_map, axis=-1).astype(np.float)
    #     return semantic_map



if __name__ == "__main__":
    from tqdm import tqdm
    ds = CityScapes('/data/detect/cityscape', mode='val')
    uni = []
    dataloader_train = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    for im, lb in tqdm(ds):
    # for ids,(im,lb) in enumerate(dataloader_train):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
        # print(ids)
    print(uni)
    print(set(uni))

