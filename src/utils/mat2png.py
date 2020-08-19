#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Created on 2017/08/23
@author: renjiao
'''
import os
import cv2
from scipy.io import loadmat as sio
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import tqdm
import shutil

def c459259l(file1,file2):
    f1 = open(file1,'r')
    f2 = open(file2,'r')
    f1cnts = f1.readlines()
    f2cnts = f2.readlines()
    dict59 = dict()
    name2label = dict()
    for tmp in f1cnts:
        tmps = tmp.strip().split(':')
        dict59[tmps[1]] = tmps[0]
    names = dict59.keys()
    for tmp in f2cnts:
        tmps = tmp.strip().split(':')
        if tmps[1] in names:
            name2label[tmps[0]] = dict59[tmps[1]]
    return name2label


def mat2png():
    path='/data/detect/VOC/VOCdevkit/VOC2010/trainval'
    files=os.listdir(path)
    labels_path= '/data/detect/VOC/VOCdevkit/VOC2010/test22' #Seglabels' #os.path.join(path,'Seglabels')
    total_num = len(files)
    name2label = c459259l('../datasets/22label.txt','/data/detect/VOC/VOCdevkit/VOC2010/labels.txt')
    keys = name2label.keys()
    for i in tqdm.tqdm(range(total_num)):
        afile = files[i]
        file_path=os.path.join(path,afile)
        if os.path.isfile(file_path):
            if os.path.getsize(file_path)==0:
                continue
            mat_idx=afile[:-4]
            mat_file=sio(file_path)
            mat_file=np.array(mat_file['LabelMap']).astype(np.int)
            h,w = mat_file.shape[:2]
            tmp_label = np.zeros([h,w])
            for tmp in keys:
                tmp_label[mat_file==int(tmp)] = int(name2label[tmp])
            tmp_label=tmp_label.astype(np.uint8) #here is a bug 459 -> 256
            # label_img=Image.fromarray(mat_file.reshape(mat_file.shape[0],mat_file.shape[1]))
            dst_path=os.path.join(labels_path,mat_idx+'.png')
            cv2.imwrite(dst_path,tmp_label)

def sel2label(labeldir,imgdir,disdir):
    fcnts = os.listdir(labeldir)
    tmpls = range(1,24)
    totalnum = len(fcnts)
    cnt =0
    for k in tqdm.tqdm(range(totalnum)):
        tmp = fcnts[k].strip()
        # fg = 0
        imgpath = os.path.join(labeldir,tmp)
        # ditpath = os.path.join(disdir,tmp.strip())
        imgorgpath = os.path.join(imgdir,tmp[:-4]+'.jpg')
        img = cv2.imread(imgpath)
        tmpim = img[:,:,0]
        imgset = set(tmpim.flatten())
        for i in tmpls:
            if i in imgset:
                tmp_dir = os.path.join(disdir,str(i))
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                ditpath = os.path.join(tmp_dir,tmp[:-4]+'.jpg')
                shutil.copyfile(imgorgpath,ditpath)
        #         fg = 1
        #         break
        # if fg==1:
            # shutil.copyfile(imgpath,ditpath)
        # if len(imgset)>1:
        #     cnt+=1
    print(cnt)

def convert20(imgdir,distdir):
    fcnts = os.listdir(imgdir)
    name2label = c459259l('../datasets/20label.txt','../datasets/59label.txt')
    total_num = len(fcnts)
    keys = name2label.keys()
    for i in tqdm.tqdm(range(total_num)):
        afile = fcnts[i]
        imgpath=os.path.join(imgdir,afile)
        imgname = afile[:-4]
        img = cv2.imread(imgpath)
        tmpimg = img[:,:,0].astype(np.int)
        h,w = tmpimg.shape[:2]
        tmp_label = np.zeros([h,w])
        for tmp in keys:
            tmp_label[tmpimg==int(tmp)] = int(name2label[tmp])
        tmp_label = tmp_label.astype(np.uint8) 
        dst_path=os.path.join(distdir,imgname+'.png')
        cv2.imwrite(dst_path,tmp_label)
    


if __name__ == '__main__':
    # name2label = c459259l('../datasets/20label.txt','../datasets/59label.txt')
    # print(name2label.keys())
    # print(name2label.values())
    # p = '/data/detect/VOC/VOCdevkit/VOC2010/Seglabels22/2008_000002.png'
    # img = cv2.imread(p)
    # print(img.shape)
    # cv2.imshow('src',img)
    # cv2.waitKey(0)
    # mat2png()
    sel2label('/data/detect/VOC/VOCdevkit/VOC2010/Seglabels24','/data/detect/VOC/VOCdevkit/VOC2010/JPEGImages','/data/detect/VOC/VOCdevkit/VOC2010/label24')
    # convert20('/data/detect/VOC/VOCdevkit/VOC2010/labels','/data/detect/VOC/VOCdevkit/VOC2010/labels20')