import numpy as np
import cv2
import time
import argparse
import tqdm
import os
import sys
import csv
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

def loadlabel(labelpath):
    f_in = open(labelpath,'r')
    ann = csv.DictReader(f_in)
    label_info = {}
    for row in ann: #ann.iterrows():
        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        class_19 = row['class_num']
        label_info[label_name] = [int(r), int(g), int(b), class_19]
    f_in.close()
    return label_info

def decodeColor(label_info,label_img):
    label_values = []
    label_values.append([0,0,0])
    for key in label_info:
        label_values.append(label_info[key][:3])
    colour_codes = np.array(label_values)
    imgsout = colour_codes[label_img.astype(int)]
    imgsout = cv2.cvtColor(np.uint8(imgsout), cv2.COLOR_RGB2BGR)
    return imgsout

def list_label_img(labelfile,labeldir,imgdir):
    lf = loadlabel(labelfile)
    img_cnts = os.listdir(imgdir)
    total_num = len(img_cnts)
    for i in tqdm.tqdm(range(total_num)):
        imgname = img_cnts[i].strip()
        imgpath = os.path.join(imgdir,imgname)
        labelpath = os.path.join(labeldir,imgname[:-4]+'.png')
        img_l = cv2.imread(labelpath)
        img = cv2.imread(imgpath)
        rt = decodeColor(lf,img_l[:,:,0])
        cv2.imshow('label',rt)
        cv2.imshow('img',img)
        cv2.waitKey(0)

def decodeimgdir(labelfile,imgdir,savedir):
    '''
    label_p: csv label file
    imgdir: segmantation label images
    savedir: output save
    '''
    lf = loadlabel(labelfile)
    img_cnts = os.listdir(imgdir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    total_num = len(img_cnts)
    for i in tqdm.tqdm(range(total_num)):
        imgname = img_cnts[i].strip()
        imgpath = os.path.join(imgdir,imgname)
        savepath = os.path.join(savedir,imgname)
        # labelpath = os.path.join(labeldir,imgname[:-4]+'.png')
        # img_l = cv2.imread(labelpath)
        img = cv2.imread(imgpath)
        rt = decodeColor(lf,img[:,:,0])
        cv2.imwrite(savepath,rt)
        # cv2.imshow('label',rt)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)


if __name__ == '__main__':
    label_p = '../datasets/voc2010v3.csv'
    imgpath = '/data/videos/mframes/video3_labels/v3_100_label.png'
    # img = cv2.imread(imgpath)
    # lf = loadlabel(label_p)
    # rt = decodeColor(lf,img[:,:,0])
    # cv2.imshow('rt',rt)
    # cv2.waitKey(0)
    # imgname = imgpath.split('/')[-1]
    # cv2.imwrite(imgname,rt)
    # list_label_img(label_p,'/data/detect/VOC/VOCdevkit/VOC2010/Seglabels22/','/data/detect/VOC/VOCdevkit/VOC2010/label22/17')
    imgdir = '/data/detect/Semantic_segmentation/imgseglabels'
    savedir = '/data/detect/Semantic_segmentation/imgmasks'
    decodeimgdir(label_p,imgdir,savedir)