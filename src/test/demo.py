# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2020/04/02 10:09
#project: senmantic segmantation
#company: 
#rversion: 0.1
#tool:   python 3.6
#modified:
#description  
####################################################
import numpy as np
import cv2
import time
import argparse
import os
import sys
import csv
import torch
import torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from deeplabv3pluss import DeeplabV3plus
from scarnet import SCAR
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from util import reverse_one_hot
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

def str2bool(raw):
    return raw.lower() in ["yes","true","t","1"]

def parms():
    parser = argparse.ArgumentParser(description='CSRnet demo')
    parser.add_argument('--save_dir', type=str, default='tmp/',
                        help='Directory for detect result')
    parser.add_argument('--modelpath', type=str,
                        default='weights/s3fd.pth', help='trained model')
    parser.add_argument('--labelpath', type=str,
                        default='', help='trained model')
    parser.add_argument('--threshold', default=0.65, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--ctx', default=True, type=str2bool,
                        help='gpu run')
    parser.add_argument('--img_dir', type=str, default='tmp/',
                        help='Directory for images')
    parser.add_argument('--file_in', type=str, default='tmp.txt',
                        help='image namesf')
    return parser.parse_args()

class GroundSeg(object):
    def __init__(self,args):
        if args.ctx and torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False
        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        self.loadmodel(args.modelpath)
        self.loadlabel(args.labelpath)
        self.save_dir = args.save_dir
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

    def loadmodel(self,modelpath):
        if self.use_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.net = DeeplabV3plus(cfgs).to(device)
        # self.net = SCAR(True).to(device)
        state_dict = torch.load(modelpath,map_location=device)
        # state_dict = self.renamedict(state_dict)
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def renamedict(self,statedict):
        state_dict_new = dict()
        for key,value in list(statedict.items()):
            if 'conv_out16' in key or 'conv_out32' in key:
                continue
            state_dict_new[key] = value
        return state_dict_new

    def proprecess(self,imglist):
        rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        imgout = []
        for img in imglist:
            h,w = img.shape[:2]
            gth = int(np.ceil(h/16.0)*16)
            gtw = int(np.ceil(w/16.0)*16)
            img = cv2.resize(img,(gtw,gth))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img.astype('float32')
            img /= 255.0
            img -= rgb_mean
            img /= rgb_std
            # img = np.transpose(img,(2,0,1))
            imgout.append(img)
        return np.array(imgout)

    def inference(self,imglist):
        t1 = time.time()
        imgs = self.proprecess(imglist)
        bt_img = torch.from_numpy(imgs)
        if self.use_cuda:
            bt_img = bt_img.cuda()
        predicts  = self.net(bt_img)
        # predicts = F.softmax(predicts, dim=1)
        t2 = time.time()
        print('inference time:',t2-t1)
        # predicts = reverse_one_hot(predicts)
        masks = self.decodeColor(predicts.cpu().numpy())
        accs = self.calarea(predicts.cpu().numpy())
        return masks,accs

    def loadlabel(self,labelpath):
        f_in = open(labelpath,'r')
        ann = csv.DictReader(f_in)
        self.label_info = {}
        for row in ann: #ann.iterrows():
            label_name = row['name']
            r = row['r']
            g = row['g']
            b = row['b']
            class_19 = row['class_num']
            self.label_info[label_name] = [int(r), int(g), int(b), class_19]
        f_in.close()

    def decodeColor(self,predicts):
        label_values = []
        label_values.append([0,0,0])
        for key in self.label_info:
            label_values.append(self.label_info[key][:3])
        colour_codes = np.array(label_values)
        imgsout = []
        for i in range(predicts.shape[0]):
            x = colour_codes[predicts[i].astype(int)]
            x = cv2.cvtColor(np.uint8(x), cv2.COLOR_RGB2BGR)
            imgsout.append(x)
        return imgsout

    def calarea(self,predicts):
        accout = []
        id_list = [10,12,13,14,16,19]
        acc = 0
        for i in range(predicts.shape[0]):
            img = predicts[i]
            h,w = img.shape[:2]
            for tid in id_list:
                mask_tmp = np.equal(img,tid)
                acc += np.sum(mask_tmp.astype(np.float))
            acc = acc/(h*w)
            accout.append(acc)
        return accout


    def __call__(self,imgpath):
        if os.path.isdir(imgpath):
            cnts = os.listdir(imgpath)
            for idx,tmp in enumerate(cnts):
                tmppath = os.path.join(imgpath,tmp.strip())
                img = cv2.imread(tmppath)
                if img is None:
                    continue
                frame,accs = self.inference([img])
                # cv2.imshow('src',img)
                # cv2.imshow('result',frame[0])
                # print('area:',accs[0])
                save_name = tmp[:-4]+'.jpg'
                savepath = os.path.join(self.save_dir,save_name)
                cv2.imwrite(savepath,frame[0])
                # cv2.waitKey(0)
        elif os.path.isfile(imgpath) and imgpath.endswith('txt'):
            # if not os.path.exists(self.save_dir):
            #     os.makedirs(self.save_dir)
            f_r = open(imgpath,'r')
            file_cnts = f_r.readlines()
            for j in tqdm(range(len(file_cnts))):
                tmp_file = file_cnts[j].strip()
                tmp_file_s = tmp_file.split('\t')
                if len(tmp_file_s)>0:
                    tmp_file = tmp_file_s[0]
                    self.real_num = int(tmp_file_s[1])
                if not tmp_file.endswith('jpg'):
                    tmp_file = tmp_file +'.jpg'
                # tmp_path = os.path.join(self.img_dir,tmp_file) 
                tmp_path = tmp_file
                if not os.path.exists(tmp_path):
                    print(tmp_path)
                    continue
                img = cv2.imread(tmp_path) 
                if img is None:
                    print('None',tmp)
                    continue
                frame = self.inference([img])[0]
                cv2.imshow('result',frame)
                #savepath = os.path.join(self.save_dir,save_name)
                #cv2.imwrite('test.jpg',frame)
                cv2.waitKey(0) 
        elif os.path.isfile(imgpath) and imgpath.endswith(('.mp4','.avi')) :
            cap = cv2.VideoCapture(imgpath)
            frame_width =  cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height =  cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # out = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 25,(frame_width, frame_height))
            if not cap.isOpened():
                print("failed open camera")
                return 0
            else: 
                while cap.isOpened():
                    _,img = cap.read()
                    frame = self.inference([img])[0]
                    # out.write(frame)
                    cv2.imshow('result',frame)
                    q=cv2.waitKey(10) & 0xFF
                    # cv2.imwrite('test_video1.jpg',frame)
                    if q == 27 or q ==ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()
        elif os.path.isfile(imgpath):
            img = cv2.imread(imgpath)
            imgname = imgpath.split('/')[-1][:-4] +'.jpg'
            if img.shape[1] > 3000:
                img = cv2.resize(img,(960,1280))
            if img is not None:
                frame,acc = self.inference([img])
                print('area',acc[0])
                cv2.imshow('result',frame[0])
                cv2.imshow('src',img)
                # cv2.imwrite(imgname,frame[0])
                key = cv2.waitKey(0) 
        else:
            print('please input the right img-path')

if __name__ == '__main__':
    args = parms()
    det = GroundSeg(args)
    det(args.file_in)
    # pa = '/data/detect/cityscape/gtFine/train/bochum/bochum_000000_000313_gtFine_labelIds.png'
    # img = cv2.imread(pa)
    # cv2.imshow('src',img)
    # cv2.waitKey(0)