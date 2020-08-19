#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import sys
import os
import argparse
import tensorflow as tf
from tensorflow.python.platform import gfile

import cv2
import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import csv

def parms():
    parser = argparse.ArgumentParser(description='CSRnet demo')
    parser.add_argument('--save_dir', type=str, default='tmp/',
                        help='Directory for detect result')
    parser.add_argument('--modelpath', type=str,
                        default='weights/s3fd.pth', help='trained model')
    parser.add_argument('--threshold', default=0.65, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--ctx', default=True, type=bool,
                        help='gpu run')
    parser.add_argument('--img_dir', type=str, default='tmp/',
                        help='Directory for images')
    parser.add_argument('--file_in', type=str, default='tmp.txt',
                        help='image namesf')
    parser.add_argument('--labelpath', type=str,
                        default='', help='trained model')
    parser.add_argument('--maskpath', type=str,
                        default='', help='trained model')
    return parser.parse_args()


class ImgSeg(object):
    def __init__(self,args):
        self.loadtfmodel(args.modelpath)
        self.threshold = args.threshold
        self.img_dir = args.img_dir
        self.real_num = 0
        self.loadlabel(args.labelpath)
        self.save_dir = args.save_dir
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # self.mask = cv2.imread(args.maskpath)
        self.kernel_size = 25

    def loadtfmodel(self,mpath):
        tf_config = tf.ConfigProto()
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        #tf_config.gpu_options = gpu_options
        tf_config.gpu_options.allow_growth=True  
        tf_config.log_device_placement=False
        self.sess = tf.Session(config=tf_config)
        # self.sess = tf.Session()
        modefile = gfile.FastGFile(mpath, 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(modefile.read())
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name='') 
        # tf.train.write_graph(graph_def, './', 'breathtest.pbtxt', as_text=True)
        # print("************begin to print graph*******************")
        # op = self.sess.graph.get_operations()
        # for m in op:
        #     # if 'input' in m.name or 'output' in m.name or 'confidence' in m.name:
        #     print(m.values()) #m.name,
        # print("********************end***************")
        # self.input_image = self.sess.graph.get_tensor_by_name('img_input:0') #img_input
        # self.cls_out = self.sess.graph.get_tensor_by_name('cls_out:0') #softmax_output
        self.input_image = self.sess.graph.get_tensor_by_name('img_input:0')
        self.cls_out = self.sess.graph.get_tensor_by_name('resnet_v1_101/logits/cls_out:0')

        
    def propress(self,img):
        rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        # img = cv2.resize(img,(1920,1080))
        h,w = img.shape[:2]
        gth = int(np.ceil(h/16.0)*16)
        gtw = int(np.ceil(w/16.0)*16)
        # gth,gtw = (1088,1920)
        img = cv2.resize(img,(gtw,gth))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img.astype('float32')
        img /= 255.0
        img -= rgb_mean
        img /= rgb_std
        return img


    def inference_img(self,imgorg):
        t1 = time.time()
        imgh,imgw = imgorg.shape[:2]
        img = self.propress(imgorg.copy())
        img = np.expand_dims(img,0)
        cls_out = self.sess.run(self.cls_out,feed_dict={self.input_image:img})
        # print("***out shape:",np.shape(output))
        # cls_out = np.argmax(conf_out,axis=-1)
        cls_out = self.rescalImage(cls_out,imgw,imgh)
        masks = self.decodeColor(cls_out)
        accs = [0]
        # accs = self.calarea(cls_out)
        t2 = time.time()
        print('consuming:',t2-t1)
        # imgs_out,bboxes = self.get_boxarea(cls_out,[imgorg])
        return masks,accs

    def rescalImage(self,imglist,imgw,imgh):
        imgout = []
        for i in range(imglist.shape[0]):
            # print(imglist[i].shape)
            tmp = cv2.resize(imglist[i].astype(np.uint8),(imgw,imgh),cv2.INTER_NEAREST)
            imgout.append(tmp)
        return np.array(imgout)
    
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
        id_list = [1] #[10,12,13,14,16,19]
        acc = 0
        for i in range(predicts.shape[0]):
            img = predicts[i]
            # h,w = img.shape[:2]
            binary_mask = np.where(self.mask[:,:,0]>250,1,0)
            total_pixes = np.sum(binary_mask)
            img +=1
            img_roi = cv2.bitwise_and(self.mask[:,:,0], img)
            for tid in id_list:
                mask_tmp = np.equal(img_roi,tid)
                acc += np.sum(mask_tmp.astype(np.float))
            acc = acc/total_pixes
            accout.append(acc)
        return accout

    def get_boxarea(self,clsmaps,framelist):
        '''
        img: gray img
        '''
        imgs_out = []
        bboxes_out = []
        convex_points = []
        for i in range(clsmaps.shape[0]):
            tmp_map = clsmaps[i]
            frame = framelist[i]
            _,tmp_map = cv2.threshold(tmp_map,1,255,cv2.THRESH_BINARY_INV)
            # cv2.imshow('thresh',tmp_map)
            kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size,1 ))
            kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.kernel_size))
            # tmp_map = cv2.dilate(tmp_map, kernelX, iterations=2)
            tmp_map = cv2.erode(tmp_map, kernelX,  iterations=2)
            tmp_map = cv2.dilate(tmp_map, kernelX,  iterations=2)
            tmp_map = cv2.erode(tmp_map, kernelY,  iterations=2)
            tmp_map = cv2.dilate(tmp_map, kernelY,  iterations=2)
            # tmp_map = cv2.medianBlur(tmp_map, 3)
            # img = cv2.medianBlur(img, 15)
            # cv2.imshow('dilate&erode', tmp_map)
            #输入的三个参数分别为：输入图像、层次类型、轮廓逼近方法
            #返回的三个返回值分别为：修改后的图像、图轮廓、层次
            # image, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours, _ = cv2.findContours(tmp_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            hull = []
            for i ,c in enumerate(contours):
                # 边界框
                x, y, w, h = cv2.boundingRect(c)
                hull.append(cv2.convexHull(c, False))
                x2 = int(x+w)
                y2 = int(y+h)
                x1 = int(x)
                y1 = int(y)
                # tmp = int(np.sum(dencity_map[x1:x2+1,y1:y2+1]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.putText(frame,str(tmp),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                boxes.append([x1,y1,x2,y2])
                # cv2.drawContours(frame, contours, i, (0, 255, 0), 1, 8)
                cv2.polylines(frame, [hull[i]], True, (0, 255, 0), 2)
            bboxes_out.append(boxes)
            convex_points.append(hull)
            imgs_out.append(frame)
        return imgs_out,bboxes_out

    def inference(self,imgpath):
        '''
        tl: x1=1415,y1=45 x2=1872,y2=90
        bl: 1770,770 1873,1024
        bl: 1390,967 1784,1024
        br: 65,970,500,1030
        '''
        if os.path.isdir(imgpath):
            cnts = os.listdir(imgpath)
            for tmp in cnts:
                tmppath = os.path.join(imgpath,tmp.strip())
                img = cv2.imread(tmppath)
                if img is None:
                    continue
                tx1,tx2,ty1,ty2 = [1415,1872,45,90]
                blx1,blx2,bly1,bly2 = [1770,1873,770,1024]
                brx1,brx2,bry1,bry2 = [65,1784,970,1030]
                img[ty1:ty2,tx1:tx2,:] =  cv2.medianBlur(img[ty1:ty2,tx1:tx2,:],19)
                img[bly1:bly2,blx1:blx2,:] = cv2.medianBlur(img[bly1:bly2,blx1:blx2,:],31)
                img[bry1:bry2,brx1:brx2,:] = cv2.medianBlur(img[bry1:bry2,brx1:brx2,:],31)
                # img[ty1:ty2,tx1:tx2,:] = cv2.bilateralFilter(img[ty1:ty2,tx1:tx2,:],31,20,5)
                # img[ty1:ty2,tx1:tx2,:] = cv2.GaussianBlur(img[ty1:ty2,tx1:tx2,:],(19,19),0)
                frame,areas = self.inference_img(img)
                # print('area >> ',areas[0])
                # cv2.imshow('result',img)
                save_name = tmp[:-4]
                savepath = os.path.join(self.save_dir,save_name)
                # cv2.imwrite(savepath+'.jpg',img)
                cv2.imwrite(savepath+'.png',frame[0])
                cv2.waitKey(10) 
        elif os.path.isfile(imgpath) and imgpath.endswith('txt'):
            f_r = open(imgpath,'r')
            file_cnts = f_r.readlines()
            for j in tqdm(range(len(file_cnts))):
                tmp_file = file_cnts[j].strip()
                # if not tmp_file.endswith('jpg'):
                    # tmp_file = tmp_file +'.jpg'
                tmp_path = os.path.join(self.img_dir,tmp_file) 
                # tmp_path = tmp_file
                if not os.path.exists(tmp_path):
                    print(tmp_path)
                    continue
                img = cv2.imread(tmp_path) 
                if img is None:
                    print('None',tmp_path)
                    continue
                frame,areas = self.inference_img(img)
                # cv2.imshow('result',frame[0])
                save_name = tmp_file.split('.')[0]+'_r.png'
                savepath = os.path.join(self.save_dir,save_name)
                cv2.imwrite(savepath,frame[0])
                # cv2.waitKey(0) 
        elif os.path.isfile(imgpath) and imgpath.endswith(('.mp4','.avi')) :
            cap = cv2.VideoCapture(imgpath)
            frame_width =  cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height =  cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # print(frame_width, frame_height)
            imgw = int(frame_width)
            imgh = int(frame_height)
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') #cv2.VideoWriter_fourcc(*"mp4v")
            # out = cv2.VideoWriter('test.mp4', fourcc, 25,(frame_width, frame_height))
            out = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc('I','4','2','0'), 25, (imgw, imgh))
            if not cap.isOpened():
                print("failed open camera")
                return 0
            else: 
                frame_cnt = 0
                boxes = []
                while cap.isOpened():
                    _,frame = cap.read()
                    frame_cnt +=1
                    if frame_cnt % 10 ==0:
                        frame,areas = self.inference_img(frame)
                    out.write(frame)
                    cv2.imshow('result',frame[0])
                    q=cv2.waitKey(10) & 0xFF
                    # cv2.imwrite('test_video1.jpg',frame)
                    if q == 27 or q ==ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()
        elif os.path.isfile(imgpath):
            img = cv2.imread(imgpath)
            imgname = imgpath.split('/')[-1].strip()
            if img is not None:
                # grab next frame
                # update FPS counter
                frame_re,show_imgs = self.inference_img(img)
                # hotmaps = self.get_hotmaps(odm_maps)
                # self.display_hotmap(hotmaps)
                # keybindings for display
                # show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=self.mask, beta=0.2, gamma=0)
                # print("area ",areas[0])
                cv2.imshow('result',frame_re[0])
                # cv2.imshow('mask',show_image)
                # cv2.imshow('bbox',show_imgs[0])
                cv2.imwrite(imgname,frame_re[0])
                # cv2.imwrite(imgname+'.png',show_imgs[0])
                key = cv2.waitKey(0) 
        else:
            print('please input the right img-path')

if __name__ == '__main__':
    args = parms()
    detector = ImgSeg(args)
    imgpath = args.file_in
    detector.inference(imgpath)