import numpy as np 
from matplotlib import pyplot as plt 
import os
import sys
import cv2
import tqdm
import json
import csv

def plothist(datadict):
    # xdata = datadict.keys()
    # ydata = []
    # for tmp in xdata:
    #     ydata.append(datadict[tmp])
    print('total plt:',len(datadict))
    fig, ax = plt.subplots(1, 1, figsize=(9, 3), sharey=True)
    # ax.bar(xdata,ydata)
    xn,bd,paths = ax.hist(datadict,bins=50)
    fw = open('../datasets/voc_maxsize.txt','w')
    for idx,tmp in enumerate(xn):
        fw.write("{}:{}\n".format(tmp,bd[idx]))
    fw.close()
    plt.savefig('../datasets/voc_maxsize.png',format='png')
    plt.show()

def get_data(imgdir):
    datas = []
    f1_cnts = os.listdir(imgdir)
    for i in tqdm.tqdm(range(len(f1_cnts))):
        tmp_f = f1_cnts[i].strip()
        tmpdir = os.path.join(imgdir,tmp_f)
        f2_cnts = os.listdir(tmpdir)
        for imgname in f2_cnts:
            imgpath = os.path.join(tmpdir,imgname.strip())
            img = cv2.imread(imgpath)
            h,w = img.shape[:2]
            datas.append(max(h,w))
    plothist(datas)

def get_color(filein,fileout):
    '''
    read labels from json file of cityscapes and save to a csv file
    '''
    with open(filein, 'r') as fr:
        labels_info = json.load(fr)
    fw = open(fileout,'w')
    fw.write("name,r,g,b,class_num\n")
    for tmp in labels_info:
        ids = int(tmp['trainId'])
        if ids >=0 and ids !=255:
            name = tmp['name'].replace(' ','_')
            r,g,b = tmp['color']
            ids +=1
            fw.write("{},{},{},{},{}\n".format(name,r,g,b,ids))
    fw.close()
    fr.close()

def get_camvid_label_img(csv_path):
    '''
    save camvid dataset color of labels into a image
    '''
    # ann = csv.read_csv(csv_path)
    f_in = open(csv_path,'r')
    ann = csv.DictReader(f_in)
    label_info = {}
    for row in ann: #ann.iterrows()
        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        class_11 = row['class_11']
        label_info[label_name] = [int(r), int(g), int(b), class_11]
    label_values = [label_info[key][:3] for key in label_info if label_info[key][3] == 1]
    keys_lab = [key for key in label_info if label_info[key][3]==1]
    label_values = np.array(label_values)
    img = np.zeros([1200,1200],dtype=np.int16)
    for i in range(11):
        img[i*100:(i+1)*100,:] = i
    label_img = label_values[img.astype(int)]
    label_img = np.uint8(label_img)
    print(keys_lab)
    for j in range(11):
        points = (int(20),int((j+1)*100))
        font=cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_scale = 1
        color = (255,255,255)
        cv2.putText(label_img,keys_lab[j], points, font, font_scale, color, 2)
    cv2.imwrite('label_img.jpg', cv2.cvtColor(label_img, cv2.COLOR_RGB2BGR))

def get_cityscape_label_img(csvpath):
    f_in = open(csvpath,'r')
    ann = csv.DictReader(f_in)
    label_info = {}
    for row in ann: #ann.iterrows():
        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        class_19 = row['class_num']
        label_info[label_name] = [int(r), int(g), int(b), class_19]
    label_values = [label_info[key][:3] for key in label_info ]
    keys_lab = [key for key in label_info]
    label_values.insert(0,[0,0,0])
    label_values = np.array(label_values)
    img = np.zeros([1200,1200],dtype=np.int16)
    for i in range(1,24):
        img[i*50:(i+1)*50,:] = i
    label_img = label_values[img.astype(int)]
    label_img = np.uint8(label_img)
    for j in range(23):
        points = (int(20),int((j+1)*50))
        font=cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_scale = 1
        color = (255,255,255)
        cv2.putText(label_img,keys_lab[j], points, font, font_scale, color, 2)
    cv2.imwrite('../datasets/voc_colorv3.jpg', cv2.cvtColor(label_img, cv2.COLOR_RGB2BGR))

def getdatalist(imgdir,outfile1,outfile2):
    fcnts = os.listdir(imgdir)
    fw = open(outfile1,'w')
    fw2 = open(outfile2,'w')
    cnt = 0
    total = len(fcnts)
    valc = total-200
    for tmp in fcnts:
        cnt+=1
        imgname = tmp.strip()[:-4]
        if cnt < valc:
            fw.write(imgname+'\n')
        else:
            fw2.write(imgname+'\n')
    fw.close()
    fw2.close()

def genlistdir1(imgdir,outfile):
    '''
    imgdir: imgdir/images.jpg
    '''
    fcnts = os.listdir(imgdir)
    fw = open(outfile,'w')
    cnt = 0
    for tmp in fcnts:
        cnt+=1
        imgname = tmp.strip()[:-4]
        fw.write(imgname+'\n')
    fw.close()
    print('total images:',cnt)

if __name__=='__main__':
    # get_data('/data/detect/VOC/VOCdevkit/VOC2010/labels')
    # get_color('../datasets/cityscapes_info.json','../datasets/cityscape.txt')
    get_cityscape_label_img('../datasets/voc2010v3.csv')
    # getdatalist('/data/detect/VOC/VOCdevkit/VOC2010/Seglabels22','../datasets/voctrain.txt','../datasets/vocval.txt')
    # genlistdir1('/home/lxy/Desktop/anshan/正常标注/','../datasets/anshan_train.txt')