import os
import sys
import numpy as np 
import cv2

def saveframe(vpath,vname,imgdir):
    if not os.path.exists(imgdir):
        os.makedirs(imgdir)
    cap = cv2.VideoCapture(vpath)
    frame_width =  cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height =  cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cnt = 0
    while cnt < total_num and cap.isOpened():
        _,frame = cap.read()
        sys.stdout.write(">\r %d / %d" %(cnt,total_num))
        sys.stdout.flush()
        if cnt ==5 :
            img = frame
            savename = vname +'_'+str(cnt)+'.jpg'
            savepath = os.path.join(imgdir,savename)
            tx1,tx2,ty1,ty2 = [1415,1872,45,90]
            blx1,blx2,bly1,bly2 = [1770,1873,770,1024]
            brx1,brx2,bry1,bry2 = [65,1784,970,1030]
            # img[ty1:ty2,tx1:tx2,:] =  cv2.medianBlur(img[ty1:ty2,tx1:tx2,:],19)
            img[bly1:bly2,blx1:blx2,:] = cv2.medianBlur(img[bly1:bly2,blx1:blx2,:],31)
            img[bry1:bry2,brx1:brx2,:] = cv2.medianBlur(img[bry1:bry2,brx1:brx2,:],31)
            cv2.imwrite(savepath,img)
            break
        cnt +=1
    cap.release()

if __name__ == '__main__':
    # vpath = '/data/videos/movies/20190324060005_2558.avi'
    # vname = 'v4'
    # sdir = '/data/videos/mframes/video4'
    # saveframe(vpath,vname,sdir)
    vdir = '/data/videos/anShan'
    vdir = '/data/videos/LangZhong'
    file_cnts = os.listdir(vdir)
    vcnt = 1
    sdir = '/data/videos/anshan_crops2'
    sdir = '/data/videos/langzhang_crops'
    for tmp in file_cnts:
        tmp = tmp.strip()
        vpath = os.path.join(vdir,tmp)
        vname = 'lz'+str(vcnt)
        vsave = 'video'+str(vcnt)
        # vname = tmp.split('-')[-1][:-4]
        # vsavedir = os.path.join(sdir,vsave)
        saveframe(vpath,vname,sdir)
        vcnt+=1