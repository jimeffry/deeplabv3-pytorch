import numpy as np 
import cv2
from matplotlib import pyplot as plt

def get_boxes(frame,bg_img):
    rectangles = []
    frame_diff = np.abs(frame - bg_img)
    frame_diff = np.where(frame_diff > 50,255,0)
    frame_diff = np.uint8(frame_diff)
    # th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    # th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    # dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (55,1 ))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 55))
    frame_diff = cv2.dilate(frame_diff, kernelX, iterations=2)
    frame_diff = cv2.erode(frame_diff, kernelX,  iterations=4)
    frame_diff = cv2.dilate(frame_diff, kernelX,  iterations=2)
    frame_diff = cv2.erode(frame_diff, kernelY,  iterations=1)
    frame_diff = cv2.dilate(frame_diff, kernelY,  iterations=2)
    frame_diff = cv2.medianBlur(frame_diff, 3)
    # image, contours, hier = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours,image = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rectangles.append([x,y,w,h])
    return rectangles

def main(videopath):
    camera = cv2.VideoCapture(videopath)
    history = 20    
    # bs = cv2.createBackgroundSubtractorKNN(detectShadows=False)  
    # bs = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=history)
    # bs.setHistory(history)
    frame_cnt = 0
    check_cnt = 0
    frame_list = []
    frame_width =  int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height =  int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    refer_fg = 1
    refer_boxes = []
    alarm_cnt = 0
    alarm_out = 0
    while camera.isOpened():
        res, frame = camera.read()
        if not res:
            break
        frame_org = frame.copy()
        # fg_mask = bs.apply(frame)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if frame_cnt < history:
            frame_cnt += 1
            frame_list.append(frame)
            continue
        # bg = bs.getUpdateBackgroundModel()
        if frame_cnt >= history and refer_fg:
            frame_list = np.array(frame_list)
            frame_bg = np.mean(frame_list,axis=0)
            refer_fg = 0
        cur_boxes = get_boxes(frame,frame_bg)
        if check_cnt == 50:
            refer_boxes = cur_boxes
            check_cnt = 0
            alarm_cnt = 0
        if check_cnt < 50:
            check_cnt +=1
            if len(refer_boxes)>0:
                for gallar in refer_boxes:
                    for prob in cur_boxes:
                        tmp = np.array(gallar) - np.array(prob)
                        tmp = np.abs(tmp)
                        if tmp[0] < 10 and tmp[1]<10 and tmp[2]<10 and tmp[3]<10:
                            alarm_cnt +=1
        if alarm_cnt > 45:
            alarm_out = 1
        for bb in cur_boxes:
            cv2.rectangle(frame_org,(int(bb[0]),int(bb[1])),(int(bb[0]+bb[2]),int(bb[1]+bb[3])),(0,255,0),2)
            cv2.putText(frame_org,str(alarm_out),(100,100),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
        for bb in refer_boxes:
            cv2.rectangle(frame_org,(int(bb[0]),int(bb[1])),(int(bb[0]+bb[2]),int(bb[1]+bb[3])),(0,0,255),1)
        cv2.imshow("src", frame_org)
        cv2.imshow("back", np.uint8(frame_bg))
        print(check_cnt,len(refer_boxes))
        if cv2.waitKey(110) & 0xff == 27:
            break
    camera.release()

def check_color(fpath):
    '''
    hsv- green: 45<h<90  43<s<255  46<v<255
    yello: 5<h<20  43<s<255  46<v<255
    red: 0<h<20 or 175<h<255,
    '''
    img = cv2.imread(fpath)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,w = hsv.shape[:2]
    # for green
    #lower_blue = np.array([5,43,43])   
    #upper_blue = np.array([20,255,255])
    # for red
    
    lower_blue1 = np.array([0,43,43])   
    upper_blue1 = np.array([20,255,255])
    lower_blue2 = np.array([175,43,43])   
    upper_blue2 = np.array([255,255,255])
    # # Threshold the HSV image to get only blue colors
    mask1 = cv2.inRange(hsv,lower_blue1,upper_blue1)
    mask2 = cv2.inRange(hsv,lower_blue2,upper_blue2)
    mask = mask1+mask2
    # # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img,mask=mask)
    cv2.imshow('frame',img)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    cv2.waitKey(0)
    mask = mask/255.0
    sum_area = np.sum(mask)
    
    # plothist(hsv[:,:,0].flatten())
    # cv2.imwrite('red1.jpg',res)
    return float(sum_area)/(h*w)
    # return 1

def plothist(datadict):
    fig, ax = plt.subplots(1, 1, figsize=(9, 3), sharey=True)
    # ax.bar(xdata,ydata)
    xn,bd,paths = ax.hist(datadict,bins=20)
    fw = open('../datasets/hsv_color_red.txt','w')
    for idx,tmp in enumerate(xn):
        fw.write("{}:{}\n".format(tmp,bd[idx]))
    fw.close()
    plt.savefig('../datasets/hsv_color_red.png',format='png')
    plt.show()

def get_boxarea(img,frame):
        '''
        img: gray img
        '''
        img = img[0]
        dencity_map = img.copy()
        imgh,imgw = img.shape[:2]
        frameh,framew = frame.shape[:2]
        # print('min',np.min(img))
        # print('max',np.max(img))
        # img = np.where(img >0.0002,255,0)
        _,img = cv2.threshold(img,0.0002,255,cv2.THRESH_BINARY)
        img = np.array(img,dtype=np.uint8)
        # cv2.imshow('thresh',img)
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size,1 ))
        kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.kernel_size))
        img = cv2.dilate(img, kernelX, iterations=2)
        img = cv2.erode(img, kernelX,  iterations=4)
        img = cv2.dilate(img, kernelX,  iterations=2)
        img = cv2.erode(img, kernelY,  iterations=1)
        img = cv2.dilate(img, kernelY,  iterations=2)
        img = cv2.medianBlur(img, 3)
        # img = cv2.medianBlur(img, 15)
        # cv2.imshow('dilate&erode', img)
        #输入的三个参数分别为：输入图像、层次类型、轮廓逼近方法
        #返回的三个返回值分别为：修改后的图像、图轮廓、层次
        # image, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        hull = []
        for i ,c in enumerate(contours):
            # 边界框
            x, y, w, h = cv2.boundingRect(c)
            hull.append(cv2.convexHull(c, False))
            if min(w,h) > 100:
                # x2 = int((x+w)/float(imgw) * framew)
                # y2 = int((y+h)/float(imgh) * frameh)
                # x1 = int(x/imgw *framew)
                # y1 = int(y/imgh *frameh)
                x1,x2,y1,y2 = int(x),int(x+w),int(y),int(y+h)
                # tmp = int(np.sum(dencity_map[x1:x2+1,y1:y2+1]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.putText(frame,str(tmp),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                boxes.append([x1,y1,x2,y2])
                cv2.polylines(frame, [hull[i]], True, (0, 255, 0), 2)
            # cv2.drawContours(frame, hull, i, (0, 255, 0), 1, 8)
            # cv2.polylines(frame, [hull[i]], True, (0, 255, 0), 2)
        return frame,boxes

if __name__ == '__main__':
    vpath = '/data/videos/mask/b6.mp4'
    # main(vpath)
    p = check_color('/home/lxy/Desktop/huo/h5.jpg')
    print(p)