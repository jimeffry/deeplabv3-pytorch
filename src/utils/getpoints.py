import cv2
import numpy as np

pts = []

def draw_roi(event, x, y, flags, param):
    global pts,img
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击，选择点
        pts.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:  # 右键点击，取消最近一次选择的点
        pts.pop()
    if event == cv2.EVENT_MBUTTONDOWN:  # 中键绘制轮廓
        mask = np.zeros(img.shape, np.uint8)
        points = np.array(pts, np.int32)
        points = points.reshape((-1, 1, 2))
 
        # 画多边形
        mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
        mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))  # 用于求 ROI
        mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))      # 用于 显示在桌面的图像
        show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)
 
        cv2.imshow("mask", mask)
        cv2.imshow("show_img", show_image)
 
        ROI = cv2.bitwise_and(mask2, img)
        cv2.imshow("ROI", ROI)
        # cv2.imwrite('v20_mask.jpg',mask2)
        cv2.waitKey(0)
 
    if len(pts) > 0:
        # 将pts中的最后一点画出来
        cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)
 
    if len(pts) > 1:
        # 画线
        for i in range(len(pts) - 1):
            cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)  # x ,y 为鼠标点击地方的坐标
            cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)
 
    cv2.imshow('image', img2)
 
 
# 创建图像与窗口并将窗口与回调函数绑定
img = cv2.imread("/data/videos/mframes/video2/v2_20.jpg")
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_roi)
print("[INFO] 单击左键：选择点，单击右键：删除上一次选择的点，单击中键：确定ROI区域")
print("[INFO] 按‘S’确定选择区域并保存")
print("[INFO] 按 ESC 退出")
 
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord("s"):
        saved_data = {"ROI": pts}
        # joblib.dump(value=saved_data, filename="config.pkl")
        print("[INFO] ROI坐标已保存到本地.")
        break
cv2.destroyAllWindows()
'''
import numpy as np 
import cv2
import sys
from time import time


selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

inteval = 1
duration = 0.01


# mouse callback function
def draw_boundingbox(event, x, y, flags, param):
    global selectingObject, initTracking, onTracking, ix, iy, cx,cy, w, h
    
    if event == cv2.EVENT_LBUTTONDOWN:
        selectingObject = True
        onTracking = False
        ix, iy = x, y
        cx, cy = x, y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        cx, cy = x, y
    
    elif event == cv2.EVENT_LBUTTONUP:
        selectingObject = False
        if(abs(x-ix)>10 and abs(y-iy)>10):
            w, h = abs(x - ix), abs(y - iy)
            ix, iy = min(x, ix), min(y, iy)
            initTracking = True
        else:
            onTracking = False
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        onTracking = False
        if(w>0):
            ix, iy = x-w/2, y-h/2
            initTracking = True



if __name__ == '__main__':

    if(len(sys.argv)==1):
        cap = cv2.VideoCapture(0)
    elif(len(sys.argv)==2):
        if(sys.argv[1].isdigit()):  # True if sys.argv[1] is str of a nonnegative integer
            cap = cv2.VideoCapture(int(sys.argv[1]))
        else:
            cap = cv2.VideoCapture(sys.argv[1])
            inteval = 30
    else:  assert(0), "too many arguments"
        #'MOSSE' 130,CSK 15, CN 9, DSST 5,DSST-LP 9, Staple 40,Staple-CA 20,KCF_CN 5,KCF_GRAY 15,KCF_HOG 20,DCF_GRAY 40，DCF_HOG 30,LDES 7,MKCFup 10,MKCFup-LP 10
        #DAT 100，BACF 15+，CSRDCF 5,CSRDCF-LP 10,SAMF 5,STRCF 15,MCCTH-Staple 15
    tracker = PyTracker('Staple')  # hog, fixed_window, multiscale

    cv2.namedWindow('tracking')
    cv2.setMouseCallback('tracking',draw_boundingbox)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        if(selectingObject):
            cv2.rectangle(frame,(ix,iy), (cx,cy), (0,255,255), 1)
        elif(initTracking):
            cv2.rectangle(frame,(ix,iy), (ix+w,iy+h), (0,255,255), 2)

            tracker.init([ix,iy,w,h], frame)

            initTracking = False
            onTracking = True
        elif(onTracking):
            t0 = time()
            boundingbox = tracker.update(frame)
            t1 = time()

            boundingbox = list(map(int, boundingbox))
            cv2.rectangle(frame,(boundingbox[0],boundingbox[1]), (boundingbox[0]+boundingbox[2],boundingbox[1]+boundingbox[3]), (0,255,255), 1)
            
            duration = 0.8*duration + 0.2*(t1-t0)
            #duration = t1-t0
            cv2.putText(frame, 'FPS: '+str(1/duration)[:4].strip('.'), (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow('tracking', frame)
        c = cv2.waitKey(inteval) & 0xFF
        if c==27 or c==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
'''