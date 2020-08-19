import numpy as np
import os
import sys

def gen_ffmpeg(dpath):
    fr = open(dpath,'r')
    cnts = fr.readlines()
    # tmp_ffp = "nohup /root/yunlivideo/svr200524/ffmpeg -y -rtsp_transport tcp -i %s -vcodec copy -an -t 600 -f mp4 /data/video/$TT-%s.mp4 &"
    tmp_ffp = "/root/video-test/ffmpeg -y -rtsp_transport tcp -i %s -vcodec copy -an -t 300 -f mp4 /home/videos/$TT-%s.mp4 &"
    tmp1 = "#!/usr/bin/bash"
    tmp2 = 'TT=`date "+%Y-%m-%d-%H-%M-%S"`'
    fpath = "../videos_wuwei/save_video%s.sh"
    fcnt =1
    tpath = fpath % str(fcnt)
    fw = open(tpath,'w')
    fw.write(tmp1+"\n")
    fw.write(tmp2+"\n")
    for i in range(len(cnts)):
        if (i+1)%30==0:
            fcnt+=1
            tpath = fpath % str(fcnt)
            fw.close()
            fw = open(tpath,'w')
            fw.write(tmp1+"\n")
            fw.write(tmp2+"\n")
        save_ffp = tmp_ffp %(cnts[i].strip(),str(i))
        fw.write(save_ffp+"\n")

def gen_crontab(dpath):
    fw = open(dpath,'w')
    # tmp = "%s %s %s 6 * root /root/yunlivideo/svr200524/videosh/save_video%s.sh"
    tmp = "%s %s %s 7 * /root/wuweicrontab/save_video%s.sh"
    d_list = [15,16,17,18]
    h_list = [7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    mt_list = [0,10,20,30,40,50]
    cnt =1
    for k in d_list:
        for i in h_list:
            for j in mt_list:
                t_cat = tmp %(str(j),str(i),str(k),str(cnt))
                cnt+=1
                fw.write(t_cat+"\n")
                if cnt ==319:
                    break
    print("over")

if __name__=='__main__':
    txtpath = '../datasets/anshan_rtl.txt'
    txtpath = '../datasets/wuwei_rtl.txt'
    # gen_ffmpeg(txtpath)
    cronp = "../datasets/wuwei_crontab.txt"
    gen_crontab(cronp)
