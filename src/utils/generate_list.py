import numpy as np
import os
import sys

def gen_ffmpeg(dpath):
    fr = open(dpath,'r')
    cnts = fr.readlines()
    # tmp_ffp = "nohup /root/yunlivideo/svr200524/ffmpeg -y -rtsp_transport tcp -i %s -vcodec copy -an -t 600 -f mp4 /data/video/$TT-%s.mp4 &"
    tmp_ffp = "/data1/yunli/YunLiVideo/svr/ffmpeg -y -rtsp_transport tcp -i %s -vcodec copy -an -t 300 -f mp4 /data1/yunli/YunLiVideo/videosave/$TT-%s.mp4 &"
    tmp1 = "#!/usr/bin/bash"
    tmp2 = 'TT=`date "+%Y-%m-%d-%H-%M-%S"`'
    fpath = "../videos_siping/save_video%s.sh"
    fcnt =193 #1
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
    tmp = "%s %s %s 9 * /data1/yunli/YunLiVideo/crontabs/save_video%s.sh"
    d_list = [9]
    h_list = [8,9,10,11,12,13,14,15,16,17]
    mt_list = [0,6,12,18,24,30,36,42,48,54]
    cnt =1 #1
    for k in d_list:
        if cnt ==3:
            break
        for i in h_list:
            if cnt == 3:
                break
            for j in mt_list:
                t_cat = tmp %(str(j),str(i),str(k),str(cnt))
                cnt+=1
                fw.write(t_cat+"\n")
                if cnt ==3:
                    break
    print("over")

if __name__=='__main__':
    txtpath = '../datasets/anshan_rtl.txt'
    txtpath = '../datasets/wuwei_rtl.txt'
    txtpath = '../datasets/siping_rtl.txt'
    txtpath = '../datasets/siping_rtl_80.txt'
    gen_ffmpeg(txtpath)
    # cronp = "../datasets/wuwei_crontab.txt"
    cronp = '../datasets/siping_crontab2.txt'
    # gen_crontab(cronp)
