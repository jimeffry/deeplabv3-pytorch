#!/bin/bash
# cd /root/yunlivideo/svr200524
# cd /home/lxy/Develop/git_prj/deeplabv3-pytorch/datasets
# function rand(){    
#     min=$1    
#     max=$(($2-$min+1))    
#     num=$RANDOM     
#     echo $(($num%$max+$min))    
# }    

# rnd=$(rand 2 5)

# #rnd=$(rand 10 20 )# 410 420， 810 820,1200 1210,1600 1610,2000 2010,2400 2410,2800 2810，3200 3210,3600 3610，
#    #4000 4010； 4400 4410；4800 4810;5200 5210;5600 5610;6000 6010;6400 6410;6800 6810;7200 7210;7600 7610；
#    # 8000 8010
# # echo $rnd
# indx=0
# # echo $indx
# cat video_test.txt | while read line
# do
#     #echo $line
#     cur_date="`date +%Y-%m-%d_%H:%M:%S`-2.gif"
#     #echo $cur_date 
#     let "indx+=1"
#     if (($indx > $rnd)); then
#         # ffmpeg -i $line -vcodec h264 /data/videos/test_v/$cur_date
#         ffmpeg -i $line -t 10 -s 320x240 -pix_fmt rgb24 /data/videos/test_v/$cur_date
#         # echo $indx 
#     fi
# done
#crontab
#0 8-18 * * *  root /root/yunlivideo/svr200524/save_video.sh
#0 6-19 * * *  root /root/yunlivideo/svr200524/save_video2.sh
ffmpeg -i /data/videos/mask/b1.mp4 -t 10 -s 320x240 -pix_fmt rgb8 /data/videos/test_v/t1.gif &
ffmpeg -i /data/videos/mask/b2.mp4 -t 10 -s 320x240 -pix_fmt rgb8 /data/videos/test_v/t2.gif &
ffmpeg -i /data/videos/mask/b3.mp4 -t 10 -s 320x240 -pix_fmt rgb8 /data/videos/test_v/t3.gif &
ffmpeg -i /data/videos/mask/b4.mp4 -t 10 -s 320x240 -pix_fmt rgb8 /data/videos/test_v/t4.gif &