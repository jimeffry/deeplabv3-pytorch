#!/usr/bin/bash

#description
#deeplabv3_tfv3.pb is trained on voc and wuwei dataset, class number is 22,backone is deeplabv3
#deeplabv3_24tf2.pb is trained on voc and wuwei dataset, class number is 24,backone is deeplabv3
#deeplabv3_wuwei_best.pth is trained on voc and wuwei dataset, class number is 22
#deeplabv3rend_voc_best is trained on voc and wuwei dataset, class number is 24,backone is deeplabv3plus + pointrend
#deeplabv3_wuwei24_best.pth is trained on voc and wuwei dataset, class number is 24; convert deeplabv24.pb, backone is deeplabv3plus
#deeplabplus_24tf.pb is trained on voc and wuwei dataset, backone is deeplabv3plus,class number is 24


# python train/train.py --num_epochs 1000 --learning_rate 1e-3  --cuda 0,2 --batch_size 2  --optimizer sgd --validation_step 200 --show_step 20 --use_gpu false --losstype multi #--mulgpu True #--pretrained_model_path /mnt/data/LXY.data/models/imgseg/bs_ohem_best.pth
#***
# python utils/processdataset.py
# python utils/mat2png.py
# python preparedata/contexvoc.py
# python utils/getpoints.py
# python utils/pjson.py
# python utils/generate_list.py
# python utils/spider.py
# python utils/load_label.py
# python utils/video.py

#** test
# python test/demo.py --modelpath /data/models/img_seg/deeplabv3_voc_best.pth --file_in /data/detect/test_seg_ims --labelpath ../datasets/voc2010v2.csv --save_dir ../datasets/voctest  
# python test/demo.py --modelpath /data/models/img_seg/deeplabv3_voc_best.pth --file_in /data/detect/test_seg_ims/d7.jpg  --labelpath ../datasets/voc2010v2.csv --save_dir ../datasets/voctest  
# python test/demo.py --modelpath /data/models/img_seg/deeplabv3_voc_best3.pth --file_in /home/lxy/Desktop/wuwei/images/2020-07-16-08-40-01-2819_24.jpg  --labelpath ../datasets/voc2010v3.csv
# python test/demo.py --modelpath /data/models/img_seg/deeplabv3_voc_best3.pth --file_in /home/lxy/Desktop/wuwei/images/  --labelpath ../datasets/voc2010v2.csv --save_dir /home/lxy/Desktop/wuwei/result_tr
# python test/demo.py --modelpath /data/models/img_seg/deeplabv3_wuwei_best.pth --file_in /home/lxy/Desktop/wuwei/images/  --labelpath ../datasets/voc2010v2.csv --save_dir /home/lxy/Desktop/wuwei/result_tr3
# python test/demo.py --modelpath /data/models/img_seg/deeplabv3rend_voc_best.pth --file_in /home/lxy/Desktop/wuwei/images/ --labelpath ../datasets/voc2010v3.csv --save_dir /home/lxy/Desktop/wuwei/result_tr3
# python test/demo.py --modelpath /data/models/img_seg/deeplabv3rend_voc_best.pth --file_in /home/lxy/Desktop/wuwei/images/2020-07-16-08-50-01-2872_24.jpg --labelpath ../datasets/voc2010v3.csv 
# python test/demo.py --modelpath /data/models/img_seg/deeplabv3_wuwei24_best.pth --file_in /home/lxy/Desktop/imgzip --labelpath ../datasets/voc2010v3.csv --save_dir /home/lxy/Desktop/imgzipresult
# python test/demo.py --modelpath /data/models/img_seg/deeplabv3_wuwei24_best.pth --file_in /data/test4.mp4 --labelpath ../datasets/voc2010v3.csv --save_dir /home/lxy/Desktop/imgzipresult

# pb test
# python test/demo_tf.py --modelpath ../models/deeplab_v.pb --file_in /data/detect/test_seg_ims/d7.jpg  --labelpath ../datasets/voc2010v2.csv --maskpath ./v20_mask.jpg
# python test/demo_tf.py --modelpath /data/models/img_seg/deeplabv3_tf.pb --file_in /data/videos/crop_frames/video4/vl4_0.jpg  --labelpath ../datasets/voc2010v2.csv --maskpath ./v20_mask.jpg
# python test/demo_tf.py --modelpath /data/models/img_seg/deeplabv3_tf.pb --file_in /data/videos/anshan_crops2  --labelpath ../datasets/voc2010v2.csv --maskpath ./v20_mask.jpg --save_dir /data/videos/anshan_crops
# python test/demo_tf.py  --modelpath  /data/models/img_seg/deeplabv3_tfv3.pb --file_in /home/lxy/Desktop/wuwei/images/  --labelpath ../datasets/voc2010v2.csv  --save_dir /home/lxy/Desktop/wuwei/results2/
# python test/demo_tf.py  --modelpath  /data/models/img_seg/deeplabv3_24tf2.pb --file_in /home/lxy/Desktop/test1.jpg  --labelpath ../datasets/voc2010v3.csv  --save_dir /home/lxy/Desktop/wuwei/results3/
# python test/demo_tf.py  --modelpath  /data/models/img_seg/deeplabplus_24tf.pb --file_in /home/lxy/Desktop/anshan/images/1930_5.jpg --labelpath ../datasets/voc2010v3.csv  
# python test/demo_tf.py  --modelpath  /data/models/img_seg/deeplabv24.pb --file_in /data/videos/langzhang_crops --labelpath ../datasets/voc2010v3.csv --save_dir /data/videos/langzhang_crops
python test/demo_tf.py  --modelpath  /data/models/img_seg/deeplabv24.pb --file_in ~/Desktop/wuwei/images/2020-07-16-08-50-01-2864_24.jpg --labelpath ../datasets/voc2010v3.csv --save_dir /home/lxy/Desktop/imgzipresult

#**convert pth to pb and pb2ptx
# python test/tr2tf.py

