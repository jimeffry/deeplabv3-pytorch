#!/usr/bin/bash
# python train/train.py --num_epochs 1000 --learning_rate 1e-3  --cuda 0,2 --batch_size 8  --optimizer sgd --validation_step 200 --show_step 20 --use_gpu True --mulgpu True #--pretrained_model_path /mnt/data/LXY.data/models/imgseg/bs_ohem_best.pth
#***
# python utils/processdataset.py
# python utils/mat2png.py
# python preparedata/contexvoc.py
python utils/getpoints.py
#** test
# python test/demo.py --modelpath /data/models/img_seg/deeplabv3_voc_best.pth --file_in /data/detect/test_seg_ims --labelpath ../datasets/voc2010v2.csv --save_dir ../datasets/voctest  
# python test/demo.py --modelpath /data/models/img_seg/deeplabv3_voc_best.pth --file_in /data/detect/test_seg_ims/d6.jpg  --labelpath ../datasets/voc2010v2.csv --save_dir ../datasets/voctest  
# python test/demo.py --modelpath /data/models/img_seg/deeplabv3_voc_best3.pth --file_in /data/videos/mframes/video2/v2_20.jpg  --labelpath ../datasets/voc2010v2.csv

# pb test
# python test/demo_tf.py --modelpath ../models/deeplab_v.pb --file_in /data/videos/mframes/video2/v2_20.jpg  --labelpath ../datasets/voc2010v2.csv --maskpath ./v20_mask.jpg

#**convert pth to pb and pb2ptx
# python test/tr2tf.py


