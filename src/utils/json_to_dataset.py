import argparse
import json
import os
import sys
import os.path as osp
import warnings
 
import PIL.Image
import yaml
 
from labelme import utils
import base64
import csv
import numpy as np
# import cv2

def loadlabel(labelpath):
    f_in = open(labelpath,'r')
    ann = csv.DictReader(f_in)
    label_info = {}
    for row in ann: 
        label_name = row['name']
        class_id = row['class_num']
        label_info[label_name] = int(class_id)
    f_in.close()
    return label_info
 
def main():
    warnings.warn("This script is aimed to demonstrate how to convert the\n"
                  "JSON file to a single image dataset, and not to handle\n"
                  "multiple JSON files to generate a real-use dataset.")
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('-i', '--label_file', default=None)
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()
 
    json_file = args.json_file
    if args.out is None:
        out_dir = osp.basename(json_file).replace('.', '_')
        out_dir = osp.join(osp.dirname(json_file), out_dir)
    else:
        out_dir = args.out
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    if args.label_file is None:
        print("please input label file")
        return 0
    else:
        label_name_to_value = loadlabel(args.label_file)
 
    count = os.listdir(json_file) 
    total_num = len(count)
    for i in range(0,total_num):
        sys.stdout.write(">\r %d / %d" %(i,total_num))
        sys.stdout.flush()
        if not count[i].endswith('json'):
            continue
        jsonpath = os.path.join(json_file, count[i])
        if os.path.isfile(jsonpath):
            data = json.load(open(jsonpath))
            # print(data['imagePath'])
            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(jsonpath), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')
            img = utils.img_b64_to_arr(imageData)
            label_name_to_value['_background_'] = 0
            # label_values, label_names = [], []
            # for shape in data['shapes']:
            #     label_name = shape['label']
            #     if label_name in label_name_to_value:
            #         label_value = label_name_to_value[label_name]
            #         # print(label_value)
            #         label_names.append(label_name)
                # else:
                #     label_value = len(label_name_to_value)
                #     label_name_to_value[label_name] = label_value
            
            # label_values must be dense
            
            # for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
            #     label_values.append(lv)
            #     label_names.append(ln)
            # assert label_values == list(range(len(label_values)))
            
            lbl,lnames = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
            
            # captions = ['{}: {}'.format(lv, ln)
                # for ln, lv in label_name_to_value.items()]
            # lbl_viz = utils.draw_label(lbl, img, captions)
            
            # out_dir = osp.basename(count[i]).replace('.', '_')
            # out_dir = osp.join(osp.dirname(count[i]), out_dir)
            # if not osp.exists(out_dir):
            #     os.mkdir(out_dir)
            imgname = count[i][:-5]+'_label.png'
            outputpath = osp.join(out_dir,imgname)
            # PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
            PIL.Image.fromarray(lbl.astype(np.uint8)).save(outputpath)
            # cv2.imwrite(osp.join(out_dir, 'label.png'),np.uint8(lbl))
            # utils.lblsave(osp.join(out_dir, 'label.png'), lbl)
            # PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))
            # with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
            #     for lbl_name in label_names:
            #         f.write(lbl_name + '\n')
 
            # warnings.warn('info.yaml is being replaced by label_names.txt')
            # info = dict(label_names=label_names)
            # with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
            #     yaml.safe_dump(info, f, default_flow_style=False)
 
            # print('Saved to: %s' % out_dir)
if __name__ == '__main__':
    main()