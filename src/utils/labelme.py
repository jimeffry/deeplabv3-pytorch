import argparse
import json
import os
import os.path as osp
import warnings
import numpy as np
import PIL.Image
import yaml
# from labelme import utils
# import labelme
import cv2
 
 
def main():
    json_file='/data/videos/mframes/video1' 
    dataset_file='/data/videos/mframes/test' 
    filecnts = os.listdir(json_file)
    for i in range(0, len(filecnts)):
        tmp = filecnts[i].strip()
        if not tmp.endswith('json'):
            continue
        path = os.path.join(json_file, tmp)
        if os.path.isfile(path):
            data = json.load(open(path))
            img = labelme.utils.img_b64_to_arr(data['imageData'])
            lbl, lbl_names = labelme.utils.labelme_shapes_to_label(img.shape, data['shapes'])
            captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
            lbl_viz = labelme.utils.draw_label(lbl, img, captions)
            out_dir = osp.basename(tmp).replace('.', '_')
            out_dir = osp.join(osp.join(dataset_file,osp.dirname(tmp)), out_dir)
            if not osp.exists(out_dir):
                os.mkdir(out_dir)
            PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
            PIL.Image.fromarray(lbl).save(osp.join(out_dir, 'label.png'))
            PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))
            with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
                for lbl_name in lbl_names:
                    f.write(lbl_name + '\n')
            warnings.warn('info.yaml is being replaced by label_names.txt')
            info = dict(label_names=lbl_names)
            with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
                yaml.safe_dump(info, f, default_flow_style=False)
            print('Saved to: %s' % out_dir)
 
 
if __name__ == '__main__':
    # main()
    pathd = '/data/videos/mframes/v1_10_json/label.png'
    img = cv2.imread(pathd)
    cv2.imshow('src',img)
    cv2.waitKey(0)
    