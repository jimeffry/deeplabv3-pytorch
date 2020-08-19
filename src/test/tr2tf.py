import os
import sys
import cv2
import torch
import onnx
from onnx_tf.backend import prepare
# import tensorflow as tf
import numpy as np
from torch.autograd import Variable

import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.platform import gfile

sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from deeplabv3pluss import DeeplabV3plus
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

def rename_dict(state_dict):
    state_dict_new = dict()
    for key,value in list(state_dict.items()):
        state_dict_new[key[7:]] = value
    return state_dict_new

def tr2onnx(modelpath):
    # Load the trained model from file
    device = 'cpu'
    # net = shufflenet_v2_x1_0(pretrained=False,num_classes=6)
    net = DeeplabV3plus(cfgs).to(device)
    state_dict = torch.load(modelpath,map_location=device)
    # state_dict = rename_dict(state_dict)
    net.load_state_dict(state_dict)
    net.eval()
    # print(net)
    # Export the trained model to ONNX
    dummy_input = Variable(torch.randn(1,1088,1920,3)) # picture will be the input to the model
    export_onnx_file = '../models/deeplab.onnx'
    torch.onnx.export(net,
                    dummy_input,
                    export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True, # 是否执行常量折叠优化
                    input_names=["img_input"], # 输入名
                    output_names=["cls_out"], # 输出名
                    # dynamic_axes={"images":{0:"batch_size"}, # 批处理变量
                    #                 "location":{0:"batch_size"},
                    #                 "confidence":{0:"batch_size"}}
    )

def onnx2tf(modelpath):
    # Load the ONNX file
    model = onnx.load(modelpath)
    # Import the ONNX model to Tensorflow
    tf_rep = prepare(model)
    # Input nodes to the model
    print('inputs:', tf_rep.inputs)
    # Output nodes from the model
    print('outputs:', tf_rep.outputs)
    # All nodes in the model
    # print('tensor_dict:')
    # print(tf_rep.tensor_dict)
    # 运行tensorflow模型
    # print('Image 1:')
    # img = cv2.imread('/data/detect/shang_crowed/part_B_final/test_data/images/IMG_2.jpg')
    # img = cv2.resize(img,(1920,1080))
    # img = np.transpose(img,(2,0,1))
    # output = tf_rep.run(np.asarray(img, dtype=np.float32)[np.newaxis,:,:, :])
    # print('The digit is classified as ', np.sum(output))
    tf_rep.export_graph('../models/waterSeg.pb')

def pbtxt_to_graphdef(filename):
    with open(filename, 'r') as f:
        graph_def = tf.GraphDef()
        file_content = f.read()
        text_format.Merge(file_content, graph_def)
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, './', '../models/deeplab_v.pb', as_text=False)


def graphdef_to_pbtxt(filename):
    with gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, './', '../models/deeplab.pbtxt', as_text=True)
    


if __name__=='__main__':
    modelpath = '/data/models/img_seg/deeplabv3_voc_best3.pth'
    # modelpath = './srd_tr.pth'
    # tr2onnx(modelpath)
    # modelpath = '../models/deeplab.onnx'
    modelpath = '/data/waterSeg_18.onnx'
    onnx2tf(modelpath)
    modelpath = '../models/deeplab.pb'
    # modelpath = '/data/models/head/csr_keras.pb'
    # graphdef_to_pbtxt(modelpath)
    modelpath = '../models/deeplab.pbtxt'
    # pbtxt_to_graphdef(modelpath)