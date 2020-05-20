import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import time
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__),'../preparedata'))
# from cityscape import CityScapes
from contexvoc import ContextVoc
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from deeplabv3pluss import DeeplabV3plus
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from util import reverse_one_hot, compute_global_accuracy,cal_miou,fast_hist,per_class_iu
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_out', type=str, default=None, help='txt file name')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--losstype', type=str, default='crossentropy', help='loss function, dice or crossentropy')
    parser.add_argument('--mulgpu', type=str2bool, default=False, help='whether user more gpu')
    return parser.parse_args()

def load_net(args):
    cropsize = [cfgs.crop_height, cfgs.crop_width]
    # dataset_train = CityScapes(cfgs.data_dir, cropsize=cropsize, mode='train')
    dataset_val = ContextVoc(cfgs.val_file,cropsize=cropsize, mode='train')
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    # build net
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # model = BiSeNet(args.num_classes, args.context_path)
    net = DeeplabV3plus(cfgs).to(device)
    if args.mulgpu:
        net = torch.nn.DataParallel(net)
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        # net.module.load_state_dict(torch.load(args.pretrained_model_path,map_location=device))
        net.load_state_dict(torch.load(args.pretrained_model_path,map_location=device))
        print('Done!')
    net.eval()
    return net,dataloader_val


def val(args, model, dataloader):
    with torch.no_grad():
        precision_record = []
        hist = torch.zeros((cfgs.num_classes, cfgs.num_classes))
        if torch.cuda.is_available():
            hist = hist.cuda()
        for i, (data, label) in enumerate(dataloader):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            # get RGB predict image
            # label = label.squeeze(1)
            t1=time.time()
            predict = model(data)
            t2 = time.time()
            print('inference time:',t2-t1)
            predict = reverse_one_hot(predict)
            if args.losstype == 'dice':
                label = reverse_one_hot(label)
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label[0], predict[0], cfgs.num_classes)
            precision_record.append(precision)
        precision_record = torch.Tensor(precision_record)
        precision = torch.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou_list = miou_list.cpu().numpy()
        miou_dict,miou = cal_miou(miou_list,cfgs.LabelFile)
        # logger.info('mIoU for validation: %.3f' % miou)
        return miou_dict,miou,precision.cpu().numpy()

def main():
    args = params()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    net,dataloader_val = load_net(args)
    # writer = SummaryWriter(log_dir=args.logdir,comment=''.format(args.optimizer, cfgs.netname))
    miou_dict,miou,acc = val(args, net, dataloader_val)
    print('miou:',miou)
    print('acc:',acc)
    fw = open(args.file_out,'w')
    keys = miou_dict.keys()
    for tmp in keys:
        fw.write('{}:{}\n'.format(tmp,miou_dict[tmp]))
    fw.close()

if __name__=='__main__':
    main()