import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
import numpy as np
import collections
import logging
import tqdm
import time
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__),'../preparedata'))
# from cityscape import CityScapes
from contexvoc import ContextVoc
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from deeplabv3pluss import DeeplabV3plus
sys.path.append(os.path.join(os.path.dirname(__file__),'../losses'))
from loss import DiceLoss,OhemCELoss,SoftmaxFocalLoss
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from util import poly_lr_scheduler
from util import reverse_one_hot, compute_global_accuracy
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--show_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--netname', type=str, default="resnet18",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--logdir', type=str, default='../logs', help='path of log data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--losstype', type=str, default='crossentropy', help='loss function, dice or crossentropy')
    parser.add_argument('--mulgpu', type=str2bool, default=True, help='whether user more gpu')
    return parser.parse_args()


def createlogger(lpath):
    if not os.path.exists(lpath):
        os.makedirs(lpath)
    logger = logging.getLogger()
    logname= time.strftime('%F-%T',time.localtime()).replace(':','-')+'.log'
    logpath = os.path.join(lpath,logname)
    hdlr = logging.FileHandler(logpath)
    logger.addHandler(hdlr)
    # logger.addHandler(logging.StreamHandler())
    logger.setLevel(level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    return logger

def train_net(args):
    cropsize = [cfgs.crop_height, cfgs.crop_width]
    # dataset_train = CityScapes(cfgs.data_dir, cropsize=cropsize, mode='train')
    dataset_train = ContextVoc(cfgs.train_file,cropsize=cropsize, mode='train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    # dataset_val = CityScapes(cfgs.data_dir,  mode='val')
    dataset_val = ContextVoc(cfgs.val_file,cropsize=cropsize, mode='train')
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=2,
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
        net.module.load_state_dict(torch.load(args.pretrained_model_path,map_location=device))
        # net.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')
    net.train()
    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(net.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), args.learning_rate)
    else:  
        print('not supported optimizer \n')
        optimizer = None
    #build loss
    if args.losstype == 'dice':
        criterion = DiceLoss()
    elif args.losstype == 'crossentropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.losstype == 'ohem':
        score_thres = 0.7
        n_min = args.batch_size * cfgs.crop_height * cfgs.crop_width //16
        criterion = OhemCELoss(thresh=score_thres, n_min=n_min)
    elif args.losstype == 'focal':
        criterion = SoftmaxFocalLoss()
    return net,optimizer,criterion,dataloader_train,dataloader_val

def main():
    args = params()
    logger = createlogger(args.logdir)
    net,optimizer,criterion,dataloader_train,dataloader_val = train_net(args)
    # writer = SummaryWriter(log_dir=args.logdir,comment=''.format(args.optimizer, cfgs.netname))
    max_miou = 0
    step = 0
    if not os.path.exists(cfgs.save_model_path):
        os.makedirs(cfgs.save_model_path)
    loss_hist = collections.deque(maxlen=200)
    rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
    rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
    for epoch in range(args.epoch_start,args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        for i, (images, target) in enumerate(dataloader_train):
            net.train()
            if args.use_gpu:
                images = images.cuda()
                target = target.cuda()
            '''
            targets = label.numpy()
            images = data.numpy()
            for i in range(targets.shape[0]):
                print(np.shape(images[i]))
                tmp_img = np.transpose(images[i],(1,2,0))
                tmp_img = tmp_img *rgb_std
                tmp_img = tmp_img + rgb_mean
                tmp_img = tmp_img * 255
                tmp_img = np.array(tmp_img,dtype=np.uint8)
                tmp_img = cv2.cvtColor(tmp_img,cv2.COLOR_RGB2BGR)
                h,w = tmp_img.shape[:2]
                gt = np.transpose(targets[i],(1,2,0))
                gt = np.array(gt,dtype=np.uint8)
                cv2.imshow('src',tmp_img)
                cv2.imshow('gt',gt)
                cv2.waitKey(0)
            '''
            output = net(images)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            # writer.add_scalar('loss_step', loss, step)
            loss_hist.append(float(loss.item()))
            if step % args.show_step == 0:
                # print('tl',images.size(),target.size())
                logger.info('epoch:{} || iter:{} || mloss:{:.6f}, cntloss:{:.6f} || lr:{:.6f}'.format(epoch,step,np.mean(loss_hist),loss.item(),optimizer.param_groups[0]['lr']))
            if step % args.validation_step == 0 :
                precision = val(args, net, dataloader_val,logger)
                # print("val precision: {:.6f}".format(precision))
                if precision > max_miou:
                    max_miou = precision
                    torch.save(net.module.state_dict(),os.path.join(cfgs.save_model_path, 'deeplabv3_voc_best.pth'))
                    logger.info("saved model: ************* step: %d" % step)
                # writer.add_scalar('step/precision_val', precision, step)
                # writer.add_scalar('step/miou val', miou, step)

def val(args, model, dataloader,logger):
    with torch.no_grad():
        model.eval()
        num = int(len(dataloader))
        precision_record = []
        # hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            # get RGB predict image
            # label = label.squeeze(1)
            predict = model(data)
            # for i in range(predicts.size(0)):
                # predict = predicts[i]
            predict = reverse_one_hot(predict)
                # get RGB label image
                # label = labels[i]
            if args.losstype == 'dice':
                label = reverse_one_hot(label)
            # label = np.array(label)
                # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
                # hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            precision_record.append(precision)
            # precision_record[0,i] = precision_record.cpu().squeeze()
        precision_record = torch.Tensor(precision_record)
        precision = torch.mean(precision_record)
        # miou_list = per_class_iu(hist)[:-1]
        # miou = np.mean(miou_list)
        logger.info('precision per pixel for test: %.6f' % precision.cpu())
        # logger.info('mIoU for validation: %.3f' % miou)
        return precision.cpu()


if __name__ == '__main__':
    main()