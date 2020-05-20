import torch.nn as nn
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import random
import numbers
import torchvision
import csv

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    #     return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr
    # return lr

def get_label_info(csv_path):
    # return label -> {label_name: [r_value, g_value, b_value, ...}
    ann = pd.read_csv(csv_path)
    label = {}
    for iter, row in ann.iterrows():
        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        class_11 = row['class_11']
        label[label_name] = [int(r), int(g), int(b), class_11]
    return label

def one_hot_it(label, label_info):
    # return semantic_map -> [H, W]
    semantic_map = np.zeros(label.shape[:-1])
    for index, info in enumerate(label_info):
        color = label_info[info]
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map[class_map] = index
        # semantic_map.append(class_map)
    # semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def one_hot_it_v11(label, label_info):
    # return semantic_map -> [H, W, class_num]
    semantic_map = np.zeros(label.shape[:-1])
    # from 0 to 11, and 11 means void
    class_index = 0
    for index, info in enumerate(label_info):
        color = label_info[info][:3]
        class_11 = label_info[info][3]
        if class_11 == 1:
            # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            # semantic_map[class_map] = index
            semantic_map[class_map] = class_index
            class_index += 1
        else:
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            semantic_map[class_map] = 11
    return semantic_map

def one_hot_it_v11_dice(label, label_info):
    # return semantic_map -> [H, W, class_num]
    semantic_map = []
    void = np.zeros(label.shape[:2])
    for index, info in enumerate(label_info):
        color = label_info[info][:3]
        class_11 = label_info[info][3]
        if class_11 == 1:
            # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            # semantic_map[class_map] = index
            semantic_map.append(class_map)
        else:
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            void[class_map] = 1
    semantic_map.append(void)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float)
    return semantic_map

def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and height as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index
    # image = image.permute(0,2,3,1)
    x = torch.argmax(image, dim=-1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]
    label_values = [label_values[key][:3] for key in label_values if label_values[key][3] == 1]
    label_values.append([0, 0, 0])
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

def compute_global_accuracy(pred, label):
    # pred = pred.flatten()
    # label = label.flatten()
    # total = len(label)
    # count = 0.0
    # for i in range(total):
    #     if pred[i] == label[i]:
    #         count = count + 1.0
    tmp = torch.eq(pred,label)
    tmp_sum = tmp.sum().float()
    b,h,w = label.size()
    return tmp_sum/ (b*h*w)

def fast_hist(a, b, n):
    '''
    a and b are label and predict respectively
    n is the number of classes
    '''
    # k = (a >= 0) & (a < n)
    # return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)
    a=a.flatten()
    b=b.flatten()
    return torch.bincount(n*a.int()+b.int(),minlength=n**2).view(n,n)


def per_class_iu(hist):
    '''
    sum(hist,1): the row sum ,is the groundtruth nums
    sum(hist,0): the sum of collocum is the predict nums
    # the bug is: the iou not consider the true coordinate
    '''
    # epsilon = 1e-5
    # return (np.diag(hist) + epsilon) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)
    ep = torch.Tensor([1e-5])
    if torch.cuda.is_available():
        ep = ep.cuda()
    diaga = torch.diag(hist)
    return (diaga+ep)/(hist.sum(dim=1)+hist.sum(dim=0)-diaga+ep)

def cal_miou(miou_list, csv_path):
    # return label -> {label_name: [r_value, g_value, b_value, ...}
    f_in = open(csv_path,'r')
    ann = csv.DictReader(f_in)
    miou_dict = {}
    for row in ann: #ann.iterrows():
        label_name = row['name']
        cnt_lb = int(row['class_num'])
        miou_dict[label_name] = miou_list[cnt_lb]
    f_in.close()
    return miou_dict, np.mean(miou_list)

class OHEM_CrossEntroy_Loss(nn.Module):
    def __init__(self, threshold, keep_num):
        super(OHEM_CrossEntroy_Loss, self).__init__()
        self.threshold = threshold
        self.keep_num = keep_num
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def forward(self, output, target):
        loss = self.loss_function(output, target).view(-1)
        loss, loss_index = torch.sort(loss, descending=True)
        threshold_in_keep_num = loss[self.keep_num]
        if threshold_in_keep_num > self.threshold:
            loss = loss[loss>self.threshold]
        else:
            loss = loss[:self.keep_num]
        return torch.mean(loss)

def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(
        group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group
