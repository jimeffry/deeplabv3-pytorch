import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class FocalLoss(nn.Module):
    '''
    :param input: 
    :param target:
    :return:
    '''
    def __init__(self, alpha=0.5, gamma=2):
        super(FocalLoss,self).__init__()
        self.gamma = gamma

    def forward(self,batchinput,target):
        n, c, h, w = batchinput.size()
        # inputs = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        inputs = batchinput.permute(0,2,3,1).contiguous().view(n,-1,c)
        targets = target.view(n,-1)
        with torch.no_grad():
            frequency = torch.FloatTensor(c,n)
            frequency = frequency.to(batchinput.device)
            for i in torch.arange(0,c):
                i = i.float()
                tmp = torch.sum(targets.eq(i),axis=1)
                i = i.long()
                frequency[i]=tmp
            frequency = torch.transpose(frequency,0,1)
            hist_num = frequency.sum(axis=1)
            # print(hist_num)
            hist_num = hist_num.unsqueeze(1).expand_as(frequency)
            # print(hist_num)
            classweights = frequency/(hist_num+1)
            targets = targets.long()
            weights = classweights.gather(1,targets)
            weights = 1-weights
        P = F.softmax(inputs, dim=2)#shape [num_samples,num_classes]
        class_mask = inputs.data.new(n*h*w,c).fill_(0)
        # class_mask = Variable(class_mask)
        ids = target.view(-1, 1).long()
        class_mask.scatter_(1, ids.data, 1.)#shape [num_samples,num_classes]  one-hot encoding
        class_mask = class_mask.view(n,h*w,c)
        # print(class_mask)
        probs = (P * class_mask).sum(2).view(n,-1) + 1e-5 #shape [num_samples,]
        log_p = probs.log()
        # print('in calculating batch_loss',weights.shape,probs.shape,log_p.shape)
        batch_loss = -weights * (torch.pow((1 - probs),self.gamma)) * log_p
        # batch_loss = -(torch.pow((1 - probs), gamma)) * log_p
        # print(batch_loss.shape)
        loss = batch_loss.mean() 
        return loss

def compute_class_weights(histogram):
    classWeights = np.ones(6, dtype=np.float32)
    normHist = histogram / np.sum(histogram)
    for i in range(6):
        classWeights[i] = 1 / (np.log(1.10 + normHist[i]))
    return classWeights

class focalloss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


if __name__=='__main__':
    a=torch.ones([2,3,5,5],dtype=torch.float32)
    b=torch.ones([2,1,5,5],dtype=torch.float32)
    crit  = FocalLoss()
    c= crit(a,b)
    print(c)