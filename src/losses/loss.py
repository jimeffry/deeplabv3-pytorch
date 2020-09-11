import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from point_features import point_sample

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5

    def forward(self, output, target):
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target)
        # intersect = (output * target).sum(-1).sum() + self.epsilon
        # denominator = ((output + target).sum(-1)).sum() + self.epsilon

        intersect = (output * target).sum(-1)
        denominator = (output + target).sum(-1)
        dice = 2*intersect / denominator
        dice = torch.mean(dice)
        return 1 - dice
        # return 1 - 2. * intersect / denominator


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.n_min = n_min
        self.criteria = nn.CrossEntropyLoss( reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        # labels = labels.squeeze(1)
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=1, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss()

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss

class Multiloss(nn.Module):
    def __init__(self,upscale):
        super(Multiloss,self).__init__()
        self.common_stride = upscale

    def forward(self,seglogits,pointlogits,pointcoords,targets):
        point_targets = (point_sample(targets.unsqueeze(1).to(torch.float),
                    pointcoords,
                    mode="nearest"
                )
                .squeeze(1)
                .to(torch.long)
            )
        print('loss',pointcoords.shape,targets.shape,point_targets.shape,pointlogits.shape)
        ploss = F.cross_entropy(
                pointlogits, point_targets, reduction="mean"
            )
        seglogits = F.interpolate(
            seglogits, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        segloss = F.cross_entropy(
            seglogits, targets, reduction="mean"
        )
        return ploss+segloss