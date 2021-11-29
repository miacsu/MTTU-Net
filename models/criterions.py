import torch
import logging
from torch.nn.functional import cross_entropy
import torch.nn.functional as F
from .util import weight_reduce_loss
from torch.autograd import Variable
# from .loss.seg_loss import ExpLog_loss
binary_cross_entropy = F.binary_cross_entropy
#idh_cross_entropy = F.cross_entropy
grade_cross_entropy = F.cross_entropy

from torch import nn


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num,loss_fn):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.loss_fn = loss_fn
        self.log_vars = nn.Parameter(torch.tensor((0.0,0.0),requires_grad=True)) #1.0, 6.0

    def forward(self, outputs,targets,weights):
        std_1 = torch.exp(self.log_vars[0]) ** 0.5
        std_2 = torch.exp(self.log_vars[1]) ** 0.5

        seg_loss, loss1,loss2,loss3 = self.loss_fn[0](outputs[0], targets[0],weights[0])
        #seg_loss_, loss1,loss2,loss3 = softmax_dice(outputs[0], targets[0],weights[0])
        #seg_loss = self.loss_fn[0](outputs[0], targets[0])
        seg_loss_1 = torch.sum(0.5 * torch.exp(-self.log_vars[0]) * seg_loss + self.log_vars[0],-1) #

        idh_loss = self.loss_fn[1](outputs[1], targets[1], weights[1])

        idh_loss_1 = torch.sum(0.5 * torch.exp(-self.log_vars[1]) * idh_loss + self.log_vars[1],-1)

        loss = torch.mean(seg_loss_1+idh_loss_1)

        return loss,seg_loss,idh_loss,loss1,loss2,loss3,std_1,std_2,self.log_vars[0],self.log_vars[1]

class MultiTaskLossWrapper_1(nn.Module):
    def __init__(self, task_num,loss_fn):
        super(MultiTaskLossWrapper_1, self).__init__()
        self.task_num = task_num
        self.loss_fn = loss_fn
        params = torch.ones(self.task_num,requires_grad=True)
        self.params = nn.Parameter(params) #1.0, 6.0

    def forward(self, outputs,targets,weights):
        std_1 = torch.log(1+self.params[0]**2)
        std_2 = torch.log(1+self.params[1]**2)

        seg_loss, loss1,loss2,loss3 = self.loss_fn[0](outputs[0], targets[0],weights[0])

        seg_loss_1 = torch.sum(0.5 / (self.params[0]**2) * seg_loss + torch.log(self.params[0]**2),-1) #

        idh_loss = self.loss_fn[1](outputs[1], targets[1], weights[1])

        idh_loss_1 = torch.sum(0.5 / (self.params[1]**2) * idh_loss + torch.log(self.params[1]**2),-1)

        loss = seg_loss_1+idh_loss_1

        return loss,seg_loss,idh_loss,loss1,loss2,loss3,std_1,std_2,self.params[0]**2,self.params[1]**2

class MultiTaskLossWrapper1(nn.Module):

    def __init__(self, task_num,loss_fn):
        super(MultiTaskLossWrapper1, self).__init__()
        self.task_num = task_num
        self.loss_fn = loss_fn
        self.vars = nn.Parameter(torch.randn(self.task_num,requires_grad=True))

    def forward(self, outputs,targets,weights):

        seg_loss, loss1,loss2,loss3 = self.loss_fn[0](outputs[0], targets[0],weights[0])

        seg_loss_1 = torch.sum(0.5 * seg_loss/ (self.vars[0]**2) + torch.log(self.vars[0]),-1)

        idh_loss = self.loss_fn[1](outputs[1], targets[1], weights[1])

        idh_loss_1 = torch.sum(0.5 * idh_loss /(self.vars[1]**2) + torch.log(self.vars[1]),-1)

        loss = torch.mean(seg_loss_1 + idh_loss_1)

        return loss,seg_loss,idh_loss,loss1,loss2,loss3,self.vars[0],self.vars[1]

# def build_expLog_loss(output, target):
#
#     explog_loss_1 = ExpLog_loss(output[:, 1, ...], (target == 1).float())
#     explog_loss_2 = ExpLog_loss(output[:, 2, ...], (target == 2).float())
#     explog_loss_3 = ExpLog_loss(output[:, 3, ...], (target == 4).float())
#     loss1 = Dice(output[:, 1, ...], (target == 1).float())
#     loss2 = Dice(output[:, 2, ...], (target == 2).float())
#     loss3 = Dice(output[:, 3, ...], (target == 4).float())
#
#     return explog_loss_1 + explog_loss_2 + explog_loss_3, 1 - loss1.data, 1 - loss2.data, 1 - loss3.data

def idh_cross_entropy(input,target,weight):

    return cross_entropy(input,target,weight=weight,ignore_index=-1)

def dice_weight_cross_entropy(pred,target,dice,weight):
    dice_weight = dice * torch.exp(dice)
    loss = cross_entropy(pred, target, weight) * dice_weight
    return loss

def dice_aware_cross_entropy(pred,target,dice_weight,weight,alpha=0.75,gamma=2.0):

    # target = torch.unsqueeze(target, 0)
    # assert pred.size() == target.size()
    pred_softmax = F.softmax(pred, 1)[0][1]
    target_1 = dice_weight
    # target = target.type_as(pred)
    if dice_weight > 0:
        focal_weight = target_1 * (target_1 > 0.0).float() + \
                       alpha * (pred_softmax - target_1).abs().pow(gamma) * \
                       (target_1 <= 0.0).float()
    else:
        logging.info("dice weight is equal to zero>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        focal_weight = (target_1 > 0.0).float() + \
                       alpha * (pred_softmax - target_1).abs().pow(gamma) * \
                       (target_1 <= 0.0).float()

    loss = cross_entropy(pred,target,weight) * focal_weight
    # loss = F.binary_cross_entropy_with_logits(
    #     pred, target, reduction='none') * focal_weight
    # loss = weight_reduce_loss(loss, weight, 'mean', None)

    return loss

def idh_focal_loss(input,target,weight):
    focalloss = FocalLoss(weight=weight)

    return focalloss(input, target)


def focal_loss(input,target,weight=None,gamma=2.0):
    input = input.float()
    target = target.long()
    ce_loss = F.cross_entropy(input, target, weight=weight, ignore_index=-1)
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
    return focal_loss

def build_masked_loss(output, target, weight=None,mask_value=-1):
    """Builds a loss function that masks based on targets

    Args:
        loss_function: The loss function to mask
        mask_value: The value to mask in the targets

    Returns:
        function: a loss function that acts like loss_function with masked inputs
    """

    return cross_entropy(output, target, weight=weight,ignore_index=mask_value)


def expand_target(x, n_class,mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:, 1, :, :, :] = (x == 1)
        xx[:, 2, :, :, :] = (x == 2)
        xx[:, 3, :, :, :] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:, 0, :, :, :] = (x == 1)
        xx[:, 1, :, :, :] = (x == 2)
        xx[:, 2, :, :, :] = (x == 3)
    return xx.to(x.device)

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
    return transposed.reshape(C, -1)

def Dice(output,target,weight=None, eps=1e-5):
    target = target.float()
    if weight is None:
        num = 2 * (output * target).sum()
        den = output.sum() + target.sum() + eps
    else:

        # sum_dims = list(range(1, output.dim()))
        #
        # num = 2 * torch.sum(weight * output * target, dim=sum_dims)
        # den= torch.sum(weight * (output + target), dim=sum_dims) + eps
        num = 2 * (weight * output * target).sum()
        den = (weight*output).sum() + (weight*target).sum() + eps
    return 1.0 - num/den


def dice_loss(logit: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None):
    """
    References
    ----------
    `Dice Loss <https://arxiv.org/abs/1606.04797>`_
    """
    if not (target.size() == logit.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))

    preds = torch.sigmoid(logit)

    sum_dims = list(range(1, logit.dim()))

    if weight is None:
        dice = 2 * torch.sum(preds * target, dim=sum_dims) / torch.sum(preds ** 2 + target ** 2, dim=sum_dims)
    else:
        dice = 2 * torch.sum(weight * preds * target, dim=sum_dims) \
               / torch.sum(weight * (preds ** 2 + target ** 2), dim=sum_dims)
    loss = 1 - dice

    return loss.mean()


def softmax_dice(output, target,weight=None):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    if weight is None:
        loss1 = Dice(output[:, 1, ...], (target == 1).float())
        loss2 = Dice(output[:, 2, ...], (target == 2).float())
        loss3 = Dice(output[:, 3, ...], (target == 3).float()) 
        # wt_dice = Dice(output>0,target>0)
        return loss1 + loss2 + loss3, 1 - loss1.data, 1 - loss2.data, 1 - loss3.data
    else:
        weight_1 = weight
        weight_2 = weight
        weight_3 = weight
        weight_1[torch.where(target != 1)] = 1
        weight_2[torch.where(target != 2)] = 1
        weight_3[torch.where(target != 4)] = 1
        loss1_ = Dice(output[:, 1, ...], (target == 1).float(), weight_1)
        loss2_ = Dice(output[:, 2, ...], (target == 2).float(), weight_2)
        loss3_ = Dice(output[:, 3, ...], (target == 4).float(), weight_3)

        loss1 = Dice(output[:, 1, ...], (target == 1).float())
        loss2 = Dice(output[:, 2, ...], (target == 2).float())
        loss3 = Dice(output[:, 3, ...], (target == 4).float())

        return loss1_+loss2_+loss3_, 1-loss1.data, 1-loss2.data, 1-loss3.data


def boundary_softmax_dice(output,boundary, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''

    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 4).float())

    boundary1 = dice_crsss_entropy(boundary[:, 1, ...].unsqueeze(1), get_boundary_3d((target == 1).float()))
    boundary2 = dice_crsss_entropy(boundary[:, 2, ...].unsqueeze(1), get_boundary_3d((target == 2).float()))
    boundary3 = dice_crsss_entropy(boundary[:, 3, ...].unsqueeze(1), get_boundary_3d((target == 4).float()))

    return loss1 + loss2 + loss3+boundary1+boundary2+boundary3, 1 - loss1.data, 1 - loss2.data, 1 - loss3.data,boundary1,boundary2,boundary3


def dice_crsss_entropy(output,target):

    bce_loss = F.binary_cross_entropy_with_logits(output, target)
    dice_loss = Dice(torch.sigmoid(output), target)
    return bce_loss+dice_loss

def get_boundary_3d(gtmasks):
    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 26, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        dtype=torch.float32, device=gtmasks.device).reshape(1, 1, 3, 3, 3).requires_grad_(False)
    boundary_targets = F.conv3d(gtmasks.unsqueeze(1), laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0
    return boundary_targets




def softmax_dice1(output, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''

    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 4).float())

    return loss1,loss2,loss3, 1-loss1.data, 1-loss2.data, 1-loss3.data

def softmax_dice2(output, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    loss0 = Dice(output[:, 0, ...], (target == 0).float())
    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 4).float())

    return loss1 + loss2 + loss3 + loss0, 1-loss1.data, 1-loss2.data, 1-loss3.data


def sigmoid_dice(output, target):
    '''
    The dice loss for using sigmoid activation function
    :param output: (b, num_class-1, d, h, w)
    :param target: (b, d, h, w)
    :return:
    '''
    loss1 = Dice(output[:, 0, ...], (target == 1).float())
    loss2 = Dice(output[:, 1, ...], (target == 2).float())
    loss3 = Dice(output[:, 2, ...], (target == 4).float())

    return loss1 + loss2 + loss3, 1-loss1.data, 1-loss2.data, 1-loss3.data


def Generalized_dice(output, target, eps=1e-5, weight_type='square'):
    if target.dim() == 4:  #(b, h, w, d)
        target[target == 4] = 3  #transfer label 4 to 3
        target = expand_target(target, n_class=output.size()[1])  #extend target from (b, h, w, d) to (b, c, h, w, d)

    output = flatten(output)[1:, ...]  # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:, ...]  # [class, N*H*W*D]

    target_sum = target.sum(-1)  # sub_class_voxels [3,1] -> 3个voxels
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :', weight_type)

    # print(class_weights)
    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2*intersect[0] / (denominator[0] + eps)
    loss2 = 2*intersect[1] / (denominator[1] + eps)
    loss3 = 2*intersect[2] / (denominator[2] + eps)

    return 1 - 2. * intersect_sum / denominator_sum, loss1, loss2, loss3


def Dual_focal_loss(output, target):

    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 4).float())
    
    if target.dim() == 4:  #(b, h, w, d)
        target[target == 4] = 3  #transfer label 4 to 3
        target = expand_target(target, n_class=output.size()[1])  #extend target from (b, h, w, d) to (b, c, h, w, d)

    target = target.permute(1, 0, 2, 3, 4).contiguous()
    output = output.permute(1, 0, 2, 3, 4).contiguous()
    target = target.view(4, -1)
    output = output.view(4, -1)
    log = 1-(target - output)**2

    return -(F.log_softmax((1-(target - output)**2), 0)).mean(), 1-loss1.data, 1-loss2.data, 1-loss3.data


class FocalLoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss1(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss1, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class FocalLoss_seg(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss_seg, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        # 
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')
        
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
