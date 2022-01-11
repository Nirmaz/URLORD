import numpy as np
from torch import nn
import torch.nn.functional as F
import torch


# class BootstrappedCE(nn.Module):
#     def __init__(self, start_warm = 1, end_warm = 10, top_p=0.15):
#         super().__init__()
#
#         self.start_warm = start_warm
#         self.end_warm = end_warm
#         self.top_p = top_p
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     def forward(self, input, target, it):
#         # print(input.size(), "input size first")
#         # print(target.size(), "input size first")
#         # print(target.long(), "target first ")
#         input = input.view(input.size()[0], -1)
#         target = target.view(input.size()[0], -1)
#         # print(input.size(), "input size")
#         # print(target.size(), "input size")
#         # print(target.long(), "target")
#         if it < self.start_warm:
#             return F.binary_cross_entropy(input, target.float()), 1.0
#
#         raw_loss = F.binary_cross_entropy(input, target.float(), reduction = 'none').view(-1)
#         # print(raw_loss.size())
#         x = torch.tensor(0).to(self.device)
#         n_raw_loss = torch.where(target.view(-1) == 1, raw_loss.float(), x.float())
#         num_pixels = raw_loss.numel()
#
#         if it > self.end_warm:
#             this_p = self.top_p
#         else:
#             this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
#
#         loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
#         loss2, _ = torch.topk(n_raw_loss, int(num_pixels * this_p), sorted=False)
#         return (loss.mean() * 0.5) +  (0.5 * loss2.mean()), this_p

class BootstrappedCE(nn.Module):
    def __init__(self, start_warm = 1, end_warm = 10, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input, target, it):
        # print(input.size(), "input size first")
        # print(target.size(), "input size first")
        # print(target.long(), "target first ")
        input = input.view(input.size()[0], -1)
        target = target.view(input.size()[0], -1)
        # print(input.size(), "input size")
        # print(target.size(), "input size")
        # print(target.long(), "target")
        if it < self.start_warm:
            return F.binary_cross_entropy(input, target.float()), 1.0

        raw_loss = F.binary_cross_entropy(input, target.float(), reduction = 'none').view(-1)
        # print(raw_loss.size())
        # x = torch.tensor(0).to(self.device)
        # n_raw_loss = torch.where(target.view(-1) == 1, raw_loss.float(), x.float())
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))

        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        # loss2, _ = torch.topk(n_raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p





def dice_func(gt_seg, prediction_seg):
    """
    compute dice coefficient
    :param gt_seg:
    :param prediction_seg:
    :return: dice coefficient between gt and predictions
    """
    seg1 = np.asarray(gt_seg).astype(np.bool)
    seg2 = np.asarray(prediction_seg).astype(np.bool)

    # Compute Dice coefficient
    intersection = np.logical_and(seg1, seg2)
    if seg1.sum() + seg2.sum() == 0:
        return 1
    return float(format(2. * intersection.sum() / (seg1.sum() + seg2.sum()), '.3f'))

def dice_loss(inputs, targets, smooth = 1):
	"""This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

	# inputs = F.sigmoid(inputs)
	# flatten label and prediction tensors
	inputs = inputs.view(-1)
	targets = targets.view(-1)
	intersection = (inputs * targets).sum()
	dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
	return 1 - dice