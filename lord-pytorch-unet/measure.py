import numpy as np








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