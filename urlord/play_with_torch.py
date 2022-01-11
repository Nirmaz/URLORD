import torch
import nibabel as nib
from os.path import join
import numpy as np
from torch import FloatTensor
from torch.autograd import Variable
if __name__ == '__main__':

    a = torch.tensor([[1,1,1], [2,2,2]])
    b = torch.tensor([[3], [3]])

    c = a * b

    a = Variable(FloatTensor([4]))

    weights = [Variable(FloatTensor([i]), requires_grad=True) for i in (2, 5, 9, 7)]

    # unpack the weights for nicer assignment
    w1, w2, w3, w4 = weights

    b = w1 * a
    c = w2 * a
    d = w3 * b + w4 * c
    L = (10 - d)

    L.register_hook(lambda grad: print(grad))
    d.register_hook(lambda grad: print(grad))
    b.register_hook(lambda grad: print(grad))
    c.register_hook(lambda grad: print(grad))
    b.register_hook(lambda grad: print(grad))

    L.backward()
    # print(f"L: {L.grad.data}")
    for index, weight in enumerate(weights, start=1):
        gradient, *_ = weight.grad.data
        print(f"Gradient of w{index} w.r.t to L: {gradient}")

    # print(b)
    # print(a)
    # print(c)
    # scan_file = nib.load("/mnt/local/nirm/placenta_new/placenta/91/volume.nii.gz")
    # nib.save(nib.Nifti1Image(scan_file.get_fdata(), np.eye(4)), '/mnt/local/nirm/placenta_new/placenta/91/aaa.nii.gz')
    #
    # scan_file = nib.load("/mnt/local/nirm/placenta_new/placenta/258/volume.nii.gz")
    # a = np.ones((scan_file.get_fdata().shape))
    # nib.save(nib.Nifti1Image(a, np.eye(4)), '/mnt/local/nirm/placenta_new/placenta/258/aaa_seg.nii.gz')