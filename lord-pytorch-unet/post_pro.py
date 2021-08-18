
from typing import List, Tuple,Union

from os.path import isfile, exists
from os import listdir, makedirs
import shutil
import json
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
import nibabel as nib
from skimage.morphology import remove_small_holes
from skimage.measure import label, regionprops
from skimage.util import view_as_windows
from skimage.filters import sobel
from skimage.transform import resize
# from preprocessing import PreProcessing
# from data_generator_seg import get_3D_patch_from_the_center_over_x_y_axes
import pickle
# import seaborn as sns
from tqdm import tqdm
from time import time, gmtime
import torch
import numpy as np
from scipy.ndimage.morphology import generate_binary_structure, binary_fill_holes
from scipy.ndimage.measurements import label
from skimage.measure import label as lab2
from skimage.measure import  regionprops


def calculate_runtime(t):
    t2 = gmtime(time() - t)
    return f'{t2.tm_hour:02.0f}:{t2.tm_min:02.0f}:{t2.tm_sec:02.0f}'



# ============================================ post processing func ====================================================

def get_main_connected_component(data):
    """
    Extract main component
    :param data:
    :return: segmentation data
    """
    labeled_array, num_features = label(data)
    i = np.argmax([np.sum(labeled_array == _) for _ in range(1, num_features + 1)]) + 1
    return labeled_array == i

def fill_holes_2d(data):
    """
    Fill holes for each slice separately
    :param data: data
    :return: filled data
    """

    filled_data = np.zeros_like(data).astype(bool)
    for i in range(data.shape[2]):
        slice = data[:, :, i]
        slice_filled = binary_fill_holes(slice).astype(bool)
        filled_data[:, :, i] = slice_filled
    return filled_data

def remove_small_components(mask, min_count):
    """
    Remove small connected components
    :param mask: mask
    :param min_count: minimum number of component voxels
    :return: result with removed small components
    """
    seg_res = np.copy(mask)
    s = generate_binary_structure(2, 2)
    labeled_array, num_features = label(mask, structure=s)
    (unique, counts) = np.unique(labeled_array, return_counts=True)
    for i in range(0, len(counts)):
        if counts[i] < min_count:
            seg_res[labeled_array == i] = 0

    return seg_res

def remove_small_components_2D(mask, min_count=5):
    """
    Remove small components in 2D slices
    :param mask: mask
    :return: largest component for ach slice
    """
    seg = np.zeros_like(mask, dtype=np.uint8)
    for i in range(0, mask.shape[2]):
        if len(np.nonzero(mask[:, :, i])[0]) > 0:
            seg[:, :, i] = remove_small_components(mask[:, :, i], min_count)

    return seg

def postprocess_prediction(pred, threshold=0.5, fill_holes=True, connected_component=True, remove_small=False,
                           fill_holes_2D=False):
    """
    postprocessing prediction
    :param pred:
    :param threshold: prediction mask
    :param fill_holes: should fill holes
    :param connected_component: should extract one main component
    :param remove_small: should remove small components in 2D
    :return: postprocessed mask
    """
    pred = pred > threshold

    if (np.count_nonzero(pred) == 0):  # no nonzero elements
        return pred

    if fill_holes:
        pred = binary_fill_holes(pred)

    if connected_component:
        pred = get_main_connected_component(pred)

    if remove_small:
        pred = remove_small_components_2D(pred)

    if fill_holes_2D:
        pred = fill_holes_2d(pred)

    return pred


# ============================================ Post processong predict =================================================
class PostProcessingSeg():


    def __init__(self,model, exp_dict, split_dict, param, min_val, max_val, data_with_gt_a: dict, margin: int, batch_size: int, patch_stride: Tuple[int, int, int] = (16, 16, 4), patch_dim: Tuple[int, int, int] = (32, 32, 8), bb = False, expand = True, unet = True, dim = 2):
        """
        init post processing find the maxoverlap for ROI
        :param cases: the cases
        :param model: the model
        :param gt: the grpund truth
        :param MARGIN: the margin for each slice
        """
        self.param = param
        self.min_val = min_val
        self.max_val = max_val
        data_gt_pre = dict()
        self.max_over_lap = 0
        self.case_to_num = dict()
        self.case_para = list()
        self.model = model
        self.exp_dict = exp_dict
        self.split_dict = split_dict
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.margin = margin
        self.patch_dim = patch_dim
        self.patch_stride = patch_stride
        self.expand = expand
        self.data_with_gt_a = data_with_gt_a
        # rebuilt all the cases
        self.model = self.model.to(self.device)
        self.unet = unet
        self.bb = bb
        self.batch_size = batch_size
        self.dim = dim
        self.predict_on_validation(data_gt_pre)

    def bbox2_3D(self, img):
        x = np.any(img, axis=(1, 2))
        y = np.any(img, axis=(0, 2))
        z = np.any(img, axis=(0, 1))

        xmin, xmax = np.where(x)[0][[0, -1]]
        ymin, ymax = np.where(y)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return xmin, xmax, ymin, ymax, zmin, zmax




    @staticmethod
    def get_borders(reg, mri_scan, patch_size, margin, mri_nifti_file_name):

        diff_col = reg.bbox[4] - reg.bbox[1]
        diff_row = reg.bbox[3] - reg.bbox[0]
        diff_width = reg.bbox[5] - reg.bbox[2]
        marg_col = int(diff_col * (1 / margin))
        marg_row = int(diff_row * (1 / margin))
        marg_width = int(diff_width * (1 / (2 * margin)))

        ymin = max(reg.bbox[1] - marg_col, 0)
        xmin = max(reg.bbox[0] - marg_row, 0)
        z_min = max(reg.bbox[2] - marg_width, 0)
        ymax = min(reg.bbox[4] + marg_col, mri_scan.shape[1])
        xmax = min(reg.bbox[3] + marg_row, mri_scan.shape[0])
        z_max = min(reg.bbox[5] + marg_width, mri_scan.shape[2])

        # making sure that the ROIs size is big enough at list as the patch size
        if xmax - xmin < patch_size[0]:
            interval = ceil((patch_size[0] - (xmax - xmin)) / 2)
            xmax = min(xmax + interval, mri_scan.shape[0])
            xmin = max(xmax - patch_size[0], 0)

        if ymax - ymin < patch_size[1]:
            interval = ceil((patch_size[1] - (ymax - ymin)) / 2)
            ymax = min(ymax + interval, mri_scan.shape[1])
            ymin = max(ymax - patch_size[1], 0)

        if z_max - z_min < patch_size[2]:
            interval = ceil((patch_size[2] - (z_max - z_min)) / 2)
            z_max = min(z_max + interval, mri_scan.shape[2])
            z_min = max(z_max - patch_size[2], 0)

        # check that the ROI is valid
        if any([np.any(np.array([xmax, ymax, z_max]) > np.array(mri_scan.shape)),
                np.any(np.array([xmin, ymin, z_min]) < 0)]):
            raise Exception(f'The shape of the case named "{mri_nifti_file_name}"'
                            f' cannot be supported by the desired patch size.\n'
                            f'The desired patch size is: {patch_size}, and the case\'s shape is: {mri_scan.shape}')

        return xmin, xmax, ymin, ymax, z_min, z_max

    @staticmethod
    def rebuild_predicted_patched_results(original_shape: Tuple[int, int, int],
                                          patches: np.ndarray, coordinates: np.ndarray,
                                          summing: bool = False) -> np.ndarray:
        """
        Rebuilding a full predicted image (segmentation actually) of the patched results.

        :param original_shape: The shape of the original image.
        :param patches: A ndarray with shape (N,x,y,z) containing the predicted patches
            - N is the number of patches (x,y,z) is the shape of the patches.
        :param coordinates: A ndarray with shape (N,3) containing the [0,0,0] real coordinate
            in the original image of each patch in 'patches'
        :param summing: If set to True the result prediction will be a sum of all the patches containing common voxels,
            otherwise it will make a logical 'or' for patches with common voxels.

        :return: The result predicted image.
        """

        result = np.zeros(original_shape, dtype=patches.dtype)
        patch_shape = patches.shape[1:]
        for c, coords_set in enumerate(coordinates):
            current_silce = np.s_[coords_set[0]:coords_set[0] + patch_shape[0],
                            coords_set[1]:coords_set[1] + patch_shape[1],
                            coords_set[2]:coords_set[2] + patch_shape[2]]
            if summing:
                result[current_silce] += patches[c]
            else:
                result[current_silce] = np.logical_or(result[current_silce], patches[c]).astype(np.float)

        return result


    def extract_3D_patches(self, im: np.ndarray,
                           window_shape: Tuple[int, int, int],
                           stride: Union[int, Tuple[int, int, int]],
                           start_from_the_end: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracting 3D patches (with overlapping - optional) from a 3D image.

        :param im: A 3D image - ndarray
        :param window_shape: A tuple (with length 3) indicating the shape of the desired patches.
        :param stride: Either a tuple (with length 3) or int indicating step size at which extraction shall
            will be performed. If integer is given, then the step is uniform in all dimensions.
        :param start_from_the_end: If set to True, the process of extracting the patches will be in such a way so that the
            first patch will begin at coordinate (0,0,0) in the image and it won't necessarily cover the higher coordinates.
            If set to False, the process of extracting the patches will be in such a way so that the last patch will end
            at coordinate (im.shape[0]-1,im.shape[1]-1,im.shape[2]-1) in the image and it won't necessarily cover the lower
            coordinates.

        :return: A tuple of the following form (patches, coords) where:
            • patches is a ndarray with shape (N,window_shape[0],window_shape[1],window_shape[2]) containing the extracted
                patches (N is the number of shape that was extracted).
            • coords is a ndarray with shape (N,3) containing the [0,0,0] real coordinate in 'im' of each patch in 'patches'.
        """

        # extracting the patches
        if start_from_the_end:
            patches = view_as_windows(np.flip(np.flip(np.flip(im, 0), 1), 2), window_shape, stride)
        else:
            patches = view_as_windows(im, window_shape, stride)

        # reshaping the patches array to the desired shape
        nX, nY, nZ, H, W, C = patches.shape
        nWindow = nX * nY * nZ
        patches = patches.reshape((nWindow, H, W, C))

        # computing the [0,0,0] real coordinate in 'im' of each patch in 'patches'
        s = stride if isinstance(stride, tuple) else (stride, stride, stride)
        coords = np.vstack([np.repeat(np.arange(0, nX * s[0], s[0]), nY * nZ),
                            np.tile(np.repeat(np.arange(0, nY * s[1], s[1]), nZ), nX),
                            np.tile(np.arange(0, nZ * s[2], s[2]), nX * nY)]).T

        if start_from_the_end:
            patches = np.flip(np.flip(np.flip(np.flip(patches, 3), 2), 1), 0)

            # calculating the with of the section, in each axis, that the patches don't cover (only at the edge, not
            # between patches).
            sec_x = im.shape[0] - (nX - 1) * s[0] - window_shape[0]
            sec_y = im.shape[1] - (nY - 1) * s[1] - window_shape[1]
            sec_z = im.shape[2] - (nZ - 1) * s[2] - window_shape[2]

            coords[:, 0] += sec_x
            coords[:, 1] += sec_y
            coords[:, 2] += sec_z

        return patches, coords



    def predict_on_validation(self, data_gt_pre: dict):
        for k, subject_id in enumerate(self.data_with_gt_a.keys()):
            if (subject_id not in  self.split_dict['cases_test_F']) and (subject_id not in  self.split_dict['cases_test_T']) and (subject_id not in  self.split_dict['cases_val_T']) and (subject_id not in  self.split_dict['cases_val_F']):
                continue

            # print(np.unique(self.data_with_gt_a[subject_id]['truth']), "unique values")
            mask_labels, num = lab2(self.data_with_gt_a[subject_id]['truth'],
                                    return_num=True, connectivity=1)
            region = regionprops(mask_labels)
            reg = sorted(region, key=lambda x: x.area)[-1]
            # print(f"len{len(regionprops(mask_labels))}")
            print(subject_id, "subject id")
            print(reg.bbox[4] - reg.bbox[1], "dif cols", 'diff_row',
                  reg.bbox[3] - reg.bbox[0])
            print(f"x:{reg.bbox[0]}:{reg.bbox[3]}",
                  f"y:{reg.bbox[1]}:{reg.bbox[4]}",
                  f"z:{reg.bbox[2]}:{reg.bbox[5]}")
            xmin, xmax, ymin, ymax, z_min, z_max = PostProcessingSeg.get_borders(
                reg, self.data_with_gt_a[subject_id]['data'], self.patch_dim,
                self.margin, subject_id)
            if not self.bb:

                xmin, xmax, ymin, ymax, z_min, z_max = 0, self.data_with_gt_a[subject_id]['data'].shape[0], 0, self.data_with_gt_a[subject_id]['data'].shape[1], 0, self.data_with_gt_a[subject_id]['data'].shape[2]

            crop_slice = np.s_[xmin:xmax, ymin: ymax, z_min: z_max]

            new_case = self.data_with_gt_a[subject_id]['data'][crop_slice]
            gt_slice = self.data_with_gt_a[subject_id]['truth'][crop_slice]
            # print(np.type(new_case))
            if np.max(new_case) > 1:
                print("hereeeeee")
                new_case = new_case.astype(np.float32)  / 255


            # print(np.unique(new_case), "unique crop slice")
            print( self.patch_dim, self.patch_stride)
            patches1, patches_coords1 = self.extract_3D_patches(new_case, window_shape= tuple(self.patch_dim), stride=tuple(self.patch_stride))
            patches_gt1, patches_gt_coords1 = self.extract_3D_patches(gt_slice, window_shape=tuple(self.patch_dim), stride=tuple(self.patch_stride))

            num_of_slices = 0
            num_of_patches = 0
            if self.dim == 3:
                print("dim3")
                patches1 = np.expand_dims(patches1, axis=-1)
                patches1_torch = torch.from_numpy(patches1.astype(np.float32)).permute(0, 4, 3, 1, 2)
                patches2_torch = torch.from_numpy(
                    np.zeros((1, patches1.shape[1], patches1.shape[2], patches1.shape[3], patches1.shape[4])).astype(
                        np.float32)).permute(0, 4, 3, 1, 2)
            elif self.dim == 2:
                print("dim2")

                num_of_patches = patches1.shape[0]
                patches1_t = np.transpose(patches1, (0,3, 1, 2))
                num_of_slices = patches1_t.shape[1]
                print(patches1.shape, "patch shape")

                patches1 = np.reshape(patches1_t, (patches1_t.shape[0] * patches1_t.shape[1],patches1_t.shape[2],patches1_t.shape[3]))
                patches2 = np.reshape(patches1_t,
                                      (patches1_t.shape[0] * patches1_t.shape[1], patches1_t.shape[2], patches1_t.shape[3]))
                patches1 = np.expand_dims(patches1, axis=-1)
                patches2 = np.expand_dims(patches2, axis=-1)
                patches1_torch = torch.from_numpy(
                    patches1.astype(np.float32)).permute(0, 3, 1, 2)
                patches2_torch = torch.from_numpy(
                    patches2.astype(np.float32)).permute(0, 3, 1, 2)
                print(patches1_torch.size(), "patches torch")
            else:
                raise Exception('bad number of dimension')


            # print(patches1_torch.size(), "patchs size")

            ones = torch.from_numpy(np.ones((patches1.shape[0],), dtype= int))
            ones2 = torch.ones((1,)).type(torch.IntTensor)

            self.model.eval()
            predictions = None
            with torch.no_grad():
                if not self.unet:
                    for i in range(0, patches1_torch.size()[0], self.batch_size):
                        print(patches1_torch[i:i+self.batch_size].size(), "patch size")
                        predictions_m = self.model(patches1_torch[i:i+self.batch_size].to(self.device), ones[i:i+self.batch_size].to(self.device), patches1_torch[i:i+self.batch_size].to(self.device),ones[i:i+self.batch_size].to(self.device))
                        if i == 0:
                            predictions = predictions_m['mask']
                        else:
                            predictions = torch.cat((predictions, predictions_m['mask']), dim = 0)

                    diff = patches1_torch.size()[0] - predictions.size()[0]
                    if diff > 0:
                        s = patches1_torch.size()[0] - diff
                        e = patches1_torch.size()[0]
                        predictions_m = self.model(patches1_torch[s:e].to(self.device), ones[s:e].to(self.device), patches2_torch[s:e].to(self.device), ones2[s:e].to(self.device))
                        predictions = torch.cat((predictions, predictions_m['mask']), dim=0)

                    # predictions = predictions.permute(0, 1, 3, 4, 2).cpu().detach().numpy()

                    # print(np.unique(predictions), "prediction unique check if we are really rounding the score" )
                else:
                    print(patches1_torch.size()[0], "patches size")
                    print(self.batch_size, "batch size")
                    for i in range(0, patches1_torch.size()[0], self.batch_size):
                        print(i, "iiiiiii")
                        predictions_m = self.model(patches1_torch[i:i + self.batch_size].to(self.device))
                        patches1_torch[i:i + self.batch_size].detach()
                        if i == 0:
                            predictions = predictions_m['mask']
                        else:
                            predictions = torch.cat(
                                (predictions, predictions_m['mask']), dim=0)

                    diff = patches1_torch.size()[0] - predictions.size()[0]
                    if diff > 0:
                        s = patches1_torch.size()[0] - diff
                        e = patches1_torch.size()[0]
                        predictions_m = self.model(patches1_torch[s:e].to(self.device))
                        predictions = torch.cat((predictions, predictions_m['mask']), dim=0)

            if self.dim == 3:
                predictions = predictions.permute(0, 1, 3, 4,2).cpu().detach().numpy()
                predictions = np.squeeze(predictions, axis=1)
            elif self.dim == 2:
                print(predictions.size(), "prediction size ")
                predictions = predictions.permute(0,2, 3, 1).cpu().detach().numpy()
                print(predictions.shape, "prediction shape ", num_of_patches, num_of_slices)
                predictions = np.reshape(predictions, (num_of_patches,num_of_slices, predictions.shape[1],predictions.shape[2],predictions.shape[3] ))
                predictions = np.transpose(predictions, (0, 4, 2, 3, 1))
                predictions = np.squeeze(predictions, axis=1)
            else:
                raise Exception('bad number of dimension')


            print(np.unique(predictions), "prediction unique check if we are really rounding the score")


            # rebuilding the patches
            predictions = (predictions >= 0.5).astype(predictions.dtype)

            print(predictions.shape, "prediction shape")
            pred = PostProcessingSeg.rebuild_predicted_patched_results(new_case.shape, predictions, patches_coords1, summing=True)
            ones = PostProcessingSeg.rebuild_predicted_patched_results(new_case.shape, np.ones_like(predictions), patches_coords1, summing=True)
            ones[ones == 0] = 1

            if ones.max() > self.max_over_lap:
                self.max_over_lap = ones.max()

            pred_label = np.round(((pred) / ones) * ones.max())
            data_gt_pre[subject_id] = dict()
            data_gt_pre[subject_id]['data'] = self.data_with_gt_a[subject_id]['data']
            data_gt_pre[subject_id]['truth'] = self.data_with_gt_a[subject_id]['truth']
            data_gt_pre[subject_id]['pred'] = np.zeros_like(self.data_with_gt_a[subject_id]['truth'])
            data_gt_pre[subject_id]['pred'][xmin:xmax, ymin: ymax, z_min: z_max] = pred_label

