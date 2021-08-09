import glob
import os
from typing import Tuple, Union, List
from skimage.util import view_as_windows
import numpy as np
# import tables
import nibabel as nib
import json
from scipy.ndimage import zoom
from skimage.measure import label, regionprops
import data_generation.preprocess
from utils.read_write_data import pickle_load, pickle_dump
from data_curation.helper_functions import move_smallest_axis_to_z, get_resolution
import matplotlib.pyplot as plt
import pickle
from math import ceil

def fetch_data_files(scans_dir, train_modalities, ext, return_subject_ids=False):
    data_files = list()
    subject_ids = list()

    if ';' not in scans_dir:#only one data path
        if(os.path.isdir(os.path.abspath(scans_dir)) == False):
            print('data dir: ' + scans_dir + 'does not exist!')
            return None
        scans_path = glob.glob(os.path.join(scans_dir, "*"))
        all_scans = scans_path
    else: #multiple data pathes
        dirs = scans_dir.split(';')
        all_scans = []
        for dir in dirs:
            if(os.path.isdir(os.path.abspath(dir)) == False):
                print('data dir: ' + dir + 'does not exist!')
                return None
            scans_path = glob.glob(os.path.join(dir, "*"))
            all_scans.extend(scans_path)

    for subject_dir in sorted(all_scans, key=os.path.basename):
        subject_ids.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in train_modalities:
            modalities = modality.split(';')
            for single_modality in modalities:
                file_path = os.path.join(subject_dir, single_modality + ".nii" + ext)
                if os.path.exists(file_path) or os.path.exists(file_path + '.gz'):
                    subject_files.append(file_path)
                    break
        data_files.append(tuple(subject_files))
    if return_subject_ids:
        return data_files, subject_ids
    else:
        return data_files


def read_img(in_file):
    filepath = os.path.abspath(in_file)
    if(os.path.exists(filepath)==False):
        filepath = filepath + '.gz'
        if(os.path.exists(filepath)==False):
            print('path' + filepath + 'does not exist!')
            return None
    print("Reading: {0}".format(in_file))
    image = nib.load(filepath)


    return image



def add_data_to_storage_train(storage_dict, data_with_gt_a, subject_id, subject_data):
    storage_dict[subject_id] = {}

    storage_dict[subject_id] = np.asarray(subject_data[0]).astype(np.float)
    print(np.unique(np.asarray(subject_data[0]).astype(np.float)), "unique 2")
    data_with_gt_a[subject_id] = {}
    data_with_gt_a[subject_id]['data'] = np.asarray(subject_data[0]).astype(np.float)
    data_with_gt_a[subject_id]['truth'] = np.asarray(subject_data[1]).astype(np.float)

    # exit()





def add_data_to_storage_test(storage_dict,data_with_gt_a, subject_id, subject_data):

    storage_dict[subject_id] = {}
    storage_dict[subject_id]['data'] = np.asarray(subject_data[0]).astype(np.float)
    storage_dict[subject_id]['truth'] = np.asarray(subject_data[1]).astype(np.float)

    data_with_gt_a[subject_id] = {}
    data_with_gt_a[subject_id]['data'] = np.asarray(subject_data[0]).astype(np.float)
    data_with_gt_a[subject_id]['truth'] = np.asarray(subject_data[1]).astype(np.float)


    if len(subject_data) > 2:
        storage_dict[subject_id]['mask'] = np.asarray(subject_data[2]).astype(np.float)


def extract_3D_patches(im: np.ndarray,
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
    # print("here 4")
    print(im.shape, "im.shape")
    print(window_shape, "window_shape")
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


        # calculating the with of the section, in each axis, that the patches don't cover (only at the edge, not
        # between patches).
    sec_x = im.shape[0] - (nX - 1) * s[0] - window_shape[0]
    sec_y = im.shape[1] - (nY - 1) * s[1] - window_shape[1]
    sec_z = im.shape[2] - (nZ - 1) * s[2] - window_shape[2]

    coords[:, 0] += sec_x
    coords[:, 1] += sec_y
    coords[:, 2] += sec_z
    # print("here 5")
    return patches, coords


def bbox2_3D(img):
    x = np.any(img, axis=(1, 2))
    y = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return xmin, xmax, ymin, ymax, zmin, zmax

def get_borders(reg, mri_scan, patch_size, margin,mri_nifti_file_name):

    diff_col = reg.bbox[4] - reg.bbox[1]
    diff_row = reg.bbox[3] - reg.bbox[0]
    diff_width = reg.bbox[5] - reg.bbox[2]
    marg_col = int(diff_col * (1 / margin))
    marg_row = int(diff_row * (1 / margin))
    marg_width = int(diff_width * (1 / (2 * margin)))

    ymin = max(reg.bbox[1] - marg_col,0)
    xmin = max(reg.bbox[0] - marg_row, 0)
    z_min =max(reg.bbox[2] - marg_width, 0)
    ymax = min(reg.bbox[4] + marg_col, mri_scan.shape[1])
    xmax = min(reg.bbox[3] + marg_row, mri_scan.shape[0])
    z_max =min(reg.bbox[5] + marg_width, mri_scan.shape[2])


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
    if any([np.any(np.array([xmax, ymax, z_max]) > np.array(mri_scan.shape)), np.any(np.array([xmin, ymin, z_min]) < 0)]):
        raise Exception(f'The shape of the case named "{mri_nifti_file_name}"'
                        f' cannot be supported by the desired patch size.\n'
                        f'The desired patch size is: {patch_size}, and the case\'s shape is: {mri_scan.shape}')

    return xmin, xmax, ymin, ymax, z_min, z_max


def create_and_store_training_patches(case: np.array, gt: np.array,
                            patchs_storage: dict ,
                            patchs_dict_with_gt: dict ,subject_id: str ,
                            model_original_dim: Tuple[int, int, int] = (128, 128, 16),
                            patch_stride: Tuple[int, int, int] = (32, 32, 8), with_without_body:bool = True, min_num_of_body_voxels:int = 1, min_num_of_plcenta_voxels:int = 32*32,num_slice_cri: int = 10,
                                      with_plcenta = True, dim: int = 3, from_bb: bool = False, margin: int = 100):
    """
    Creating and saving patches for model training purpose with a shape (x+y, y+x, z) where (x, y ,z) is the original
    patches shape the model needs.

    :param CTs: A list of nifti files' names of CT scans (ordered relatively to 'masks_GT' and 'ROIs').
    :param masks_GT: A list of nifti files' names of masks-GT segmentations (ordered relatively to 'CTs' and 'ROIs').
    :param ROIs: A list of nifti files' names of roi segmentations (ordered relatively to 'CTs' and 'masks_GT') in order
        to know where to crop the images.
    :param results_directory_path: The path of the directory where to save there the results.
    :param expand: If set to True, the patches will be extracted from the image in a double size (over x, y axes) for
        augmentation purpose. If set to False (by default) the patches will be as the 'model_original_dim' size.
    :param model_original_dim: The shape of the patches the model would accept.
    :param clip_intensities: A tuple (with shape 2) indicating the interesting intensities for predicting (e.g. for
            liver intensities (by default): clip_intensities=(-150,150)).
    :param patch_stride: A tuple (with length 3) indicating step size at which extraction shall will be performed at the
        patches extraction.
    :param with_lesion: If set to True (by default), the result patches will be those that have lesions in them. If set
        to False, the result patches will be those that have not lesions in them.
    :param min_num_of_lesion_voxels: The minimum number of lesion voxels in order to be considered as a patch that has
        lesions in it.
    :param num_slice_cri: if we work in 3D the demand is te number of slices that fit the critra of number of pixels with body or placenta

    """
    # print("here 1")
    patch_size = model_original_dim

    dict_parameters = dict(
        model_original_dim = model_original_dim,
        patch_stride = patch_stride,
        with_without_body = with_without_body,
        min_num_of_body_voxels = min_num_of_body_voxels,
        min_num_of_plcenta_voxels = min_num_of_plcenta_voxels,
        num_slice_cri = num_slice_cri,
        with_plcenta =with_plcenta,
        dim = dim,
        from_bb = from_bb,
        margin = margin
    )
    # print("here 2")
    # cropping the cases to the bbox of the ROI
    if from_bb:
        mask_labels, num = label(gt, return_num=True, connectivity=1)
        region = regionprops(mask_labels)
        reg = sorted(region, key=lambda x: x.area)[-1]
        # print(f"len{len(regionprops(mask_labels))}")
        print(subject_id, "subject id")
        # print(reg.bbox[4] - reg.bbox[1],"dif cols", 'diff_row', reg.bbox[3] - reg.bbox[0])
        # print(f"x:{reg.bbox[0]}:{reg.bbox[3]}",f"y:{reg.bbox[1]}:{reg.bbox[4]}",f"z:{reg.bbox[2]}:{reg.bbox[5]}" )
        xmin, xmax, ymin, ymax, z_min, z_max = get_borders(reg, case, patch_size, margin, subject_id)
    else:
        xmin, xmax, ymin, ymax, z_min, z_max = 0, case.shape[0], 0, case.shape[1], 0, case.shape[2]
        # print("here 3")

    crop_slice = np.s_[xmin:xmax, ymin: ymax, z_min: z_max]

    new_case = case[crop_slice]
    gt_slice = gt[crop_slice]
    # print(np.unique(new_case), "unique crop slice")
    # extracting the patches
    # print(new_case.shape, "new_cse")
    # print(patch_size, "patch size")
    patches1, patches_coords1 = extract_3D_patches(new_case, window_shape = patch_size, stride=patch_stride)
    print("gt box")
    patches_gt1, patches_gt_coords1 = extract_3D_patches(gt_slice, window_shape = patch_size, stride=patch_stride)

    # print(f"finsih creat path len patchs {patches_gt1.shape[0]}")
    # plt.title(f"case 0")
    # plt.imshow(new_case[:,:,16], cmap='gray')
    # plt.show()
    for (patches, patches_gt, patches_coords) in [(patches1, patches_gt1, patches_coords1)]:
        print("Im in the loop")
        for j, patch in enumerate(patches):
            current_x_coord = 0 + patches_coords[j, 0]
            current_y_coord = 0 + patches_coords[j, 1]
            current_z_coord = 0 + patches_coords[j, 2]
            if dim == 2:
                for k in range(patch.shape[2]):
                    body_pixels = np.zeros((patch.shape[0], patch.shape[1]))
                    body_pixels[patch[:,:,k] > 0.1] = 1
                    num_of_body_voxels = body_pixels.sum()
                    num_of_plcenta_voxels = patches_gt[j][:,:,k].sum()

                    # print(num_of_body_voxels)
                    if (with_without_body and num_of_body_voxels < min_num_of_body_voxels) or (num_of_plcenta_voxels < min_num_of_plcenta_voxels and with_plcenta):
                        continue
                    patchs_storage[f'{subject_id}_{current_x_coord}_{current_y_coord}_{current_z_coord}_{k}'] = patch[:,:,k]
                    patchs_dict_with_gt[f'{subject_id}_{current_x_coord}_{current_y_coord}_{current_z_coord}_{k}'] = patches_gt[j][:,:,k]
            else:
                # print("go into 3d world")
                body_pixels = np.zeros_like((patch))
                body_pixels[patch > -100] = 1
                num_of_body_voxels = body_pixels.sum()
                num_of_plcenta_voxels = patches_gt[j].sum()

                if (with_without_body and (num_of_body_voxels < min_num_of_body_voxels*num_slice_cri)) or (num_of_plcenta_voxels < (min_num_of_plcenta_voxels*num_slice_cri) and with_plcenta):
                    # print("continue that bum")
                    continue
                print("here nir")
                # print("hopa hey save")
                patchs_storage[f'{subject_id}_{current_x_coord}_{current_y_coord}_{current_z_coord}'] = patch
                patchs_dict_with_gt[f'{subject_id}_{current_x_coord}_{current_y_coord}_{current_z_coord}'] = patches_gt[j]
                # print("hopa hey finish savesave")

        # print("finish all")
        return dict_parameters



def write_image_data_to_file(image_files,data_with_gt_a, data_storage, subject_ids, scale=None, preproc=None, rescale_res=None, metadata_path=None, default_res= [1.56,1.56,3], train = True):

    for subject_id, set_of_files in zip(subject_ids, image_files):
        images = [read_img(_) for _ in set_of_files]
        subject_data = [image.get_data() for image in images]

        # print()
        subject_data[0], swap_axis = move_smallest_axis_to_z(subject_data[0])
        subject_data[1], swap_axis = move_smallest_axis_to_z(subject_data[1])
        if len(subject_data)==3: #mask also exists
            subject_data[2], swap_axis = move_smallest_axis_to_z(subject_data[2])
        # print(np.unique(subject_data[0]).shape, "unique shape4")
        if rescale_res is not None and metadata_path is not None:
            curr_resolution = get_resolution(subject_id, metadata_path=metadata_path)
            if(curr_resolution is None):
                curr_resolution = default_res
                print('resolution of case ' + subject_id + ' is not in metadata file, using default resolution of ' + str(default_res))
            scale_factor = [curr_resolution[0]/rescale_res[0], curr_resolution[1]/rescale_res[1],
                            curr_resolution[2]/rescale_res[2]]
            subject_data[0] = zoom(subject_data[0], scale_factor)
            subject_data[1] = zoom(subject_data[1], scale_factor, order=0)
            # print(subject_data[0].shape,"shape data" )
            if len(subject_data)==3: #mask also exists
                subject_data[2] = zoom(subject_data[2], scale_factor, order=0)
            if scale is not None:
                print('both rescale_res and scale parameters are given. Using only rescale resolution parameter!')

            # print(np.unique(subject_data[0]).shape, "unique shape3")
        elif scale is not None:
            subject_data[0] = zoom(subject_data[0], scale)
            subject_data[1] = zoom(subject_data[1], scale, order=0)
            if len(subject_data)==3: #mask also exists
                subject_data[2] = zoom(subject_data[2], scale, order=0)

        print(np.unique(subject_data[0]).shape, "unique shape1")
        # print(subject_data[0])
        if preproc is not None:
            print(preproc, "preproc")
            subject_data[0] = preproc(subject_data[0])

        # print(np.unique(subject_data[0]).shape, "unique shape")

        if train:
            add_data_to_storage_train(data_storage,data_with_gt_a, subject_id, subject_data)
        else:
            add_data_to_storage_test(data_storage,data_with_gt_a, subject_id, subject_data)

    return data_storage

def normalize_data(data, min, max):
    data -= min
    data /= max
    print(np.max(data), "max data", np.min(data), "min data", "after normlziation")
    return data

def normalize_data_storage(data_dict: dict, train: bool = True):
    means = list()
    stds = list()
    for key in data_dict:
        if train:
            data = data_dict[key]
        else:
            data = data_dict[key]['data']
        means.append(data.mean(axis=(-1, -2, -3)))
        stds.append(data.std(axis=(-1, -2, -3)))
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    for key in data_dict:
        if train:
            data_dict[key] = normalize_data(data_dict[key], mean, std)
            print(np.unique(data_dict[key]), "    unique    ")
        else:
            data_dict[key]['data'] = normalize_data(data_dict[key]['data'], mean, std)
    return data_dict, mean, std


def normalize_data_storage_each(data_dict: dict, train: bool = True):
    for key in data_dict:
        data = data_dict[key]
        mean = data.mean(axis=(-1, -2, -3))
        std = data.std(axis=(-1, -2, -3))
        data_dict[key] = normalize_data(data, mean, std)
    return data_dict, None, None

def write_data_to_file(training_data_files,data_withgt_file, out_file,out_file_patchs ,out_file_patchs_gt, patches_param_file,subject_ids = None, normalize='all', scale=None, preproc=None, rescale_res=None, metadata_path=None, train = True, store_patches = True, dim = 3):
    """
    Takes in a set of training images and writes those images to an hdf5 file.
    :param data_files: List of tuples containing the training data files. The modalities should be listed in
    the same order in each tuple. The last item in each tuple must be the labeled image.
    Example: [('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'),
              ('sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz')]
    :param out_file: Where the hdf5 file will be written to.
    :param truth_dtype: Default is 8-bit unsigned integer.
    :return: Location of the hdf5 file with the image data written to it.
    """


    data_dict = {}
    patchs_dict = {}
    patchs_gt = {}
    data_with_gt_a = {}
    write_image_data_to_file(training_data_files,data_with_gt_a, data_dict, subject_ids, scale = scale, preproc = preproc, rescale_res = rescale_res, metadata_path = metadata_path, train = train)

    for subject_id in data_dict.keys():
        subject_data_2 = data_dict[subject_id]
        subject_data =  data_with_gt_a[subject_id]['data']
        subject_gt=  data_with_gt_a[subject_id]['truth']

        # print(np.max(subject_data_2), "subject data max")


    if isinstance(normalize, str):
        _, mean, std = {
            'all': normalize_data_storage,
            'each': normalize_data_storage_each
        }[normalize](data_dict, train)
    else:
        mean, std = None, None

    if isinstance(normalize, str):
        _, mean, std = {
            'all': normalize_data_storage,
            'each': normalize_data_storage_each
        }[normalize](data_with_gt_a, False)
    else:
        mean, std = None, None

    patches_parameters  = None
    if store_patches:

        for subject_id in data_dict.keys():
            print(f"subject id for checking unet {subject_id}")
            subject_data = data_with_gt_a[subject_id]['data']
            subject_gt = data_with_gt_a[subject_id]['truth']

            patches_parameters = create_and_store_training_patches(subject_data ,subject_gt ,  patchs_dict, patchs_gt, subject_id, dim = dim)


    # no norm
    # if isinstance(normalize, str):
    #     _, _, _ = {
    #         'all': normalize_data_storage,
    #         'each': normalize_data_storage_each
    #     }[normalize](patchs_dict, train)
    # else:
    #     mean, std = None, None
    #
    #
    if patches_parameters != None:
        print("dump 1")
        pickle_dump(patches_parameters, patches_param_file)
    print("dump 2")
    # print(f"data_withgt_file{data_withgt_file}")
    # print(f"data_withgt_file{data_with_gt_a}")
    pickle_dump(data_with_gt_a, data_withgt_file)
    print("dump 3")
    pickle_dump(data_dict, out_file)
    print("dump 4")
    pickle_dump(patchs_dict, out_file_patchs)
    print("dump 5")
    pickle_dump(patchs_gt, out_file_patchs_gt)

    return out_file, (mean, std)

def open_data_file(filename):
    return pickle_load(filename)

def create_load_hdf5(normalization, data_dir, scans_dir, train_modalities, ext, overwrite = False, preprocess=None, scale= None, rescale_res=None, metadata_path=None, train = True, store_patches = True, dim = 3):
    """
    This function normalizes raw data and creates hdf5 file if needed
    Returns loaded hdf5 file
    """
    data_file = os.path.join(data_dir, "fetal_data.h5")
    data_withgt_file = os.path.join(data_dir, "fetal_data_withgt.h5")
    data_file_patchs = os.path.join(data_dir, "fetal_data_patches.h5")#path to normalized data hdf5 file
    data_file_patchs_gt = os.path.join(data_dir, "fetal_data_patches_gt.h5")#path to normalized data hdf5 file
    patches_param_file = os.path.join(data_dir, "patches_param.h5")#path to normalized data hdf5 file
    print('opening data file at: ' + data_file)
    # convert input images into an hdf5 file
    if overwrite or not os.path.exists(data_file):
        training_files, subject_ids = fetch_data_files(scans_dir, train_modalities, ext, return_subject_ids=True)
        # print(training_files, "training file")
        if preprocess is not None:
            preproc_func = getattr(data_generation.preprocess, preprocess)
        else:
            preproc_func = None

        _, (mean, std) = write_data_to_file(training_files,data_withgt_file, data_file, data_file_patchs,data_file_patchs_gt,patches_param_file, subject_ids=subject_ids, normalize=normalization, preproc=preproc_func, scale = scale, rescale_res = rescale_res, metadata_path = metadata_path, train = train, store_patches = store_patches, dim = dim)



        with open(os.path.join(data_dir, 'norm_params.json'), mode='w') as f:
            json.dump({'mean': mean, 'std': std}, f)

    data_file_opened = open_data_file(data_file)
    return data_file_opened






if __name__ == '__main__':
    # create_load_hdf5(normalization = "all", scans_dir = '/cs/casmip/nirm/embryo_project_version1/embryo_data_raw/before_preprocess/DRS_3/TRAIN/Labeled/placenta/' ,data_dir = '/cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA_3/TRAIN/Labeled/placenta', train_modalities = [ "volume", "truth"], ext ="",overwrite =True, preprocess = "window_1_99", scale = None, train = True, store_patches= True, dim = 3)
    print("finish 1")
    # create_load_hdf5(normalization = "all", scans_dir = '/cs/casmip/nirm/embryo_project_version1/embryo_data_raw/before_preprocess/DRS_3/TRAIN/Labeled/TRUFI/' , data_dir = '/cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA_3/TRAIN/Labeled/TRUFI', train_modalities = [ "volume", "truth"], ext = "", overwrite = True, preprocess = "window_1_99", scale = [0.5, 0.5, 1],rescale_res = [1.56,.56,3], metadata_path = None, train = True, store_patches= True, dim = 3)
    print("finish 2")
    # create_load_hdf5(normalization = "all", scans_dir = '/cs/casmip/nirm/embryo_project_version1/embryo_data_raw/before_preprocess/DRS_3/TRAIN/Unlabeled/TRUFI/' , data_dir = '/cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA_3/TRAIN/Unlabeled/TRUFI', train_modalities = [ "volume", "truth"], ext = "", overwrite = True, preprocess = "window_1_99", scale = [0.5, 0.5, 1],rescale_res = [1.56,.56,3], metadata_path = None, train = True, store_patches= True, dim = 3)
    print("finish 3")
    # create_load_hdf5(normalization = "all", scans_dir = '/cs/casmip/nirm/embryo_project_version1/embryo_data_raw/before_preprocess/DRS_3/VALIDATION/placenta/' , data_dir = '/cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA_3/VALIDATION/placenta', train_modalities = [ "volume", "truth"], ext = "", overwrite = True, preprocess = "window_1_99", scale = [0.5, 0.5, 1],rescale_res = [1.56,.56,3], metadata_path = None, train = True, store_patches= True, dim = 3)
    create_load_hdf5(normalization = "all", scans_dir = '/cs/casmip/nirm/embryo_project_version1/embryo_data_raw/before_preprocess/DRS_2/VALIDATION/placenta/' , data_dir = '/cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA_2/VALIDATION/placenta', train_modalities = [ "volume", "truth"], ext = "", overwrite = True, preprocess = "window_1_99", scale = [0.5, 0.5, 1],rescale_res = [1.56,.56,3], metadata_path = None, train = True, store_patches= True, dim = 3)
    create_load_hdf5(normalization = "all", scans_dir = '/cs/casmip/nirm/embryo_project_version1/embryo_data_raw/before_preprocess/DRS_1/VALIDATION/placenta/' , data_dir = '/cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA_1/VALIDATION/placenta', train_modalities = [ "volume", "truth"], ext = "", overwrite = True, preprocess = "window_1_99", scale = [0.5, 0.5, 1],rescale_res = [1.56,.56,3], metadata_path = None, train = True, store_patches= True, dim = 3)
    print("finish 5")
    # create_load_hdf5(normalization = "all", scans_dir = '/cs/casmip/nirm/embryo_project_version1/embryo_data_raw/before_preprocess/DRS_3/VALIDATION/TRUFI/' , data_dir = '/cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA_3/VALIDATION/TRUFI', train_modalities = [ "volume", "truth"], ext = "", overwrite = True, preprocess = "window_1_99", scale = [0.5, 0.5, 1],rescale_res = [1.56,.56,3], metadata_path = None, train = True, store_patches= True, dim = 3)
    print("finish 6")
    # create_load_hdf5(normalization = "all", scans_dir= '/cs/casmip/nirm/embryo_project_version1/embryo_data_raw/before_preprocess/DRS_3/TEST/TRUFI/' , data_dir = '/cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA_3/TEST/TRUFI', train_modalities = [ "volume", "truth"], ext = "", overwrite = True, preprocess = "window_1_99", scale = [0.5, 0.5, 1],rescale_res = [1.56,.56,3], metadata_path = None, train = True, store_patches= True, dim = 3)
    print("finish 7")
    # create_load_hdf5(normalization = "all", data_dir = '/cs/casmip/nirm/embryo_project_version1/DATA_NEW/placenta' ,scans_dir = '/cs/casmip/nirm/embryo_project_version1/DATA-RAW/placenta', train_modalities = [ "volume", "truth"], ext ="",overwrite =True, preprocess = "window_1_99", scale = None, train = True, store_patches= True)
    # create_load_hdf5(normalization = "all", data_dir = '/cs/casmip/nirm/embryo_project_version1/DATA_NEW/TRUFI', scans_dir = '/cs/casmip/nirm/embryo_project_version1/DATA-RAW/TRUFI', train_modalities = [ "volume", "truth"], ext = "", overwrite = True, preprocess = "window_1_99", scale = [0.5, 0.5, 1],rescale_res = [1.56,.56,3], metadata_path = None, train = True, store_patches= True)
    exit()
    with open(os.path.join('/cs/casmip/nirm/embryo_project_version1/DATA_NEWLS1/', 'placenta', 'fetal_data_patches.h5'), "rb") as opened_file:
        fiesta_dataset = pickle.load(opened_file)

    imgs_fiesta = np.array(list(fiesta_dataset.values()))
    # print(imgs_fiesta.shape, "fiesta")
    with open(os.path.join('/cs/casmip/nirm/embryo_project_version1/DATA_NEWLS1/placenta/',
                           'fetal_data_patches_gt.h5'), "rb") as opened_file:

        fiesta_dataset_gt = pickle.load(opened_file)

    imgs_fiesta = np.array(list(fiesta_dataset.values()))
    imgs_fiesta_gt = np.array(list(fiesta_dataset_gt.values()))
    print(imgs_fiesta.shape, "fiesta")
    print(imgs_fiesta_gt.shape, "fiesta")
    plt.imshow(imgs_fiesta[0], cmap='gray')
    plt.show()

    plt.imshow(imgs_fiesta_gt[0], cmap='gray')
    plt.show()
    #

# ---------------------------DATA _NEWLS-----------------------------------------------------------------------------
# create_load_hdf5(normalization = "all", data_dir = '/cs/casmip/nirm/embryo_project_version1/DATA_NEWLS/placenta/' ,scans_dir = '/cs/casmip/nirm/embryo_project_version1/DATA-RAW/placenta', train_modalities = [ "volume", "truth"], ext ="",overwrite =False, preprocess = "window_1_99", scale = None, train = True, store_patches= True)
# create_load_hdf5(normalization = "all", data_dir = '/cs/casmip/nirm/embryo_project_version1/DATA_NEWLS/TRUFI/' , scans_dir = '/cs/casmip/nirm/embryo_project_version1/DATA-RAW/TRUFI', train_modalities = [ "volume", "truth"], ext = "", overwrite = False, preprocess = "window_1_99", scale = [0.5, 0.5, 1],rescale_res = [1.56,.56,3], metadata_path = None, train = True, store_patches= True)
# model_original_dim: Tuple[int, int, int] = (128, 128, 20),
#                             patch_stride: Tuple[int, int, int] = (32, 32, 28), with_without_body:bool = True, min_num_of_body_voxels:int = 32*32, min_num_of_plcenta_voxels:int = 4*4, with_plcenta = True):

# ------------------------DATA_NEWLSj--------------------------------------------------------------------------------
#  create_load_hdf5(normalization = "all", data_dir = '/cs/casmip/nirm/embryo_project_version1/DATA_NEWLS/placenta/' ,scans_dir = '/cs/casmip/nirm/embryo_project_version1/DATA-RAW/placenta', train_modalities = [ "volume", "truth"], ext ="",overwrite =False, preprocess = "window_1_99", scale = None, train = True, store_patches= True)
#     create_load_hdf5(normalization = "all", data_dir = '/cs/casmip/nirm/embryo_project_version1/DATA_NEWLS/TRUFI/' , scans_dir = '/cs/casmip/nirm/embryo_project_version1/DATA-RAW/TRUFI', train_modalities = [ "volume", "truth"], ext = "", overwrite = False, preprocess = "window_1_99", scale = [0.5, 0.5, 1],rescale_res = [1.56,.56,3], metadata_path = None, train = True, store_patches= True)
# model_original_dim: Tuple[int, int, int] = (128, 128, 20),
# patch_stride: Tuple[int, int, int] = (16, 16,
                                      # 28), with_without_body:bool = True, min_num_of_body_voxels:int = 32 * 32, min_num_of_plcenta_voxels:int = 16 * 16, with_plcenta = True

# ------------------------DATA_NEWLS1--------------------------------------------------------------------------------
# model_original_dim: Tuple[int, int, int] = (128, 128, 20),
# patch_stride: Tuple[int, int, int] = (32, 32,
#                                       28), with_without_body:bool = True, min_num_of_body_voxels:int = 50 * 50, min_num_of_plcenta_voxels:int = 32 * 32, with_plcenta = True):
#   create_load_hdf5(normalization = "all", data_dir = '/cs/casmip/nirm/embryo_project_version1/DATA_NEWLS1/placenta/' ,scans_dir = '/cs/casmip/nirm/embryo_project_version1/DATA-RAW/placenta', train_modalities = [ "volume", "truth"], ext ="",overwrite =False, preprocess = "window_1_99", scale = None, train = True, store_patches= True)
#     create_load_hdf5(normalization = "all", data_dir = '/cs/casmip/nirm/embryo_project_version1/DATA_NEWLS1/TRUFI/' , scans_dir = '/cs/casmip/nirm/embryo_project_version1/DATA-RAW/TRUFI', train_modalities = [ "volume", "truth"], ext = "", overwrite = False, preprocess = "window_1_99", scale = [0.5, 0.5, 1],rescale_res = [1.56,.56,3], metadata_path = None, train = True, store_patches= True)
