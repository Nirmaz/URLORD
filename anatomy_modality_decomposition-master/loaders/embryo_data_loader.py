
import os
import nibabel as nib
import numpy as np
from skimage import transform

import utils.data_utils
from loaders.base_loader import Loader

from loaders.data import Data
from parameters import conf
import logging

from loaders.assets import AssetManager


class EmbL(Loader):
    """
    Abstract class defining the behaviour of loaders for different datasets.
    """
    def __init__(self):

        super(EmbL, self).__init__()

        self.base_dir = "/cs/labs/josko/nirm/embryo_project_version1/embryo_data"
        self.data_l_name = "DSRA12d_TL"
        self.data_u_name = "DSRA12d_TU"
        self.data_v_name = "DSRA12d_VA"
        self.data_t_name = "DSRA12d_TE"

        self.num_volumes = 0
        self.input_shape = (None, None, 1)
        self.data_folder = None
        # self.volumes = sorted(self.splits()[0]['training'] +
        #                       self.splits()[0]['validation'] +
        #                       self.splits()[0]['test'])
        self.log = None
        assets = AssetManager(self.base_dir)
        data = np.load(assets.get_preprocess_file_path(self.data_l_name))
        data_u = np.load(assets.get_preprocess_file_path(self.data_u_name))
        data_v = np.load(assets.get_preprocess_file_path(self.data_v_name))
        data_t = np.load(assets.get_preprocess_file_path(self.data_t_name))

        if np.max(data['imgs']) > 1:
            print("dividing....")
            self.imgs_l = data['imgs'].astype(np.float32) / 255
            self.imgs_u = data_u['imgs'].astype(np.float32) / 255
            self.imgs_v = data_v['imgs'].astype(np.float32) / 255
            self.imgs_t = data_t['imgs'].astype(np.float32) / 255
            self.segs_l = data['segs']
            self.classes_l = data['classes']
            self.segs_u = data_u['segs']
            self.classes_u = data_u['classes']
            self.segs_v = data_v['segs']
            self.classes_v = data_v['classes']
            self.segs_t = data_t['segs']
            self.classes_t = data_t['classes']
        # print(segs )
        else:
            self.imgs_l = data['imgs'].astype(np.float32)
            self.imgs_u = data_t['imgs'].astype(np.float32)
            self.imgs_v = data_v['imgs'].astype(np.float32)
            self.imgs_t = data_t['imgs'].astype(np.float32)
            self.segs_l = data['segs']
            self.classes_l = data['classes']
            self.segs_u = data_t['segs']
            self.classes_u = data_t['classes']
            self.segs_v = data_v['segs']
            self.classes_v = data_v['classes']
            self.segs_t = data_t['segs']
            self.classes_t = data_t['classes']
        self.log = logging.getLogger('embl')
        self.input_shape = (self.imgs_l.shape[1], self.imgs_l.shape[2], 1)
        # print(self.input_shape, "INPUT Shape")
        self.num_masks = 1

    def splits(self):
        """
        :return: an array of splits into validation, test and train indices
        """
        pass


    def load_labelled_data(self, split, split_type, modality, normalise=True, value_crop=True, downsample=1):
        """
        Load labelled data from saved numpy arrays.
        Assumes a naming convention of numpy arrays as:
        <dataset_name>_images.npz, <dataset_name>_masks_lv.npz, <dataset_name>_masks_myo.npz etc.
        If numpy arrays are not found, then data is loaded from sources and saved in numpy arrays.

        :param split:       the split number, e.g. 0, 1
        :param split_type:  the split type, e.g. training, validation, test, all (for all data)
        :param modality:    modality to load if the dataset has multimodal data
        :param normalise:   True/False: normalise images to [-1, 1]
        :param value_crop:  True/False: crop values between 5-95 percentiles
        :param downsample:  downsample image ratio - used for for testing
        :return:            a Data object containing the loaded data
        """
        if split_type == 'training':
            # mud = np.zeros((self.imgs_l.shape[0],))
            mud = np.array(['MR'] * self.imgs_l.shape[0])
            d = Data(self.imgs_l, self.segs_l, np.arange(self.segs_l.shape[0]), mud )
            return d
        if split_type == 'validation':
            mud = np.array(['MR'] * self.imgs_t.shape[0])
            return Data(self.imgs_t, self.segs_t, np.arange(self.segs_t.shape[0]), mud)
        if split_type == 'test':
            mud = np.array(['MR'] * self.imgs_t.shape[0])
            return Data(self.imgs_t, self.segs_t, np.arange(self.segs_t.shape[0]), mud)

        raise Exception("bad split type")



    def load_unlabelled_data(self, split, split_type, modality='MR', normalise=True, value_crop=True):
        """
        Load unlabelled data from saved numpy arrays.
        Assumes a naming convention of numpy arrays as ul_<dataset_name>_images.npz
        If numpy arrays are not found, then data is loaded from sources and saved in numpy arrays.
        :param split:       the split number, e.g. 0, 1
        :param split_type:  the split type, e.g. training, validation, test, all (for all data)
        :param modality:    modality to load if the dataset has multimodal data
        :param normalise:   True/False: normalise images to [-1, 1]
        :param value_crop:  True/False: crop values between 5-95 percentiles
        :return:            a Data object containing the loaded data
        """
        mud = np.array(['MR'] * self.imgs_u.shape[0])
        return Data(self.imgs_u, self.segs_u, np.arange(self.segs_u.shape[0]), mud)



    def load_all_data(self, split, split_type, modality='MR', normalise=True, value_crop=True):
        """
        Load all images (labelled and unlabelled) from saved numpy arrays.
        Assumes a naming convention of numpy arrays as all_<dataset_name>_images.npz
        If numpy arrays are not found, then data is loaded from sources and saved in numpy arrays.
        :param split:       the split number, e.g. 0, 1
        :param split_type:  the split type, e.g. training, validation, test, all (for all data)
        :param modality:    modality to load if the dataset has multimodal data
        :param normalise:   True/False: normalise images to [-1, 1]
        :param value_crop:  True/False: crop values between 5-95 percentiles
        :return:            a Data object containing the loaded data
        """
        mud = np.array(['MR'] * (self.imgs_l.shape[0] + self.imgs_u.shape[0]))
        return Data(np.concatenate((self.imgs_u,self.imgs_l)), np.concatenate((self.segs_u,self.segs_l)),np.arange(np.concatenate((self.imgs_u,self.imgs_l)).shape[0]), mud)


    def load_raw_labelled_data(self, normalise=True, value_crop=True):
        """
        Load raw data, do preprocessing e.g. normalisation, resampling, value cropping etc
        :param normalise:  True or False to normalise data
        :param value_crop: True or False to crop in the 5-95 percentiles or not.
        :return:           a pair of arrays (images, index)
        """
        pass


    def load_raw_unlabelled_data(self, include_labelled, normalise=True, value_crop=True):
        """
        Load raw data, do preprocessing e.g. normalisation, resampling, value cropping etc
        :param normalise:  True or False to normalise data
        :param value_crop: True or False to crop in the 5-95 percentiles or not.
        :return:           a pair of arrays (images, index)
        """

    def base_load_unlabelled_images(self, dataset, split, split_type, include_labelled, normalise, value_crop):
        """
        Load only images.
        :param dataset:             dataset name
        :param split:               the split number, e.g. 0, 1
        :param split_type:          the split type, e.g. training, validation, test, all (for all data)
        :param include_labelled:    include images from ES, ED phases that are labelled. Can be True/False
        :param normalise:           True or False to normalise data
        :param value_crop:          True or False to crop in the 5-95 percentiles or not.
        :return:                    a tuple of images and index arrays.
        """
        npz_prefix_type = 'ul_' if not include_labelled else 'all_'
        npz_prefix = npz_prefix_type + 'norm_' if normalise else npz_prefix_type + 'unnorm_'

        # Load saved numpy array
        if os.path.exists(os.path.join(self.data_folder, npz_prefix + dataset + '_images.npz')):
            images = np.load(os.path.join(self.data_folder, npz_prefix + dataset + '_images.npz'))['arr_0']
            index  = np.load(os.path.join(self.data_folder, npz_prefix + dataset + '_index.npz'))['arr_0']
            self.log.debug('Loaded compressed ' + dataset + ' unlabelled data of shape ' + str(images.shape))
        # Load from source
        else:
            images, index = self.load_raw_unlabelled_data(include_labelled, normalise, value_crop)
            images = np.expand_dims(images, axis=3)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + dataset + '_images'), images)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + dataset + '_index'), index)

        assert split_type in ['training', 'validation', 'test', 'all'], 'Unknown split_type: ' + split_type

        if split_type == 'all':
            return images, index

        volumes = self.splits()[split][split_type]
        images = np.concatenate([images[index == v] for v in volumes])
        index  = np.concatenate([index[index==v] for v in volumes])
        return images, index