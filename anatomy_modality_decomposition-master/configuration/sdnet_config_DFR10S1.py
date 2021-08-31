import numpy as np
import os
from configuration import discriminator_config, unet_config_acdc
from loaders import DFR10S1_data_loader
import json
#
loader = DFR10S1_data_loader
params = {
    'seed': 1,
    'folder': 'experiment_sdnet_acdc',
    'data_len': 0,
    'epochs': 100,
    'batch_size': 32,
    'pool_size': 50,
    'split': 0,
    'load_pre': True,
    'description': '',
    'dataset_name': 'emb',
    'test_dataset': 'emb',
    'input_shape': loader.DFR10S1().input_shape,
    'image_downsample': 1,
    'modality': 'MR',
    'prefix': 'norm',                         # Prefix used to load a dataset, e.g. norm_baseline for data_by_dog
    'augment': True,
    'model': 'sdnet.SDNet',
    'executor': 'embryo_sdnet_exec.EmbSDNetExecutor',
    'l_mix': 1,
    'ul_mix': 1,
    'rounding': 'encoder',
    'num_mask_channels': 8,
    'num_z': 8,
    'w_adv_M': 10,
    'w_rec_X': 1,
    'w_rec_Z': 1,
    'w_kl': 0.01,
    'w_sup_M': 10,
    'w_dc': 0,
    'lr': 0.0001,
    'decay': 0.0001,
    'model_dir': "/cs/casmip/nirm/embryo_project_version1/embyo_projects_codes/anatomy_modality_decomposition-master/" + 'experiment_DFR10S1',
    'results': "/cs/casmip/nirm/embryo_project_version1/embyo_projects_codes/anatomy_modality_decomposition-master/" + 'experiment_DFR10S1'+ '/results',

    'path_patch': "/cs/casmip/nirm/embryo_project_version1/Brain_data_raw/Brain_after/patch2D_merge3",
    'path_data_set': "/cs/casmip/nirm/embryo_project_version1/Brain_data_raw/dataset2d/DFR_10_2d_S1",
    'path_data': "/cs/casmip/nirm/embryo_project_version1/Brain_data_raw/Brain_after/FR_FSE"
}







d_mask_params = discriminator_config.params
d_mask_params['downsample_blocks'] = 4
d_mask_params['filters'] = 64
d_mask_params['lr'] = 0.0001
d_mask_params['name'] = 'D_Mask'
d_mask_params['decay'] = 0.0001
d_mask_params['output'] = '1D'

anatomy_encoder_params = unet_config_acdc.params
anatomy_encoder_params['normalise'] = 'batch'
anatomy_encoder_params['downsample'] = 4
anatomy_encoder_params['filters'] = 64
anatomy_encoder_params['out_channels'] = params['num_mask_channels']


def get():
    shp = params['input_shape']
    ratio = params['image_downsample']
    shp = (int(np.round(shp[0] / ratio)), int(np.round(shp[1] / ratio)), shp[2])

    params['input_shape'] = shp
    d_mask_params['input_shape'] = (shp[:-1]) + (loader.EmbL().num_masks,)
    anatomy_encoder_params['input_shape'] = shp

    params.update({'anatomy_encoder_params': anatomy_encoder_params,
                   'd_mask_params': d_mask_params})
    return params

if __name__ == 'main':
    path_patch = "/cs/casmip/nirm/embryo_project_version1/Brain_data_raw/Brain_after/patch2D_merge3"
    with open(os.path.join(path_patch, 'config.json'), 'rb') as patches_param_file:
        data_param = json.load(patches_param_file)

    path_data_set = "/cs/casmip/nirm/embryo_project_version1/Brain_data_raw/dataset2d/DFR_10_2d_S1"
    with open(os.path.join(path_data_set, 'param_dict.json'), 'rb') as patches_param_file:
        split = json.load(patches_param_file)

    path_data = "/cs/casmip/nirm/embryo_project_version1/Brain_data_raw/Brain_after/FR_FSE"
    with open(os.path.join(path_data, 'fetal_data_withgt.h5'), 'rb') as patches_param_file:
        data_withgt = json.load(patches_param_file)
