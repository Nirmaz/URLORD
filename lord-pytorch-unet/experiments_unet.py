from os import listdir, mkdir
from os.path import join
import pickle
EXP_PATH = '/cs/casmip/nirm/embryo_project_version1/EXP_FOLDER/experiments_unet'
EXP_PATH_C = '/cs/casmip/nirm/embryo_project_version1/EXP_FOLDER/experiments_unet_comp'
EXP_FP = '/cs/casmip/nirm/embryo_project_version1/EXP_FOLDER/exp_fp'
import json
import wandb

def init_expe_directory_n(path, get_num_l_exp = False, init_dir = True):
    """
    init directory for the experiments
    :return: the number of the new experiment
    """

    experiments_folders = listdir(path)
    max_folder = 0
    for folder in experiments_folders:
        if max_folder <= int(folder):
            max_folder = int(folder)

    if get_num_l_exp:
        return max_folder

    if init_dir:
        mkdir(join(path, str(max_folder + 1)))
        mkdir(join(path, str(max_folder + 1), 'results'))
        mkdir(join(path, str(max_folder + 1), 'config'))
        mkdir(join(path, str(max_folder + 1), 'logging'))

    return max_folder + 1



#
def init_expe_directory(get_num_exp = False):
    """
    init directory for the experiments
    :return: the number of the new experiment
    """

    experiments_folders = listdir(EXP_PATH )
    max_folder = 0
    for folder in experiments_folders:
        if max_folder <= int(folder):
            max_folder = int(folder)

    if get_num_exp:
        return max_folder

    mkdir(join(EXP_PATH, str(max_folder + 1)))
    mkdir(join(EXP_PATH, str(max_folder + 1), 'results'))
    mkdir(join(EXP_PATH, str(max_folder + 1), 'config'))
    mkdir(join(EXP_PATH, str(max_folder + 1), 'logging'))
    return max_folder + 1

def init_expe_directoryc(get_num_exp = False):
    """
    init directory for the experiments
    :return: the number of the new experiment
    """

    experiments_folders = listdir(EXP_PATH_C)
    max_folder = 0
    for folder in experiments_folders:
        if max_folder <= int(folder):
            max_folder = int(folder)

    if get_num_exp:
        return max_folder

    mkdir(join(EXP_PATH_C, str(max_folder + 1)))
    mkdir(join(EXP_PATH_C, str(max_folder + 1), 'results'))
    mkdir(join(EXP_PATH_C, str(max_folder + 1), 'config'))
    mkdir(join(EXP_PATH_C, str(max_folder + 1), 'logging'))
    return max_folder + 1


def write_arguments_to_file(args, path):

    commandline_args = open(join( path, 'commandline_args.txt'),'w')
    for i, arg in enumerate(args):
        if i == 0:
            commandline_args.write(arg)
        if i != 0 and i % 2 == 0:
            commandline_args.write('\n' + arg)

        if i % 2 == 1:
            commandline_args.write(' ' + arg)

def jason_dump(item, out_file):

    with open(out_file,  mode='w') as opened_file:
        json.dump(item, opened_file)