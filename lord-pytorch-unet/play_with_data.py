import torch
from os import listdir
if __name__ == '__main__':
    path = '/cs/casmip/nirm/embryo_project_version1/embryo_data_raw/before_preprocess/placenta'
    a = listdir(path)
    print(a)
    path =  '/cs/casmip/nirm/embryo_project_version1/embryo_data_raw/before_preprocess/TRUFI'
    b = listdir(path)
    print(b)
