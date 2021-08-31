#!/bin/csh
cd /cs/casmip/nirm/embryo_project_version1/venu-pytorch/bin
source activate.csh
cd /cs/casmip/nirm/embryo_project_version1/embyo_projects_codes/lord-pytorch-unet

#nohup python lord_exp.py --base-dir /cs/casmip/nirm/embryo_project_version1/EXP_FOLDER/exp_fp2 run_exp --exp-dict-dir /cs/casmip/nirm/embryo_project_version1/embyo_projects_codes/exp_config/exp_2d/
#python3 lord_exp.py --base-dir /cs/casmip/nirm/embryo_project_version1/EXP_FOLDER/exp3d_remote run_exp --exp-dict-dir /cs/casmip/nirm/embryo_project_version1/embyo_projects_codes/exp_config/exp_3d_remote
python3 lord_exp.py --base-dir /cs/casmip/nirm/embryo_project_version1/EXP_FOLDER/exp3d_plac run_exp --exp-dict-dir /cs/casmip/nirm/embryo_project_version1/embyo_projects_codes/exp_config/exp_3d
#python3 lord_exp.py --base-dir /cs/casmip/nirm/embryo_project_version1/EXP_FOLDER/exp2d_plac run_exp --exp-dict-dir /cs/casmip/nirm/embryo_project_version1/embyo_projects_codes/exp_config/exp_2d_placenta

#DF_1_2d_dict , DF_2_2d_dict, DF_all_2d_dict, DS_2d_dict, D_0_2d_dict, D_1_2d_dict

#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_16_3d_S1/TRAIN/Labeled --data-name DF_16_3d_S1_TL
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_16_3d_S1/TRAIN/ULabeled --data-name DF_16_3d_S1_TU
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_16_3d_S1/VALIDATION --data-name DF_16_3d_S1_VA
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_16_3d_S1/TEST --data-name DF_16_3d_S1_TE
#
#
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_16_3d_S2/TRAIN/Labeled --data-name DF_16_3d_S2_TL
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_16_3d_S2/TRAIN/ULabeled --data-name DF_16_3d_S2_TU
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_16_3d_S2/VALIDATION --data-name DF_16_3d_S2_VA
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_16_3d_S2/TEST --data-name DF_16_3dS2_TE
#
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_16_3d_S3/TRAIN/Labeled --data-name DF_16_3d_S3_TL
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_16_3d_S3/TRAIN/ULabeled --data-name DF_16_3d_S3_TU
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_16_3d_S3/VALIDATION --data-name DF_16_3d_S3_VA
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_16_3d_S3/TEST --data-name DF_16_3d_S3_TE
#
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_5_3d_S1/TRAIN/Labeled --data-name DF_5_3d_S1_TL
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_5_3d_S1/TRAIN/ULabeled --data-name DF_5_3d_S1_TU
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_5_3d_S1/VALIDATION --data-name DF_5_3d_S1_VA
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_5_3d_S1/TEST --data-name DF_5_3d_S1_TE
#
#
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_5_3d_S2/TRAIN/Labeled --data-name DF_5_3d_S2_TL
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_5_3d_S2/TRAIN/ULabeled --data-name DF_5_3d_S2_TU
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_5_3d_S2/VALIDATION --data-name DF_5_3d_S2_VA
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_5_3d_S2/TEST --data-name DF_5_3d_S2_TE
#
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_5_3d_S3/TRAIN/Labeled --data-name DF_5_3d_S3_TL
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_5_3d_S3/TRAIN/ULabeled --data-name DF_5_3d_S3_TU
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_5_3d_S3/VALIDATION --data-name DF_5_3d_S3_VA
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_5_3d_S3/TEST --data-name DF_5_3d_S3_TE
#
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_8_3d_S1/TRAIN/Labeled --data-name DF_8_3d_S1_TL
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_8_3d_S1/TRAIN/ULabeled --data-name DF_8_3d_S1_TU
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_8_3d_S1/VALIDATION --data-name DF_8_3d_S1_VA
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_8_3d_S1/TEST --data-name DF_8_3d_S1_TE
#
#
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_8_3d_S2/TRAIN/Labeled --data-name DF_8_3d_S2_TL
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_8_3d_S2/TRAIN/ULabeled --data-name DF_8_3d_S2_TU
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_8_3d_S2/VALIDATION --data-name DF_8_3d_S2_VA
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_8_3d_S2/TEST --data-name DF_8_3d_S2_TE
#
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_8_3d_S3/TRAIN/Labeled --data-name DF_8_3d_S3_TL
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_8_3d_S3/TRAIN/ULabeled --data-name DF_8_3d_S3_TU
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_8_3d_S3/VALIDATION --data-name DF_8_3d_S3_VA
#python lord_unet.py --base-dir /mnt/local/nirm/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_8_3d_S3/TEST --data-name DF_8_3d_S3_TE

