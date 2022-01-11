#!/bin/csh
cd /cs/casmip/nirm/embryo_project_version1/venu-pytorch/bin
source activate.csh
cd /cs/casmip/nirm/embryo_project_version1/embyo_projects_codes/urlord
#module load torch
#nohup python lord_exp.py --base-dir /cs/casmip/nirm/embryo_project_version1/EXP_FOLDER/exp_fp2 run_exp --exp-dict-dir /cs/casmip/nirm/embryo_project_version1/embyo_projects_codes/exp_config/exp_2d/
#python3 lord_exp.py --base-dir /cs/casmip/nirm/embryo_project_version1/EXP_FOLDER/exp3d_remote run_exp --exp-dict-dir /cs/casmip/nirm/embryo_project_version1/embyo_projects_codes/exp_config/exp_3d_remote
#python3 lord_exp.py --base-dir /cs/casmip/nirm/embryo_project_version1/EXP_FOLDER/exp3d_plac run_exp --exp-dict-dir /cs/casmip/nirm/embryo_project_version1/embyo_projects_codes/exp_config/exp_3d
#python3 lord_exp.py --base-dir /cs/casmip/nirm/embryo_project_version1/EXP_FOLDER/exp2d_plac run_exp --exp-dict-dir /cs/casmip/nirm/embryo_project_version1/embyo_projects_codes/exp_config/exp_2d_placenta

python3 lord_exp.py --base-dir /cs/casmip/nirm/embryo_project_version1/EXP_FOLDER/exp2d_plac run_exp --exp-dict-dir /cs/casmip/nirm/embryo_project_version1/embyo_projects_codes/exp_config/exp_2d/

#python3 lord_exp.py --base-dir /cs/casmip/nirm/embryo_project_version1/EXP_FOLDER/exp2d_plac run_exp --exp-dict-dir /cs/casmip/nirm/embryo_project_version1/embyo_projects_codes/exp_config/exp_2d/
#python3 lord_exp.py --base-dir /cs/casmip/nirm/embryo_project_version1/EXP_FOLDER/exp2d_plac run_exp --exp-dict-dir /cs/casmip/nirm/embryo_project_version1/embyo_projects_codes/exp_config/exp_2d/
#DF_1_2d_dict , DF_2_2d_dict, DF_all_2d_dict, DS_2d_dict, D_0_2d_dict, D_1_2d_dict

#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_3d_S1/TRAIN/Labeled --data-name DF_16_3d_S1_TL
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_3d_S1/TRAIN/ULabeled --data-name DF_16_3d_S1_TU
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_3d_S1/VALIDATION --data-name DF_16_3d_S1_VA
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_3d_S1/TEST --data-name DF_16_3d_S1_TE
#
##
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_3d_S2/TRAIN/Labeled --data-name DF_16_3d_S2_TL
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_3d_S2/TRAIN/ULabeled --data-name DF_16_3d_S2_TU
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_3d_S2/VALIDATION --data-name DF_16_3d_S2_VA
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_3d_S2/TEST --data-name DF_16_3dS2_TE
##
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_3d_S3/TRAIN/Labeled --data-name DF_16_3d_S3_TL
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_3d_S3/TRAIN/ULabeled --data-name DF_16_3d_S3_TU
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_3d_S3/VALIDATION --data-name DF_16_3d_S3_VA
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_3d_S3/TEST --data-name DF_16_3d_S3_TE
##
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_3d_S1/TRAIN/Labeled --data-name DF_5_3d_S1_TL
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_3d_S1/TRAIN/ULabeled --data-name DF_5_3d_S1_TU
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_3d_S1/VALIDATION --data-name DF_5_3d_S1_VA
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_3d_S1/TEST --data-name DF_5_3d_S1_TE
##
##
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_3d_S2/TRAIN/Labeled --data-name DF_5_3d_S2_TL
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_3d_S2/TRAIN/ULabeled --data-name DF_5_3d_S2_TU
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_3d_S2/VALIDATION --data-name DF_5_3d_S2_VA
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_3d_S2/TEST --data-name DF_5_3d_S2_TE
##
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_3d_S3/TRAIN/Labeled --data-name DF_5_3d_S3_TL
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_3d_S3/TRAIN/ULabeled --data-name DF_5_3d_S3_TU
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_3d_S3/VALIDATION --data-name DF_5_3d_S3_VA
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_3d_S3/TEST --data-name DF_5_3d_S3_TE
##
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_3d_S1/TRAIN/Labeled --data-name DF_8_3d_S1_TL
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_3d_S1/TRAIN/ULabeled --data-name DF_8_3d_S1_TU
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_3d_S1/VALIDATION --data-name DF_8_3d_S1_VA
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_3d_S1/TEST --data-name DF_8_3d_S1_TE
##
##
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_3d_S2/TRAIN/Labeled --data-name DF_8_3d_S2_TL
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_3d_S2/TRAIN/ULabeled --data-name DF_8_3d_S2_TU
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_3d_S2/VALIDATION --data-name DF_8_3d_S2_VA
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_3d_S2/TEST --data-name DF_8_3d_S2_TE
##
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_3d_S3/TRAIN/Labeled --data-name DF_8_3d_S3_TL
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_3d_S3/TRAIN/ULabeled --data-name DF_8_3d_S3_TU
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_3d_S3/VALIDATION --data-name DF_8_3d_S3_VA
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/split_dataset/DF_8_3d_S3/TEST --data-name DF_8_3d_S3_TE

# ========================================================== 2D ========================================================

#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_2d_S1_small/TRAIN/Labeled --data-name DF_16_2d_S1_TL_small
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_2d_S1_small/TRAIN/ULabeled --data-name DF_16_2d_S1_TU_small
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_2d_S1_small/VALIDATION --data-name DF_16_2d_S1_VA_small
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_2d_S1_small/TEST --data-name DF_16_2d_S1_TE_small
#
##
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_2d_S2_small/TRAIN/Labeled --data-name DF_16_2d_S2_TL_small
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_2d_S2_small/TRAIN/ULabeled --data-name DF_16_2d_S2_TU_small
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_2d_S2_small/VALIDATION --data-name DF_16_2d_S2_VA_small
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_2d_S2_small/TEST --data-name DF_16_2d_S2_TE_small
##
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_2d_S3_small/TRAIN/Labeled --data-name DF_16_2d_S3_TL_small
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_2d_S3_small/TRAIN/ULabeled --data-name DF_16_2d_S3_TU_small
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_2d_S3_small/VALIDATION --data-name DF_16_2d_S3_VA_small
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_16_2d_S3_small/TEST --data-name DF_16_2d_S3_TE_small
##
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_2d_S1_small/TRAIN/Labeled --data-name DF_5_2d_S1_TL_small
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_2d_S1_small/TRAIN/ULabeled --data-name DF_5_2d_S1_TU_small
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_2d_S1_small/VALIDATION --data-name DF_5_2d_S1_VA_small
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_2d_S1_small/TEST --data-name DF_5_2d_S1_TE_small
##
##
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_2d_S2/TRAIN/Labeled --data-name DF_5_2d_S2_TL
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_2d_S2/TRAIN/ULabeled --data-name DF_5_2d_S2_TU
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_2d_S2/VALIDATION --data-name DF_5_2d_S2_VA
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_2d_S2/TEST --data-name DF_5_2d_S2_TE
##
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_2d_S3/TRAIN/Labeled --data-name DF_5_2d_S3_TL
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_2d_S3/TRAIN/ULabeled --data-name DF_5_2d_S3_TU
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_2d_S3/VALIDATION --data-name DF_5_2d_S3_VA
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_5_2d_S3/TEST --data-name DF_5_2d_S3_TE
##
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_2d_S1/TRAIN/Labeled --data-name DF_8_2d_S1_TL
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_2d_S1/TRAIN/ULabeled --data-name DF_8_2d_S1_TU
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_2d_S1/VALIDATION --data-name DF_8_2d_S1_VA
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_2d_S1/TEST --data-name DF_8_2d_S1_TE
##
##
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_2d_S2/TRAIN/Labeled --data-name DF_8_2d_S2_TL
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_2d_S2/TRAIN/ULabeled --data-name DF_8_2d_S2_TU
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_2d_S2/VALIDATION --data-name DF_8_2d_S2_VA
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_2d_S2/TEST --data-name DF_8_2d_S2_TE
##
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_2d_S3/TRAIN/Labeled --data-name DF_8_2d_S3_TL
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_2d_S3/TRAIN/ULabeled --data-name DF_8_2d_S3_TU
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_2d_S3/VALIDATION --data-name DF_8_2d_S3_VA
#python lord_unet.py --base-dir /mnt/local/nirm/placenta_data/data_storage preprocess --dataset-id embryos_dataset --dataset-path /mnt/local/nirm/placenta_data/split_dataset/DF_8_2d_S3/TEST --data-name DF_8_3d_S3_TE
#
