#!/bin/csh
cd /cs/casmip/nirm/embryo_project_version1/venu-pytorch/bin
source activate.csh
cd /cs/casmip/nirm/embryo_project_version1/embyo_projects_codes/lord-pytorch-unet

python3 lord_exp.py --base-dir /cs/casmip/nirm/embryo_project_version1/EXP_FOLDER/exp2d_remote run_exp --exp-dict-dir /cs/casmip/nirm/embryo_project_version1/embyo_projects_codes/exp_config/exp_brain_2d_remote

#==========================load data ===================================================================================
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_5_2d_S1/TRAIN/Labeled --data-name DFR_5_2d_S1_TL
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_5_2d_S1/TRAIN/ULabeled --data-name DFR_5_2d_S1_TU
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_5_2d_S1/VALIDATION --data-name DFR_5_2d_S1_VA
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_5_2d_S1/TEST --data-name DFR_5_2d_S1_TE
#
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_5_2d_S2/TRAIN/Labeled --data-name DFR_5_2d_S2_TL
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_5_2d_S2/TRAIN/ULabeled --data-name DFR_5_2d_S2_TU
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_5_2d_S2/VALIDATION --data-name DFR_5_2d_S2_VA
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_5_2d_S2/TEST --data-name DFR_5_2d_S2_TE
#
#
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_5_2d_S3/TRAIN/Labeled --data-name DFR_5_2d_S3_TL
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_5_2d_S3/TRAIN/ULabeled --data-name DFR_5_2d_S3_TU
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_5_2d_S3/VALIDATION --data-name DFR_5_2d_S3_VA
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_5_2d_S3/TEST --data-name DFR_5_2d_S3_TE
##
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_10_2d_S1/TRAIN/Labeled --data-name DFR_10_2d_S1_TL
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_10_2d_S1/TRAIN/ULabeled --data-name DFR_10_2d_S1_TU
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_10_2d_S1/VALIDATION --data-name DFR_10_2d_S1_VA
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_10_2d_S1/TEST --data-name DFR_10_2d_S1_TE
#
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_10_2d_S2/TRAIN/Labeled --data-name DFR_10_2d_S2_TL
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_10_2d_S2/TRAIN/ULabeled --data-name DFR_10_2d_S2_TU
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_10_2d_S2/VALIDATION --data-name DFR_10_2d_S2_VA
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_10_2d_S2/TEST --data-name DFR_10_2d_S2_TE
#
#
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_10_2d_S3/TRAIN/Labeled --data-name DFR_10_2d_S3_TL
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_10_2d_S3/TRAIN/ULabeled --data-name DFR_10_2d_S3_TU
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_10_2d_S3/VALIDATION --data-name DFR_10_2d_S3_VA
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_10_2d_S3/TEST --data-name DFR_10_2d_S3_TE
#
#
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_24_2d/TRAIN/Labeled --data-name DFR_24_2d_TL
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_24_2d/TRAIN/ULabeled --data-name DFR_24_2d_TU
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_24_2d/VALIDATION --data-name DFR_24_2d_VA
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_24_2d/TEST --data-name DFR_24_2d_TE
#
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_44_2d/TRAIN/Labeled --data-name DFR_44_2d_TL
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_44_2d/VALIDATION --data-name DFR_44_2d_VA
#python lord_unet.py --base-dir /mnt/local/nirm/Brain/Brain_data preprocess --dataset-id brain_dataset --dataset-path /mnt/local/nirm/Brain/2d_dataset/DFR_44_2d/TEST --data-name DFR_44_2d_TE
