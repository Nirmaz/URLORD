#!/bin/csh
cd /cs/casmip/nirm/embryo_project_version1/venu-pytorch/bin
source activate.csh
cd /cs/casmip/nirm/embryo_project_version1/embyo_projects_codes/lord-pytorch-unet
#python lord_exp.py --base-dir /cs/labs/josko/nirm/embryo_project_version1/EXP_FOLDER/exp_fp2 run_exp --exp-dict-dir /cs/labs/josko/nirm/embryo_project_version1/EXP_FOLDER/exp_config/exp_1

#python lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data preprocess_onegr --dataset-id embryos_dataset_onegr --dataset-path /cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA12d/TRAIN/Labeled --data-name DSRA12d_TL --t 1 --f 1
#python lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data preprocess_onegr --dataset-id embryos_dataset_onegr --dataset-path /cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA12d/TRAIN/ULabeled --data-name DSRA12d_TU --t 1 --f 0
#python lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data preprocess_onegr --dataset-id embryos_dataset_onegr --dataset-path /cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA12d/VALIDATION --data-name DSRA12d_VA --t 1 --f 1
#python lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data preprocess_onegr --dataset-id embryos_dataset_onegr --dataset-path /cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA12d/TEST --data-name DSRA12d_TE --t 1 --f 0
#
#
#python lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data preprocess_onegr --dataset-id embryos_dataset_onegr --dataset-path /cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA12d/TRAIN/Labeled --data-name DSRA12d_TL --t 1 --f 1
#python lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data preprocess_onegr --dataset-id embryos_dataset_onegr --dataset-path /cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA12d/TRAIN/Unlabeled --data-name DSRA12d_TU --t 1 --f 0
#python lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data preprocess_onegr --dataset-id embryos_dataset_onegr --dataset-path /cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA12d/VALIDATION --data-name DSRA12d_VA --t 1 --f 1
#python lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data preprocess_onegr --dataset-id embryos_dataset_onegr --dataset-path /cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA12d/TEST --data-name DSRA12d_TE --t 1 --f 0


# python lord_exp.py --base-dir /cs/labs/josko/nirm/embryo_project_version1/EXP_FOLDER/exp_fp2 run_exp --exp-dict-dir /cs/labs/josko/nirm/embryo_project_version1/EXP_FOLDER/exp_config/exp_1
#cd /cs/casmip/nirm/embryo_project_version1/embyo_projects_codes/embryo_main
#nohup python3 create_data.py

#python lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data t_and_c_ULord3D --data-l-name DSRA_1_TL --data-u-name DSRA_1_TU --data-v-name DSRA_1_VA --data-t-name DSRA_1_TE  --model-name-u u --model-name-l ulord3d_model_film_sf_16 --load-model 0 --g-config 1 --p-config /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/film_sf_16 --p-uconfig /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/film_sf_16 --p-dataparam /cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA_1/TEST/TRUFI --unet 1 --lord 1 --t-u 1 --t-l 1


#python lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data t_and_c_ULord3D --data-l-name DSRA_1_TL --data-u-name DSRA_1_TU --data-v-name DSRA_1_VA --model-name ulord3d_model_film_sf_8 --load-model 1 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0 --g-config 0 --p-config '' --p-uconfig '' --p-dataparam '/cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA_1/TRAIN/Labeled/placenta' --unet 0 --lord 1 --t-u 1 ----t-l 0
#python lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data preprocess_onegr --dataset-id embryos_dataset_onegr --dataset-path /cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA_1/TRAIN/Labeled --data-name DSRA_1_TL --t 1 --f 1
#python lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data preprocess_onegr --dataset-id embryos_dataset_onegr --dataset-path /cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA_1/TRAIN/Unlabeled --data-name DSRA_1_TU --t 1 --f 0
#python lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data preprocess_onegr --dataset-id embryos_dataset_onegr --dataset-path /cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA_1/VALIDATION --data-name DSRA_1_VA --t 1 --f 1
#python lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data preprocess_onegr --dataset-id embryos_dataset_onegr --dataset-path /cs/casmip/nirm/embryo_project_version1/embryo_data_raw/after_preprocess/DRSA_1/TEST --data-name DSRA_1_TE --t 1 --f 0
#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord3d --data-l-name DSRA_1_TL --data-u-name DSRA_1_TU --data-v-name DSRA_1_VA --data-t-name DSRA_1_TE --model-name ulord3d_model_film_sf_8 --load-model 0 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0 --g-config 1 --p-config /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/film_sf_8 --p-uconfig /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/film_sf_8/
#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_unet3D --data-l-name DSRA_1_TL --data-v-name DSRA_1_VA --data-t-name DSRA_1_TE --model-name unet3d_model --load-model 0 --g-config 1 --p-config /cs/casmip/nirm/embryo_project_version1/config_for_many_run_U/lr_6_sf_8/ --p-uconfig /cs/casmip/nirm/embryo_project_version1/config_for_many_run_U/lr_6_sf_8/




#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord --data-l-name embryo_data_2_labeled_1 --data-u-name embryo_data_2_ulabled_1 --model-name ulord_model_2_0t --load-model 0 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0
#python3 lord_unet.py  --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord --data-l-name embryo_data_1_labeled_1 --data-u-name embryo_data_1_ulabled_1 --model-name ulord_model_1_1 --load-model 0 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0
#python3 lord_unet.py  --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord --data-l-name embryo_data_1_labeled_2 --data-u-name embryo_data_1_ulabled_2 --model-name ulord_model_1_2 --load-model 0 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0
#python3 lord_unet.py  --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord --data-l-name embryo_data_1_labeled_3 --data-u-name embryo_data_1_ulabled_3 --model-name ulord_model_1_3 --load-model 0 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0
#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data split-samples-ulord --input-data-name embryo_data3d_1 --labled-data-name embryo_data3d_1_labeled_1 --ulabled-data-name embryo_data3d_1_ulabled_1 --f-split 1 --t-split 0.1
#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data split-samples-ulord --input-data-name embryo_data3d_1 --labled-data-name embryo_data3d_1_labeled_2 --ulabled-data-name embryo_data3d_1_ulabled_2 --f-split 1 --t-split 0.2
#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data split-samples-ulord --input-data-name embryo_data3d_1 --labled-data-name embryo_data3d_1_labeled_3 --ulabled-data-name embryo_data3d_1_ulabled_3 --f-split 1 --t-split 0.3
#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord3d --data-l-name embryo_dataset_train_l --data-u-name embryo_dataset_train_ul --data-v-name embryo_dataset_val --model-name ulord3d_model --load-model 0 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0 --g-config 0 --p-config '' --p-uconfig ''
#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_unet3D --data-l-name embryo_dataset_train_l --data-v-name embryo_dataset_val --model-name unet3d_model --load-model 0 --g-config 0 --p-config '' --p-uconfig ''

#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_unet3D --data-l-name embryo_dataset_train_l --data-v-name embryo_dataset_val --model-name unet3d_model_lr_4_sf_2 --load-model 1 --g-config 1 --p-config /cs/casmip/nirm/embryo_project_version1/config_for_many_run_U/lr_4_sf_2/ --p-uconfig /cs/casmip/nirm/embryo_project_version1/config_for_many_run_U/lr_4_sf_2/

#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord3d --data-l-name embryo_dataset_train_l --data-u-name embryo_dataset_train_ul --data-v-name embryo_dataset_val --model-name ulord3d_model_lr_4_sf_21 --load-model 0 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0 --g-config 1 --p-config /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_4_sf_2/ --p-uconfig /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_4_sf_2/


#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord3d --data-l-name embryo_dataset_train_l --data-u-name embryo_dataset_train_ul --data-v-name embryo_dataset_val --model-name ulord3d_model_lr_4_sf_4 --load-model 0 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0 --g-config 1 --p-config /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_4_sf_4/ --p-uconfig /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_4_sf_4/
#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord3d --data-l-name embryo_dataset_train_l --data-u-name embryo_dataset_train_ul --data-v-name embryo_dataset_val --model-name ulord3d_model_lr_4_sf_8 --load-model 0 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0 --g-config 1 --p-config /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_4_sf_8/ --p-uconfig /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_4_sf_8/

#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord3d --data-l-name embryo_dataset_train_l --data-u-name embryo_dataset_train_ul --data-v-name embryo_dataset_val --model-name ulord3d_model_lr_5_sf_2 --load-model 0 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0 --g-config 1 --p-config /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_5_sf_2/ --p-uconfig /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_5_sf_2/
#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord3d --data-l-name embryo_dataset_train_l --data-u-name embryo_dataset_train_ul --data-v-name embryo_dataset_val --model-name ulord3d_model_lr_5_sf_4 --load-model 0 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0 --g-config 1 --p-config /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_5_sf_4/ --p-uconfig /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_5_sf_4/
#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord3d --data-l-name embryo_dataset_train_l --data-u-name embryo_dataset_train_ul --data-v-name embryo_dataset_val --model-name ulord3d_model_lr_5_sf_8 --load-model 0 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0 --g-config 1 --p-config /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_5_sf_8/ --p-uconfig /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_5_sf_8/
#
#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord3d --data-l-name embryo_dataset_train_l --data-u-name embryo_dataset_train_ul --data-v-name embryo_dataset_val --model-name ulord3d_model_lr_4_sf_2 --load-model 1 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0 --g-config 1 --p-config /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_4_sf_2/ --p-uconfig /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_4_sf_2/
#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord3d --data-l-name embryo_dataset_train_l --data-u-name embryo_dataset_train_ul --data-v-name embryo_dataset_val --model-name ulord3d_model_lr_4_sf_4 --load-model 1 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0 --g-config 1 --p-config /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_4_sf_4/ --p-uconfig /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_4_sf_4/
#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord3d --data-l-name embryo_dataset_train_l --data-u-name embryo_dataset_train_ul --data-v-name embryo_dataset_val --model-name ulord3d_model_lr_4_sf_8 --load-model 1 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0 --g-config 1 --p-config /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_4_sf_8/ --p-uconfig /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_4_sf_8/
#
#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord3d --data-l-name embryo_dataset_train_l --data-u-name embryo_dataset_train_ul --data-v-name embryo_dataset_val --model-name ulord3d_model_lr_5_sf_2 --load-model 1 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0 --g-config 1 --p-config /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_5_sf_2/ --p-uconfig /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_5_sf_2/
#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord3d --data-l-name embryo_dataset_train_l --data-u-name embryo_dataset_train_ul --data-v-name embryo_dataset_val --model-name ulord3d_model_lr_5_sf_4 --load-model 1 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0 --g-config 1 --p-config /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_5_sf_4/ --p-uconfig /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_5_sf_4/
#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord3d --data-l-name embryo_dataset_train_l --data-u-name embryo_dataset_train_ul --data-v-name embryo_dataset_val --model-name ulord3d_model_lr_5_sf_8 --load-model 1 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0 --g-config 1 --p-config /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_5_sf_8/ --p-uconfig /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_5_sf_8/

#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord3d --data-l-name embryo_dataset_train_l --data-u-name embryo_dataset_train_ul --data-v-name embryo_dataset_val --model-name ulord3d_model --load-model 0 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0 --g-config 1 --p-config /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_6_sf_2/ --p-uconfig /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_6_sf_2/
#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord3d --data-l-name embryo_dataset_train_l --data-u-name embryo_dataset_train_ul --data-v-name embryo_dataset_val --model-name ulord3d_model --load-model 0 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0 --g-config 1 --p-config /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_6_sf_4/ --p-uconfig /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_6_sf_4/
#python3 lord_unet.py --base-dir /cs/casmip/nirm/embryo_project_version1/embryo_data train_ulord3d --data-l-name embryo_dataset_train_l --data-u-name embryo_dataset_train_ul --data-v-name embryo_dataset_val --model-name ulord3d_model --load-model 0 --seg-loss 1 --seg-gard 1 --recon-loss 1 --r-content 0 --r-rclass 0 --g-config 1 --p-config /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_6_sf_8/ --p-uconfig /cs/casmip/nirm/embryo_project_version1/config_for_many_runs/lr_6_sf_8/