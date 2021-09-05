import os
import numpy as np


unet_emb_result_DF_3_dice = [0.60, 0.5,0.711  ]
unet_emb_result_DF_3_recall = [0.76, 0.75,0.75 ]
unet_emb_result_DF_3_pre = [0.5, 0.38, 0.68 ]

urlord_emb_result_DF_3_dice = [0.55, 0.45, 0.69]
urlord_emb_result_DF_3_recall = [0.76, 0.66, 0.66 ]
urlord_emb_result_DF_3_pre = [0.44, 0.37, 0.733 ]

unet_emb_result_DF_5_dice = [0.67, 0.71,0.61]
unet_emb_result_DF_5_recall = [0.78, 0.85,0.69]
unet_emb_result_DF_5_pre = [0.6, 0.62, 0.54 ]

urlord_emb_result_DF_5_dice = [0.71,0.68 , 0.623]
urlord_emb_result_DF_5_recall = [0.71,0.67, 0.67 ]
urlord_emb_result_DF_5_pre = [0.71, 0.69, 0.5878 ]

unet_emb_result_DF_8_dice = [0.599, 0.74,0.75 ]
unet_emb_result_DF_8_recall = [0.65, 0.82,0.83]
unet_emb_result_DF_8_pre = [0.55, 0.68, 0.68 ]

urlord_emb_result_DF_8_dice = [0.727,0.71 , 0.76]
urlord_emb_result_DF_8_recall = [0.73,0.79, 0.83 ]
urlord_emb_result_DF_8_pre = [0.73, 0.65, 0.69]

unet_emb_result_DF_16_dice = [0.59, 0.78,0.75]
unet_emb_result_DF_16_recall = [0.67, 0.81,0.83]
unet_emb_result_DF_16_pre = [0.53, 0.77, 0.68 ]

urlord_emb_result_DF_16_dice = [0.73,0.765 , 0.76]
urlord_emb_result_DF_16_recall = [0.75,0.75, 0.83 ]
urlord_emb_result_DF_16_pre = [0.71, 0.79, 0.69]


dice_arr_unet = [unet_emb_result_DF_3_dice, unet_emb_result_DF_5_dice, unet_emb_result_DF_8_dice, unet_emb_result_DF_16_dice]
recall_arr_unet = [unet_emb_result_DF_3_recall, unet_emb_result_DF_5_recall, unet_emb_result_DF_8_recall, unet_emb_result_DF_16_recall]
pre_arr_unet = [unet_emb_result_DF_3_pre, unet_emb_result_DF_5_pre, unet_emb_result_DF_8_pre, unet_emb_result_DF_16_pre]

dice_arr_urlord = [urlord_emb_result_DF_3_dice, urlord_emb_result_DF_5_dice, urlord_emb_result_DF_8_dice, urlord_emb_result_DF_16_dice]
recall_arr_urlord = [urlord_emb_result_DF_3_recall, urlord_emb_result_DF_5_recall, urlord_emb_result_DF_8_recall, urlord_emb_result_DF_16_recall]
pre_arr_urlord = [urlord_emb_result_DF_3_pre, urlord_emb_result_DF_5_pre, urlord_emb_result_DF_8_pre, urlord_emb_result_DF_16_pre]



if __name__ == '__main__':

    # working on dice, showing results

    dice_arr_unet =np.array(dice_arr_unet) * 100
    recall_arr_unet  = np.array(recall_arr_unet) * 100
    pre_arr_unet = np.array(pre_arr_unet) * 100

    dice_arr_urlord  = np.array(dice_arr_urlord) * 100
    recall_arr_urlord = np.array(recall_arr_urlord) * 100
    pre_arr_urlord = np.array(pre_arr_urlord) * 100


    dice_meanarr_unet = np.mean(dice_arr_unet, axis= 1)
    recall_meanarr_unet  = np.mean(recall_arr_unet, axis= 1)
    pre_meanarr_unet = np.mean(pre_arr_unet, axis= 1)

    dice_meanarr_urlord  = np.mean(dice_arr_urlord, axis= 1)
    recall_meanarr_urlord = np.mean(recall_arr_urlord, axis= 1)
    pre_meanarr_urlord = np.mean(pre_arr_urlord, axis= 1)


    dice_maxarr_unet = np.max(dice_arr_unet, axis= 1)
    recall_maxarr_unet  = np.max(recall_arr_unet, axis= 1)
    pre_maxarr_unet = np.max(pre_arr_unet, axis= 1)

    dice_maxarr_urlord  = np.max(dice_arr_urlord, axis= 1)
    recall_maxarr_urlord = np.max(recall_arr_urlord, axis= 1)
    pre_maxarr_urlord = np.max(pre_arr_urlord, axis= 1)

    dice_minarr_unet = np.min(dice_arr_unet, axis= 1)
    recall_minarr_unet  = np.min(recall_arr_unet, axis= 1)
    pre_minarr_unet = np.min(pre_arr_unet, axis= 1)

    dice_minarr_urlord  = np.min(dice_arr_urlord, axis= 1)
    recall_minarr_urlord = np.min(recall_arr_urlord, axis= 1)
    pre_minarr_urlord = np.min(pre_arr_urlord, axis= 1)

    dice_std_unet = np.std(dice_arr_unet, axis= 1)
    recall_std_unet  = np.std(recall_arr_unet, axis= 1)
    pre_std_unet = np.std(pre_arr_unet, axis= 1)

    dice_std_urlord  = np.std(dice_arr_urlord, axis= 1)
    recall_std_urlord = np.std(recall_arr_urlord, axis= 1)
    pre_std_urlord = np.std(pre_arr_urlord, axis= 1)



    print("dice mean unet\n")
    print(dice_std_urlord)
    print("Recall mean unet\n")
    print(recall_std_urlord)

    print("pre mean unet\n")
    print( pre_std_urlord)
    #
    # print("\n")
    # print("dice URLORD\n")
    # print(dice_meanarr_urlord)
    # print("max dice unet\n")
    # print(dice_maxarr_unet)
    # print("\n")
    # print("max dice URLORD\n")
    # print(dice_maxarr_urlord)
    # print("min dice unet\n")
    # print(dice_minarr_unet)
    # print("\n")
    # print("min dice URLORD\n")
    # print(dice_minarr_urlord)
    # print("\n")
    # print("min dice URLORD\n")
    # print(dice_minarr_urlord)
    # print("std dice unet\n")
    # print(dice_std_unet)
    # print("\n")
    # print("max dice URLORD\n")
    # print(dice_std_urlord )

