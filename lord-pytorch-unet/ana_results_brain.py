import os
import numpy as np

unet_brain_result_DF_1_dice = [0.73, 0.561, 0.62]
urlord_brain_result_DF_1_dice = [0.72, 0.4194, 0.67]
sdknet_brain_result_DF_1_dice = [0.72, 0.4194, 0.67]

unet_brain_result_DF_3_dice = [0.838, 0.8, 0.789]
urlord_brain_result_DF_3_dice = [0.748, 0.779, 0.766]
sdknet_brain_result_DF_3_dice = [0.7338, 0.6772, 0.758]


unet_brain_result_DF_5_dice = [0.827, 0.850,0.878]
urlord_brain_result_DF_5_dice = [0.8198,0.852 , 0.858]
sdknet_brain_result_DF_5_dice = [0.771,0.7939 , 0.63]


unet_brain_result_DF_10_dice = [0.885, 0.8699,0.8799]
urlord_brain_result_DF_10_dice = [0.871,0.878 , 0.8606]
sdknet_brain_result_DF_10_dice = [0.825,0.7387, 0.822]


unet_brain_result_DF_24_dice = [0.912, 0.897, 0.909]
urlord_brain_result_DF_24_dice = [0.905,0.880 , 0.908]
sdknet_brain_result_DF_24_dice = [0.89,0.8286 , 0.8773]



dice_arr_unet = [unet_brain_result_DF_3_dice, unet_brain_result_DF_5_dice, unet_brain_result_DF_10_dice, unet_brain_result_DF_24_dice]

dice_arr_urlord = [urlord_brain_result_DF_3_dice, urlord_brain_result_DF_5_dice, urlord_brain_result_DF_10_dice, urlord_brain_result_DF_24_dice]
dice_arr_sdknet = [sdknet_brain_result_DF_3_dice, sdknet_brain_result_DF_5_dice, sdknet_brain_result_DF_10_dice, sdknet_brain_result_DF_24_dice]




if __name__ == '__main__':

    # working on dice, showing results

    dice_arr_unet =np.array(dice_arr_unet)
    dice_arr_urlord  = np.array(dice_arr_urlord)
    dice_arr_sdknet  = np.array(dice_arr_sdknet)

    dice_meanarr_unet = np.mean(dice_arr_unet, axis= 1)
    dice_meanarr_urlord  = np.mean(dice_arr_urlord, axis= 1)
    dice_meanarr_sdknet  = np.mean(dice_arr_sdknet, axis= 1)

    dice_maxarr_unet = np.max(dice_arr_unet, axis= 1)
    dice_maxarr_urlord  = np.max(dice_arr_urlord, axis= 1)
    dice_maxarr_sdknet  = np.max(dice_arr_sdknet, axis= 1)


    dice_minarr_unet = np.min(dice_arr_unet, axis= 1)
    dice_minarr_urlord  = np.min(dice_arr_urlord, axis= 1)
    dice_minarr_sdknet  = np.min(dice_arr_sdknet, axis= 1)


    dice_std_unet = np.std(dice_arr_unet, axis= 1)
    dice_std_urlord = np.std(dice_arr_urlord, axis= 1)
    dice_std_sdknet = np.std(dice_arr_sdknet, axis= 1)




    print("dice mean urlord\n")
    print(dice_std_unet)

    print("Recall mean unet\n")
    print(dice_std_urlord)
    # print(recall_std_urlord)

    print("pre mean unet\n")
    print(dice_minarr_sdknet)
    #