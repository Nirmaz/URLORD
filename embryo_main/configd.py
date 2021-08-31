
config_patches3d =dict(
	data_set_name = 'patch3D',
	patches_params = dict(
                            model_original_dim = (128, 128, 16),
                            patch_stride = (64, 64, 14),
							with_without_body = True,
							min_num_of_body_voxels = 1,
							min_num_of_plcenta_voxels = 10,
							num_slice_cri = 2,
							with_plcenta = True,
							dim = 3,
							from_bb = False,
							from_h_bb = True,
							margin = 100
						)
)


config_patches2d =dict(

	data_set_name = 'patch2D',
	patches_params = dict(
                            model_original_dim = (128, 128, 14),
                            patch_stride = (32, 32, 14),
							with_without_body = True,
							min_num_of_body_voxels = 1,
							min_num_of_plcenta_voxels = 10,
							num_slice_cri = 16,
							with_plcenta = True,
							dim = 2,
							from_bb = False,
							from_h_bb = True,
							margin = 100
						)
)

all_F = ['228', '225', '234', '211', '91', '208', '201', '81', '93', '104', '176', '197', '78', '191', '232', '224', '223', '135', '189', '177', '95', '76', '258', '174', '188', '170', '205', '203', '209', '200', '184'],
all_T= [ 'ABR', 'GG','Pat2426_Se22_Res1.25_1.25_Spac3.5', 'Pat876_Se17_Res1.25_1.25_Spac3.5', 'HM', 'ILZ', 'Pat3547_Se18_Res0.7421875_0.7421875_Spac3.5', 'RA', 'Pat2771_Se28_Res1.25_1.25_Spac3.5', 'Pat1029_Se17_Res0.7421875_0.7421875_Spac3.5', 'ZM', 'HOY', 'Pat1837_Se18_Res0.9375_0.9375_Spac3.5', 'Pat2175_Se23_Res0.5859375_0.5859375_Spac3.0', 'SS', 'Pat928_Se21_Res1.25_1.25_Spac3.5', 'Pat1274_Se16_Res1.34375_1.34375_Spac3.5', 'Pat1133_Se24_Res1.25_1.25_Spac3.5', 'Pat438_Se19_Res0.78125_0.78125_Spac4.0', 'RR', 'GKY', 'Pat3011_Se24_Res0.7421875_0.7421875_Spac3.5', 'FC', 'Pat2957_Se25_Res0.78125_0.78125_Spac4.0', 'AP', 'Pat1563_Se13_Res1.25_1.25_Spac3.5', 'Pat663_Se20_Res1.25_1.25_Spac3.5', 'KZ', 'GN', 'Pat3201_Se22_Res0.78125_0.78125_Spac4.0', 'Pat978_Se16_Res1.25_1.25_Spac3.5', 'Pat2227_Se19_Res0.7421875_0.7421875_Spac3.5', 'AH', 'Pat1374_Se26_Res0.78125_0.78125_Spac4.0', 'Pat911_Se31_Res0.78125_0.78125_Spac4.0', 'DA']

#-----------------------------------------------------------------------------------------------------------------------


# DF_X_2d_dict exp =====================================================================================================



DF_1_2d_dict =dict(
	data_set_name = 'DF_1_2d',
	data_set_name_patches_name = 'patch2D',
	cases_train_F_l = ['91'],
	cases_train_T_l = [],
	cases_train_F_u = ['208', '201','81', '93', '104', '176', '197', '78', '191', '232', '224', '223', '135', '189', '177'],
	cases_train_T_u = [],
	cases_val_F = ['228', '225', '234', '95', '76'],
	cases_val_T = [],
	cases_test_T = [],
	cases_test_F = ['203', '209', '200', '184', '211', '258', '174', '188', '170', '205'],
)

DF_8_2d_dict =dict(
	data_set_name = 'DF_8_2d',
	data_set_name_patches_name = 'patch2D',
	cases_train_F_l = ['91', '208', '201','81', '93', '104', '176', '197'],
	cases_train_T_l = [],
	cases_train_F_u = ['78', '191', '232', '224', '223', '135', '189', '177', '203', '209', '200', '184', '211'],
	cases_train_T_u = [],
	cases_val_F = ['228', '225', '234', '95', '76'],
	cases_val_T = [],
	cases_test_T = [],
	cases_test_F = ['258', '174', '188', '170', '205'],
)

DF_all_2d_dict =dict(
	data_set_name = 'DF_all_2d',
	data_set_name_patches_name = 'patch2D',
	cases_train_F_l = ['91', '208', '201', '81', '93', '104', '176', '197', '78', '191', '232', '224', '223', '135', '189'],
	cases_train_T_l = [],
	cases_train_F_u = ['177'],
	cases_train_T_u = [],
	cases_val_F = ['228', '225', '234', '95', '76'],
	cases_val_T = [],
	cases_test_T = [],
	cases_test_F = ['203', '209', '200', '184', '211', '258', '174', '188', '170', '205'],
)

# D_X_2d ==============================================================================================================

D_1_2d_dict =dict(

	data_set_name = 'D_1_2d',
	data_set_name_patches_name = 'patch2D',
	cases_train_F_l = ['228', '225', '234', '211', '91', '208', '201', '81', '93', '104', '176', '197', '78', '191', '232', '224', '223', '135', '189', '177', '95', '76', '258', '174', '188', '170', '205', '203', '209', '200', '184'],
	cases_train_T_l = ['ABR'],
	cases_train_F_u = [],
	cases_train_T_u = ['Pat2771_Se28_Res1.25_1.25_Spac3.5','Pat1029_Se17_Res0.7421875_0.7421875_Spac3.5','Pat1837_Se18_Res0.9375_0.9375_Spac3.5', 'Pat2175_Se23_Res0.5859375_0.5859375_Spac3.0','Pat928_Se21_Res1.25_1.25_Spac3.5', 'Pat1274_Se16_Res1.34375_1.34375_Spac3.5', 'Pat1133_Se24_Res1.25_1.25_Spac3.5', 'Pat438_Se19_Res0.78125_0.78125_Spac4.0','Pat3011_Se24_Res0.7421875_0.7421875_Spac3.5','Pat2957_Se25_Res0.78125_0.78125_Spac4.0', 'Pat1563_Se13_Res1.25_1.25_Spac3.5', 'Pat663_Se20_Res1.25_1.25_Spac3.5','Pat3201_Se22_Res0.78125_0.78125_Spac4.0', 'Pat978_Se16_Res1.25_1.25_Spac3.5','Pat2227_Se19_Res0.7421875_0.7421875_Spac3.5','Pat1374_Se26_Res0.78125_0.78125_Spac4.0', 'Pat911_Se31_Res0.78125_0.78125_Spac4.0'],
	cases_val_F = [],
	cases_val_T = ['GG','Pat2426_Se22_Res1.25_1.25_Spac3.5','Pat876_Se17_Res1.25_1.25_Spac3.5', 'Pat3547_Se18_Res0.7421875_0.7421875_Spac3.5'],
	cases_test_T = ['HM', 'ILZ', 'RA', 'ZM', 'HOY', 'SS', 'RR', 'GKY', 'FC', 'AP',  'KZ', 'GN', 'AH',  'DA'],
	cases_test_F = [],
)


DS_2d_dict = dict(

	data_set_name = 'DS_2d',
	data_set_name_patches_name = 'patch2D',
	cases_train_F_l = [ '91', '208', '201', '81', '93', '104', '176', '197', '78', '191', '232', '224', '223', '135', '189', '177', '95', '76', '258', '174', '188', '170', '205', '203', '209', '200', '184'],
	cases_train_T_l = ['Pat2771_Se28_Res1.25_1.25_Spac3.5','Pat1029_Se17_Res0.7421875_0.7421875_Spac3.5','Pat1837_Se18_Res0.9375_0.9375_Spac3.5', 'Pat2175_Se23_Res0.5859375_0.5859375_Spac3.0','Pat928_Se21_Res1.25_1.25_Spac3.5', 'Pat1274_Se16_Res1.34375_1.34375_Spac3.5', 'Pat1133_Se24_Res1.25_1.25_Spac3.5', 'Pat438_Se19_Res0.78125_0.78125_Spac4.0','Pat3011_Se24_Res0.7421875_0.7421875_Spac3.5','Pat2957_Se25_Res0.78125_0.78125_Spac4.0', 'Pat1563_Se13_Res1.25_1.25_Spac3.5', 'Pat663_Se20_Res1.25_1.25_Spac3.5','Pat3201_Se22_Res0.78125_0.78125_Spac4.0', 'Pat978_Se16_Res1.25_1.25_Spac3.5','Pat2227_Se19_Res0.7421875_0.7421875_Spac3.5','Pat1374_Se26_Res0.78125_0.78125_Spac4.0', 'Pat911_Se31_Res0.78125_0.78125_Spac4.0'],
	cases_train_F_u = [],
	cases_train_T_u = ['Pat3547_Se18_Res0.7421875_0.7421875_Spac3.5'],
	cases_val_F = [],
	cases_val_T = ['GG', 'ABR' ,'Pat2426_Se22_Res1.25_1.25_Spac3.5','Pat876_Se17_Res1.25_1.25_Spac3.5'],
	cases_test_T = ['HM', 'ILZ', 'RA', 'ZM', 'HOY', 'SS', 'RR', 'GKY', 'FC', 'AP',  'KZ', 'GN', 'AH',  'DA'],
	cases_test_F = [],
)


D_0_2d_dict =dict(

	data_set_name = 'D_0_2d',
	data_set_name_patches_name = 'patch2D',
	cases_train_F_l = [ '228', '225', '234', '211', '91', '208', '201', '81', '93', '104', '176', '197', '78', '191', '232', '224', '223', '135', '189', '177', '95', '76', '258', '174', '188', '170', '205', '203', '209', '200', '184'],
	cases_train_T_l = [],
	cases_train_F_u = [],
	cases_train_T_u = ['ABR', 'Pat2771_Se28_Res1.25_1.25_Spac3.5','Pat1029_Se17_Res0.7421875_0.7421875_Spac3.5','Pat1837_Se18_Res0.9375_0.9375_Spac3.5', 'Pat2175_Se23_Res0.5859375_0.5859375_Spac3.0','Pat928_Se21_Res1.25_1.25_Spac3.5', 'Pat1274_Se16_Res1.34375_1.34375_Spac3.5', 'Pat1133_Se24_Res1.25_1.25_Spac3.5', 'Pat438_Se19_Res0.78125_0.78125_Spac4.0','Pat3011_Se24_Res0.7421875_0.7421875_Spac3.5','Pat2957_Se25_Res0.78125_0.78125_Spac4.0', 'Pat1563_Se13_Res1.25_1.25_Spac3.5', 'Pat663_Se20_Res1.25_1.25_Spac3.5','Pat3201_Se22_Res0.78125_0.78125_Spac4.0', 'Pat978_Se16_Res1.25_1.25_Spac3.5','Pat2227_Se19_Res0.7421875_0.7421875_Spac3.5','Pat1374_Se26_Res0.78125_0.78125_Spac4.0', 'Pat911_Se31_Res0.78125_0.78125_Spac4.0'],
	cases_val_F = [],
	cases_val_T = ['GG','Pat2426_Se22_Res1.25_1.25_Spac3.5','Pat876_Se17_Res1.25_1.25_Spac3.5', 'Pat3547_Se18_Res0.7421875_0.7421875_Spac3.5'],
	cases_test_T = ['HM', 'ILZ', 'RA', 'ZM', 'HOY', 'SS', 'RR', 'GKY', 'FC', 'AP',  'KZ', 'GN', 'AH',  'DA'],
	cases_test_F = [],
)



# ======================================================================================================================

# ================================================ 3D ==================================================================
# ================================================ D_X_3d ==============================================================

D_5_3d_dict =dict(

	data_set_name = 'D_5_3d',
	data_set_name_patches_name = 'patch3D',
	cases_train_F_l = ['228', '225', '234', '211', '91', '208', '201', '81', '93', '104', '176', '197', '78', '191', '232', '224', '223', '135', '189', '177', '95', '76', '258', '174', '188', '170', '205', '203', '209', '200', '184'],
	cases_train_T_l = ['ABR', 'HM', 'ILZ', 'RR', 'GKY'],
	cases_train_F_u = [],
	cases_train_T_u = ['Pat2771_Se28_Res1.25_1.25_Spac3.5','Pat1029_Se17_Res0.7421875_0.7421875_Spac3.5','Pat1837_Se18_Res0.9375_0.9375_Spac3.5', 'Pat2175_Se23_Res0.5859375_0.5859375_Spac3.0','Pat928_Se21_Res1.25_1.25_Spac3.5', 'Pat1274_Se16_Res1.34375_1.34375_Spac3.5', 'Pat1133_Se24_Res1.25_1.25_Spac3.5', 'Pat438_Se19_Res0.78125_0.78125_Spac4.0','Pat3011_Se24_Res0.7421875_0.7421875_Spac3.5','Pat2957_Se25_Res0.78125_0.78125_Spac4.0', 'Pat1563_Se13_Res1.25_1.25_Spac3.5', 'Pat663_Se20_Res1.25_1.25_Spac3.5','Pat3201_Se22_Res0.78125_0.78125_Spac4.0', 'Pat978_Se16_Res1.25_1.25_Spac3.5','Pat2227_Se19_Res0.7421875_0.7421875_Spac3.5','Pat1374_Se26_Res0.78125_0.78125_Spac4.0', 'Pat911_Se31_Res0.78125_0.78125_Spac4.0', 'Pat2426_Se22_Res1.25_1.25_Spac3.5','Pat876_Se17_Res1.25_1.25_Spac3.5', 'Pat3547_Se18_Res0.7421875_0.7421875_Spac3.5'],
	cases_val_F = [],
	cases_val_T = ['GG', 'RA', 'ZM', 'HOY', 'SS'],
	cases_test_T = ['FC', 'AP',  'KZ', 'GN', 'AH',  'DA'],
	cases_test_F = [],
)


# D_1_3d_dict =dict(
#
# 	data_set_name = 'D_1_3d',
# 	data_set_name_patches_name = 'patch3D',
# 	cases_train_F_l = ['228', '225', '234', '211', '91', '208', '201', '81', '93', '104', '176', '197', '78', '191', '232', '224', '223', '135', '189', '177', '95', '76', '258', '174', '188', '170', '205'],
# 	cases_train_T_l = ['ABR'],
# 	cases_train_F_u = [],
# 	cases_train_T_u = ['Pat2771_Se28_Res1.25_1.25_Spac3.5','Pat1029_Se17_Res0.7421875_0.7421875_Spac3.5','Pat1837_Se18_Res0.9375_0.9375_Spac3.5', 'Pat2175_Se23_Res0.5859375_0.5859375_Spac3.0','Pat928_Se21_Res1.25_1.25_Spac3.5', 'Pat1274_Se16_Res1.34375_1.34375_Spac3.5', 'Pat1133_Se24_Res1.25_1.25_Spac3.5', 'Pat438_Se19_Res0.78125_0.78125_Spac4.0','Pat3011_Se24_Res0.7421875_0.7421875_Spac3.5','Pat2957_Se25_Res0.78125_0.78125_Spac4.0', 'Pat1563_Se13_Res1.25_1.25_Spac3.5', 'Pat663_Se20_Res1.25_1.25_Spac3.5','Pat3201_Se22_Res0.78125_0.78125_Spac4.0', 'Pat978_Se16_Res1.25_1.25_Spac3.5','Pat2227_Se19_Res0.7421875_0.7421875_Spac3.5','Pat1374_Se26_Res0.78125_0.78125_Spac4.0', 'Pat911_Se31_Res0.78125_0.78125_Spac4.0', 'Pat2426_Se22_Res1.25_1.25_Spac3.5','Pat876_Se17_Res1.25_1.25_Spac3.5', 'Pat3547_Se18_Res0.7421875_0.7421875_Spac3.5'],
# 	cases_val_F = [],
# 	cases_val_T = ['GG',  '203', '209', '200', '184'],
# 	cases_test_T = [ 'HM', 'ILZ', 'RA', 'ZM', 'HOY', 'SS', 'RR', 'GKY', 'FC', 'AP',  'KZ', 'GN', 'AH',  'DA'],
# 	cases_test_F = [],
# )
#
# D_1_3d_dict =dict(
#
# 	data_set_name = 'D_1_3d',
# 	data_set_name_patches_name = 'patch3D',
# 	cases_train_F_l = ['228', '225', '234', '211', '91', '208', '201', '81', '93', '104', '176', '197', '78', '191', '232', '224', '223', '135', '189', '177', '95', '76', '258', '174', '188', '170', '205', '203', '209', '200', '184'],
# 	cases_train_T_l = ['ABR', 'GG'],
# 	cases_train_F_u = [],
# 	cases_train_T_u = ['Pat2771_Se28_Res1.25_1.25_Spac3.5','Pat1029_Se17_Res0.7421875_0.7421875_Spac3.5','Pat1837_Se18_Res0.9375_0.9375_Spac3.5', 'Pat2175_Se23_Res0.5859375_0.5859375_Spac3.0','Pat928_Se21_Res1.25_1.25_Spac3.5', 'Pat1274_Se16_Res1.34375_1.34375_Spac3.5', 'Pat1133_Se24_Res1.25_1.25_Spac3.5', 'Pat438_Se19_Res0.78125_0.78125_Spac4.0','Pat3011_Se24_Res0.7421875_0.7421875_Spac3.5','Pat2957_Se25_Res0.78125_0.78125_Spac4.0', 'Pat1563_Se13_Res1.25_1.25_Spac3.5', 'Pat663_Se20_Res1.25_1.25_Spac3.5','Pat3201_Se22_Res0.78125_0.78125_Spac4.0', 'Pat978_Se16_Res1.25_1.25_Spac3.5','Pat2227_Se19_Res0.7421875_0.7421875_Spac3.5','Pat1374_Se26_Res0.78125_0.78125_Spac4.0', 'Pat911_Se31_Res0.78125_0.78125_Spac4.0'],
# 	cases_val_F = [],
# 	cases_val_T = ['Pat2426_Se22_Res1.25_1.25_Spac3.5','Pat876_Se17_Res1.25_1.25_Spac3.5', 'Pat3547_Se18_Res0.7421875_0.7421875_Spac3.5'],
# 	cases_test_T = [ 'HM', 'ILZ', 'RA', 'ZM', 'HOY', 'SS', 'RR', 'GKY', 'FC', 'AP',  'KZ', 'GN', 'AH',  'DA'],
# 	cases_test_F = [],
# )
#
# D_2_3d_dict = dict(
#
# 	data_set_name = 'DS_2d',
# 	data_set_name_patches_name = 'patch2D',
# 	cases_train_F_l = [ '91', '208', '201', '81', '93', '104', '176', '197', '78', '191', '232', '224', '223', '135', '189', '177', '95', '76', '258', '174', '188', '170', '205', '203', '209', '200', '184'],
# 	cases_train_T_l = ['Pat2771_Se28_Res1.25_1.25_Spac3.5','Pat1029_Se17_Res0.7421875_0.7421875_Spac3.5','Pat1837_Se18_Res0.9375_0.9375_Spac3.5', 'Pat2175_Se23_Res0.5859375_0.5859375_Spac3.0','Pat928_Se21_Res1.25_1.25_Spac3.5', 'Pat1274_Se16_Res1.34375_1.34375_Spac3.5', 'Pat1133_Se24_Res1.25_1.25_Spac3.5', 'Pat438_Se19_Res0.78125_0.78125_Spac4.0','Pat3011_Se24_Res0.7421875_0.7421875_Spac3.5','Pat2957_Se25_Res0.78125_0.78125_Spac4.0', 'Pat1563_Se13_Res1.25_1.25_Spac3.5', 'Pat663_Se20_Res1.25_1.25_Spac3.5','Pat3201_Se22_Res0.78125_0.78125_Spac4.0', 'Pat978_Se16_Res1.25_1.25_Spac3.5','Pat2227_Se19_Res0.7421875_0.7421875_Spac3.5','Pat1374_Se26_Res0.78125_0.78125_Spac4.0', 'Pat911_Se31_Res0.78125_0.78125_Spac4.0'],
# 	cases_train_F_u = [],
# 	cases_train_T_u = ['Pat3547_Se18_Res0.7421875_0.7421875_Spac3.5'],
# 	cases_val_F = [],
# 	cases_val_T = ['Pat2426_Se22_Res1.25_1.25_Spac3.5','Pat876_Se17_Res1.25_1.25_Spac3.5'],
# 	cases_test_T = ['HM', 'ILZ', 'RA', 'ZM', 'HOY', 'SS', 'RR', 'GKY', 'FC', 'AP',  'KZ', 'GN', 'AH',  'DA'],
# 	cases_test_F = [],
# )
#
#
# D_0_2d_dict =dict(
#
# 	data_set_name = 'D_0_2d',
# 	data_set_name_patches_name = 'patch2D',
# 	cases_train_F_l = [ '228', '225', '234', '211', '91', '208', '201', '81', '93', '104', '176', '197', '78', '191', '232', '224', '223', '135', '189', '177', '95', '76', '258', '174', '188', '170', '205', '203', '209', '200', '184'],
# 	cases_train_T_l = [],
# 	cases_train_F_u = [],
# 	cases_train_T_u = ['ABR', 'Pat2771_Se28_Res1.25_1.25_Spac3.5','Pat1029_Se17_Res0.7421875_0.7421875_Spac3.5','Pat1837_Se18_Res0.9375_0.9375_Spac3.5', 'Pat2175_Se23_Res0.5859375_0.5859375_Spac3.0','Pat928_Se21_Res1.25_1.25_Spac3.5', 'Pat1274_Se16_Res1.34375_1.34375_Spac3.5', 'Pat1133_Se24_Res1.25_1.25_Spac3.5', 'Pat438_Se19_Res0.78125_0.78125_Spac4.0','Pat3011_Se24_Res0.7421875_0.7421875_Spac3.5','Pat2957_Se25_Res0.78125_0.78125_Spac4.0', 'Pat1563_Se13_Res1.25_1.25_Spac3.5', 'Pat663_Se20_Res1.25_1.25_Spac3.5','Pat3201_Se22_Res0.78125_0.78125_Spac4.0', 'Pat978_Se16_Res1.25_1.25_Spac3.5','Pat2227_Se19_Res0.7421875_0.7421875_Spac3.5','Pat1374_Se26_Res0.78125_0.78125_Spac4.0', 'Pat911_Se31_Res0.78125_0.78125_Spac4.0'],
# 	cases_val_F = [],
# 	cases_val_T = ['GG','Pat2426_Se22_Res1.25_1.25_Spac3.5','Pat876_Se17_Res1.25_1.25_Spac3.5', 'Pat3547_Se18_Res0.7421875_0.7421875_Spac3.5'],
# 	cases_test_T = ['HM', 'ILZ', 'RA', 'ZM', 'HOY', 'SS', 'RR', 'GKY', 'FC', 'AP',  'KZ', 'GN', 'AH',  'DA'],
# 	cases_test_F = [],
# )

print("nir")
# ================================================ 3D  DF_X ============================================================

DF_3_3d_S1_dict =dict(
	data_set_name = 'DF_3_3d_S1',
	data_set_name_patches_name = 'patch3D',
	cases_train_F_l = ['78', '191', '232'],
	cases_train_T_l = [],
	cases_train_F_u = ['91','208', '201',  '224', '223', '135', '189', '177', '228', '225', '234', '95', '76', '104', '176', '197','81', '93'],
	cases_train_T_u = [],
	cases_val_F = ['203', '209', '200', '184', '211'],
	cases_val_T = [],
	cases_test_T = [],
	cases_test_F = ['258', '174', '188', '170', '205'],
)

DF_3_3d_S2_dict =dict(
	data_set_name = 'DF_3_3d_S2',
	data_set_name_patches_name = 'patch3D',
	cases_train_F_l = ['91','208', '201'],
	cases_train_T_l = [],
	cases_train_F_u = ['78', '191', '232', '224', '223', '135', '189', '177', '228', '225', '234', '95', '76', '104', '176', '197','81', '93'],
	cases_train_T_u = [],
	cases_val_F = ['203', '209', '200', '184', '211'],
	cases_val_T = [],
	cases_test_T = [],
	cases_test_F = ['258', '174', '188', '170', '205'],
)


DF_3_3d_S3_dict =dict(
	data_set_name = 'DF_3_3d_S3',
	data_set_name_patches_name = 'patch3D',
	cases_train_F_l = ['224', '223', '135'],
	cases_train_T_l = [],
	cases_train_F_u = ['91','208', '201', '78', '191', '232',  '189', '177', '228', '225', '234', '95', '76', '104', '176', '197','81', '93'],
	cases_train_T_u = [],
	cases_val_F = ['203', '209', '200', '184', '211'],
	cases_val_T = [],
	cases_test_T = [],
	cases_test_F = ['258', '174', '188', '170', '205'],
)


DF_5_3d_S1_dict =dict(
	data_set_name = 'DF_5_3d_S1',
	data_set_name_patches_name = 'patch3D',
	cases_train_F_l = ['91','208', '201','81', '93'],
	cases_train_T_l = [],
	cases_train_F_u = ['78', '191', '232', '224', '223', '135', '189', '177', '228', '225', '234', '95', '76', '104', '176', '197'],
	cases_train_T_u = [],
	cases_val_F = ['203', '209', '200', '184', '211'],
	cases_val_T = [],
	cases_test_T = [],
	cases_test_F = ['258', '174', '188', '170', '205'],
)

DF_5_3d_S2_dict=dict(
	data_set_name = 'DF_5_3d_S2',
	data_set_name_patches_name = 'patch3D',
	cases_train_F_l = ['78', '191', '232', '224', '223'],
	cases_train_T_l = [],
	cases_train_F_u = ['91','208', '201','81', '93', '135', '189', '177', '228', '225', '234', '95', '76', '104', '176', '197'],
	cases_train_T_u = [],
	cases_val_F = ['203', '209', '200', '184', '211'],
	cases_val_T = [],
	cases_test_T = [],
	cases_test_F = ['258', '174', '188', '170', '205'],
)

DF_5_3d_S3_dict=dict(
	data_set_name = 'DF_5_3d_S3',
	data_set_name_patches_name = 'patch3D',
	cases_train_F_l = [ '135', '189', '177', '228', '225'],
	cases_train_T_l = [],
	cases_train_F_u = ['78', '191', '232', '224', '223', '91','208', '201','81', '93', '234', '95', '76', '104', '176', '197'],
	cases_train_T_u = [],
	cases_val_F = ['203', '209', '200', '184', '211'],
	cases_val_T = [],
	cases_test_T = [],
	cases_test_F = ['258', '174', '188', '170', '205'],
)

DF_8_3d_dict =dict(
	data_set_name = 'DF_8_3d',
	data_set_name_patches_name = 'patch3D',
	cases_train_F_l = ['91','208', '201','81', '93', '104', '176', '197'],
	cases_train_T_l = [],
	cases_train_F_u = ['78', '191', '232', '224', '223', '135', '189', '177', '228', '225', '234', '95', '76'],
	cases_train_T_u = [],
	cases_val_F = ['203', '209', '200', '184', '211'],
	cases_val_T = [],
	cases_test_T = [],
	cases_test_F = ['258', '174', '188', '170', '205'],
)


DF_8_3d_S1_dict =dict(
	data_set_name = 'DF_8_3d_S1',
	data_set_name_patches_name = 'patch3D',
	cases_train_F_l = ['91','208', '201','81', '78', '191', '232', '224'],
	cases_train_T_l = [],
	cases_train_F_u = [ '93', '104', '176', '197', '223', '135', '189', '177', '228', '225', '234', '95', '76'],
	cases_train_T_u = [],
	cases_val_F = ['203', '209', '200', '184', '211'],
	cases_val_T = [],
	cases_test_T = [],
	cases_test_F = ['258', '174', '188', '170', '205'],
)


DF_8_3d_S2_dict =dict(
	data_set_name = 'DF_8_3d_S2',
	data_set_name_patches_name = 'patch3D',
	cases_train_F_l = ['78', '191', '232', '224', '223', '135', '189', '177'],
	cases_train_T_l = [],
	cases_train_F_u = ['91','208', '201','81', '93', '104', '176', '197', '228', '225', '234', '95', '76'],
	cases_train_T_u = [],
	cases_val_F = ['203', '209', '200', '184', '211'],
	cases_val_T = [],
	cases_test_T = [],
	cases_test_F = ['258', '174', '188', '170', '205'],
)


DF_8_3d_S3_dict =dict(
	data_set_name = 'DF_8_3d_S3',
	data_set_name_patches_name = 'patch3D',
	cases_train_F_l = ['225', '234', '95', '76', '223', '135', '189', '177'],
	cases_train_T_l = [],
	cases_train_F_u = ['91','208', '201','81', '93', '104', '176', '197', '228', '78', '191', '232', '224'],
	cases_train_T_u = [],
	cases_val_F = ['203', '209', '200', '184', '211'],
	cases_val_T = [],
	cases_test_T = [],
	cases_test_F = ['258', '174', '188', '170', '205'],
)


DF_16_3d_S1_dict =dict(
	data_set_name = 'DF_16_3d_S1',
	data_set_name_patches_name = 'patch3D',
	cases_train_F_l = ['91','208', '201','81', '93', '104', '176', '197', '78', '191', '232', '224', '223', '135', '189', '177'],
	cases_train_T_l = [],
	cases_train_F_u = ['228', '225', '234', '95', '76'],
	cases_train_T_u = [],
	cases_val_F = ['203', '209', '200', '184', '211'],
	cases_val_T = [],
	cases_test_T = [],
	cases_test_F = ['258', '174', '188', '170', '205'],
)


DF_16_3d_S2_dict =dict(
	data_set_name = 'DF_16_3d_S2',
	data_set_name_patches_name = 'patch3D',
	cases_train_F_l = ['203', '209', '200', '184', '211', '104', '176', '197', '78', '191', '232', '228', '225', '234', '95', '76'],
	cases_train_T_l = [],
	cases_train_F_u = [ '224', '223', '135', '189', '177'],
	cases_train_T_u = [],
	cases_val_F = ['91','208', '201','81', '93'],
	cases_val_T = [],
	cases_test_T = [],
	cases_test_F = ['258', '174', '188', '170', '205'],
)


DF_16_3d_S3_dict =dict(
	data_set_name = 'DF_16_3d_S3',
	data_set_name_patches_name = 'patch3D',
	cases_train_F_l = ['203', '209', '200', '184', '211', '228', '225', '234', '95', '76', '91','208', '201','81', '93', '177'],
	cases_train_T_l = [],
	cases_train_F_u = ['104', '176', '197', '78', '191'],
	cases_train_T_u = [],
	cases_val_F = ['232', '224', '223', '135', '189'],
	cases_val_T = [],
	cases_test_T = [],
	cases_test_F = ['258', '174', '188', '170', '205'],
)





DF_21_3d_dict =dict(
	data_set_name = 'DF_21_3d',
	data_set_name_patches_name = 'patch3D',
	cases_train_F_l = ['91', '208', '201','81', '93', '104', '176', '197', '78', '191', '232', '224', '223', '135', '189', '177', '228', '225', '234', '95', '76'],
	cases_train_T_l = [],
	cases_train_F_u = [],
	cases_train_T_u = [],
	cases_val_F = ['203', '209', '200', '184', '211'],
	cases_val_T = [],
	cases_test_T = [],
	cases_test_F = ['258', '174', '188', '170', '205'],
)

DT_5_3d_dict =dict(

	data_set_name = 'DT_5_3d',
	data_set_name_patches_name = 'patch3D',
	cases_train_F_l = [],
	cases_train_T_l = ['ABR', 'HM', 'ILZ', 'RR', 'GKY'],
	cases_train_F_u = [],
	cases_train_T_u = ['Pat2771_Se28_Res1.25_1.25_Spac3.5','Pat1029_Se17_Res0.7421875_0.7421875_Spac3.5','Pat1837_Se18_Res0.9375_0.9375_Spac3.5', 'Pat2175_Se23_Res0.5859375_0.5859375_Spac3.0','Pat928_Se21_Res1.25_1.25_Spac3.5', 'Pat1274_Se16_Res1.34375_1.34375_Spac3.5', 'Pat1133_Se24_Res1.25_1.25_Spac3.5', 'Pat438_Se19_Res0.78125_0.78125_Spac4.0','Pat3011_Se24_Res0.7421875_0.7421875_Spac3.5','Pat2957_Se25_Res0.78125_0.78125_Spac4.0', 'Pat1563_Se13_Res1.25_1.25_Spac3.5', 'Pat663_Se20_Res1.25_1.25_Spac3.5','Pat3201_Se22_Res0.78125_0.78125_Spac4.0', 'Pat978_Se16_Res1.25_1.25_Spac3.5','Pat2227_Se19_Res0.7421875_0.7421875_Spac3.5','Pat1374_Se26_Res0.78125_0.78125_Spac4.0', 'Pat911_Se31_Res0.78125_0.78125_Spac4.0', 'Pat2426_Se22_Res1.25_1.25_Spac3.5','Pat876_Se17_Res1.25_1.25_Spac3.5', 'Pat3547_Se18_Res0.7421875_0.7421875_Spac3.5'],
	cases_val_F = [],
	cases_val_T = ['GG', 'RA', 'ZM', 'HOY', 'SS'],
	cases_test_T = ['FC', 'AP',  'KZ', 'GN', 'AH',  'DA'],
	cases_test_F = [],
)

# ================================================ 3D ==================================================================



config_data_set =dict(

	data_set_name = 'DS_3d',
	data_set_name_patches_name = 'patch2D',
	cases_train_F_l = [ '91', '208', '201', '81', '93', '104', '176', '197', '78', '191', '232', '224', '223', '135', '189', '177', '95', '76', '258', '174', '188', '170', '205', '203', '209', '200', '184'],
	cases_train_T_l = ['ABR', 'GG', 'Pat2426_Se22_Res1.25_1.25_Spac3.5','Pat876_Se17_Res1.25_1.25_Spac3.5', 'Pat3547_Se18_Res0.7421875_0.7421875_Spac3.5', 'Pat2771_Se28_Res1.25_1.25_Spac3.5','Pat1029_Se17_Res0.7421875_0.7421875_Spac3.5','Pat1837_Se18_Res0.9375_0.9375_Spac3.5', 'Pat2175_Se23_Res0.5859375_0.5859375_Spac3.0','Pat928_Se21_Res1.25_1.25_Spac3.5', 'Pat1274_Se16_Res1.34375_1.34375_Spac3.5', 'Pat1133_Se24_Res1.25_1.25_Spac3.5', 'Pat438_Se19_Res0.78125_0.78125_Spac4.0','Pat3011_Se24_Res0.7421875_0.7421875_Spac3.5','Pat2957_Se25_Res0.78125_0.78125_Spac4.0', 'Pat1563_Se13_Res1.25_1.25_Spac3.5', 'Pat663_Se20_Res1.25_1.25_Spac3.5','Pat3201_Se22_Res0.78125_0.78125_Spac4.0', 'Pat978_Se16_Res1.25_1.25_Spac3.5','Pat2227_Se19_Res0.7421875_0.7421875_Spac3.5','Pat1374_Se26_Res0.78125_0.78125_Spac4.0', 'Pat911_Se31_Res0.78125_0.78125_Spac4.0'],
	cases_train_F_u = [],
	cases_train_T_u = [],
	cases_val_F = ['228', '225', '234', '211'],
	cases_val_T = [],
	cases_test_T = ['HM', 'ILZ', 'RA', 'ZM', 'HOY', 'SS', 'RR', 'GKY', 'FC', 'AP',  'KZ', 'GN', 'AH',  'DA'],
	cases_test_F = [],

)









config_data_set_2d_exp2 =dict(

	data_set_name = 'DF_6_2d',
	data_set_name_patches_name = 'patch2D',
	cases_train_F_l = [ '91', '208', '201','81', '93', '104'],
	cases_train_T_l = [],
	cases_train_F_u = ['176', '197', '78', '191', '232', '224', '223', '135', '189', '177', '95', '76', '258', '174', '188', '170', '205'],
	cases_train_T_u = [],
	cases_val_F = ['228', '225', '234'],
	cases_val_T = [],
	cases_test_T = [],
	cases_test_F = ['203', '209', '200', '184', '211'],
)