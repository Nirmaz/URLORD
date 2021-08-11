
config_data_set =dict(

	data_set_name = 'DRSA13d',
	cases_train_F_l = [ '91', '208', '201', '81', '93', '104', '176', '197', '78', '191', '232', '224', '223', '135', '189', '177', '95', '76', '258', '174', '188', '170', '205', '203', '209', '200', '184'],
	cases_train_T_l = ['ABR'],
	cases_train_F_u = [],
	cases_train_T_u = ['Pat2426_Se22_Res1.25_1.25_Spac3.5','Pat876_Se17_Res1.25_1.25_Spac3.5', 'Pat3547_Se18_Res0.7421875_0.7421875_Spac3.5', 'Pat2771_Se28_Res1.25_1.25_Spac3.5','Pat1029_Se17_Res0.7421875_0.7421875_Spac3.5','Pat1837_Se18_Res0.9375_0.9375_Spac3.5', 'Pat2175_Se23_Res0.5859375_0.5859375_Spac3.0','Pat928_Se21_Res1.25_1.25_Spac3.5', 'Pat1274_Se16_Res1.34375_1.34375_Spac3.5', 'Pat1133_Se24_Res1.25_1.25_Spac3.5', 'Pat438_Se19_Res0.78125_0.78125_Spac4.0','Pat3011_Se24_Res0.7421875_0.7421875_Spac3.5','Pat2957_Se25_Res0.78125_0.78125_Spac4.0', 'Pat1563_Se13_Res1.25_1.25_Spac3.5', 'Pat663_Se20_Res1.25_1.25_Spac3.5','Pat3201_Se22_Res0.78125_0.78125_Spac4.0', 'Pat978_Se16_Res1.25_1.25_Spac3.5','Pat2227_Se19_Res0.7421875_0.7421875_Spac3.5','Pat1374_Se26_Res0.78125_0.78125_Spac4.0', 'Pat911_Se31_Res0.78125_0.78125_Spac4.0'],
	cases_val_F = ['228', '225', '234', '211'],
	cases_val_T = ['GG'],
	cases_test_T = ['HM', 'ILZ', 'RA', 'ZM', 'HOY', 'SS', 'RR', 'GKY', 'FC', 'AP',  'KZ', 'GN', 'AH',  'DA'],
	cases_test_F = [],
	patches_params = dict(
                            model_original_dim = (128, 128, 16),
                            patch_stride = (32, 32, 16),
							with_without_body = True,
							min_num_of_body_voxels = 1,
							min_num_of_plcenta_voxels = 32*32,
							num_slice_cri = 16,
							with_plcenta = True,
							dim = 3,
							from_bb = False,
							from_h_bb = False,
							margin = 100
						)
)


config_data_set_2d =dict(

	data_set_name = 'DRSA12d',
	cases_train_F_l = [ '91', '208', '201', '81', '93', '104', '176', '197', '78', '191', '232', '224', '223', '135', '189', '177', '95', '76', '258', '174', '188', '170', '205', '203', '209', '200', '184'],
	cases_train_T_l = ['ABR'],
	cases_train_F_u = [],
	cases_train_T_u = ['Pat2426_Se22_Res1.25_1.25_Spac3.5','Pat876_Se17_Res1.25_1.25_Spac3.5', 'Pat3547_Se18_Res0.7421875_0.7421875_Spac3.5', 'Pat2771_Se28_Res1.25_1.25_Spac3.5','Pat1029_Se17_Res0.7421875_0.7421875_Spac3.5','Pat1837_Se18_Res0.9375_0.9375_Spac3.5', 'Pat2175_Se23_Res0.5859375_0.5859375_Spac3.0','Pat928_Se21_Res1.25_1.25_Spac3.5', 'Pat1274_Se16_Res1.34375_1.34375_Spac3.5', 'Pat1133_Se24_Res1.25_1.25_Spac3.5', 'Pat438_Se19_Res0.78125_0.78125_Spac4.0','Pat3011_Se24_Res0.7421875_0.7421875_Spac3.5','Pat2957_Se25_Res0.78125_0.78125_Spac4.0', 'Pat1563_Se13_Res1.25_1.25_Spac3.5', 'Pat663_Se20_Res1.25_1.25_Spac3.5','Pat3201_Se22_Res0.78125_0.78125_Spac4.0', 'Pat978_Se16_Res1.25_1.25_Spac3.5','Pat2227_Se19_Res0.7421875_0.7421875_Spac3.5','Pat1374_Se26_Res0.78125_0.78125_Spac4.0', 'Pat911_Se31_Res0.78125_0.78125_Spac4.0'],
	cases_val_F = ['228', '225', '234', '211'],
	cases_val_T = ['GG'],
	cases_test_T = ['HM', 'ILZ', 'RA', 'ZM', 'HOY', 'SS', 'RR', 'GKY', 'FC', 'AP',  'KZ', 'GN', 'AH',  'DA'],
	cases_test_F = [],
	patches_params = dict(
                            model_original_dim = (128, 128, 16),
                            patch_stride = (32, 32, 8),
							with_without_body = True,
							min_num_of_body_voxels = 1,
							min_num_of_plcenta_voxels = 42*32,
							num_slice_cri = 16,
							with_plcenta = True,
							dim = 3,
							from_bb = False,
							from_h_bb = False,
							margin = 100
						)
)

all_f = ['228', '225', '234', '211', '91', '208', '201', '81', '93', '104', '176', '197', '78', '191', '232', '224', '223', '135', '189', '177', '95', '76', '258', '174', '188', '170', '205', '203', '209', '200', '184'],
all_t= [ 'ABR', 'GG','Pat2426_Se22_Res1.25_1.25_Spac3.5', 'Pat876_Se17_Res1.25_1.25_Spac3.5', 'HM', 'ILZ', 'Pat3547_Se18_Res0.7421875_0.7421875_Spac3.5', 'RA', 'Pat2771_Se28_Res1.25_1.25_Spac3.5', 'Pat1029_Se17_Res0.7421875_0.7421875_Spac3.5', 'ZM', 'HOY', 'Pat1837_Se18_Res0.9375_0.9375_Spac3.5', 'Pat2175_Se23_Res0.5859375_0.5859375_Spac3.0', 'SS', 'Pat928_Se21_Res1.25_1.25_Spac3.5', 'Pat1274_Se16_Res1.34375_1.34375_Spac3.5', 'Pat1133_Se24_Res1.25_1.25_Spac3.5', 'Pat438_Se19_Res0.78125_0.78125_Spac4.0', 'RR', 'GKY', 'Pat3011_Se24_Res0.7421875_0.7421875_Spac3.5', 'FC', 'Pat2957_Se25_Res0.78125_0.78125_Spac4.0', 'AP', 'Pat1563_Se13_Res1.25_1.25_Spac3.5', 'Pat663_Se20_Res1.25_1.25_Spac3.5', 'KZ', 'GN', 'Pat3201_Se22_Res0.78125_0.78125_Spac4.0', 'Pat978_Se16_Res1.25_1.25_Spac3.5', 'Pat2227_Se19_Res0.7421875_0.7421875_Spac3.5', 'AH', 'Pat1374_Se26_Res0.78125_0.78125_Spac4.0', 'Pat911_Se31_Res0.78125_0.78125_Spac4.0', 'DA']
