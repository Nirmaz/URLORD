config_unet_nn = dict(

	in_channels = 1,
    out_channels  = 6,
    normalization = 'batch',
    activation = 'relu',
    n_blocks = 4,
    start_filters = 16,
    conv_mode = 'same',
    dim = 2,
    up_mode = 'upsample',
    activation_upsample = 'leaky',
    normalization_upsample='instance',
    last_activation = 'Softmax',
    apply_last_act = True,
    droupout = 0.1

)

config_unet_nn_3d = dict(
    # in channel, anatomy rep and out channel came together
	in_channels = 1,
    anatomy_rep = 6,
    out_channels  =  6,
    normalization = 'batch',
    activation = 'relu',
    n_blocks = 4,
    start_filters = 16,
    conv_mode = 'same',
    dim = 3,
    up_mode = 'upsample',
    activation_upsample = 'leaky',
    normalization_upsample='instance',
    last_activation = 'Softmax',
    apply_last_act = True,
    droupout = 0.1

)
# unet to cpmpare
config_unet_nn_3d_c = dict(
    # in channel, anatomy rep and out channel came together
	in_channels = 1,
    anatomy_rep = 1,
    out_channels  =  1,
    normalization = 'batch',
    activation = 'relu',
    n_blocks = 4,
    start_filters = 16,
    conv_mode = 'same',
    dim = 3,
    up_mode = 'upsample',
    activation_upsample = 'leaky',
    normalization_upsample='instance',
    last_activation = 'sigmoid',
    apply_last_act = True,
    droupout = 0.1

)