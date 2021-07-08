config_unet_nn = dict(

	in_channels = 1,
    out_channels  = 2,
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

)