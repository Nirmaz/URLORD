from torch import nn
import torch



@torch.jit.script
def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
    """
    Center-crops the encoder_layer to the size of the decoder_layer,
    so that merging (concatenation) between levels/blocks is possible.
    This is only necessary for input sizes != 2**n for 'same' padding and always required for 'valid' padding.
    """
    if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
        print("bad bad bad autocrop")
        ds = encoder_layer.shape[2:]
        es = decoder_layer.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        if encoder_layer.dim() == 4:  # 2D
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)
                            ]
        elif encoder_layer.dim() == 5:  # 3D
            assert ds[2] >= es[2]
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2),
                            ((ds[2] - es[2]) // 2):((ds[2] + es[2]) // 2),
                            ]
    return encoder_layer, decoder_layer


def conv_layer(dim: int):
    if dim == 3:
        return nn.Conv3d
    elif dim == 2:
        return nn.Conv2d


def get_conv_layer(in_channels: int,
                   out_channels: int,
                   kernel_size: int = 3,
                   stride: int = 1,
                   padding: int = 1,
                   bias: bool = True,
                   dim: int = 2):
    return conv_layer(dim)(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=bias)


def conv_transpose_layer(dim: int):
    if dim == 3:
        return nn.ConvTranspose3d
    elif dim == 2:
        return nn.ConvTranspose2d


def get_up_layer(in_channels: int,
                 out_channels: int,
                 kernel_size: int = 2,
                 stride: int = 2,
                 dim: int = 3,
                 up_mode: str = 'transposed',
                 ):
    if up_mode == 'transposed':
        return conv_transpose_layer(dim)(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    else:
        return nn.Upsample(scale_factor=2.0)


def maxpool_layer(dim: int):
    if dim == 3:
        return nn.MaxPool3d
    elif dim == 2:
        return nn.MaxPool2d


def get_maxpool_layer(kernel_size: int = 2,
                      stride: int = 2,
                      padding: int = 0,
                      dim: int = 2):
    return maxpool_layer(dim=dim)(kernel_size=kernel_size, stride=stride, padding=padding)


def get_activation(activation: str):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'Softmax':
        return nn.Softmax(dim = 1)


def get_normalization(normalization: str,
                      num_channels: int,
                      dim: int):
    if normalization == 'batch':
        if dim == 3:
            return nn.BatchNorm3d(num_channels)
        elif dim == 2:
            return nn.BatchNorm2d(num_channels)
    elif normalization == 'instance':
        if dim == 3:
            return nn.InstanceNorm3d(num_channels)
        elif dim == 2:
            return nn.InstanceNorm2d(num_channels)
    elif 'group' in normalization:
        num_groups = int(normalization.partition('group')[-1])  # get the group size from string
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, layer_1, layer_2):
        x = torch.cat((layer_1, layer_2), 1)

        return x


class DownBlock(nn.Module):
    """
    A helper Module that performs 2 Convolutions and 1 MaxPool.
    An activation follows each convolution.
    A normalization layer follows each convolution.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pooling: bool = True,
                 activation: str = 'relu',
                 normalization: str = None,
                 dim: str = 2,
                 conv_mode: str = 'same'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.normalization = normalization
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.dim = dim
        self.activation = activation

        # conv layers
        self.conv1 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True, dim = self.dim)
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True, dim=self.dim)

        # pooling layer
        if self.pooling:
            self.pool = get_maxpool_layer(kernel_size = 2, stride = 2, padding = 0, dim=self.dim)

        # activation layers
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        # normalization layers
        if self.normalization:
            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim= self.dim)

    def forward(self, x):
        y = self.conv1(x)  # convolution 1
        if self.normalization:
            y = self.norm1(y)
        y = self.act1(y)  # activation 1
          # normalization 1
        y = self.conv2(y)
        if self.normalization:
            y = self.norm2(y)
        # convolution 2
        y = self.act2(y)  # activation 2
       # normalization 2

        before_pooling = y  # save the outputs before the pooling operation
        if self.pooling:
            y = self.pool(y)  # pooling
        return y, before_pooling


class UpBlock(nn.Module):
    """
    A helper Module that performs 2 Convolutions and 1 UpConvolution/Upsample.
    An activation follows each convolution.
    A normalization layer follows each convolution.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: str = 'relu',
                 normalization: str = None,
                 dim: int = 3,
                 conv_mode: str = 'same',
                 up_mode: str = 'transposed',
                 activation_upsample = 'leaky',
                 normalization_upsample = 'instance'
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.dim = dim
        self.activation = activation
        self.up_mode = up_mode
        self.activation_upsample = activation_upsample
        self.normalization_upsample = normalization_upsample

        # upconvolution/upsample layer
        self.up = get_up_layer(self.in_channels, self.out_channels, kernel_size=2, stride=2, dim=self.dim,
                               up_mode=self.up_mode)

        # conv layers
        self.conv_upsample = get_conv_layer(self.in_channels, self.out_channels, kernel_size=3, stride = 1, padding = 0,
                                    bias = True, dim = self.dim)
        self.conv0 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0,
                                    bias=True, dim=self.dim)
        self.conv1 = get_conv_layer(2 * self.out_channels, self.out_channels, kernel_size=3, stride=1,
                                    padding=self.padding,
                                    bias=True, dim=self.dim)
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True, dim=self.dim)

        # activation layers
        self.act_upsample = get_activation(self.activation_upsample)

        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        # normalization layers
        if self.normalization:
            self.norm_upsample = get_normalization(normalization = self.normalization_upsample, num_channels=self.out_channels,
                                           dim=self.dim)

            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)

        # concatenate layer
        self.concat = Concatenate()

    def forward(self, encoder_layer, decoder_layer):
        """ Forward pass
        Arguments:
            encoder_layer: Tensor from the encoder pathway
            decoder_layer: Tensor from the decoder pathway (to be up'd)
        """
        # print(decoder_layer.size(), "decoder_layer")
        # print(encoder_layer.size(), "encoder_layer")
        up_layer = self.up(decoder_layer)  # up-convolution/up-sampling
        # print(up_layer.size(), "up_layer")
        cropped_encoder_layer, dec_layer = autocrop(encoder_layer, up_layer)  # cropping

        # if self.up_mode != 'transposed':
        #     # We need to reduce the channel dimension with a conv layer
        up_layer = self.conv0(up_layer)  # convolution 0
        if self.normalization:
            up_layer = self.norm_upsample(up_layer)
        up_layer = self.act_upsample(up_layer)  # activation 0
         # normalization 0

        merged_layer = self.concat(up_layer, cropped_encoder_layer)  # concatenation
        # convolution 1
        y = self.conv1(merged_layer)
        if self.normalization:
            y = self.norm1(y)
        y = self.act1(y)  # activation 1
        # normalization 1
        y = self.conv2(y)  # convolution 2
        if self.normalization:
            y = self.norm2(y)
        y = self.act2(y)  # acivation 2
          # normalization 2
        return y


class UNet(nn.Module):
    def __init__(self,config_unet):
        super().__init__()
        # print(f'in channel:', config_unet)
        self.in_channels = config_unet['in_channels']
        self.out_channels = config_unet['out_channels']
        self.n_blocks = config_unet['n_blocks']
        self.start_filters = config_unet['start_filters']
        self.activation = config_unet['activation']
        self.normalization = config_unet['normalization']
        self.conv_mode = config_unet['conv_mode']
        self.dim = config_unet['dim']
        self.up_mode = config_unet['up_mode']
        self.down_blocks = []
        self.up_blocks = []
        self.apply_last_act = config_unet['apply_last_act']

        # create encoder path
        for i in range(self.n_blocks):
            num_filters_in = self.in_channels if i == 0 else num_filters_out
            num_filters_out = self.start_filters * (2 ** i)
            pooling = True if i < self.n_blocks - 1 else False

            down_block = DownBlock(in_channels = num_filters_in,
                                   out_channels = num_filters_out,
                                   pooling=pooling,
                                   activation=self.activation,
                                   normalization=self.normalization,
                                   conv_mode=self.conv_mode,
                                   dim = self.dim)

            self.down_blocks.append(down_block)

        # create decoder path (requires only n_blocks-1 blocks)
        for i in range(self.n_blocks - 1):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            up_block = UpBlock(in_channels = num_filters_in,
                               out_channels = num_filters_out,
                               activation = self.activation,
                               normalization = self.normalization,
                               conv_mode = self.conv_mode,
                               dim = self.dim,
                               up_mode = self.up_mode,
                               activation_upsample = config_unet['activation_upsample'],
                               normalization_upsample = config_unet['normalization_upsample'])

            self.up_blocks.append(up_block)

        # final convolution
        self.conv_final = get_conv_layer(num_filters_out, self.out_channels, kernel_size = 1, stride=1, padding=0,
                                         bias=True, dim=self.dim)

        # add the list of modules to current module
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)
        self.last_activation = get_activation(config_unet['last_activation'])

        # initialize the weights
        self.initialize_parameters()

    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.weight, **kwargs)  # weights

    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.bias, **kwargs)  # bias

    def initialize_parameters(self,
                              method_weights=nn.init.xavier_uniform_,
                              method_bias=nn.init.zeros_,
                              kwargs_weights={},
                              kwargs_bias={}
                              ):
        for module in self.modules():
            self.weight_init(module, method_weights, **kwargs_weights)  # initialize weights
            self.bias_init(module, method_bias, **kwargs_bias)  # initialize bias

    def forward(self, x: torch.tensor):
        encoder_output = []

        # Encoder pathway
        for module in self.down_blocks:
            x, before_pooling = module(x)
            encoder_output.append(before_pooling)

        # Decoder pathway
        for i, module in enumerate(self.up_blocks):
            before_pool = encoder_output[-(i + 2)]
            # print(before_pool.size(), f"bfore pool{i}")
            # print(x.size(), "x size pytorch")
            x = module(before_pool, x)

        x = self.conv_final(x)
        if self.apply_last_act:
            x = self.last_activation(x)
            # print(x[0,:,0,0], "after_softmax")


        return x

    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys() if '_' not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'


class Segmentntor(nn.Module):

        def __init__(self, config_unet):
            super().__init__()

            self.conv1 = get_conv_layer(config_unet['out_channels'], config_unet['start_filters'] , kernel_size = 3, stride = 1, padding = 1, bias=True, dim = config_unet['dim'])
            self.conv2 = get_conv_layer(config_unet['start_filters'], config_unet['start_filters'], kernel_size=3, stride=1, padding = 1,bias = True, dim = config_unet['dim'])
            self.conv3 = get_conv_layer(config_unet['start_filters'], 1, kernel_size = 1,
                                        stride=1, padding=0, bias=True, dim=config_unet['dim'])

            self.act1 = nn.LeakyReLU(negative_slope=0.1)
            self.act2 = nn.LeakyReLU(negative_slope=0.1)
            self.act3 = nn.Sigmoid()
            self.normalize1 = nn.BatchNorm2d(config_unet['start_filters'])
            self.normalize2 = nn.BatchNorm2d(config_unet['start_filters'])

        def forward(self, x):
            # print(x.size(), "start")
            x = self.conv1(x)
            x = self.normalize1(x)
            x = self.act1(x)
            x = self.conv2(x)
            x = self.normalize2(x)
            x = self.act2(x)
            x = self.conv3(x)
            x = self.act3(x)
            # print(x.size(), "end")
            return x

