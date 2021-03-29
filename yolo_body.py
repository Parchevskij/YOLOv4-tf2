from act_func import Mish
from tools import compose
from functools import wraps
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, Add, Concatenate, \
    MaxPooling2D, UpSampling2D
from tensorflow.keras.initializers import RandomNormal


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_kw = dict()
    darknet_kw['kernel_initializer'] = RandomNormal(mean=0.0, stddev=0.01)
    darknet_kw['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_kw.update(kwargs)
    return Conv2D(*args, **darknet_kw)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def DarknetConv2D_BN_Mish(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())


def resblock_body(x, num_filters, num_blocks, all_narrow=True):
    zeropad = ZeroPadding2D(((1,0),(1,0)))(x)
    zeropad = DarknetConv2D_BN_Mish(num_filters, (3,3), strides=(2, 2))(zeropad)
    half_filters = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1, 1))(zeropad)
    conv_mish = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1, 1))(zeropad)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Mish(num_filters//2, (1,1)),
                DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3, 3)))(conv_mish)
        conv_mish = Add()([conv_mish, y])
    x = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1, 1))(conv_mish)
    route = Concatenate()([x, half_filters])
    return DarknetConv2D_BN_Mish(num_filters, (1, 1))(route)


def darknet_body(x):
    x = DarknetConv2D_BN_Mish(32, (3,3))(x)
    x = resblock_body(x, 64, 1, False)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def yolo4_body(inputs, num_anchors, num_classes):
    backbone = Model(inputs, darknet_body(inputs))

    y19 = DarknetConv2D_BN_Leaky(512, (1, 1))(backbone.output)
    y19 = DarknetConv2D_BN_Leaky(1024, (3, 3))(y19)
    y19 = DarknetConv2D_BN_Leaky(512, (1, 1))(y19)
    maxpool1 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(y19)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(y19)
    maxpool3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(y19)
    y19 = Concatenate()([maxpool1, maxpool2, maxpool3, y19])
    y19 = DarknetConv2D_BN_Leaky(512, (1, 1))(y19)
    y19 = DarknetConv2D_BN_Leaky(1024, (3, 3))(y19)
    y19 = DarknetConv2D_BN_Leaky(512, (1, 1))(y19)

    y19_upsample = compose(DarknetConv2D_BN_Leaky(256, (1, 1)), UpSampling2D(2))(y19)

    y38 = DarknetConv2D_BN_Leaky(256, (1, 1))(backbone.layers[204].output)
    y38 = Concatenate()([y38, y19_upsample])
    y38 = DarknetConv2D_BN_Leaky(256, (1, 1))(y38)
    y38 = DarknetConv2D_BN_Leaky(512, (3, 3))(y38)
    y38 = DarknetConv2D_BN_Leaky(256, (1, 1))(y38)
    y38 = DarknetConv2D_BN_Leaky(512, (3, 3))(y38)
    y38 = DarknetConv2D_BN_Leaky(256, (1, 1))(y38)

    y38_upsample = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(y38)

    y76 = DarknetConv2D_BN_Leaky(128, (1, 1))(backbone.layers[131].output)
    y76 = Concatenate()([y76, y38_upsample])
    y76 = DarknetConv2D_BN_Leaky(128, (1, 1))(y76)
    y76 = DarknetConv2D_BN_Leaky(256, (3, 3))(y76)
    y76 = DarknetConv2D_BN_Leaky(128, (1, 1))(y76)
    y76 = DarknetConv2D_BN_Leaky(256, (3, 3))(y76)
    y76 = DarknetConv2D_BN_Leaky(128, (1, 1))(y76)

    y76_output = DarknetConv2D_BN_Leaky(256, (3, 3))(y76)
    y76_output = DarknetConv2D(num_anchors*(num_classes+5), (1, 1))(y76_output)

    y76_downsample = ZeroPadding2D(((1, 0), (1, 0)))(y76)
    y76_downsample = DarknetConv2D_BN_Leaky(256, (3, 3), strides=(2, 2))(y76_downsample)
    y38 = Concatenate()([y76_downsample, y38])
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)
    y38 = DarknetConv2D_BN_Leaky(512, (3,3))(y38)
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)
    y38 = DarknetConv2D_BN_Leaky(512, (3,3))(y38)
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)

    y38_output = DarknetConv2D_BN_Leaky(512, (3, 3))(y38)
    y38_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(y38_output)

    y38_downsample = ZeroPadding2D(((1, 0), (1, 0)))(y38)
    y38_downsample = DarknetConv2D_BN_Leaky(512, (3,3), strides=(2,2))(y38_downsample)
    y19 = Concatenate()([y38_downsample, y19])
    y19 = DarknetConv2D_BN_Leaky(512, (1, 1))(y19)
    y19 = DarknetConv2D_BN_Leaky(1024, (3, 3))(y19)
    y19 = DarknetConv2D_BN_Leaky(512, (1, 1))(y19)
    y19 = DarknetConv2D_BN_Leaky(1024, (3, 3))(y19)
    y19 = DarknetConv2D_BN_Leaky(512, (1, 1))(y19)

    y19_output = DarknetConv2D_BN_Leaky(1024, (3, 3))(y19)
    y19_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(y19_output)

    yolo4_model = Model(inputs, [y19_output, y38_output, y76_output])

    return yolo4_model

