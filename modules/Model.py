import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, ZeroPadding2D, BatchNormalization, Flatten
from tensorflow.keras.regularizers import l2

# TODO: modify darknet layers to fit JPG NN sizes

# TODO: probably don't need yolov3 sized darknet; scale down number of nodes/layers to save compute/memory resources


def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True):
    if downsample:
        input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strides,
                  padding=padding, use_bias=not bn, kernel_regularizer=l2(0.0005),
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                  bias_initializer=tf.constant_initializer(0.))(input_layer)
    if bn:
        conv = BatchNormalization()(conv)
    if activate == True:
        conv = LeakyReLU(alpha=0.1)(conv)

    return conv


def residual_block(input_layer, input_channel, filter_num1, filter_num2):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1))
    conv = convolutional(conv, filters_shape=(3, 3, filter_num1, filter_num2))

    residual_output = short_cut + conv
    return residual_output


def darknet53(input_data):
    input_data = convolutional(input_data, (3, 3, 3, 32))
    input_data = convolutional(input_data, (3, 3, 32, 64), downsample=True)

    for i in range(1):
        input_data = residual_block(input_data, 64, 32, 64)

    input_data = convolutional(input_data, (3, 3, 64, 128), downsample=True)

    for i in range(2):
        input_data = residual_block(input_data, 128, 64, 128)

    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 256, 128, 256)

    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True)

    for i in range(4):
        input_data = residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data


def jpgEnhanceModel(input_layer):
    route_1, route_2, conv = darknet53(input_layer)

    return Flatten()(route_1 + route_2 + conv)
