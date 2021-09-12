import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D
from modules.NNConfig import INPUT_SHAPE


def EQLRI_model():
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same"
    }

    inputs = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)

    layer1 = Conv2D(filters=128, kernel_size=(9, 9), **conv_args)(inputs)

    layer2 = Conv2D(filters=64, kernel_size=(1, 1), **conv_args)(layer1)

    layer3 = Conv2D(filters=3, kernel_size=(5, 5), **conv_args)(layer2)

    output = layer3

    return inputs, output


def kerasExample_model():
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same"
    }

    inputs = Input(shape=(None, None, 3), dtype=tf.dtypes.float32)

    layer1 = Conv2D(filters=64, kernel_size=5, **conv_args)(inputs)

    layer2 = Conv2D(filters=64, kernel_size=5, **conv_args)(layer1)

    layer3 = Conv2D(filters=32, kernel_size=3, **conv_args)(layer2)

    output = Conv2D(filters=3, kernel_size=3, **conv_args)(layer3)

    return inputs, output
