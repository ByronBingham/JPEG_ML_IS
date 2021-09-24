import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add
from modules.NNConfig import INPUT_SHAPE, MPRRN_FILTERS_PER_LAYER, MPRRN_FILTER_SHAPE, MPRRN_RRU_PER_IRB, MPRRN_IRBS


def EQLRI_model():
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same"
    }

    inputs = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)

    layer1 = Conv2D(filters=64, kernel_size=(9, 9), **conv_args)(inputs)

    layer2 = Conv2D(filters=64, kernel_size=(1, 1), **conv_args)(layer1)

    layer3 = Conv2D(filters=3, kernel_size=(5, 5), **conv_args)(layer2)

    output = layer3

    return tf.keras.Model(inputs, output)


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

    return tf.keras.Model(inputs, output)


def STRRN():
    structure_input = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)
    texture_input = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)

    structure_mprrn = MPRRN(structure_input, rrusPerIrb=MPRRN_RRU_PER_IRB, irbCount=MPRRN_IRBS)
    texture_mprrn = MPRRN(texture_input, rrusPerIrb=MPRRN_RRU_PER_IRB, irbCount=MPRRN_IRBS)

    aggregator_input = Add()([structure_mprrn, texture_mprrn])

    aggregator = MPRRN(aggregator_input, rrusPerIrb=1, irbCount=1)

    model = tf.keras.Model(inputs=[structure_input, texture_input], outputs=aggregator)

    return model


def MPRRN(inputs, rrusPerIrb, irbCount):
    conv_1 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')(inputs)

    irbs = []
    sums = []
    for i in range(irbCount):
        if i == 0:
            irbs.append(MPRRN_IRB(conv_1, rrusPerIrb))
            sums.append(Add()([irbs[i], conv_1]))
        else:
            irbs.append(MPRRN_IRB(sums[i - 1], rrusPerIrb))
            sums.append(Add()([irbs[i], conv_1]))

    conv_2 = Conv2D(filters=INPUT_SHAPE[-1], kernel_size=MPRRN_FILTER_SHAPE, padding='same')(sums[-1])

    out = Add()([inputs, conv_2])

    return out


def MPRRN_RRU(inputs):
    bn_1 = BatchNormalization()(inputs)
    relu_1 = ReLU()(bn_1)
    conv_1 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')(relu_1)

    bn_2 = BatchNormalization()(conv_1)
    relu_2 = ReLU()(bn_2)
    conv_2 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')(relu_2)

    sum = Add()([conv_2, inputs])

    return sum


def MPRRN_IRB(inputs, rruCount):
    rrus = []

    for i in range(rruCount):
        if i == 0:
            rrus.append(MPRRN_RRU(inputs))
        else:
            rrus.append(MPRRN_RRU(rrus[i - 1]))

    return rrus[-1]


modelSwitch = {
    'eqlri': EQLRI_model(),
    'kerasexample': kerasExample_model(),
    'strrn': STRRN()
}
