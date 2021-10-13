import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Activation, Conv2DTranspose, Layer, \
    Dropout
from modules.NNConfig import INPUT_SHAPE, MPRRN_FILTERS_PER_LAYER, MPRRN_FILTER_SHAPE, MPRRN_RRU_PER_IRB, MPRRN_IRBS, \
    DROPOUT_RATE, STRUCTURE_FILTERS_PER_LAYER, TEXTURE_FILTERS_PER_LAYER


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


def MPRRN(inputs, rrusPerIrb, irbCount, filtersPerLayer=MPRRN_FILTERS_PER_LAYER):
    conv_1 = Conv2D(filters=filtersPerLayer, kernel_size=MPRRN_FILTER_SHAPE, padding='same')(inputs)

    conv_2 = Conv2D(filters=filtersPerLayer, kernel_size=MPRRN_FILTER_SHAPE, padding='same')
    conv_3 = Conv2D(filters=filtersPerLayer, kernel_size=MPRRN_FILTER_SHAPE, padding='same')

    irbs = []
    sums = []
    for i in range(irbCount):
        if i == 0:
            irbs.append(MPRRN_IRB(conv_1, rrusPerIrb, conv_2, conv_3))
            sums.append(Add()([irbs[i], conv_1]))
        else:
            irbs.append(MPRRN_IRB(sums[i - 1], rrusPerIrb, conv_2, conv_3))
            sums.append(Add()([irbs[i], conv_1]))

    conv_4 = Conv2D(filters=INPUT_SHAPE[-1], kernel_size=MPRRN_FILTER_SHAPE, padding='same')(sums[-1])

    out = Add()([inputs, conv_4])

    return out


def MPRRN_RRU(inputs, conv_in_1, conv_in_2):
    bn_1 = BatchNormalization()(inputs)
    relu_1 = ReLU()(bn_1)
    conv_1 = conv_in_1(relu_1)

    bn_2 = BatchNormalization()(conv_1)
    relu_2 = ReLU()(bn_2)
    conv_2 = conv_in_2(relu_2)

    sum = Add()([conv_2, inputs])

    return sum


def MPRRN_IRB(inputs, rruCount, conv_in_1, conv_in_2):
    rrus = []

    for i in range(rruCount):
        if i == 0:
            rrus.append(MPRRN_RRU(inputs, conv_in_1, conv_in_2))
        else:
            rrus.append(MPRRN_RRU(rrus[i - 1], conv_in_1, conv_in_2))

    return rrus[-1]


#####################################
# Custom Models
#####################################
def MPRRN_RRU_4layer(inputs, conv_in_1, conv_in_2, deconv_in_1, deconv_in_2):
    bn_1 = BatchNormalization()(inputs)
    relu_1 = ReLU()(bn_1)
    conv_1 = conv_in_1(relu_1)

    bn_2 = BatchNormalization()(conv_1)
    relu_2 = ReLU()(bn_2)
    conv_2 = conv_in_2(relu_2)

    bn_3 = BatchNormalization()(conv_2)
    relu_3 = ReLU()(bn_3)
    deconv_1 = deconv_in_1(relu_3)

    bn_4 = BatchNormalization()(deconv_1)
    relu_4 = ReLU()(bn_4)
    deconv_2 = deconv_in_2(relu_4)

    sum = Add()([deconv_2, inputs])

    return sum


def MPRRN_IRB_4layer(inputs, rruCount, conv_in_1, conv_in_2, deconv_in_1, deconv_in_2):
    rrus = []

    for i in range(rruCount):
        if i == 0:
            rrus.append(MPRRN_RRU_4layer(inputs, conv_in_1, conv_in_2, deconv_in_1, deconv_in_2))
        else:
            rrus.append(MPRRN_RRU_4layer(rrus[i - 1], conv_in_1, conv_in_2, deconv_in_1, deconv_in_2))

    return rrus[-1]


def MPRRN_encodeDecode(inputs, rrusPerIrb, irbCount):
    conv_1 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')(inputs)

    conv_2 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same', strides=2)
    conv_3 = Conv2DTranspose(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same', strides=2)

    irbs = []
    sums = []
    for i in range(irbCount):
        if i == 0:
            irbs.append(MPRRN_IRB(conv_1, rrusPerIrb, conv_2, conv_3))
            sums.append(Add()([irbs[i], conv_1]))
        else:
            irbs.append(MPRRN_IRB(sums[i - 1], rrusPerIrb, conv_2, conv_3))
            sums.append(Add()([irbs[i], conv_1]))

    conv_4 = Conv2D(filters=INPUT_SHAPE[-1], kernel_size=MPRRN_FILTER_SHAPE, padding='same')(sums[-1])

    out = Add()([inputs, conv_4])

    return out


def MPRRN_encodeDecode_4layer(inputs, rrusPerIrb, irbCount):
    conv_1 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')(inputs)

    conv_2 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same', strides=2)
    conv_3 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same', strides=2)
    deconv_1 = Conv2DTranspose(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same',
                               strides=2)
    deconv_2 = Conv2DTranspose(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same',
                               strides=2)

    irbs = []
    sums = []
    for i in range(irbCount):
        if i == 0:
            irbs.append(MPRRN_IRB_4layer(conv_1, rrusPerIrb, conv_2, conv_3, deconv_1, deconv_2))
            sums.append(Add()([irbs[i], conv_1]))
        else:
            irbs.append(MPRRN_IRB_4layer(sums[i - 1], rrusPerIrb, conv_2, conv_3, deconv_1, deconv_2))
            sums.append(Add()([irbs[i], conv_1]))

    conv_4 = Conv2D(filters=INPUT_SHAPE[-1], kernel_size=MPRRN_FILTER_SHAPE, padding='same')(sums[-1])

    out = Add()([inputs, conv_4])

    return out


def STRRN_encodeDecode():
    structure_input = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)
    texture_input = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)

    structure_mprrn = MPRRN_encodeDecode(structure_input, rrusPerIrb=MPRRN_RRU_PER_IRB, irbCount=MPRRN_IRBS)
    texture_mprrn = MPRRN_encodeDecode(texture_input, rrusPerIrb=MPRRN_RRU_PER_IRB, irbCount=MPRRN_IRBS)

    aggregator_input = Add()([structure_mprrn, texture_mprrn])

    aggregator = MPRRN_encodeDecode(aggregator_input, rrusPerIrb=1, irbCount=1)

    model = tf.keras.Model(inputs=[structure_input, texture_input], outputs=aggregator)

    return model


def MPRRN_only():
    inputs = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)

    mprrn = MPRRN(inputs, rrusPerIrb=MPRRN_RRU_PER_IRB, irbCount=MPRRN_IRBS)

    model = tf.keras.Model(inputs=inputs, outputs=mprrn)

    return model


def MPRRN_structure():
    inputs = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)

    mprrn = MPRRN(inputs, rrusPerIrb=MPRRN_RRU_PER_IRB, irbCount=MPRRN_IRBS,
                  filtersPerLayer=STRUCTURE_FILTERS_PER_LAYER)

    model = tf.keras.Model(inputs=inputs, outputs=mprrn)

    return model


def MPRRN_texture():
    inputs = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)

    mprrn = MPRRN(inputs, rrusPerIrb=MPRRN_RRU_PER_IRB, irbCount=MPRRN_IRBS, filtersPerLayer=TEXTURE_FILTERS_PER_LAYER)

    model = tf.keras.Model(inputs=inputs, outputs=mprrn)

    return model


def MPRRN_only_encodeDecode():
    inputs = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)

    mprrn = MPRRN_encodeDecode(inputs, rrusPerIrb=MPRRN_RRU_PER_IRB, irbCount=MPRRN_IRBS)

    model = tf.keras.Model(inputs=inputs, outputs=mprrn)

    return model


def MPRRN_only_encodeDecode_4layer():
    inputs = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)

    mprrn = MPRRN_encodeDecode_4layer(inputs, rrusPerIrb=MPRRN_RRU_PER_IRB, irbCount=MPRRN_IRBS)

    model = tf.keras.Model(inputs=inputs, outputs=mprrn)

    return model


def MPRRN_no_IRB_residual(inputs, rrusPerIrb, irbCount):
    conv_1 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')(inputs)

    conv_2 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')
    conv_3 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')

    irbs = []
    for i in range(irbCount):
        if i == 0:
            irbs.append(MPRRN_IRB(conv_1, rrusPerIrb, conv_2, conv_3))
        else:
            irbs.append(MPRRN_IRB(irbs[i - 1], rrusPerIrb, conv_2, conv_3))

    conv_4 = Conv2D(filters=INPUT_SHAPE[-1], kernel_size=MPRRN_FILTER_SHAPE, padding='same')(irbs[-1])

    return conv_4


def STRRN_no_IRB_residual():
    structure_input = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)
    texture_input = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)

    structure_mprrn = MPRRN_no_IRB_residual(structure_input, rrusPerIrb=MPRRN_RRU_PER_IRB, irbCount=MPRRN_IRBS)
    texture_mprrn = MPRRN_no_IRB_residual(texture_input, rrusPerIrb=MPRRN_RRU_PER_IRB, irbCount=MPRRN_IRBS)

    aggregator_input = Add()([structure_mprrn, texture_mprrn])

    aggregator = MPRRN_no_IRB_residual(aggregator_input, rrusPerIrb=1, irbCount=1)

    model = tf.keras.Model(inputs=[structure_input, texture_input], outputs=aggregator)

    return model


def MPRRN_no_IRB_residual_encodeDecode(inputs, rrusPerIrb, irbCount):
    conv_1 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')(inputs)

    conv_2 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same', strides=2)
    conv_3 = Conv2DTranspose(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same', strides=2)

    irbs = []
    for i in range(irbCount):
        if i == 0:
            irbs.append(MPRRN_IRB(conv_1, rrusPerIrb, conv_2, conv_3))
        else:
            irbs.append(MPRRN_IRB(irbs[i - 1], rrusPerIrb, conv_2, conv_3))

    conv_4 = Conv2D(filters=INPUT_SHAPE[-1], kernel_size=MPRRN_FILTER_SHAPE, padding='same')(irbs[-1])

    return conv_4


def STRRN_no_IRB_residual_encodeDecode():
    structure_input = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)
    texture_input = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)

    structure_mprrn = MPRRN_no_IRB_residual_encodeDecode(structure_input, rrusPerIrb=MPRRN_RRU_PER_IRB,
                                                         irbCount=MPRRN_IRBS)
    texture_mprrn = MPRRN_no_IRB_residual_encodeDecode(texture_input, rrusPerIrb=MPRRN_RRU_PER_IRB, irbCount=MPRRN_IRBS)

    aggregator_input = Add()([structure_mprrn, texture_mprrn])

    aggregator = MPRRN_no_IRB_residual(aggregator_input, rrusPerIrb=1, irbCount=1)

    model = tf.keras.Model(inputs=[structure_input, texture_input], outputs=aggregator)

    return model


def hourglass_6():
    inputs = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)

    conv1 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same', strides=2,
                   activation='relu')(inputs)
    dropout1 = Dropout(rate=DROPOUT_RATE)(conv1)
    batchNorm1 = BatchNormalization()(dropout1)
    conv2 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same', strides=2,
                   activation='relu')(batchNorm1)
    dropout2 = Dropout(rate=DROPOUT_RATE)(conv2)
    batchNorm2 = BatchNormalization()(dropout2)
    conv3 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same', strides=2,
                   activation='relu')(batchNorm2)
    dropout3 = Dropout(rate=DROPOUT_RATE)(conv3)
    batchNorm3 = BatchNormalization()(dropout3)

    deconv1 = Conv2DTranspose(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same',
                              strides=2, activation='relu')(batchNorm3)
    dropout4 = BatchNormalization()(deconv1)
    batchNorm4 = BatchNormalization()(dropout4)
    sum1 = Add()([conv2, batchNorm4])

    deconv2 = Conv2DTranspose(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same',
                              strides=2, activation='relu')(sum1)
    dropout5 = Dropout(rate=DROPOUT_RATE)(deconv2)
    batchNorm5 = BatchNormalization()(dropout5)
    sum2 = Add()([conv1, batchNorm5])

    deconv3 = Conv2DTranspose(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same',
                              strides=2, activation='relu')(sum2)
    dropout6 = Dropout(rate=DROPOUT_RATE)(deconv3)
    batchNorm6 = BatchNormalization()(dropout6)

    conv4 = Conv2D(filters=1, kernel_size=MPRRN_FILTER_SHAPE, padding='same', strides=1, activation='relu')(
        batchNorm6)
    batchNorm7 = BatchNormalization()(conv4)

    model = tf.keras.Model(inputs=inputs, outputs=batchNorm7)

    return model


modelSwitch = {
    'eqlri': EQLRI_model,
    'kerasexample': kerasExample_model,
    'strrn': STRRN,
    'strrn_encodedecode': STRRN_encodeDecode,
    'mprrn_only': MPRRN_only,
    'mprrn_structure': MPRRN_structure,
    'mprrn_texture': MPRRN_texture,
    'mprrn_encodedecode': MPRRN_only_encodeDecode,
    'mprrn_encodedecode_4layer': MPRRN_only_encodeDecode_4layer,
    'strrn_no_irb_residual': STRRN_no_IRB_residual,
    'strrn_no_irb_residual_encodedecode': STRRN_no_IRB_residual_encodeDecode,
    'hourglass_6': hourglass_6
}
