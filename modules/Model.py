import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Activation, Conv2DTranspose, Layer, \
    Dropout, Concatenate
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

    layer3 = Conv2D(filters=INPUT_SHAPE[2], kernel_size=(5, 5), **conv_args)(layer2)

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


def MPRRN_RRU_w1x1(inputs, conv_in_1, conv_in_2, conv_in_3):
    bn_1 = BatchNormalization()(inputs)
    relu_1 = ReLU()(bn_1)
    conv_1 = conv_in_1(relu_1)

    bn_2 = BatchNormalization()(conv_1)
    relu_2 = ReLU()(bn_2)
    conv_2 = conv_in_2(relu_2)

    bn_3 = BatchNormalization()(conv_2)
    relu_3 = ReLU()(bn_3)
    conv_3 = conv_in_3(relu_3)

    sum = Add()([conv_3, inputs])

    return sum


def MPRRN_IRB_w1x1(inputs, rruCount, conv_in_1, conv_in_2, conv_3):
    rrus = []

    for i in range(rruCount):
        if i == 0:
            rrus.append(MPRRN_RRU_w1x1(inputs, conv_in_1, conv_in_2, conv_3))
        else:
            rrus.append(MPRRN_RRU_w1x1(rrus[i - 1], conv_in_1, conv_in_2, conv_3))

    return rrus[-1]


def MPRRN_w1x1(inputs, rrusPerIrb, irbCount, filtersPerLayer=MPRRN_FILTERS_PER_LAYER):
    conv_1 = Conv2D(filters=filtersPerLayer, kernel_size=MPRRN_FILTER_SHAPE, padding='same')(inputs)

    conv_2 = Conv2D(filters=filtersPerLayer, kernel_size=MPRRN_FILTER_SHAPE, padding='same')
    conv_3 = Conv2D(filters=filtersPerLayer, kernel_size=1, padding='same')
    conv_4 = Conv2D(filters=filtersPerLayer, kernel_size=MPRRN_FILTER_SHAPE, padding='same')

    irbs = []
    sums = []
    for i in range(irbCount):
        if i == 0:
            irbs.append(MPRRN_IRB_w1x1(conv_1, rrusPerIrb, conv_2, conv_3, conv_4))
            sums.append(Add()([irbs[i], conv_1]))
        else:
            irbs.append(MPRRN_IRB_w1x1(sums[i - 1], rrusPerIrb, conv_2, conv_3, conv_4))
            sums.append(Add()([irbs[i], conv_1]))

    conv_5 = Conv2D(filters=INPUT_SHAPE[-1], kernel_size=MPRRN_FILTER_SHAPE, padding='same')(sums[-1])

    out = Add()([inputs, conv_5])

    return out


def MPRRN_only_w1x1():
    inputs = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)

    mprrn = MPRRN_w1x1(inputs, rrusPerIrb=MPRRN_RRU_PER_IRB, irbCount=MPRRN_IRBS)

    conv1 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=1, padding='same')(mprrn)
    out = Conv2D(filters=1, kernel_size=1, padding='same')(conv1)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    return model


def DualChannelLayer_3(struct_in, text_in, struct_conv_1, struct_conv_2, text_conv_1, text_conv_2, struct_inter,
                       text_inter, one_by_conv, struct_net_in, text_net_in):
    struct_rru = MPRRN_RRU(struct_in, struct_conv_1, struct_conv_2)
    text_rru = MPRRN_RRU(text_in, text_conv_1, text_conv_2)

    cat = Concatenate(axis=-1)([struct_rru, text_rru])
    comb_conv = one_by_conv(cat)

    struct_inter_1 = struct_inter(cat)
    text_inter_1 = text_inter(cat)

    struct_out = Add()([struct_inter_1, comb_conv, struct_net_in])
    text_out = Add()([text_inter_1, comb_conv, text_net_in])

    return struct_out, text_out


def DualChannelInterconnect():
    structure_input = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)
    texture_input = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)

    struct_in = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')(
        structure_input)
    text_in = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')(
        texture_input)

    struct_conv_1 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')
    struct_conv_2 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')

    text_conv_1 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')
    text_conv_2 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')

    struct_inter = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')
    text_inter = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')

    one_by_conv = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=1, padding='same')

    layers = []
    for i in range(2):
        if i == 0:
            layers.append(DualChannelLayer_3(struct_in, text_in, struct_conv_1, struct_conv_2, text_conv_1, text_conv_2,
                                             struct_inter,
                                             text_inter, one_by_conv, struct_in, text_in))
        else:
            layers.append(
                DualChannelLayer_3(layers[i - 1][0], layers[i - 1][1], struct_conv_1, struct_conv_2,
                                   text_conv_1, text_conv_2, struct_inter,
                                   text_inter, one_by_conv, struct_in, text_in))

    cat = Concatenate(axis=-1)(layers[-1])

    aggr = MPRRN(inputs=cat, rrusPerIrb=1, irbCount=1)

    out = Conv2D(filters=1, kernel_size=3, padding='same')(aggr)

    model = tf.keras.Model(inputs=[structure_input, texture_input], outputs=out)

    return model


def DualChannelInterconnect_struct_encodedecode():
    structure_input = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)
    texture_input = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)

    struct_in = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')(
        structure_input)
    text_in = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')(
        texture_input)

    struct_conv_1 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same', strides=2)
    struct_conv_2 = Conv2DTranspose(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same',
                                    strides=2)

    text_conv_1 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')
    text_conv_2 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')

    struct_inter = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')
    text_inter = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same')

    one_by_conv = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=1, padding='same')

    layers = []
    for i in range(3):
        if i == 0:
            layers.append(DualChannelLayer_3(struct_in, text_in, struct_conv_1, struct_conv_2, text_conv_1, text_conv_2,
                                             struct_inter,
                                             text_inter, one_by_conv, struct_in, text_in))
        else:
            layers.append(
                DualChannelLayer_3(layers[i - 1][0], layers[i - 1][1], struct_conv_1, struct_conv_2,
                                   text_conv_1, text_conv_2, struct_inter,
                                   text_inter, one_by_conv, struct_in, text_in))

    cat = Concatenate(axis=-1)(layers[-1])

    aggr = MPRRN(inputs=cat, rrusPerIrb=1, irbCount=1)

    out = Conv2D(filters=1, kernel_size=3, padding='same')(aggr)

    model = tf.keras.Model(inputs=[structure_input, texture_input], outputs=out)

    return model


def DC_Hourglass_Interconnect_2():
    structure_input = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)
    texture_input = Input(shape=INPUT_SHAPE, dtype=tf.dtypes.float32)

    struct_conv_1 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same', strides=2, activation='relu')(
        structure_input)
    text_conv_1 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same', strides=2, activation='relu')(
        texture_input)

    struct_sum_1 = Add()([struct_conv_1, text_conv_1])
    text_sum_1 = Add()([struct_conv_1, text_conv_1])

    struct_conv_2 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same', strides=2, activation='relu')(
        struct_sum_1)
    text_conv_2 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same', strides=2, activation='relu')(
        text_sum_1)

    struct_sum_2 = Add()([struct_conv_2, text_conv_2])
    text_sum_2 = Add()([struct_conv_2, text_conv_2])

    struct_conv_3 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same', strides=2, activation='relu')(
        struct_sum_2)
    text_conv_3 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same', strides=2, activation='relu')(
        text_sum_2)

    struct_sum_3 = Add()([struct_conv_3, text_conv_3])
    text_sum_3 = Add()([struct_conv_3, text_conv_3])

    struct_conv_4 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same', activation='relu')(
        struct_sum_3)
    text_conv_4 = Conv2D(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same', activation='relu')(
        text_sum_3)

    struct_sum_4 = Add()([struct_conv_4, text_conv_4])
    text_sum_4 = Add()([struct_conv_4, text_conv_4])

    struct_deconv_1 = Conv2DTranspose(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same',
                                      strides=2, activation='relu')(struct_sum_4)
    text_deconv_1 = Conv2DTranspose(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same',
                                    strides=2, activation='relu')(text_sum_4)

    struct_sum_5 = Add()([struct_deconv_1, text_deconv_1])
    text_sum_5 = Add()([struct_deconv_1, text_deconv_1])

    struct_deconv_2 = Conv2DTranspose(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same',
                                      strides=2, activation='relu')(struct_sum_5)
    text_deconv_2 = Conv2DTranspose(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same',
                                    strides=2, activation='relu')(text_sum_5)

    struct_sum_6 = Add()([struct_deconv_2, text_deconv_2])
    text_sum_6 = Add()([struct_deconv_2, text_deconv_2])

    struct_deconv_3 = Conv2DTranspose(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same',
                                      strides=2, activation='relu')(struct_sum_6)
    text_deconv_3 = Conv2DTranspose(filters=MPRRN_FILTERS_PER_LAYER, kernel_size=MPRRN_FILTER_SHAPE, padding='same',
                                    strides=2, activation='relu')(text_sum_6)

    struct_sum_7 = Add()([struct_deconv_3, text_deconv_3])
    text_sum_7 = Add()([struct_deconv_3, text_deconv_3])

    # Aggregate

    agg_sum = Add()([struct_sum_7, text_sum_7])

    aggr = MPRRN(inputs=agg_sum, rrusPerIrb=1, irbCount=1)

    out = Conv2D(filters=1, kernel_size=1, padding='same')(aggr)

    model = tf.keras.Model(inputs=[structure_input, texture_input], outputs=out)

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
    'hourglass_6': hourglass_6,
    'mprrn_only_w1x1': MPRRN_only_w1x1,
    'dualchannelinterconnect_4': DualChannelInterconnect,
    'dualchannelinterconnect_struct_encodedecode': DualChannelInterconnect_struct_encodedecode,
    'dc_hourglass_interconnect_2': DC_Hourglass_Interconnect_2
}
