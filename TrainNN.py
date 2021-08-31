import tensorflow as tf
import tensorflow.keras.layers as Layers
from modules import Model


def TF_Init():
    tf.compat.v1.enable_control_flow_v2()
    print("eager or not: " + str(tf.executing_eagerly()))

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(tf.config.experimental.list_physical_devices('GPU'))

    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except Exception as e:
        print(e)


def createModel(input_layer):
    return Model.jpgEnhanceModel(input_layer)


model = createModel(Layers.Input(shape=(96, 96, 3), dtype=tf.dtypes.float16))

print(model)
