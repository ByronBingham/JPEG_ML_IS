import tensorflow as tf
import datetime as dt

from modules import Model
from modules.NNConfig import EPOCHS, LEARNING_RATE


model = None
trainData = None
testData = None

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


def main():
    TF_Init()
    model = Model.EQLRI_model()


def train():
    time1 = dt.datetime.now()

    for epoch in range(EPOCHS):
        do_epoch()
        do_test()



def do_epoch():
    batches = 0
    for batch in trainData:
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)

    do_eval()

def do_eval():


def do_test():
    for batch in testData:



