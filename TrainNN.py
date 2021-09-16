import tensorflow as tf
import datetime as dt
import numpy as np

from modules import Model
from modules.NNConfig import EPOCHS, LEARNING_RATE, GRAD_NORM, NN_MODEL, BATCH_SIZE
from modules.Dataset import JPEGDataset, BATCH_COMPRESSED, BATCH_PAD_MASK, BATCH_TARGET
from modules.Losses import MGE_MSE_combinedLoss


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


class TrainNN:
    model = None
    trainData = None
    testData = None

    def __init__(self):
        TF_Init()
        self.model = Model.modelSwitch[NN_MODEL]

        self.trainData = JPEGDataset('train')
        self.testData = JPEGDataset('validation')

        self.info = NN_MODEL + "_" + str(EPOCHS) + "epochs_batchSize" + str(BATCH_SIZE) + "_learningRate" + str(
            LEARNING_RATE)
        self.psnrTrainCsv = "./stats/psnr_train_" + self.info + ".csv"
        self.lossTrainCsv = "./stats/loss_train_" + self.info + ".csv"
        self.psnrTestCsv = "./stats/psnr_test_" + self.info + ".csv"
        self.lossTestCsv = "./stats/loss_test_" + self.info + ".csv"

    def train(self):
        time1 = dt.datetime.now()

        for epoch in range(EPOCHS):
            self.do_epoch(epoch)
            print("\nEpoch " + str(epoch) + " finished. Starting test step\n")
            self.do_test(epoch)

    def do_epoch(self, epoch):
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        batches = 0
        total_loss = 0.0
        total_psnr = 0.0

        for batch in self.trainData:
            with tf.GradientTape() as tape:
                tape.watch(self.model.trainable_variables)
                model_out = self.model(batch[BATCH_COMPRESSED], training=True)

                # multiply output by the padding mask to make sure padded areas are 0
                model_out = tf.math.multiply(model_out, batch[BATCH_PAD_MASK])

                loss = MGE_MSE_combinedLoss(model_out, batch[BATCH_TARGET])
                psnr = tf.image.psnr(batch[BATCH_TARGET], model_out, max_val=1.0)

                total_loss += np.average(loss)
                total_psnr += np.sum(psnr)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                gradients = [tf.clip_by_norm(g, GRAD_NORM) for g in gradients]
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            print("Batch " + str(batches + 1) + " Complete")
            batches = batches + 1

        avg_loss = total_loss / batches
        avg_psnr = total_psnr / batches

        lossFile = open(self.lossTrainCsv)
        psnrFile = open(self.psnrTrainCsv)
        lossFile.write(str(epoch) + "," + str(avg_loss))
        psnrFile.write(str(epoch) + "," + str(avg_psnr))
        lossFile.close()
        psnrFile.close()

        self.do_eval()

    def do_eval(self):
        print("Eval step NYI")

    def do_test(self, epoch):
        total_loss = 0.0
        total_psnr = 0.0
        batches = 0

        for batch in self.testData:
            model_out = self.model(batch[BATCH_COMPRESSED], training=True)

            loss = MGE_MSE_combinedLoss(model_out, batch[BATCH_TARGET])
            psnr = tf.image.psnr(batch[BATCH_TARGET], model_out, max_val=1.0)
            total_loss += np.average(loss)
            total_psnr += np.sum(psnr)
            batches += 1

        avg_loss = total_loss / batches
        avg_psnr = total_psnr / batches

        lossFile = open(self.lossTestCsv)
        psnrFile = open(self.psnrTestCsv)
        lossFile.write(str(epoch) + "," + str(avg_loss))
        psnrFile.write(str(epoch) + "," + str(avg_psnr))
        lossFile.close()
        psnrFile.close()


trainNn = TrainNN()

trainNn.train()
