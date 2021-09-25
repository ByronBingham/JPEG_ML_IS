import os
import shutil
import tensorflow as tf
import datetime as dt
import numpy as np

from modules import Model
from modules.NNConfig import EPOCHS, LEARNING_RATE, GRAD_NORM, NN_MODEL, BATCH_SIZE, SAMPLE_IMAGES, JPEG_QUALITY, \
    ADAM_EPSILON, LOAD_WEIGHTS, CHECKPOINTS_PATH, LEARNING_RATE_DECAY_INTERVAL, LEARNING_RATE_DECAY, TEST_BATCH_SIZE, \
    SAVE_AND_CONTINUE, ACCURACY_PSNR_THRESHOLD, MPRRN_RRU_PER_IRB, MPRRN_IRBS
from modules.Dataset import JPEGDataset, BATCH_COMPRESSED, BATCH_PAD_MASK, BATCH_TARGET, preprocessInputsForSTRRN
from modules.Losses import MGE_MSE_combinedLoss
from PIL import Image
from pathlib import Path


def TF_Init():
    tf.compat.v1.enable_control_flow_v2()
    print("eager or not: " + str(tf.executing_eagerly()))

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(tf.config.experimental.list_physical_devices('GPU'))
    print("Num TPUs Available: ", len(tf.config.experimental.list_physical_devices('TPU')))
    print(tf.config.experimental.list_physical_devices('TPU'))

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

        if NN_MODEL == 'strrn':
            self.info = NN_MODEL + "_MPRRNs" + str(MPRRN_RRU_PER_IRB) + "_IRBs" + str(MPRRN_IRBS) + "_" + str(
                EPOCHS) + "epochs_batchSize" + str(BATCH_SIZE) + "_learningRate" + str(
                LEARNING_RATE)
        else:
            self.info = NN_MODEL + "_" + str(EPOCHS) + "epochs_batchSize" + str(BATCH_SIZE) + "_learningRate" + str(
                LEARNING_RATE)

        self.saveFile = self.info + ".save"
        self.psnrTrainCsv = "./stats/psnr_train_" + self.info + ".csv"
        self.ssimTrainCsv = "./stats/loss_train_" + self.info + ".csv"
        self.lossTrainCsv = "./stats/ssim_train_" + self.info + ".csv"
        self.accuracyTrainCsv = "./stats/accuracy_train_" + self.info + ".csv"
        self.psnrTestCsv = "./stats/psnr_test_" + self.info + ".csv"
        self.lossTestCsv = "./stats/loss_test_" + self.info + ".csv"
        self.ssimTestCsv = "./stats/ssim_test_" + self.info + ".csv"
        self.accuracyTestCsv = "./stats/accuracy_test_" + self.info + ".csv"
        self.psnrValidationCsv = "./stats/psnr_validation_" + self.info + ".csv"
        self.lossValidationCsv = "./stats/loss_validation_" + self.info + ".csv"
        self.ssimValidationCsv = "./stats/ssim_validation_" + self.info + ".csv"
        self.accuracyValidationCsv = "./stats/accuracy_Validation_" + self.info + ".csv"

        self.model = Model.modelSwitch[NN_MODEL]

        self.startingEpoch = 0
        if not SAVE_AND_CONTINUE:
            TrainNN.clean_dirs()
            TrainNN.create_dirs()
            if LOAD_WEIGHTS:
                self.load_weights()
        else:
            self.continueTraining()
            self.load_weights()

    def train(self):
        learningRate = LEARNING_RATE

        for epoch in range(EPOCHS):
            if epoch % LEARNING_RATE_DECAY_INTERVAL == 0:
                learningRate = learningRate / LEARNING_RATE_DECAY
            self.do_epoch(epoch, learningRate)
            self.save_weights()
            print("\nEpoch " + str(epoch) + " finished. Starting test step\n")
            self.do_test(epoch)
            self.sample_output_images(epoch)
            self.saveEpoch(epoch + 1)

    def do_epoch(self, epoch, learningRate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate, epsilon=ADAM_EPSILON)
        batches = 0
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_accuracy = 0.0
        trainData = JPEGDataset('train', BATCH_SIZE)

        for batch in trainData:
            structureIn, textureIn = None, None
            if NN_MODEL == 'strrn':
                structureIn, textureIn = preprocessInputsForSTRRN(batch[BATCH_COMPRESSED])
            for c in range(3):  # do separate training pass for each RGB channel
                with tf.GradientTape() as tape:
                    tape.watch(self.model.trainable_variables)
                    if NN_MODEL == 'strrn':
                        model_out = self.model([structureIn[..., c:c + 1], textureIn[..., c:c + 1]], training=True)
                    else:
                        model_out = self.model(batch[BATCH_COMPRESSED][..., c:c + 1], training=True)

                    # DEBUG
                    # self.saveNNOutput(model_out, "NN_Output.png")
                    # self.saveNNOutput(batch[BATCH_COMPRESSED], "This_should_be_NN_input.png")
                    # self.saveNNOutput(batch[BATCH_TARGET], "This_should_be_target_data.png")
                    # DEBUG

                    # multiply output by the padding mask to make sure padded areas are 0
                    model_out = tf.math.multiply(model_out, batch[BATCH_PAD_MASK][..., c:c + 1])

                    loss = MGE_MSE_combinedLoss(model_out, batch[BATCH_TARGET][..., c:c + 1])
                    psnr = tf.image.psnr(batch[BATCH_TARGET][..., c:c + 1], model_out, max_val=1.0)
                    ssim = tf.image.ssim(batch[BATCH_TARGET][..., c:c + 1], model_out, max_val=1.0)

                    total_loss += np.average(loss)
                    total_psnr += np.average(psnr)
                    total_ssim += np.average(ssim)
                    if np.average(psnr) > ACCURACY_PSNR_THRESHOLD:
                        total_accuracy += 1

                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    gradients = [tf.clip_by_norm(g, GRAD_NORM) for g in gradients]
                    optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            print("Batch " + str(batches + 1) + " Complete")
            batches = batches + 1

        avg_loss = total_loss / batches / 3
        avg_psnr = total_psnr / batches / 3
        avg_ssim = total_ssim / batches / 3
        avg_accuracy = total_accuracy / batches / 3

        lossFile = open(self.lossTrainCsv, "a")
        psnrFile = open(self.psnrTrainCsv, "a")
        ssimFile = open(self.ssimTrainCsv, "a")
        accuracyFile = open(self.accuracyTrainCsv, "a")
        lossFile.write(str(epoch) + "," + str(avg_loss) + "\n")
        psnrFile.write(str(epoch) + "," + str(avg_psnr) + "\n")
        ssimFile.write(str(epoch) + "," + str(avg_ssim) + "\n")
        accuracyFile.write(str(epoch) + "," + str(avg_accuracy) + "\n")
        lossFile.close()
        psnrFile.close()
        ssimFile.close()
        accuracyFile.close()

        self.do_validation(epoch)

    def do_validation(self, epoch):
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_accuracy = 0.0
        batches = 0
        testData = JPEGDataset('validation', BATCH_SIZE)

        for batch in testData:
            structureIn, textureIn = None, None
            if NN_MODEL == 'strrn':
                structureIn, textureIn = preprocessInputsForSTRRN(batch[BATCH_COMPRESSED])

            for c in range(3):
                if NN_MODEL == 'strrn':
                    model_out = self.model([structureIn[..., c:c + 1], textureIn[..., c:c + 1]])
                else:
                    model_out = self.model(batch[BATCH_COMPRESSED][..., c:c + 1])

                loss = MGE_MSE_combinedLoss(model_out, batch[BATCH_TARGET][..., c:c + 1])
                psnr = tf.image.psnr(batch[BATCH_TARGET][..., c:c + 1], model_out, max_val=1.0)
                ssim = tf.image.ssim(batch[BATCH_TARGET][..., c:c + 1], model_out, max_val=1.0)

                total_loss += np.average(loss)
                total_psnr += np.average(psnr)
                total_ssim += np.average(ssim)
                if np.average(psnr) > ACCURACY_PSNR_THRESHOLD:
                    total_accuracy += 1

            batches += 1

        avg_loss = total_loss / batches / 3
        avg_psnr = total_psnr / batches / 3
        avg_ssim = total_ssim / batches / 3
        avg_accuracy = total_accuracy / batches / 3

        lossFile = open(self.lossValidationCsv, "a")
        psnrFile = open(self.psnrValidationCsv, "a")
        ssimFile = open(self.ssimValidationCsv, "a")
        accuracyFile = open(self.accuracyValidationCsv, "a")
        lossFile.write(str(epoch) + "," + str(avg_loss) + "\n")
        psnrFile.write(str(epoch) + "," + str(avg_psnr) + "\n")
        ssimFile.write(str(epoch) + "," + str(avg_ssim) + "\n")
        accuracyFile.write(str(epoch) + "," + str(avg_accuracy) + "\n")
        lossFile.close()
        psnrFile.close()
        ssimFile.close()
        accuracyFile.close()

    def do_test(self, epoch):
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_accuracy = 0.0
        batches = 0
        testData = JPEGDataset('test', TEST_BATCH_SIZE)

        for batch in testData:
            structureIn, textureIn = None, None
            if NN_MODEL == 'strrn':
                structureIn, textureIn = preprocessInputsForSTRRN(batch[BATCH_COMPRESSED])

            for c in range(3):
                if NN_MODEL == 'strrn':
                    model_out = self.model([structureIn[..., c:c + 1], textureIn[..., c:c + 1]])
                else:
                    model_out = self.model(batch[BATCH_COMPRESSED][..., c:c + 1])

                loss = MGE_MSE_combinedLoss(model_out, batch[BATCH_TARGET][..., c:c + 1])
                psnr = tf.image.psnr(batch[BATCH_TARGET][..., c:c + 1], model_out, max_val=1.0)
                ssim = tf.image.ssim(batch[BATCH_TARGET][..., c:c + 1], model_out, max_val=1.0)

                total_loss += np.average(loss)
                total_psnr += np.average(psnr)
                total_ssim += np.average(ssim)
                if np.average(psnr) > ACCURACY_PSNR_THRESHOLD:
                    total_accuracy += 1

            batches += 1

        avg_loss = total_loss / batches / 3
        avg_psnr = total_psnr / batches / 3
        avg_ssim = total_ssim / batches / 3
        avg_accuracy = total_accuracy / batches / 3

        lossFile = open(self.lossTestCsv, "a")
        psnrFile = open(self.psnrTestCsv, "a")
        ssimFile = open(self.ssimTestCsv, "a")
        accuracyFile = open(self.accuracyTestCsv, "a")
        lossFile.write(str(epoch) + "," + str(avg_loss) + "\n")
        psnrFile.write(str(epoch) + "," + str(avg_psnr) + "\n")
        ssimFile.write(str(epoch) + "," + str(avg_ssim) + "\n")
        accuracyFile.write(str(epoch) + "," + str(avg_accuracy) + "\n")
        lossFile.close()
        psnrFile.close()
        ssimFile.close()
        accuracyFile.close()

    @staticmethod
    def sample_compress():
        for file in SAMPLE_IMAGES:
            pil_img = Image.open("./sampleImages" + file + ".png")
            pil_img.save("./sampleImages" + file + ".png" + ".compressed.jpg", format="JPEG", quality=JPEG_QUALITY)

    def sample_output_images(self, epoch):
        for file in SAMPLE_IMAGES:
            pil_img = Image.open("./sampleImages" + file + ".png" + ".compressed.jpg")
            nn_input = np.array(pil_img, dtype='float32') / 255.0
            nn_input = np.expand_dims(nn_input, axis=0)

            structureIn, textureIn = None, None
            if NN_MODEL == 'strrn':
                structureIn, textureIn = preprocessInputsForSTRRN(np.asarray(nn_input))

            channels_out = []
            for c in range(3):
                if NN_MODEL == 'strrn':
                    model_out = self.model([structureIn[..., c:c + 1], textureIn[..., c:c + 1]])
                else:
                    model_out = self.model(nn_input[..., c:c + 1])

                channels_out.append(np.asarray(model_out))

            arr = np.array(channels_out)
            image_out = np.concatenate(arr, axis=-1)

            self.saveNNOutput(image_out, "./sampleImageOutputs/" + file + "_" + self.info + "_" + str(epoch) + ".png")

    @staticmethod
    def saveNNOutput(output, file):
        output = output * 255.0
        output = np.array(output).astype('uint8')
        out_img = Image.fromarray(output[0])
        out_img.save(file, format="PNG")

    def load_weights(self):
        # load model checkpoint if exists
        try:
            self.model.load_weights(CHECKPOINTS_PATH + "modelCheckpoint_" + self.info)
            print("Weights loaded")
        except Exception as e:
            print("Weights not loaded. Will create new weights")

    def save_weights(self):
        self.model.save_weights(CHECKPOINTS_PATH + "/modelCheckpoint_" + self.info)

    def save_training_results(self):
        if not os.path.exists("./savedResults/" + self.info):
            os.mkdir("./savedResults/" + self.info)
        if os.path.exists("./savedResults/" + self.info + "./checkpoints"):
            shutil.rmtree("./savedResults/" + self.info + "./checkpoints")
        shutil.copytree(src="./checkpoints", dst="./savedResults/" + self.info + "./checkpoints", dirs_exist_ok=True)
        if os.path.exists("./savedResults/" + self.info + "./sampleImageOutputs"):
            shutil.rmtree("./savedResults/" + self.info + "./sampleImageOutputs")
        shutil.copytree(src="./sampleImageOutputs", dst="./savedResults/" + self.info + "./sampleImageOutputs",
                        dirs_exist_ok=True)
        if os.path.exists("./savedResults/" + self.info + "./stats"):
            shutil.rmtree("./savedResults/" + self.info + "./stats")
        shutil.copytree(src="./stats", dst="./savedResults/" + self.info + "./stats", dirs_exist_ok=True)

    @staticmethod
    def create_dirs():
        os.mkdir("stats")
        os.mkdir("checkpoints")
        os.mkdir("sampleImageOutputs")
        Path("stats/.gitkeep").touch()
        Path("checkpoints/.gitkeep").touch()
        Path("sampleImageOutputs/.gitkeep").touch()

    @staticmethod
    def clean_dirs():
        try:
            shutil.rmtree("stats")
            shutil.rmtree("checkpoints")
            shutil.rmtree("sampleImageOutputs")
        except FileNotFoundError:
            print("Nothing to delete here")

    def continueTraining(self):
        if os.path.exists(self.saveFile):
            save = open(self.saveFile, 'r')
            epoch = save.read()
            self.startingEpoch = int(epoch)
        else:
            print("No Save point found. Starting from epoch 0")
            file = open(self.saveFile, 'w')
            file.write('0')
            file.close()
            self.startingEpoch = 0

    def saveEpoch(self, epoch):
        save = open(self.saveFile, 'w')
        save.write(str(epoch))
        self.save_training_results()


trainNn = TrainNN()
trainNn.sample_compress()

trainNn.train()

# trainNn.load_weights()
# trainNn.sample_output_images(99)
