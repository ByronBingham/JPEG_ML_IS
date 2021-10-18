import os
import shutil
import tensorflow as tf
import numpy as np

from modules import Model
from modules.NNConfig import EPOCHS, LEARNING_RATE, GRAD_NORM, NN_MODEL, BATCH_SIZE, SAMPLE_IMAGES, JPEG_QUALITY, \
    ADAM_EPSILON, LOAD_WEIGHTS, CHECKPOINTS_PATH, LEARNING_RATE_DECAY_INTERVAL, LEARNING_RATE_DECAY, TEST_BATCH_SIZE, \
    SAVE_AND_CONTINUE, ACCURACY_PSNR_THRESHOLD, MPRRN_RRU_PER_IRB, MPRRN_IRBS, L0_GRADIENT_MIN_LAMDA, \
    DATASET_EARLY_STOP, DUAL_CHANNEL_MODELS, EVEN_PAD_DATA, MPRRN_FILTER_SHAPE, MPRRN_TRAINING, PRETRAINED_MPRRN_PATH, \
    PRETRAINED_STRUCTURE, PRETRAINED_TEXTURE, STRUCTURE_MODEL, TEXTURE_MODEL, TRAIN_DIFF
from modules.Dataset import JPEGDataset, BATCH_COMPRESSED, BATCH_TARGET, preprocessDataForSTRRN, \
    preprocessInputsForSTRRN
from modules.Losses import JPEGLoss
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
    models = None
    trainData = None
    testData = None

    def __init__(self):
        TF_Init()

        if NN_MODEL in 'strrn mprrn_only':
            self.info = NN_MODEL + "_MPRRNs" + str(MPRRN_RRU_PER_IRB) + "_IRBs" + str(
                MPRRN_IRBS) + "_QL" + str(
                JPEG_QUALITY) + "_L0Lmb" + str(L0_GRADIENT_MIN_LAMDA)
        else:
            self.info = NN_MODEL

        if DATASET_EARLY_STOP:
            self.info = "minirun_" + self.info

        if MPRRN_TRAINING == 'structure' or MPRRN_TRAINING == 'texture' or MPRRN_TRAINING == 'aggregator':
            self.info = MPRRN_TRAINING + self.info

        if TRAIN_DIFF:
            self.info = "_diff" + self.info

        self.info = self.info + "_QL" + str(JPEG_QUALITY) + "filterShape" + "_batchSize" + str(
            BATCH_SIZE) + "_learningRate" + str(
            LEARNING_RATE) + "_filterShape" + str(MPRRN_FILTER_SHAPE)

        self.saveFile = self.info + ".save"
        self.psnrTrainCsv = "./stats/psnr_train_" + self.info + ".csv"
        self.ssimTrainCsv = "./stats/ssim_train_" + self.info + ".csv"
        self.lossTrainCsv = "./stats/loss_train_" + self.info + ".csv"
        self.accuracyTrainCsv = "./stats/accuracy_train_" + self.info + ".csv"
        self.psnrTestCsv = "./stats/psnr_test_" + self.info + ".csv"
        self.lossTestCsv = "./stats/loss_test_" + self.info + ".csv"
        self.ssimTestCsv = "./stats/ssim_test_" + self.info + ".csv"
        self.accuracyTestCsv = "./stats/accuracy_test_" + self.info + ".csv"
        self.psnrValidationCsv = "./stats/psnr_validation_" + self.info + ".csv"
        self.lossValidationCsv = "./stats/loss_validation_" + self.info + ".csv"
        self.ssimValidationCsv = "./stats/ssim_validation_" + self.info + ".csv"
        self.accuracyValidationCsv = "./stats/accuracy_Validation_" + self.info + ".csv"

        if not os.path.exists("./stats"):
            os.mkdir("./stats")
        if not os.path.exists("./sampleImageOutputs"):
            os.mkdir("./sampleImageOutputs")
        if not os.path.exists("./checkpoints"):
            os.mkdir("./checkpoints")

        self.models = [Model.modelSwitch[NN_MODEL](), Model.modelSwitch[NN_MODEL](), Model.modelSwitch[NN_MODEL]()]
        self.models = np.asarray(self.models)

        self.structureModels = []
        self.textureModels = []

        if MPRRN_TRAINING == 'aggregator':
            for c in range(3):
                self.structureModels.append(Model.modelSwitch[STRUCTURE_MODEL]())
                self.textureModels.append(Model.modelSwitch[TEXTURE_MODEL]())
                self.structureModels[c].load_weights(
                    PRETRAINED_MPRRN_PATH + "modelCheckpoint_ch" + str(c) + "_" + PRETRAINED_STRUCTURE)
                self.textureModels[c].load_weights(
                    PRETRAINED_MPRRN_PATH + "modelCheckpoint_ch" + str(c) + "_" + PRETRAINED_TEXTURE)

            self.structureModels = np.asarray(self.structureModels)
            self.textureModels = np.asarray(self.textureModels)

        self.startingEpoch = 0
        self.best_psnr = 0.0

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

        for epoch in range(self.startingEpoch, EPOCHS):
            # if epoch % LEARNING_RATE_DECAY_INTERVAL == 0:
            #    learningRate = learningRate / LEARNING_RATE_DECAY
            self.do_epoch(epoch, learningRate)
            self.save_weights()
            print("\nEpoch " + str(epoch) + " finished. Starting test step\n")
            self.do_test(epoch)
            self.sample_output_images(epoch)
            self.saveEpoch(epoch + 1)

    def get_model_out(self, c, batch):
        compressed_structure = batch['compressed_structure']
        compressed_texture = batch['compressed_texture']
        compressed = batch['compressed']

        model_out = None
        structure_out = None
        texture_out = None

        if MPRRN_TRAINING == 'aggregator':
            structure_out = self.structureModels[c](compressed_structure[..., c:c + 1])
            texture_out = self.textureModels[c](compressed_texture[..., c:c + 1])
        if NN_MODEL in DUAL_CHANNEL_MODELS:
            model_out = self.models[c]([compressed_structure[..., c:c + 1], compressed_texture[..., c:c + 1]],
                                       training=True)

        if MPRRN_TRAINING == 'structure':
            model_out = self.models[c](compressed_structure[..., c:c + 1], training=True)
        elif MPRRN_TRAINING == 'texture':
            model_out = self.models[c](compressed_texture[..., c:c + 1], training=True)
        elif MPRRN_TRAINING == 'aggregator':
            agg_in = np.add(structure_out, texture_out)
            model_out = self.models[c](agg_in)

        else:
            model_out = self.models[c](compressed[..., c:c + 1], training=True)

        return model_out

    def get_model_out_and_metrics(self, c, batch):
        original = batch['original']
        target_structure = batch['target_structure']
        target_texture = batch['target_texture']
        compressed_structure = batch['compressed_structure']
        compressed_texture = batch['compressed_texture']
        compressed = batch['compressed']
        diff = batch['diff']

        psnr = None
        ssim = None
        loss = None

        model_out = self.get_model_out(c, batch)

        if MPRRN_TRAINING == 'structure':
            loss = JPEGLoss(model_out, target_structure[..., c:c + 1], compressed_structure[..., c:c + 1])
            psnr = tf.image.psnr(target_structure[..., c:c + 1], model_out, max_val=1.0)
            ssim = tf.image.ssim(target_structure[..., c:c + 1], model_out, max_val=1.0)
        elif MPRRN_TRAINING == 'texture':
            loss = JPEGLoss(model_out, target_texture[..., c:c + 1], compressed_texture[..., c:c + 1])
            psnr = tf.image.psnr(target_texture[..., c:c + 1], model_out, max_val=1.0)
            ssim = tf.image.ssim(target_texture[..., c:c + 1], model_out, max_val=1.0)
        elif MPRRN_TRAINING == 'aggregator':
            loss = JPEGLoss(model_out, original[..., c:c + 1], compressed[..., c:c + 1])
            psnr = tf.image.psnr(original[..., c:c + 1], model_out, max_val=1.0)
            ssim = tf.image.ssim(original[..., c:c + 1], model_out, max_val=1.0)

        elif TRAIN_DIFF:
            loss = JPEGLoss(model_out, diff[..., c:c + 1], compressed[..., c:c + 1])
            psnr = tf.image.psnr(diff[..., c:c + 1], model_out, max_val=1.0)
            ssim = tf.image.ssim(diff[..., c:c + 1], model_out, max_val=1.0)
        else:
            loss = JPEGLoss(model_out, original[..., c:c + 1], compressed[..., c:c + 1])
            psnr = tf.image.psnr(original[..., c:c + 1], model_out, max_val=1.0)
            ssim = tf.image.ssim(original[..., c:c + 1], model_out, max_val=1.0)

        return model_out, loss, psnr, ssim

    def do_epoch(self, epoch, learningRate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate, epsilon=ADAM_EPSILON)
        batches = 0
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_accuracy = 0.0
        trainData = JPEGDataset('train', BATCH_SIZE)

        for batch in trainData:

            for c in range(3):  # do separate training pass for each RGB channel

                with tf.GradientTape() as tape:
                    tape.watch(self.models[c].trainable_variables)
                    model_out, loss, psnr, ssim = self.get_model_out_and_metrics(c, batch)

                    total_loss += np.average(loss)
                    total_psnr += np.average(psnr)
                    total_ssim += np.average(ssim)
                    if np.average(psnr) > ACCURACY_PSNR_THRESHOLD:
                        total_accuracy += 1

                    gradients = tape.gradient(loss, self.models[c].trainable_variables)
                    gradients = [tf.clip_by_norm(g, GRAD_NORM) for g in gradients]
                    optimizer.apply_gradients(zip(gradients, self.models[c].trainable_variables))

            print("Batch " + str(batches + 1) + " Complete", end="\r")
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
            original = batch['original']
            target_structure = batch['target_structure']
            target_texture = batch['target_texture']
            compressed_structure = batch['compressed_structure']
            compressed_texture = batch['compressed_texture']
            compressed = batch['compressed']

            for c in range(3):

                model_out, loss, psnr, ssim = self.get_model_out_and_metrics(c, batch)

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

            for c in range(3):

                model_out, loss, psnr, ssim = self.get_model_out_and_metrics(c, batch)

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

        # save weights if this is the best psnr performance
        if avg_psnr > self.best_psnr:
            if os.path.exists(CHECKPOINTS_PATH + "best/"):
                shutil.rmtree(CHECKPOINTS_PATH + "best/")
            shutil.copytree(src=CHECKPOINTS_PATH, dst=CHECKPOINTS_PATH + "best/")
            self.best_psnr = avg_psnr

    @staticmethod
    def sample_compress():
        for file in SAMPLE_IMAGES:
            pil_img = Image.open("./sampleImages/" + file + ".png")
            pil_img.save("./sampleImages/" + file + ".png" + ".compressed.jpg", format="JPEG", quality=JPEG_QUALITY)

    def sample_output_images(self, epoch):
        for file in SAMPLE_IMAGES:
            pil_img = Image.open("./sampleImages/" + file + ".png" + ".compressed.jpg")
            nn_input = np.array(pil_img, dtype='float32') / 255.0
            nn_input = np.expand_dims(nn_input, axis=0)

            # pad data for conv/deconv layers
            if EVEN_PAD_DATA > 1:
                if (nn_input.shape[1] % EVEN_PAD_DATA) != 0:  # if shape is odd, pad to make even
                    nn_input = np.pad(nn_input,
                                      [(0, 0), (0, EVEN_PAD_DATA - (nn_input.shape[1] % EVEN_PAD_DATA)), (0, 0),
                                       (0, 0)],
                                      constant_values=0)
                if (nn_input.shape[2] % EVEN_PAD_DATA) != 0:
                    nn_input = np.pad(nn_input,
                                      [(0, 0), (0, EVEN_PAD_DATA - (nn_input.shape[2] % EVEN_PAD_DATA)), (0, 0)],
                                      constant_values=0)

            structureIn, textureIn = None, None
            if (NN_MODEL in DUAL_CHANNEL_MODELS) or (
                    MPRRN_TRAINING == 'structure' or MPRRN_TRAINING == 'texture' or MPRRN_TRAINING == 'aggregator'):
                structureIn, textureIn = preprocessInputsForSTRRN(np.asarray(nn_input))

            channels_out = []
            for c in range(3):
                batch = {
                    'compressed_structure': structureIn,
                    'compressed_texture': textureIn,
                    'compressed': nn_input
                }
                model_out = self.get_model_out(c, batch)

                channels_out.append(np.asarray(model_out))

            arr = np.array(channels_out)
            image_out = np.concatenate(arr, axis=-1)

            self.saveNNOutput(image_out, "./sampleImageOutputs/" + file + "_" + self.info + "_" + str(epoch) + ".png")

            if TRAIN_DIFF:
                diff_out = image_out + nn_input
                self.saveNNOutput(diff_out,
                                  "./sampleImageOutputs/" + file + "_" + self.info + "_" + str(epoch) + ".diff.png")

    @staticmethod
    def saveNNOutput(output, file):
        output = np.clip(output, a_min=0.0, a_max=1.0)
        output = output * 255.0
        output = np.array(output).astype('uint8')
        out_img = Image.fromarray(output[0])
        out_img.save(file, format="PNG")

    def load_weights(self):
        # load model checkpoint if exists
        try:
            for c in range(3):
                self.models[c].load_weights(CHECKPOINTS_PATH + "modelCheckpoint_ch" + str(c) + "_" + self.info)
            print("Weights loaded")
        except Exception as e:
            print("Weights not loaded. Will create new weights")
            self.clean_dirs()

    def save_weights(self):
        for c in range(3):
            self.models[c].save_weights(CHECKPOINTS_PATH + "modelCheckpoint_ch" + str(c) + "_" + self.info)

    def save_training_results(self):
        if not os.path.exists("../savedResults/"):
            os.mkdir("../savedResults/")
        if not os.path.exists("../savedResults/" + self.info):
            os.mkdir("../savedResults/" + self.info)
        if os.path.exists("../savedResults/" + self.info + "./checkpoints"):
            shutil.rmtree("../savedResults/" + self.info + "./checkpoints")
        shutil.copytree(src="./checkpoints", dst="../savedResults/" + self.info + "./checkpoints")
        if os.path.exists("../savedResults/" + self.info + "./sampleImageOutputs"):
            shutil.rmtree("../savedResults/" + self.info + "./sampleImageOutputs")
        shutil.copytree(src="./sampleImageOutputs", dst="../savedResults/" + self.info + "./sampleImageOutputs")
        if os.path.exists("../savedResults/" + self.info + "./stats"):
            shutil.rmtree("../savedResults/" + self.info + "./stats")
        shutil.copytree(src="./stats", dst="../savedResults/" + self.info + "./stats")

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
            os.mkdir("stats")
            shutil.rmtree("checkpoints")
            os.mkdir("checkpoints")
            shutil.rmtree("sampleImageOutputs")
            os.mkdir("sampleImageOutputs")
        except FileNotFoundError:
            print("Nothing to delete here")

    def continueTraining(self):
        if os.path.exists(self.saveFile):
            save = open(self.saveFile, 'r')
            epoch = save.readline()
            best_psnr = save.readline()
            save.close()
            self.startingEpoch = int(epoch)
            self.best_psnr = float(best_psnr)
        else:
            print("No Save point found. Starting from epoch 0")
            file = open(self.saveFile, 'w')
            file.write("0\n")
            file.write("0.0\n")
            file.close()
            self.startingEpoch = 0

    def saveEpoch(self, epoch):
        save = open(self.saveFile, 'w')
        save.write(str(epoch) + "\n")
        save.write(str(self.best_psnr) + "\n")
        save.close()
        self.save_training_results()


trainNn = TrainNN()
trainNn.sample_compress()

trainNn.train()

# Init datasets
# trainData = JPEGDataset('train', BATCH_SIZE)
# testData = JPEGDataset('test', BATCH_SIZE)
# validationData = JPEGDataset('validation', BATCH_SIZE)
