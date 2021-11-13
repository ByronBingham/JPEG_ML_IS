import os
import shutil
import tensorflow as tf
import numpy as np

from modules import Model
from modules.NNConfig import NN_MODEL, ACCURACY_PSNR_THRESHOLD, DUAL_CHANNEL_MODELS, MPRRN_TRAINING, \
    PRETRAINED_STRUCTURE_PATH, PRETRAINED_TEXTURE_PATH, TRAIN_DIFF, SAVE_TEST_OUT, USE_CPU_FOR_HIGH_MEMORY, \
    TOP_HALF_MODEL, TEST_DATASET, TEST_CHECKPOINT_DIR, STRUCTURE_MODEL, TEXTURE_MODEL
from modules.Dataset import JPEGDataset
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


class TestNN:
    models = None
    trainData = None
    testData = None

    def __init__(self):
        TF_Init()

        self.info = NN_MODEL

        if MPRRN_TRAINING == 'structure' or MPRRN_TRAINING == 'texture' or MPRRN_TRAINING == 'aggregator':
            self.info = MPRRN_TRAINING + self.info

        if TRAIN_DIFF:
            self.info = "_diff" + self.info

        self.info = "TEST_" + TEST_DATASET + "_" + self.info

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
                    PRETRAINED_STRUCTURE_PATH + "modelCheckpoint_ch" + str(c) + "_" + STRUCTURE_MODEL)
                self.textureModels[c].load_weights(
                    PRETRAINED_TEXTURE_PATH + "modelCheckpoint_ch" + str(c) + "_" + TEXTURE_MODEL)

            self.structureModels = np.asarray(self.structureModels)
            self.textureModels = np.asarray(self.textureModels)

        self.dcOutModels = []

        if "dc_hourglass_interconnect_bottom_half_" in NN_MODEL:
            for c in range(3):
                self.dcOutModels.append(Model.modelSwitch[TOP_HALF_MODEL]())
                self.dcOutModels[c].load_weights(
                    TEST_CHECKPOINT_DIR + "modelCheckpoint_ch" + str(c) + "_" + TOP_HALF_MODEL)

        self.startingEpoch = 0
        self.best_psnr = 0.0

        TestNN.create_dirs()
        TestNN.clean_dirs()
        self.load_weights()

    def test(self):

        datasets = [JPEGDataset("test", 1, TEST_DATASET)]

        for dataset in datasets:
            if USE_CPU_FOR_HIGH_MEMORY:
                with tf.device("CPU:0"):
                    self.do_test(0, dataset)
            else:
                self.do_test(0, dataset)

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
        if "dc_hourglass_interconnect_bottom_half_" in NN_MODEL:
            structure_out, texture_out = self.dcOutModels[c](
                [compressed_structure[..., c:c + 1], compressed_texture[..., c:c + 1]])
        if NN_MODEL in DUAL_CHANNEL_MODELS:
            return self.models[c]([compressed_structure[..., c:c + 1], compressed_texture[..., c:c + 1]],
                                  training=True)

        if MPRRN_TRAINING == 'structure':
            return self.models[c](compressed_structure[..., c:c + 1], training=True)
        elif MPRRN_TRAINING == 'texture':
            return self.models[c](compressed_texture[..., c:c + 1], training=True)
        elif MPRRN_TRAINING == 'aggregator':
            agg_in = np.add(structure_out, texture_out)
            return self.models[c](agg_in)

        elif 'dc_hourglass_interconnect_top_half' in NN_MODEL:
            structure_out, texture_out = self.models[c](
                [compressed_structure[..., c:c + 1], compressed_texture[..., c:c + 1]])
            return structure_out, texture_out

        else:
            return self.models[c](compressed[..., c:c + 1], training=True)

    def get_model_out_and_metrics(self, c, batch):
        original = batch['original']
        compressed = batch['compressed']

        model_out = self.get_model_out(c, batch)

        if MPRRN_TRAINING == 'aggregator' or ("dc_hourglass_interconnect_bottom_half_" in NN_MODEL):
            loss = JPEGLoss(model_out, original[..., c:c + 1], compressed[..., c:c + 1])
            psnr = tf.image.psnr(original[..., c:c + 1], model_out, max_val=1.0)
            ssim = tf.image.ssim(original[..., c:c + 1], model_out, max_val=1.0)
        else:
            loss = JPEGLoss(model_out, original[..., c:c + 1], compressed[..., c:c + 1])
            psnr = tf.image.psnr(original[..., c:c + 1], model_out, max_val=1.0)
            ssim = tf.image.ssim(original[..., c:c + 1], model_out, max_val=1.0)

        return model_out, loss, psnr, ssim

    def do_test(self, epoch, dataset):
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_accuracy = 0.0
        batches = 0

        for batch in dataset:

            for c in range(3):

                model_out, loss, psnr, ssim = self.get_model_out_and_metrics(c, batch)

                if SAVE_TEST_OUT:
                    self.savePrediction(model_out, batch['original'], batches)

                total_loss += np.average(loss)
                total_psnr += np.average(psnr)
                total_ssim += np.average(ssim)
                if np.average(psnr) > ACCURACY_PSNR_THRESHOLD:
                    total_accuracy += 1

            print("Completed: " + str(batches), end="\r")
            batches += 1

        print("Finished testing, creating stat files")
        if os.path.exists("predictions_best"):
            shutil.rmtree("predictions_best")
        shutil.copytree(src="predictions", dst="predictions_best")

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

        self.save_testing_results()

    @staticmethod
    def savePrediction(model_out, original, number):
        model_out = np.asarray(model_out[0])
        original = np.asarray(original)

        np.save(file="predictions/" + str(number) + "_prediction", arr=model_out)
        np.save(file="predictions/" + str(number) + "_original", arr=original)

    @staticmethod
    def saveNNOutput(output, file):
        output = np.clip(output, a_min=0.0, a_max=1.0)
        output = output * 255.0
        output = np.array(output).astype('uint8')
        out_img = Image.fromarray(output[0])
        out_img.save(file, format="PNG")

    def load_weights(self):
        # load model checkpoint if exists
        for c in range(3):
            self.models[c].load_weights(TEST_CHECKPOINT_DIR + "modelCheckpoint_ch" + str(c) + "_" + NN_MODEL)
        print("Weights loaded")


    def save_testing_results(self):
        if not os.path.exists("../savedResults/"):
            os.mkdir("../savedResults/")
        if not os.path.exists("../savedResults/" + self.info):
            os.mkdir("../savedResults/" + self.info)
        else:
            shutil.rmtree("../savedResults/" + self.info)
            os.mkdir("../savedResults/" + self.info)
            shutil.copy(src="modules/NNConfig.py", dst="../savedResults/" + self.info + "/" + self.info + ".config")
        if os.path.exists("../savedResults/" + self.info + "./checkpoints"):
            shutil.rmtree("../savedResults/" + self.info + "./checkpoints")
        shutil.copytree(src="./checkpoints", dst="../savedResults/" + self.info + "./checkpoints")
        if os.path.exists("../savedResults/" + self.info + "./sampleImageOutputs"):
            shutil.rmtree("../savedResults/" + self.info + "./sampleImageOutputs")
        shutil.copytree(src="./sampleImageOutputs", dst="../savedResults/" + self.info + "./sampleImageOutputs")
        if os.path.exists("../savedResults/" + self.info + "./stats"):
            shutil.rmtree("../savedResults/" + self.info + "./stats")
        shutil.copytree(src="./stats", dst="../savedResults/" + self.info + "./stats")
        if os.path.exists("../savedResults/" + self.info + "./predictions_best"):
            shutil.rmtree("../savedResults/" + self.info + "./predictions_best")
        shutil.copytree(src="./predictions_best", dst="../savedResults/" + self.info + "./predictions_best")

    @staticmethod
    def create_dirs():
        if not os.path.exists("stats"):
            os.mkdir("stats")
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        if not os.path.exists("sampleImageOutputs"):
            os.mkdir("sampleImageOutputs")
        if not os.path.exists("predictions"):
            os.mkdir("predictions")
        if not os.path.exists("predictions_best"):
            os.mkdir("predictions_best")
        Path("stats/.gitkeep").touch()
        Path("checkpoints/.gitkeep").touch()
        Path("sampleImageOutputs/.gitkeep").touch()
        Path("predictions/.gitkeep").touch()
        Path("predictions_best/.gitkeep").touch()

    @staticmethod
    def clean_dirs():
        try:
            shutil.rmtree("stats")
            os.mkdir("stats")
            shutil.rmtree("checkpoints")
            os.mkdir("checkpoints")
            shutil.rmtree("sampleImageOutputs")
            os.mkdir("sampleImageOutputs")
            shutil.rmtree("predictions")
            os.mkdir("predictions")
            shutil.rmtree("predictions_best")
            os.mkdir("predictions_best")
        except FileNotFoundError:
            print("Nothing to delete here")


# TrainNN.sample_preprocess()

testNn = TestNN()
testNn.test()

# Init datasets
# trainData = JPEGDataset('train', BATCH_SIZE)
# testData = JPEGDataset('test', BATCH_SIZE)
# validationData = JPEGDataset('validation', BATCH_SIZE)
