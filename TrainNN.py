import os
import shutil
import tensorflow as tf
import numpy as np

from L0GradientMin.l0_gradient_minimization import l0_gradient_minimization_2d
from modules import Model
from modules.NNConfig import EPOCHS, LEARNING_RATE, GRAD_NORM, NN_MODEL, BATCH_SIZE, SAMPLE_IMAGES, JPEG_QUALITY, \
    ADAM_EPSILON, LOAD_WEIGHTS, CHECKPOINTS_PATH, TEST_BATCH_SIZE, \
    SAVE_AND_CONTINUE, ACCURACY_PSNR_THRESHOLD, MPRRN_RRU_PER_IRB, MPRRN_IRBS, L0_GRADIENT_MIN_LAMDA, \
    DATASET_EARLY_STOP, DUAL_CHANNEL_MODELS, EVEN_PAD_DATA, MPRRN_FILTER_SHAPE, MPRRN_TRAINING, PRETRAINED_MPRRN_PATH, \
    PRETRAINED_STRUCTURE_PATH, PRETRAINED_TEXTURE_PATH, STRUCTURE_MODEL, TEXTURE_MODEL, TRAIN_DIFF, LOSS_FUNCTION, \
    L0_GRADIENT_MIN_BETA_MAX, SAVE_TEST_OUT, USE_CPU_FOR_HIGH_MEMORY, TRAINING_DATASET, TOP_HALF_MODEL, IMAGE_CHANNELS
from modules.Dataset import JPEGDataset, preprocessInputsForSTRRN
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

        if NN_MODEL in 'strrn mprrn_only mprrn_encodedecode mprrn_encodedecode_4layer':
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
        if IMAGE_CHANNELS == 1:
            self.info = "greyscale" + self.info
        elif IMAGE_CHANNELS == 3:
            self.info = "rgb" + self.info

        self.info = self.info + "_QL" + str(JPEG_QUALITY) + "filterShape" + "_batchSize" + str(
            BATCH_SIZE) + "_learningRate" + str(LEARNING_RATE) + "_filterShape" + str(
            MPRRN_FILTER_SHAPE) + "_" + TRAINING_DATASET + "_loss_" + LOSS_FUNCTION

        self.saveFile = self.info + ".save"
        self.psnrTrainCsv = "./stats/psnr_train.csv"
        self.ssimTrainCsv = "./stats/ssim_train.csv"
        self.lossTrainCsv = "./stats/loss_train.csv"
        self.accuracyTrainCsv = "./stats/accuracy_train.csv"
        self.psnrTestCsv = "./stats/psnr_test.csv"
        self.lossTestCsv = "./stats/loss_test.csv"
        self.ssimTestCsv = "./stats/ssim_test.csv"
        self.accuracyTestCsv = "./stats/accuracy_test.csv"
        self.psnrValidationCsv = "./stats/psnr_validation.csv"
        self.lossValidationCsv = "./stats/loss_validation.csv"
        self.ssimValidationCsv = "./stats/ssim_validation.csv"
        self.accuracyValidationCsv = "./stats/accuracy_Validation.csv"

        if not os.path.exists("./stats"):
            os.mkdir("./stats")
        if not os.path.exists("./sampleImageOutputs"):
            os.mkdir("./sampleImageOutputs")
        if not os.path.exists("./checkpoints"):
            os.mkdir("./checkpoints")

        self.models = []
        for c in range(IMAGE_CHANNELS):
            self.models.append(Model.modelSwitch[NN_MODEL]())
        self.models = np.asarray(self.models)

        self.structureModels = []
        self.textureModels = []

        if MPRRN_TRAINING == 'aggregator':
            for c in range(IMAGE_CHANNELS):
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
            for c in range(IMAGE_CHANNELS):
                self.dcOutModels.append(Model.modelSwitch[TOP_HALF_MODEL]())
                self.dcOutModels[c].load_weights(
                    PRETRAINED_MPRRN_PATH + "dc_hourglass_grey_checkpoint/modelCheckpoint_ch" + str(
                        c) + "_" + TOP_HALF_MODEL)

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
            if USE_CPU_FOR_HIGH_MEMORY:
                with tf.device("CPU:0"):
                    self.do_test(epoch)
                    self.sample_output_images(epoch)
                    self.saveEpoch(epoch + 1)
            else:
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
        target_structure = batch['target_structure']
        target_texture = batch['target_texture']
        compressed_structure = batch['compressed_structure']
        compressed_texture = batch['compressed_texture']
        compressed = batch['compressed']
        if TRAIN_DIFF:
            diff = batch['diff']

        model_out = self.get_model_out(c, batch)

        if MPRRN_TRAINING == 'structure':
            loss = JPEGLoss(model_out, target_structure[..., c:c + 1], compressed_structure[..., c:c + 1])
            psnr = tf.image.psnr(target_structure[..., c:c + 1], model_out, max_val=1.0)
            ssim = tf.image.ssim(target_structure[..., c:c + 1], model_out, max_val=1.0)
        elif MPRRN_TRAINING == 'texture':
            loss = JPEGLoss(model_out, target_texture[..., c:c + 1], compressed_texture[..., c:c + 1])
            psnr = tf.image.psnr(target_texture[..., c:c + 1], model_out, max_val=1.0)
            ssim = tf.image.ssim(target_texture[..., c:c + 1], model_out, max_val=1.0)
        elif MPRRN_TRAINING == 'aggregator' or ("dc_hourglass_interconnect_bottom_half_" in NN_MODEL):
            loss = JPEGLoss(model_out, original[..., c:c + 1], compressed[..., c:c + 1])
            psnr = tf.image.psnr(original[..., c:c + 1], model_out, max_val=1.0)
            ssim = tf.image.ssim(original[..., c:c + 1], model_out, max_val=1.0)

        elif TRAIN_DIFF:
            loss = JPEGLoss(model_out, diff[..., c:c + 1], compressed[..., c:c + 1])
            psnr = tf.image.psnr(diff[..., c:c + 1], model_out, max_val=1.0)
            ssim = tf.image.ssim(diff[..., c:c + 1], model_out, max_val=1.0)

        elif 'dc_hourglass_interconnect_top_half' in NN_MODEL:
            loss = [JPEGLoss(model_out[0], target_structure[..., c:c + 1], compressed_structure[..., c:c + 1]),
                    JPEGLoss(model_out[1], target_texture[..., c:c + 1], compressed_texture[..., c:c + 1])]
            psnr = [tf.image.psnr(target_structure[..., c:c + 1], model_out[0], max_val=1.0),
                    tf.image.psnr(target_texture[..., c:c + 1], model_out[1], max_val=1.0)]
            ssim = [tf.image.ssim(target_structure[..., c:c + 1], model_out[0], max_val=1.0),
                    tf.image.ssim(target_texture[..., c:c + 1], model_out[1], max_val=1.0)]
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

            for c in range(IMAGE_CHANNELS):  # do separate training pass for each RGB channel

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

        avg_loss = total_loss / batches / IMAGE_CHANNELS
        avg_psnr = total_psnr / batches / IMAGE_CHANNELS
        avg_ssim = total_ssim / batches / IMAGE_CHANNELS
        avg_accuracy = total_accuracy / batches / IMAGE_CHANNELS

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

            for c in range(IMAGE_CHANNELS):

                model_out, loss, psnr, ssim = self.get_model_out_and_metrics(c, batch)

                total_loss += np.average(loss)
                total_psnr += np.average(psnr)
                total_ssim += np.average(ssim)
                if np.average(psnr) > ACCURACY_PSNR_THRESHOLD:
                    total_accuracy += 1

            batches += 1

        avg_loss = total_loss / batches / IMAGE_CHANNELS
        avg_psnr = total_psnr / batches / IMAGE_CHANNELS
        avg_ssim = total_ssim / batches / IMAGE_CHANNELS
        avg_accuracy = total_accuracy / batches / IMAGE_CHANNELS

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

            for c in range(IMAGE_CHANNELS):

                model_out, loss, psnr, ssim = self.get_model_out_and_metrics(c, batch)

                if SAVE_TEST_OUT:
                    self.savePrediction(model_out, batch['original'], batches)

                total_loss += np.average(loss)
                total_psnr += np.average(psnr)
                total_ssim += np.average(ssim)
                if np.average(psnr) > ACCURACY_PSNR_THRESHOLD:
                    total_accuracy += 1

            batches += 1

        avg_loss = total_loss / batches / IMAGE_CHANNELS
        avg_psnr = total_psnr / batches / IMAGE_CHANNELS
        avg_ssim = total_ssim / batches / IMAGE_CHANNELS
        avg_accuracy = total_accuracy / batches / IMAGE_CHANNELS

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

            if SAVE_TEST_OUT:
                if os.path.exists("predictions_best"):
                    shutil.rmtree("predictions_best")
                shutil.copytree(src="predictions", dst="predictions_best")

        if SAVE_TEST_OUT:
            if os.path.exists("predictions"):
                shutil.rmtree("predictions")
            os.mkdir("predictions")

    @staticmethod
    def savePrediction(model_out, original, number):
        model_out = np.asarray(model_out[0])
        original = np.asarray(original)

        np.save(file="predictions/" + str(number) + "_prediction", arr=model_out)
        np.save(file="predictions/" + str(number) + "_original", arr=original)

    @staticmethod
    def sample_preprocess():
        for file in SAMPLE_IMAGES:
            # compression
            pil_img = Image.open("./sampleImages/" + file + ".png")
            pil_img.save("./sampleImages/" + file + ".png" + ".compressed.jpg", format="JPEG", quality=JPEG_QUALITY)

            # diff
            original = np.asarray(pil_img) / 255.0
            compressed = np.asarray(Image.open("./sampleImages/" + file + ".png" + ".compressed.jpg")) / 255.0
            diff = original - compressed
            diff = np.clip(diff, a_min=0.0, a_max=1.0)
            diff = diff * 255.0
            diff = diff.astype('uint8')
            diff_img = Image.fromarray(diff)
            diff_img.save("./sampleImages/" + file + ".diff.jpg")

            # smoothing
            pil_img_c = Image.open("./sampleImages/" + file + ".compressed.jpg")
            pil_img_c = np.asarray(pil_img_c)
            pil_img_c = pil_img_c / 255.0
            smoothed_img_c = l0_gradient_minimization_2d(pil_img_c, lmd=L0_GRADIENT_MIN_LAMDA,
                                                         beta_max=L0_GRADIENT_MIN_BETA_MAX)
            smoothed_img_c = np.clip(smoothed_img_c, a_min=0.0, a_max=1.0)

            smoothed_img_c = smoothed_img_c * 255.0
            smoothed_img_c = smoothed_img_c.astype('uint8')
            out = Image.fromarray(smoothed_img_c)
            out.save("./sampleImages/" + file + ".png" + ".compressed.smoothed.png", format="PNG")

            # texture
            texture = pil_img_c - smoothed_img_c
            texture = np.clip(texture, a_min=0.0, a_max=1.0)
            texture = texture * 255.0
            texture = texture.astype('uint8')
            texture_out = Image.fromarray(texture)
            texture_out.save("./sampleImages/" + file + ".compressed.texture.png", format="PNG")

            # structure uncompressed
            pil_img = Image.open("./sampleImages/" + file + ".png")
            pil_img = np.asarray(pil_img)
            pil_img = pil_img / 255.0
            smoothed_img = l0_gradient_minimization_2d(pil_img, lmd=L0_GRADIENT_MIN_LAMDA,
                                                       beta_max=L0_GRADIENT_MIN_BETA_MAX)
            smoothed_img = np.clip(smoothed_img, a_min=0.0, a_max=1.0)

            smoothed_img = smoothed_img * 255.0
            smoothed_img = smoothed_img.astype('uint8')
            out = Image.fromarray(smoothed_img)
            out.save("./sampleImages/" + file + ".smoothed.png", format="PNG")

            # texture uncompressed
            texture = pil_img - smoothed_img
            texture = np.clip(texture, a_min=0.0, a_max=1.0)
            texture = texture * 255.0
            texture = texture.astype('uint8')
            texture_out = Image.fromarray(texture)
            texture_out.save("./sampleImages/" + file + ".texture.png", format="PNG")

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
            for c in range(IMAGE_CHANNELS):
                batch = {
                    'compressed_structure': structureIn,
                    'compressed_texture': textureIn,
                    'compressed': nn_input
                }
                model_out = self.get_model_out(c, batch)

                if 'dc_hourglass_interconnect_top_half' in NN_MODEL:
                    model_out = np.append(model_out[0], model_out[1], axis=1)

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
        if IMAGE_CHANNELS == 1:
            output = np.repeat(a=output, repeats=3, axis=-1)
        out_img = Image.fromarray(output[0])
        out_img.save(file, format="PNG")

    def load_weights(self):
        # load model checkpoint if exists
        try:
            for c in range(IMAGE_CHANNELS):
                self.models[c].load_weights(CHECKPOINTS_PATH + "modelCheckpoint_ch" + str(c) + "_" + self.info)
            print("Weights loaded")
        except Exception as e:
            print("Weights not loaded. Will create new weights")
            self.clean_dirs()

    def save_weights(self):
        for c in range(IMAGE_CHANNELS):
            self.models[c].save_weights(CHECKPOINTS_PATH + "modelCheckpoint_ch" + str(c) + "_" + self.info)

    def save_training_results(self):
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
        os.mkdir("stats")
        os.mkdir("checkpoints")
        os.mkdir("sampleImageOutputs")
        os.mkdir("predictions")
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


# TrainNN.sample_preprocess()

trainNn = TrainNN()
trainNn.train()

# Init datasets
# trainData = JPEGDataset('train', BATCH_SIZE)
# testData = JPEGDataset('test', BATCH_SIZE)
# validationData = JPEGDataset('validation', BATCH_SIZE)
