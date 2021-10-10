from io import BytesIO

from PIL import Image

import os
import numpy as np

from L0GradientMin.l0_gradient_minimization import l0_gradient_minimization_2d
from modules.NNConfig import L0_GRADIENT_MIN_LAMDA, L0_GRADIENT_MIN_BETA_MAX, JPEG_QUALITY

DATASET_PATH = 'e:/datasets/div2k_dataset/downloads/extracted/'
OUTPUT_PATH = 'e:/datasets/div2k_dataset/preprocessed/'

FILE_SUFFIX = '.png'

PATCH_SIZE = 31
STRIDE = 21

SEGMENT_IMAGES = True
AUGMENT_IMAGES = True
STRRN_PREPROCESSING = True


def STRRN_processing_and_save(original, compressed, out_dir, count):
    # process compressed data (input)
    originalCompressed = compressed / 255.0
    compressedStructure = l0_gradient_minimization_2d(originalCompressed,
                                                      lmd=L0_GRADIENT_MIN_LAMDA,
                                                      beta_max=L0_GRADIENT_MIN_BETA_MAX)
    compressedStructure = np.clip(compressedStructure, a_min=0.0, a_max=1.0)
    compressedTexture = np.subtract(originalCompressed, compressedStructure)

    # process target data
    originalTarget = original / 255.0
    targetStructure = l0_gradient_minimization_2d(originalTarget, lmd=L0_GRADIENT_MIN_LAMDA,
                                                  beta_max=L0_GRADIENT_MIN_BETA_MAX)
    targetStructure = np.clip(targetStructure, a_min=0.0, a_max=1.0)
    targetTexture = np.subtract(originalTarget, targetStructure)

    originalTarget = originalTarget.astype('float32')
    originalCompressed = originalCompressed.astype('float32')
    compressedStructure = compressedStructure.astype('float32')
    compressedTexture = compressedTexture.astype('float32')
    targetStructure = targetStructure.astype('float32')
    targetTexture = targetTexture.astype('float32')

    originalTarget.tofile(out_dir + str(count) + ".original.ndarray")
    originalCompressed.tofile(out_dir + str(count) + ".compressed.ndarray")
    compressedStructure.tofile(out_dir + str(count) + ".compressed_structure.ndarray")
    compressedTexture.tofile(out_dir + str(count) + ".compressed_texture.ndarray")
    targetStructure.tofile(out_dir + str(count) + ".target_structure.ndarray")
    targetTexture.tofile(out_dir + str(count) + ".target_texture.ndarray")


def saveImages(img, out_dir, count):
    buffer = BytesIO()
    pil_img = Image.fromarray(img)
    pil_img.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    pil_img = Image.open(buffer)
    pil_img = np.asarray(pil_img)

    if STRRN_PREPROCESSING:
        STRRN_processing_and_save(img, pil_img, out_dir, count)
    else:
        originalTarget = img / 255.0
        originalCompressed = pil_img / 255.0

        originalTarget = originalTarget.astype('float32')
        originalCompressed = originalCompressed.astype('float32')

        originalTarget.tofile(out_dir + str(count) + ".original.ndarray")
        originalCompressed.tofile(out_dir + str(count) + ".compressed.ndarray")


def preprocess():
    # find all files
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    for r, d, f in os.walk(DATASET_PATH):
        images = 0
        for file in f:
            if FILE_SUFFIX in file:

                imgs = []

                # open file and augment data
                image_file = Image.open(r + "/" + file)
                imgs.append(np.asarray(image_file))
                if AUGMENT_IMAGES:
                    imgs.append(np.asarray(image_file.rotate(90, expand=True)))
                    imgs.append(np.asarray(image_file.rotate(180, expand=True)))
                    imgs.append(np.asarray(image_file.rotate(270, expand=True)))
                    imgs.append(np.asarray(image_file.transpose(Image.FLIP_LEFT_RIGHT)))
                    imgs.append(np.asarray(image_file.transpose(Image.FLIP_TOP_BOTTOM)))

                out_dir = OUTPUT_PATH + os.path.splitext(os.path.basename(file))[0] + "/"
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)

                imgs = np.asarray(imgs)

                # spit images into tiles

                count = 0
                for image in imgs:
                    x_stride = 0
                    y_stride = 0

                    if SEGMENT_IMAGES:
                        while x_stride < image.shape[0] - PATCH_SIZE - 1:
                            while y_stride < image.shape[1] - PATCH_SIZE - 1:
                                tmp = image[x_stride:(x_stride + PATCH_SIZE), y_stride:(y_stride + PATCH_SIZE), ...]

                                saveImages(tmp, out_dir, count)

                                count += 1

                                y_stride += STRIDE
                            x_stride += STRIDE
                    else:
                        saveImages(image, out_dir, count)
                        count += 1

                print("Finished processing image " + str(images) + " ", end="\r")
                images += 1

    print("Finished all pre-processing")


preprocess()
