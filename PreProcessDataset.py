import shutil
from io import BytesIO

from PIL import Image, ImageOps
from multiprocessing import Pool
from Lib.pathlib import Path

import os
import numpy as np

from L0GradientMin.l0_gradient_minimization import l0_gradient_minimization_2d
from modules.NNConfig import L0_GRADIENT_MIN_LAMDA, L0_GRADIENT_MIN_BETA_MAX

DATASET_PATH = Path('e:/datasets/urban100_dataset/image_SRF_4')
OUTPUT_PATH = 'e:/datasets/urban100_dataset/greyscale_preprocessed/'

FILE_SUFFIX = '*HR.png'
DIFF_FILE_SUFFIX = '.original.npy'

SAVE_FORMAT = 'npy'  # 'png'
OUTPUT_SEPARATE_DIR = False
SUB_DIR_FOR_EACH_EXAMPLE = True

PATCH_SIZE = 128
STRIDE = 87

SEGMENT_IMAGES = False
AUGMENT_IMAGES = False
STRRN_PREPROCESSING = True
INCLUDE_DIFFS = False
CONVERT_TO_GREYSCALE = True

JPEG_QUALITY = 10

SKIP = -1
PROCESSES = 15
MAX_TASKS_PER_CHILD = 1


def STRRN_processing_and_save(original, compressed, out_dir, count):
    # process compressed data (input)
    originalCompressed = compressed / 255.0
    compressedStructure = l0_gradient_minimization_2d(originalCompressed,
                                                      lmd=L0_GRADIENT_MIN_LAMDA,
                                                      beta_max=L0_GRADIENT_MIN_BETA_MAX)
    compressedTexture = np.subtract(originalCompressed, compressedStructure)

    # process target data
    originalTarget = original / 255.0
    targetStructure = l0_gradient_minimization_2d(originalTarget, lmd=L0_GRADIENT_MIN_LAMDA,
                                                  beta_max=L0_GRADIENT_MIN_BETA_MAX)
    targetTexture = np.subtract(originalTarget, targetStructure)

    originalTarget = originalTarget.astype('float32')
    originalCompressed = originalCompressed.astype('float32')
    compressedStructure = compressedStructure.astype('float32')
    compressedTexture = compressedTexture.astype('float32')
    targetStructure = targetStructure.astype('float32')
    targetTexture = targetTexture.astype('float32')

    np.save(out_dir + str(count) + ".original" + "QL" + str(JPEG_QUALITY), originalTarget)
    np.save(out_dir + str(count) + ".compressed" + "QL" + str(JPEG_QUALITY), originalCompressed)
    np.save(out_dir + str(count) + ".compressed_structure" + "QL" + str(JPEG_QUALITY), compressedStructure)
    np.save(out_dir + str(count) + ".compressed_texture" + "QL" + str(JPEG_QUALITY), compressedTexture)
    np.save(out_dir + str(count) + ".target_structure" + "QL" + str(JPEG_QUALITY), targetStructure)
    np.save(out_dir + str(count) + ".target_texture" + "QL" + str(JPEG_QUALITY), targetTexture)


def saveImages(img, out_dir, count, img_num):
    buffer = BytesIO()
    pil_img = Image.fromarray(img)
    pil_img.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    pil_img = Image.open(buffer)

    if CONVERT_TO_GREYSCALE:
        img = Image.fromarray(img)
        img = ImageOps.grayscale(img)
        pil_img = ImageOps.grayscale(pil_img)

    pil_img = np.asarray(pil_img)
    img = np.asarray(img)

    if STRRN_PREPROCESSING:
        STRRN_processing_and_save(img, pil_img, out_dir, count)
    else:
        if SAVE_FORMAT == 'npy':
            originalTarget = img / 255.0
            originalCompressed = pil_img / 255.0

            originalTarget = originalTarget.astype('float32')
            originalCompressed = originalCompressed.astype('float32')

            np.save(out_dir + str(img_num) + "_" + str(count) + "QL" + str(JPEG_QUALITY) + ".original", originalTarget)
            np.save(out_dir + str(img_num) + "_" + str(count) + "QL" + str(JPEG_QUALITY) + ".compressed",
                    originalCompressed)
        elif SAVE_FORMAT == 'png':
            originalTarget = Image.fromarray(img)
            originalCompressed = Image.fromarray(pil_img)

            originalTarget.save(
                out_dir + "original/" + str(img_num) + "_" + "QL" + str(JPEG_QUALITY) + "_" + str(count) + ".png")
            originalCompressed.save(
                out_dir + "compressed/" + str(img_num) + "_" + "QL" + str(JPEG_QUALITY) + "_" + str(count) + ".png")


def process(file, image_num):
    imgs = []
    if SUB_DIR_FOR_EACH_EXAMPLE:
        out_dir = OUTPUT_PATH + str(image_num)
    else:
        out_dir = OUTPUT_PATH

    if SUB_DIR_FOR_EACH_EXAMPLE:
        # open file and augment data
        if os.path.exists(out_dir):
            return
        if os.path.exists(out_dir + "_tmp"):
            shutil.rmtree(out_dir + "_tmp")

    image_file = Image.open(str(file))
    tmp = np.asarray(image_file)[..., 0:3]  # Remove alpha channel
    imgs.append(tmp)

    if AUGMENT_IMAGES:
        imgs.append(np.asarray(image_file.rotate(90, expand=True)))
        imgs.append(np.asarray(image_file.rotate(180, expand=True)))
        imgs.append(np.asarray(image_file.rotate(270, expand=True)))
        imgs.append(np.asarray(image_file.transpose(Image.FLIP_LEFT_RIGHT)))
        imgs.append(np.asarray(image_file.transpose(Image.FLIP_TOP_BOTTOM)))

    if SUB_DIR_FOR_EACH_EXAMPLE:
        if not os.path.exists(out_dir + "_tmp"):
            os.mkdir(out_dir + "_tmp")
    else:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    imgs = np.asarray(imgs)

    count = 0
    for image in imgs:
        x_stride = 0
        y_stride = 0

        # spit images into tiles
        if SEGMENT_IMAGES:
            while x_stride < image.shape[0] - PATCH_SIZE - 1:
                while y_stride < image.shape[1] - PATCH_SIZE - 1:
                    tmp = image[x_stride:(x_stride + PATCH_SIZE), y_stride:(y_stride + PATCH_SIZE), ...]

                    if SUB_DIR_FOR_EACH_EXAMPLE:
                        saveImages(tmp, out_dir + "_tmp/", count, image_num)
                    else:
                        saveImages(tmp, out_dir, count, image_num)

                    count += 1

                    y_stride += STRIDE
                x_stride += STRIDE
        else:
            if SUB_DIR_FOR_EACH_EXAMPLE:
                saveImages(image, out_dir + "_tmp/", count, image_num)
            else:
                saveImages(image, out_dir, count, image_num)

            count += 1
    if SUB_DIR_FOR_EACH_EXAMPLE:
        os.rename(src=out_dir + "_tmp", dst=out_dir)
    print("Finished processing image " + str(image_num))


ready_list = []


def callback(arg):
    global ready_list
    ready_list.append(arg)


def preprocess():
    # find all files
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    if OUTPUT_SEPARATE_DIR:
        if not os.path.exists(OUTPUT_PATH + "original"):
            os.mkdir(OUTPUT_PATH + "original")
        if not os.path.exists(OUTPUT_PATH + "compressed"):
            os.mkdir(OUTPUT_PATH + "compressed")

    files = []
    image_num = 0
    for f in DATASET_PATH.glob(FILE_SUFFIX):
        if image_num < SKIP:
            image_num += 1
            continue
        else:
            files.append((f, image_num))
            image_num += 1

    with Pool(processes=15, maxtasksperchild=4) as p:
        p.starmap(func=process, iterable=files)

    p.close()
    p.join()
    print("Finished all pre-processing")


def diffProcess(r, file, image_num):
    baseFileName = file.replace('.original.npy', '')

    original = np.load(r + "/" + file)
    compressed = np.load(r + "/" + baseFileName + '.compressed.npy')

    diff = original - compressed

    if os.path.exists(r + "/" + baseFileName + '.diff.npy'):
        os.remove(r + "/" + baseFileName + '.diff.npy')

    np.save(file=r + "/" + baseFileName + '.diff', arr=diff)

    print("Finished processing image " + str(image_num))


def preProcessDiffs():
    # find all files
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    files = []
    image_num = 0
    for r, d, f in os.walk(OUTPUT_PATH):
        for file in f:
            if image_num < SKIP:
                image_num += 1
                continue

            if DIFF_FILE_SUFFIX in file:
                files.append((r, file, image_num))
                image_num += 1

    print("Total files found: " + str(image_num))

    with Pool(processes=PROCESSES, maxtasksperchild=MAX_TASKS_PER_CHILD) as p:
        p.starmap(func=diffProcess, iterable=files)

    p.close()
    p.join()
    print("Finished all pre-processing")


if __name__ == '__main__':
    preprocess()
    # preProcessDiffs()
