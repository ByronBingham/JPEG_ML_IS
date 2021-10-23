import shutil
from io import BytesIO

from PIL import Image
from multiprocessing import Pool

import os
import numpy as np

from L0GradientMin.l0_gradient_minimization import l0_gradient_minimization_2d
from modules.NNConfig import L0_GRADIENT_MIN_LAMDA, L0_GRADIENT_MIN_BETA_MAX, JPEG_QUALITY

DATASET_PATH = 'e:/datasets/div2k_dataset/downloads/extracted/'
OUTPUT_PATH = 'e:/datasets/div2k_dataset/preprocessed/tile128/'

FILE_SUFFIX = '.png'
DIFF_FILE_SUFFIX = '.original.npy'

PATCH_SIZE = 128
STRIDE = 87

SEGMENT_IMAGES = True
AUGMENT_IMAGES = True
STRRN_PREPROCESSING = True

SKIP = -1


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

    np.save(out_dir + str(count) + ".original", originalTarget)
    np.save(out_dir + str(count) + ".compressed", originalCompressed)
    np.save(out_dir + str(count) + ".compressed_structure", compressedStructure)
    np.save(out_dir + str(count) + ".compressed_texture", compressedTexture)
    np.save(out_dir + str(count) + ".target_structure", targetStructure)
    np.save(out_dir + str(count) + ".target_texture", targetTexture)


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

        np.save(out_dir + str(count) + ".original", originalTarget)
        np.save(out_dir + str(count) + ".compressed", originalCompressed)


def process(r, file, image_num):
    imgs = []
    out_dir = OUTPUT_PATH + os.path.splitext(os.path.basename(file))[0]

    # open file and augment data
    if os.path.exists(out_dir):
        return
    if os.path.exists(out_dir + "_tmp"):
        shutil.rmtree(out_dir + "_tmp")

    image_file = Image.open(r + "/" + file)
    imgs.append(np.asarray(image_file))
    if AUGMENT_IMAGES:
        imgs.append(np.asarray(image_file.rotate(90, expand=True)))
        imgs.append(np.asarray(image_file.rotate(180, expand=True)))
        imgs.append(np.asarray(image_file.rotate(270, expand=True)))
        imgs.append(np.asarray(image_file.transpose(Image.FLIP_LEFT_RIGHT)))
        imgs.append(np.asarray(image_file.transpose(Image.FLIP_TOP_BOTTOM)))

    if not os.path.exists(out_dir + "_tmp"):
        os.mkdir(out_dir + "_tmp")

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

                    saveImages(tmp, out_dir + "_tmp/", count)

                    count += 1

                    y_stride += STRIDE
                x_stride += STRIDE
        else:
            saveImages(image, out_dir + "_tmp/", count)
            count += 1

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

    files = []
    for r, d, f in os.walk(DATASET_PATH):
        image_num = 0
        for file in f:
            if image_num < SKIP:
                image_num += 1
                continue

            if FILE_SUFFIX in file:
                files.append((r, file, image_num))
                image_num += 1

    with Pool(processes=15, maxtasksperchild=4) as p:
        print(p.starmap(func=process, iterable=files))

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

    with Pool(processes=15, maxtasksperchild=4) as p:
        p.starmap(func=diffProcess, iterable=files)

    p.close()
    p.join()
    print("Finished all pre-processing")


if __name__ == '__main__':
    # preprocess()
    preProcessDiffs()
