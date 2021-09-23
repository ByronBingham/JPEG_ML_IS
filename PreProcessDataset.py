from PIL import Image

import os
import numpy as np

DATASET_PATH = 'x:/div2k_dataset/downloads/extracted/'
OUTPUT_PATH = 'x:/div2k_dataset/preprocessed/'

PATCH_SIZE = 31
STRIDE = 21


def preprocess():
    # find all files
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    for r, d, f in os.walk(DATASET_PATH):
        for file in f:
            if '.png' in file:

                # open file and augment data
                augImg = []
                image_file = Image.open(r + "/" + file)
                augImg.append(np.asarray(image_file))
                augImg.append(np.asarray(image_file.rotate(90, expand=True)))
                augImg.append(np.asarray(image_file.rotate(180, expand=True)))
                augImg.append(np.asarray(image_file.rotate(270, expand=True)))
                augImg.append(np.asarray(image_file.transpose(Image.FLIP_LEFT_RIGHT)))
                augImg.append(np.asarray(image_file.transpose(Image.FLIP_TOP_BOTTOM)))

                # augImg = np.asarray(augImg)

                aug_dir = OUTPUT_PATH + os.path.splitext(os.path.basename(file))[0] + "/"
                os.mkdir(aug_dir)

                # spit images into tiles
                count = 0
                for image in augImg:
                    x_stride = 0
                    y_stride = 0
                    while x_stride < image.shape[0] - PATCH_SIZE - 1:
                        while y_stride < image.shape[1] - PATCH_SIZE - 1:
                            tmp = image[x_stride:(x_stride + PATCH_SIZE), y_stride:(y_stride + PATCH_SIZE), ...]
                            Image.fromarray(tmp).save(fp=aug_dir + str(count) + ".png", format='PNG')

                            count += 1
                            y_stride += STRIDE

                        x_stride += STRIDE


preprocess()
