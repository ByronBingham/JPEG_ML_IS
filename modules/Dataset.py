import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import random as rand
import datasets.urban100_dataset

from PIL import Image
from io import BytesIO
from L0GradientMin.l0_gradient_minimization import l0_gradient_minimization_2d

from modules.NNConfig import DATASET_PREFETCH, JPEG_QUALITY, L0_GRADIENT_MIN_LAMDA, \
    L0_GRADIENT_MIN_BETA_MAX, TRAINING_DATASET, DATASETS_DIR

BATCH_COMPRESSED = 0
BATCH_TARGET = 1
BATCH_PAD_MASK = 2


class JPEGDataset(object):

    def __init__(self, dataset_type, batch_size):
        self.batch_size = batch_size

        if dataset_type == 'train':
            self.ds = tf.keras.utils.image_dataset_from_directory(directory=TRAINING_DATASET, batch_size=1,
                                                                  image_size=(31, 31), validation_split=0.2,
                                                                  subset='training', seed=int(rand.random() * (2 ^ 64)))
        elif dataset_type == 'validation':
            self.ds = tf.keras.utils.image_dataset_from_directory(directory=TRAINING_DATASET, batch_size=1,
                                                                  image_size=(31, 31), validation_split=0.2,
                                                                  subset='validation',
                                                                  seed=int(rand.random() * (2 ^ 64)))
        elif dataset_type == 'test':
            self.ds = tfds.load('urban100_dataset', data_dir=DATASETS_DIR)['train']

        self.ds = self.ds.prefetch(DATASET_PREFETCH)
        self.ds_iter = iter(self.ds)

        self.dataset_type = dataset_type

        # DEBUG
        self.test = 0

    def __iter__(self):
        return self

    def __next__(self):
        target_images = []
        compressed_images = []
        i = 0

        '''
        # debug code
        if self.test >= 1:
            raise StopIteration
        '''

        for e in range(self.batch_size):
            # add original image to targets
            example = next(self.ds_iter)
            if example is None:
                raise StopIteration
            if self.dataset_type == 'test':
                img = np.asarray(example['image'])
            else:
                img = np.asarray(example[0][0])
            img = img.astype('float32')
            img = img / 255.0
            target_images.append(img)

            # compress image and add to inputs
            if self.dataset_type == 'test':
                tmp = np.asarray(example['image']).astype('uint8')
            else:
                tmp = np.asarray(example[0][0]).astype('uint8')
            pil_img = Image.fromarray(tmp)
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG", quality=JPEG_QUALITY)
            pil_img = Image.open(buffer)
            pil_img = np.asarray(pil_img)
            pil_img = pil_img.astype('float32')
            pil_img = pil_img / 255.0
            compressed_images.append(pil_img)

            if i >= self.batch_size - 1:
                break

            i = i + 1

        # pad all images to the largest width and height
        max_w = 0
        max_h = 0
        for img in target_images:
            if len(img) > max_w:
                max_w = len(img)
            if len(img[0]) > max_h:
                max_h = len(img[0])

        '''
            TF only takes tensors that have the same dimensions. Since images will be different dimensions
            we need to pad them to all be the same dimensions for each batch.
        '''
        pad_mask = []
        tmp_target = []
        for img in target_images:
            mask = np.ones(shape=img.shape)
            pad = [(0, max_w - len(img)), (0, max_h - len(img[0])), (0, 0)]
            mask = np.pad(mask, pad_width=pad, constant_values=0)
            mask = mask.astype('float32')
            img = np.pad(img, pad_width=pad, constant_values=0)

            tmp_target.append(img)
            pad_mask.append(mask)

        tmp_compressed = []
        for img in compressed_images:
            pad = [(0, max_w - len(img)), (0, max_h - len(img[0])), (0, 0)]
            img = np.pad(img, pad_width=pad, constant_values=0)

            tmp_compressed.append(img)

        target_images = tmp_target
        compressed_images = tmp_compressed

        target_images = np.asarray(target_images)
        compressed_images = np.asarray(compressed_images)
        pad_mask = np.asarray(pad_mask)

        # DEBUG
        self.test += 1
        return compressed_images, target_images, pad_mask


def preprocessInputsForSTRRN(batch_compressed):
    '''

    :param batch_compressed: compressed images only (no targets or masks)
    :return:
    '''
    batch_structure = []
    batch_texture = []

    for b in range(batch_compressed.shape[0]):
        originalCompressed = batch_compressed[b]
        imageStructure = l0_gradient_minimization_2d(originalCompressed, lmd=L0_GRADIENT_MIN_LAMDA,
                                                     beta_max=L0_GRADIENT_MIN_BETA_MAX)
        imageTexture = np.subtract(originalCompressed, imageStructure)

        batch_structure.append(imageStructure)
        batch_texture.append(imageTexture)

    batch_structure = np.asarray(batch_structure)
    batch_texture = np.asarray(batch_texture)

    return batch_structure, batch_texture
