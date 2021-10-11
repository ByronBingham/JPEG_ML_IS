import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import random as rand
import datasets.urban100_dataset
import datasets.div2k_preprocessed

from PIL import Image
from io import BytesIO
from L0GradientMin.l0_gradient_minimization import l0_gradient_minimization_2d

from modules.NNConfig import DATASET_PREFETCH, JPEG_QUALITY, L0_GRADIENT_MIN_LAMDA, \
    L0_GRADIENT_MIN_BETA_MAX, TRAINING_DATASET, DATASETS_DIR, DATASET_EARLY_STOP, TRAIN_EARLY_STOP, \
    VALIDATION_EARLY_STOP, TEST_EARLY_STOP, EVEN_PAD_DATA, TEST_BATCH_SIZE

BATCH_COMPRESSED = 0
BATCH_TARGET = 1
BATCH_PAD_MASK = 2


class JPEGDataset(object):

    def __init__(self, dataset_type, batch_size):
        self.batch_size = batch_size

        if dataset_type == 'train':
            self.ds = self.ds = tfds.load('div2k_preprocessed', data_dir=DATASETS_DIR, shuffle_files=True,
                                          split='train', batch_size=batch_size)
        elif dataset_type == 'validation':
            self.ds = self.ds = tfds.load('div2k_preprocessed', data_dir=DATASETS_DIR, shuffle_files=True,
                                          split='validation', batch_size=batch_size)
        elif dataset_type == 'test':
            self.ds = tfds.load('urban100_dataset', data_dir=DATASETS_DIR, batch_size=TEST_BATCH_SIZE)['test']

        self.ds = self.ds.prefetch(DATASET_PREFETCH)
        self.ds_iter = iter(self.ds)

        self.dataset_type = dataset_type

        self.batches = 0

    def __iter__(self):
        return self

    def __next__(self):
        target_images = []
        compressed_images = []
        i = 0

        if DATASET_EARLY_STOP:
            if self.dataset_type == 'train':
                if self.batches >= TRAIN_EARLY_STOP:
                    raise StopIteration
            if self.dataset_type == 'validation':
                if self.batches >= VALIDATION_EARLY_STOP:
                    raise StopIteration
            if self.dataset_type == 'test':
                if self.batches >= TEST_EARLY_STOP:
                    raise StopIteration

        batch = next(self.ds_iter)

        self.batches += 1

        return batch


def preprocessDataForSTRRN(batch):
    """

    :param batch: compressed and target data
    :return:
    """
    structure_in = []
    texture_in = []
    structure_target = []
    texture_target = []

    batch = np.asarray(batch)

    for b in range(batch.shape[1]):
        # process compressed data (input)
        originalCompressed = batch[0][b]
        compressedStructure = l0_gradient_minimization_2d(originalCompressed, lmd=L0_GRADIENT_MIN_LAMDA,
                                                          beta_max=L0_GRADIENT_MIN_BETA_MAX)
        compressedTexture = np.subtract(originalCompressed, compressedStructure)

        structure_in.append(compressedStructure)
        texture_in.append(compressedTexture)

        # process target data
        originalTarget = batch[1][b]
        targetStructure = l0_gradient_minimization_2d(originalTarget, lmd=L0_GRADIENT_MIN_LAMDA,
                                                      beta_max=L0_GRADIENT_MIN_BETA_MAX)
        targetTexture = np.subtract(originalTarget, targetStructure)

        structure_target.append(targetStructure)
        texture_target.append(targetTexture)

    structure_in = np.asarray(structure_in)
    texture_in = np.asarray(texture_in)
    structure_target = np.asarray(structure_target)
    texture_target = np.asarray(texture_target)

    structure_in = structure_in.astype('float32')
    texture_in = texture_in.astype('float32')
    structure_target = structure_target.astype('float32')
    texture_target = texture_target.astype('float32')

    return structure_in, texture_in, structure_target, texture_target


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
