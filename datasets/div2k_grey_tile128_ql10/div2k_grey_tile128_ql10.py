"""div2k_preprocessed dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

from PIL import Image
from Lib.pathlib import Path

_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

_CITATION = """
"""


class Div2k_grey_tile128_ql10(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for div2k_tile128 dataset."""

    VERSION = tfds.core.Version('1.2.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
        '1.1.0': 'Switched to saving data as png',
        '1.2.0': 'Optimized texture data'
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'original': tfds.features.Tensor(shape=(None, None, 1), dtype=tf.dtypes.float32, encoding='zlib'),
                'target_structure': tfds.features.Tensor(shape=(None, None, 1), dtype=tf.dtypes.float32,
                                                         encoding='zlib'),
                'target_texture': tfds.features.Tensor(shape=(None, None, 1), dtype=tf.dtypes.float32, encoding='zlib'),
                'compressed_structure': tfds.features.Tensor(shape=(None, None, 1), dtype=tf.dtypes.float32,
                                                             encoding='zlib'),
                'compressed_texture': tfds.features.Tensor(shape=(None, None, 1), dtype=tf.dtypes.float32,
                                                           encoding='zlib'),
                'compressed': tfds.features.Tensor(shape=(None, None, 1), dtype=tf.dtypes.float32, encoding='zlib')
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=(None),  # Set to `None` to disable
            homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        path_train = Path("e:/datasets/div2k_dataset/greyscale_preprocessed_tile128_QL10/train")
        path_validation = Path("e:/datasets/div2k_dataset/greyscale_preprocessed_tile128_QL10/validation")

        return {
            'train': self._generate_examples(path_train),
            'validation': self._generate_examples(path_validation)
        }

    def _generate_examples(self, path):
        """Yields examples."""
        for f in path.glob('*/*original.png'):
            original = np.asarray(Image.open(str(f)))

            baseName = str(f).replace("original.png", "")

            target_structure = np.asarray(Image.open(baseName + "target_structure.png"))
            compressed_structure = np.asarray(Image.open(baseName + "compressed_structure.png"))
            compressed = np.asarray(Image.open(baseName + "compressed.png"))

            original = (original / 255.0).astype('float32')
            target_structure = (target_structure / 255.0).astype('float32')
            compressed_structure = (compressed_structure / 255.0).astype('float32')
            compressed = (compressed / 255.0).astype('float32')

            original = np.expand_dims(original, axis=-1)
            target_structure = np.expand_dims(target_structure, axis=-1)
            compressed_structure = np.expand_dims(compressed_structure, axis=-1)
            compressed = np.expand_dims(compressed, axis=-1)

            target_texture = original - target_structure
            compressed_texture = compressed - compressed_structure

            yield str(f), {
                    'original': original,
                    'target_structure': target_structure,
                    'target_texture': target_texture,
                    'compressed_structure': compressed_structure,
                    'compressed_texture': compressed_texture,
                    'compressed': compressed
                }
