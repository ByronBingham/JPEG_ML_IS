"""urban100_dataset_4x dataset."""

import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

from PIL import Image
from Lib.pathlib import Path

_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

_CITATION = """
"""


class Urban100Dataset4x(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for urban100_dataset_4x dataset."""

    VERSION = tfds.core.Version('1.1.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
        '1.1.0': 'Added difference feature.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'original': tfds.features.Tensor(shape=(None, None, 3), dtype=tf.dtypes.float32, encoding='zlib'),
                'target_structure': tfds.features.Tensor(shape=(None, None, 3), dtype=tf.dtypes.float32,
                                                         encoding='zlib'),
                'target_texture': tfds.features.Tensor(shape=(None, None, 3), dtype=tf.dtypes.float32, encoding='zlib'),
                'compressed_structure': tfds.features.Tensor(shape=(None, None, 3), dtype=tf.dtypes.float32,
                                                             encoding='zlib'),
                'compressed_texture': tfds.features.Tensor(shape=(None, None, 3), dtype=tf.dtypes.float32,
                                                           encoding='zlib'),
                'compressed': tfds.features.Tensor(shape=(None, None, 3), dtype=tf.dtypes.float32, encoding='zlib'),
                'diff': tfds.features.Tensor(shape=(None, None, 3), dtype=tf.dtypes.float32, encoding='zlib')
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

        path = Path("e:/datasets/urban100_dataset/preprocessed/image_SRF_4")

        return {
            'test': self._generate_examples(path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        for f in path.glob('*/*original.npy'):
            original = np.load(str(f))

            baseName = str(f).replace("original.npy", "")

            target_structure = np.load(baseName + "target_structure.npy")
            target_texture = np.load(baseName + "target_texture.npy")
            compressed_structure = np.load(baseName + "compressed_structure.npy")
            compressed_texture = np.load(baseName + "compressed_texture.npy")
            compressed = np.load(baseName + "compressed.npy")
            diff = np.load(baseName + "diff.npy")

            yield str(f), {
                'original': original,
                'target_structure': target_structure,
                'target_texture': target_texture,
                'compressed_structure': compressed_structure,
                'compressed_texture': compressed_texture,
                'compressed': compressed,
                'diff': diff
            }
