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


class Urban100_greyscale_ql10(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for div2k_tile128 dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'original': tfds.features.Tensor(shape=(None, None, 1), dtype=tf.dtypes.float32, encoding='zlib'),
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

        path_train = Path("e:/datasets/urban100_dataset/greyscale_preprocessed_QL10/")

        return {
            'test': self._generate_examples(path_train),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        for f in path.glob('*/*original.npy'):
            original = np.load(str(f))

            baseName = str(f).replace("original.npy", "")

            compressed = np.load(baseName + "compressed.npy")

            original = np.expand_dims(original, axis=-1)
            compressed = np.expand_dims(compressed, axis=-1)

            yield str(f), {
                'original': original,
                'compressed': compressed
            }
