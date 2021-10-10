"""urban100_dataset dataset."""

import tensorflow_datasets as tfds
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


class Urban100Dataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for urban100_dataset dataset."""

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
                'original': tfds.features.Image(shape=(None, None, 3)),
                'target_structure': tfds.features.Image(shape=(None, None, 3)),
                'target_texture': tfds.features.Image(shape=(None, None, 3)),
                'compressed_structure': tfds.features.Image(shape=(None, None, 3)),
                'compressed_texture': tfds.features.Image(shape=(None, None, 3)),
                'compressed': tfds.features.Image(shape=(None, None, 3))
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
        # path = dl_manager.download_and_extract('https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip')

        path = Path("e:/datasets/urban100_dataset/preprocessed/image_SRF_2")

        return {
            'test': self._generate_examples(path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        for f in path.glob('*original.ndarray'):
            original = np.fromfile(str(f), dtype="float32")
            original = np.asarray(original)

            baseName = str(f).replace("original.ndarray", "")

            target_structure = np.fromfile(baseName + "target_structure.ndarray", dtype='float32')
            target_texture = np.fromfile(baseName + "target_texture.ndarray", dtype='float32')
            compressed_structure = np.fromfile(baseName + "compressed_structure.ndarray", dtype='float32')
            compressed_texture = np.fromfile(baseName + "compressed_texture.ndarray", dtype='float32')
            compressed = np.fromfile(baseName + "compressed.ndarray", dtype='float32')

            yield str(f), {
                'original': original,
                'target_structure': target_structure,
                'target_texture': target_texture,
                'compressed_structure': compressed_structure,
                'compressed_texture': compressed_texture,
                'compressed': compressed
            }
