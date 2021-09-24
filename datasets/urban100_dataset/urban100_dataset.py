"""urban100_dataset dataset."""

import tensorflow_datasets as tfds
import numpy as np

from PIL import Image

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
                'image': tfds.features.Image(shape=(None, None, 3))
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
        path = dl_manager.download_and_extract('https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip')

        return {
            'train': self._generate_examples(path / 'image_SRF_2'),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(urban100_dataset): Yields (key, example) tuples from the dataset
        for f in path.glob('*HR.png'):
            img = Image.open(str(f))
            img = np.asarray(img)
            yield str(f), {
                'image': img,
            }
