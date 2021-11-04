"""bsds500_test dataset."""

import tensorflow_datasets as tfds
from . import bsds500_test


class Bsds500TestTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for bsds500_test dataset."""
  # TODO(bsds500_test):
  DATASET_CLASS = bsds500_test.Bsds500Test
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
