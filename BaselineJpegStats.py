import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

from modules.Dataset import JPEGDataset, preprocessDataForSTRRN

ds = JPEGDataset('test', batch_size=1, dataset_name="bsd_s500_test_dataset")

total_psnr_structure = 0.0
total_ssim_structure = 0.0
total_psnr_texture = 0.0
total_ssim_texture = 0.0
total_psnr_diff = 0.0
total_ssim_diff = 0.0
total_strrn_psnr = 0.0
total_strrn_ssim = 0.0

examples = 0
for example in ds:
    original = example['original']
    target_structure = example['target_structure']
    target_texture = example['target_texture']
    compressed_structure = example['compressed_structure']
    compressed_texture = example['compressed_texture']
    compressed = example['compressed']

    '''
    org_mono = np.asarray(original[0]) * 255.0
    org_mono = org_mono.astype('uint8')
    org_mono = Image.fromarray(org_mono)
    org_mono = ImageOps.grayscale(org_mono)
    org_mono = np.asarray(org_mono) / 255.0
    org_mono = np.expand_dims(org_mono, axis=-1)

    cmp_mono = np.asarray(compressed[0]) * 255.0
    cmp_mono = cmp_mono.astype('uint8')
    cmp_mono = Image.fromarray(cmp_mono)
    cmp_mono = ImageOps.grayscale(cmp_mono)
    cmp_mono = np.asarray(cmp_mono) / 255.0
    cmp_mono = np.expand_dims(cmp_mono, axis=-1)
    '''

    total_psnr_structure += np.average(tf.image.psnr(compressed_structure, target_structure, max_val=1.0))
    total_ssim_structure += np.average(tf.image.ssim(compressed_structure, target_structure, max_val=1.0))

    total_psnr_texture += np.average(tf.image.psnr(compressed_texture, target_texture, max_val=1.0))
    total_ssim_texture += np.average(tf.image.ssim(compressed_texture, target_texture, max_val=1.0))

    total_strrn_psnr += np.average(tf.image.psnr(compressed, original, max_val=1.0))
    total_strrn_ssim += np.average(tf.image.ssim(compressed, original, max_val=1.0))

    examples += 1

print("Average PSNR(structure) of test dataset: " + str(total_psnr_structure / examples))
print("Average SSIM(structure) of test dataset: " + str(total_ssim_structure / examples))
print("Average PSNR(texture) of test dataset: " + str(total_psnr_texture / examples))
print("Average SSIM(texture) of test dataset: " + str(total_ssim_texture / examples))
print("Average PSNR(STRRRN) of test dataset: " + str(total_strrn_psnr / examples))
print("Average SSIM(STRRRN) of test dataset: " + str(total_strrn_ssim / examples))
