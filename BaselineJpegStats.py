import tensorflow as tf
import numpy as np

from modules.Dataset import JPEGDataset, preprocessDataForSTRRN

ds = JPEGDataset('test', batch_size=1)

total_psnr_structure = 0.0
total_ssim_structure = 0.0
total_psnr_texture = 0.0
total_ssim_texture = 0.0
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
