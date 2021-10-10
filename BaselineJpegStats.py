import tensorflow as tf
import numpy as np

from modules.Dataset import JPEGDataset, preprocessDataForSTRRN

ds = JPEGDataset('test', batch_size=1)

total_psnr_structure = 0.0
total_ssim_structure = 0.0
total_psnr_texture = 0.0
total_ssim_texture = 0.0
total_psnr =0.0
total_ssim = 0.0

examples = 0
for example in ds:

    structure_in, texture_in, structure_target, texture_target = preprocessDataForSTRRN(example)

    total_psnr_structure += np.average(tf.image.psnr(structure_in, structure_target, max_val=1.0))
    total_psnr_texture += np.average(tf.image.psnr(texture_in, texture_target, max_val=1.0))

    total_ssim_structure += np.average(tf.image.ssim(structure_in, structure_target, max_val=1.0))
    total_ssim_texture += np.average(tf.image.ssim(texture_in, texture_target, max_val=1.0))

    total_psnr += np.average(tf.image.psnr(example[0], example[1], max_val=1.0))
    total_ssim += np.average(tf.image.ssim(example[0], example[1], max_val=1.0))
    examples += 1

print("Average PSNR(structure) of test dataset: " + str(total_psnr_structure / examples))
print("Average SSIM(structure) of test dataset: " + str(total_ssim_structure / examples))
print("Average PSNR(texture) of test dataset: " + str(total_psnr_texture / examples))
print("Average SSIM(texture) of test dataset: " + str(total_ssim_texture / examples))
print("Average PSNR of test dataset: " + str(total_psnr / examples))
print("Average SSIM of test dataset: " + str(total_ssim / examples))
