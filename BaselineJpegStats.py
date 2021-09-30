import tensorflow as tf
import numpy as np

from modules.Dataset import JPEGDataset

ds = JPEGDataset('test', batch_size=1)

total_psnr = 0.0
total_ssim = 0.0

examples = 0
for example in ds:

    total_psnr += np.average(tf.image.psnr(example[0], example[1], max_val=1.0))
    total_ssim += np.average(tf.image.ssim(example[0], example[1], max_val=1.0))
    examples += 1

print("Average PSNR of test dataset: " + str(total_psnr / examples))
print("Average SSIM of test dataset: " + str(total_ssim / examples))
