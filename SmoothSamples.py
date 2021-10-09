import numpy as np

from L0GradientMin.l0_gradient_minimization import l0_gradient_minimization_2d
from modules.NNConfig import SAMPLE_IMAGES, L0_GRADIENT_MIN_LAMDA, L0_GRADIENT_MIN_BETA_MAX
from PIL import Image

for file in SAMPLE_IMAGES:
    # compressed images
    pil_img_c = Image.open("./sampleImages/" + file + ".png" + ".compressed.jpg")
    pil_img_c = np.asarray(pil_img_c)
    pil_img_c = pil_img_c / 255.0
    smoothed_img_c = l0_gradient_minimization_2d(pil_img_c, lmd=L0_GRADIENT_MIN_LAMDA,
                                               beta_max=L0_GRADIENT_MIN_BETA_MAX)
    smoothed_img_c = np.clip(smoothed_img_c, a_min=0.0, a_max=1.0)

    smoothed_img_c = smoothed_img_c * 255.0
    smoothed_img_c = smoothed_img_c.astype('uint8')
    out = Image.fromarray(smoothed_img_c)
    out.save("./sampleImages/" + file + ".png" + ".compressed.smoothed.png", format="PNG")

    # uncompressed images
    pil_img = Image.open("./sampleImages/" + file + ".png")
    pil_img = np.asarray(pil_img)
    pil_img = pil_img / 255.0
    smoothed_img = l0_gradient_minimization_2d(pil_img, lmd=L0_GRADIENT_MIN_LAMDA,
                                               beta_max=L0_GRADIENT_MIN_BETA_MAX)
    smoothed_img = np.clip(smoothed_img, a_min=0.0, a_max=1.0)

    smoothed_img = smoothed_img * 255.0
    smoothed_img = smoothed_img.astype('uint8')
    out = Image.fromarray(smoothed_img)
    out.save("./sampleImages/" + file + ".png" + ".smoothed.png", format="PNG")
