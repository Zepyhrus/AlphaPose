import math
import numpy as np

import skimage
import cv2

def random_noise(img):
    modes = [
        "gaussian",
        "localvar",
        "poisson",
        "salt",
        "pepper",
        "s&p",
        "speckle",
        None
    ]
    mode = modes[np.random.randint(len(modes))]

    if mode is None:
        return img
    else:
        img_noised = skimage.util.random_noise(img, mode=mode)

        return (img_noised*255).astype(np.uint8)


def random_blur(img):
    """
    Random blur image
    """
    modes = ['gaussian', 'median', None ]
    mode = modes[np.random.randint(len(modes))]
    factor = img.shape[0] // 32
    ksize = 2 * np.random.randint(factor) + 1    # random kernel size

    if mode is None:
        return img
    elif mode == 'gaussian':
        return cv2.GaussianBlur(img, (ksize, ksize), 1)
    elif mode == 'median':
        return cv2.medianBlur(img, ksize)
    else:
        raise 'wrong blur mode!'



def random_brightness(img):
    """
    Random adjust image brightness and contrast
    """
    gf = 0.3
    gamma = np.clip(np.random.randn() * gf + 1, 1/(1+gf*3), 1+gf*3)

    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype(np.uint8)

    return cv2.LUT(img, table)


