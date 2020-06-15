import math
import numpy as np
from numpy import random

import skimage
import cv2


def random_pass(img):
    """
    Do nothing, random placeholder
    """
    return img

def random_noise(img):
    modes = [
        "gaussian",
        "localvar",
        "poisson",
        "salt",
        "pepper",
        "s&p",
        "speckle"
    ]
    mode = modes[np.random.randint(len(modes))]
    img_noised = skimage.util.random_noise(img, mode=mode)

    return (img_noised*255).astype(np.uint8)


def random_blur(img):
    """
    Random blur image
    """
    modes = ['gaussian', 'median']
    mode = modes[np.random.randint(len(modes))]
    factor = img.shape[0] // 64
    ksize = 2 * np.random.randint(factor//2, factor+1) + 1    # random kernel size


    if mode == 'gaussian':
        return cv2.GaussianBlur(img, (2*ksize+1, 2*ksize+1), 1)
    elif mode == 'median':
        return cv2.medianBlur(img, ksize)
    else:
        raise 'wrong blur mode!'



def random_brightness(img):
    """
    Random adjust image brightness and contrast
    """
    if np.random.rand() < 0.3:
        gf = 0.3
        gamma = np.clip(np.random.randn() * gf + 1, 1/(1+gf*3), 1+gf*3)

        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype(np.uint8)

        return cv2.LUT(img, table)
    else:
        img = img.astype(np.float32)
        ratio = img.mean() / 3

        aug = (np.random.rand() * 2 - 1) * ratio
        img += aug

        img = np.clip(img, 0, 255).astype(np.uint8)

        return img


def random_hsv(img):
    img = img.astype(np.float32)
    
    mode = random.randint(2)

    if mode == 1:
        if random.randint(2):
            alpha = random.uniform(0.5, 1.5)
            img *= alpha
            img = np.clip(img, 0, 255)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if random.randint(2):
        img[..., 1] *= random.uniform(0.5, 1.5)

    if random.randint(2):
        img[..., 0] += random.uniform(-18, 18)
        img[..., 0][img[..., 0] > 360] -= 360
        img[..., 0][img[..., 0] < 0] += 360

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    if mode == 0:
        if random.randint(2):
            alpha = random.uniform(0.5, 1.5)
            img *= alpha
            img = np.clip(img, 0, 255)

    if random.randint(2):
        img = img[..., random.permutation(3)]
    
    img = img.astype(np.uint8)

    return img