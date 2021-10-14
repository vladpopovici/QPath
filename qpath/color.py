# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = 0.1
__all__ = ['rgb2ycbcr', 'ycbcr2rgb',
           'R_', 'G_', 'B_', 'RGBA2RGB']

"""
COLOR: a few functions to complement the COLOR module in scikit-image package.
@author: vlad
"""

import numpy as np
from skimage.util import img_as_uint
from .mask import binary_mask


def rgb2ycbcr(im):
    """
    RGB2YCBCR: converts an RGB image into YCbCr (YUV) color space.

    :param im: numpy.ndarray
      [m x n x 3] image
    """

    if im.ndim != 3:
        raise ValueError('Input image must be RGB.')
    h, w, c = im.shape
    if c != 3:
        raise ValueError('Input image must be a 3-channel (RGB) image.')

    if im.dtype != np.uint8:
        im = img_as_uint(im)

    ycc = np.array([[0.257, 0.439, -0.148],
                    [0.504, -0.368, -0.291],
                    [0.098, -0.071, 0.439]])

    im = im.reshape((h * w, c))

    r = np.dot(im, ycc).reshape((h, w, c))
    r[:, :, 0] += 16
    r[:, :, 1:3] += 128

    im_res = np.array(np.round(r), dtype=im.dtype)

    return im_res


def ycbcr2rgb(im):
    """
    YCBCR2RGB: converts an YCbCr (YUV) in RGB color space.

    :param im: numpy.ndarray
      [m x n x 3] image
    """

    if im.ndim != 3:
        raise ValueError('Input image must be YCbCr.')
    h, w, c = im.shape
    if c != 3:
        raise ValueError('Input image must be a 3-channel (YCbCr) image.')

    if im.dtype != np.uint8:
        im = img_as_uint(im)

    iycc = np.array([[1.164, 1.164, 1.164],
                     [0, -0.391, 2.018],
                     [1.596, -0.813, 0]])

    r = im.reshape((h * w, c))

    r[:, 0] -= 16.0
    r[:, 1:3] -= 128.0
    r = np.dot(r, iycc)
    r[r < 0] = 0
    r[r > 255] = 255
    r = np.round(r)
    # x = r[:,2]; r[:,2] = r[:,0]; r[:,0] = x

    im_res = np.array(r.reshape((h, w, c)), dtype=np.uint8)

    return im_res


def R_(_img):
    return _img[:, :, 0]


def G_(_img):
    return _img[:, :, 1]


def B_(_img):
    return _img[:, :, 2]


def RGBA2RGB(img: np.ndarray,
        with_masking: bool=True,
        nonzero_alpha_is_foreground:bool=True,
        alpha_cutoff: np.uint8=127,
        background_level: np.uint8 = 255) -> np.ndarray:
    """Removes the alpha channel with eventual masking. Many WSI use the
    alpha channel to store the mask for foreground.

    :param img: (nmupy.ndarray) a NumPy array with image data (m x n x 4),
        with channel ordering R,G,B,A.
    :param with_masking: (bool) if True, the masking from the alpha channel
        is applied to the rest of the channles
    :param nonzero_alpha_is_foreground: (bool) indicates that nonzero alpha
        values are indicating pixels of the foreground
    :param alpha_cutoff: (uint8) cutoff for alpha mask
    :param backgound_level: (uint8) sets all the pixels in background (according
        to alpha channel) to this value. Typically, is set to 255 indicating
        that background is white.
    :return: a new RGB image
    """
    new_img = img[...,:3].copy()
    if with_masking:
        mask = binary_mask(img[...,3].squeeze(),
            level=alpha_cutoff,
            mode = 'below' if nonzero_alpha_is_foreground else 'above')
        for k in np.arange(0,3):
            new_img[mask==1, k] = background_level

    return new_img
