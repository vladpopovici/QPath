# -*- coding: utf-8 -*-

# STAIN: stain deconvolution and normalization


#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################


__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = 0.1
__all__ = ['rgb2he',
           'rgb2he_macenko',
           'MacenkoNormalizer',
           'VahadaneNormalizer',
           'ReinhardNormalizer',
           'StainNormalizerFactory',
           'getNormalizer']

import numpy as np
from scipy.linalg import eig
from abc import ABC, abstractmethod
import numpy as np
import spams
import cv2 as cv

from skimage.exposure import rescale_intensity


def rgb2he(img):
    """Stain separation for H&E slides: estimate the H- and E- signal intensity
    and the residuals.

    Args:
        img (numpy.ndarray): a H x W x 3 image array

    Returns:
        3 numpy arrays of size H x W with signal scaled to [0,1] corresponding
        to estimated intensities of Haematoxylin, Eosine and background/resodual
        components.
    """
    # This implementation follows http://web.hku.hk/~ccsigma/color-deconv/color-deconv.html

    assert (img.ndim == 3)
    assert (img.shape[2] == 3)

    height, width, _ = img.shape

    img = -np.log((img + 1.0) / img.max())

    D = np.array([[ 1.92129515,  1.00941672, -2.34107612],
                  [-2.34500192,  0.47155124,  2.65616872],
                  [ 1.21495282, -0.99544467,  0.2459345 ]])

    rgb = img.swapaxes(2, 0).reshape((3, height*width))
    heb = np.dot(D, rgb)
    res_img = heb.reshape((3, width, height)).swapaxes(0, 2)

    return rescale_intensity(res_img[:,:,0], out_range=(0,1)), \
           rescale_intensity(res_img[:,:,1], out_range=(0,1)), \
           rescale_intensity(res_img[:,:,2], out_range=(0,1))


def rgb2he_macenko(img, D=None, alpha=1.0, beta=0.15, white=255.0,
                   return_deconvolution_matrix=False):
    """
    Performs stain separation from RGB images using the method in
    M Macenko, et al. "A method for normalizing histology slides for quantitative analysis",
    IEEE ISBI, 2009. dx.doi.org/10.1109/ISBI.2009.5193250

    Args:
        img (numpy.ndarray): RGB input image
        D (numpy.ndarray): a deconvolution matrix. If None, one will be computed from the image
        alpha (float): tolerance for pseudo-min/-max
        beta (float): OD threshold for transparent pixels
        white (float): white level (in each channel)
        return_deconvolution_matrix (bool): if True, the deconvolution matrix is also returned

    Returns:
        three 2d arrays for H-, E- and remainder channels, respectively.
        If return_deconvolution_matrix is True, the deconvolution matrix is also returned.
    """

    assert (img.ndim == 3)
    assert (img.shape[2] == 3)

    I = img.reshape((img.shape[0] * img.shape[1], 3))
    OD = -np.log((I + 1.0) / white)  # optical density

    if D is None:
        # the deconvolution matrix is not provided so one has to be estimated from the
        # image
        rows = (OD >= beta).all(axis=1)
        if not any(rows):
            # no rows with all pixels above the threshold
            raise RuntimeError('optical density below threshold')

        ODhat = OD[rows, :]  # discard transparent pixels

        u, V, _ = eig(np.cov(ODhat.T))
        idx = np.argsort(u)  # get a permutation to sort eigenvalues increasingly
        V = V[:, idx]        # sort eigenvectors
        theta = np.dot(ODhat, V[:, 1:3])  # project optical density onto the eigenvectors
                                          # corresponding to the largest eigenvalues
        phi = np.arctan2(theta[:,1], theta[:,0])
        min_phi, max_phi = np.percentile(phi, [alpha, 100.0-alpha], axis=None)

        u1 = np.dot(V[:,1:3], np.array([[np.cos(min_phi)],[np.sin(min_phi)]]))
        u2 = np.dot(V[:,1:3], np.array([[np.cos(max_phi)],[np.sin(max_phi)]]))

        if u1[0] > u2[0]:
            D = np.hstack((u1, u2)).T
        else:
            D = np.hstack((u2, u1)).T

        D = np.vstack((D, np.cross(D[0,],D[1,])))
        D = D / np.reshape(np.repeat(np.linalg.norm(D, axis=1), 3), (3,3), order=str('C'))

    img_res = np.linalg.solve(D.T, OD.T).T
    img_res = np.reshape(img_res, img.shape, order=str('C'))

    if not return_deconvolution_matrix:
        D = None

    return rescale_intensity(img_res[:,:,0], out_range=(0, 1)), \
           rescale_intensity(img_res[:,:,1], out_range=(0,1)), \
           rescale_intensity(img_res[:,:,2], out_range=(0,1)), \
           D
# end rgb2he_macenko

def getNormalizer(method='macenko'):
    method = method.lower()
    if method == 'macenko':
        return StainNormalizerFactory.getMacenkoNormalizer()
    elif method == 'reinhard':
        return StainNormalizerFactory.getReinhardNormalizer()
    elif method == 'vahadane':
        return StainNormalizerFactory.getVahadaneNormalizer()
    else:
        raise RuntimeError('Unkown normalization method')


# Most of the code below is inspired/adapted from
#
#  https://github.com/JisuiWang/Stain_Normalization

class StainNormalizer(ABC):
    def __init__(self):
        self.target_means = None
        self.target_stds  = None

    @abstractmethod
    def fit(self, target):
        pass

    @abstractmethod
    def apply(self, I):
        pass

    @abstractmethod
    def H(self, I):
        pass

    @abstractmethod
    def save(self, file):
        pass

    @abstractmethod
    def load(self, file):
        pass


## MacenkoNormalizer
class MacenkoNormalizer(StainNormalizer):
    """
    Stain normalization based on the method of:

    M. Macenko et al., ‘A method for normalizing histology slides for quantitative analysis’,
    in 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, 2009, pp. 1107–1110.
    """

    def _get_stain_matrix(self, I, beta=0.15, alpha=1):
        """
        Get stain matrix (2x3)
        :param I:
        :param beta:
        :param alpha:
        :return:
        """
        OD = RGB_to_OD(I).reshape((-1, 3))
        OD = (OD[(OD > beta).any(axis=1), :])
        _, V = np.linalg.eigh(np.cov(OD, rowvar=False))
        V = V[:, [2, 1]]
        if V[0, 0] < 0: V[:, 0] *= -1
        if V[0, 1] < 0: V[:, 1] *= -1
        That = np.dot(OD, V)
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)
        v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
        v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
        if v1[0] > v2[0]:
            HE = np.array([v1, v2])
        else:
            HE = np.array([v2, v1])
        return normalize_rows(HE)

    ###

    def __init__(self):
        self.stain_matrix_target = None
        self.target_concentrations = None

    def save(self, file):
        np.savez_compressed(file, SMT=self.stain_matrix_target, TC=self.target_concentrations)


    def load(self, file):
        d = np.load(file)
        self.stain_matrix_target = d['STM']
        self.target_concentrations = d['TC']

    def fit(self, target):
        target = standardize_brightness(target)
        self.stain_matrix_target = self._get_stain_matrix(target)
        self.target_concentrations = get_concentrations(target, self.stain_matrix_target)

    def target_stains(self):
        return OD_to_RGB(self.stain_matrix_target)

    def apply(self, I):
        I = standardize_brightness(I)
        stain_matrix_source = self._get_stain_matrix(I)
        source_concentrations = get_concentrations(I, stain_matrix_source)
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= (maxC_target / maxC_source)
        return (255 * np.exp(-1 * np.dot(source_concentrations,
                                         self.stain_matrix_target).reshape(I.shape))).astype(np.uint8)

    def H(self, I):
        I = standardize_brightness(I)
        h, w, c = I.shape
        stain_matrix_source = self._get_stain_matrix(I)
        source_concentrations = get_concentrations(I, stain_matrix_source)
        _H = source_concentrations[:, 0].reshape(h, w)
        _H = np.exp(-1 * _H)
        return _H
## end MacenkoNormalizer


## ReinhardNormalizer
class ReinhardNormalizer(StainNormalizer):
    """
    Normalize a patch stain to the target image using the method of:

    E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’,
    IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.
    """

    def _lab_split(self, I):
        """
        Convert from RGB uint8 to LAB and split into channels
        :param I: uint8
        :return:
        """
        I = cv.cvtColor(I, cv.COLOR_RGB2LAB)
        I = I.astype(np.float32)
        I1, I2, I3 = cv.split(I)
        I1 /= 2.55
        I2 -= 128.0
        I3 -= 128.0
        return I1, I2, I3


    def _merge_back(self, I1, I2, I3):
        """
        Take seperate LAB channels and merge back to give RGB uint8
        :param I1:
        :param I2:
        :param I3:
        :return:
        """
        I1 *= 2.55
        I2 += 128.0
        I3 += 128.0
        I = np.clip(cv.merge((I1, I2, I3)), 0, 255).astype(np.uint8)
        return cv.cvtColor(I, cv.COLOR_LAB2RGB)


    def _get_mean_std(self, I):
        """
        Get mean and standard deviation of each channel
        :param I: uint8
        :return:
        """
        I1, I2, I3 = self._lab_split(I)
        m1, sd1 = cv.meanStdDev(I1)
        m2, sd2 = cv.meanStdDev(I2)
        m3, sd3 = cv.meanStdDev(I3)
        means = m1, m2, m3
        stds = sd1, sd2, sd3
        return means, stds


    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def save(self, file):
        np.savez_compressed(file, TM=self.target_means, TS=self.target_stds)

    def load(self, file):
        d = np.load(file)
        self.target_means = d['TM']
        self.target_stds = d['TS']

    def fit(self, target):
        target = standardize_brightness(target)
        self.target_means, self.target_stds = self._get_mean_std(target)

    def apply(self, I):
        I = standardize_brightness(I)
        I1, I2, I3 = self._lab_split(I)
        means, stds = self._get_mean_std(I)
        norm1 = ((I1 - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((I2 - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((I3 - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]
        return self._merge_back(norm1, norm2, norm3)


    def H(self, I):
        pass

## end ReinhardNormalizer


## VahadaneNormalizer
class VahadaneNormalizer(StainNormalizer):
    """
    Stain normalization inspired by method of:

    A. Vahadane et al., ‘Structure-Preserving Color Normalization and
    Sparse Stain Separation for Histological Images’,
    IEEE Transactions on Medical Imaging, vol. 35, no. 8, pp. 1962–1971, Aug. 2016.
    """

    def _get_stain_matrix(self, I, threshold=0.8, lamda=0.1):
        """
        Get 2x3 stain matrix. First row H and second row E
        :param I:
        :param threshold:
        :param lamda:
        :return:
        """
        mask = notwhite_mask(I, thresh=threshold).reshape((-1,))
        OD = RGB_to_OD(I).reshape((-1, 3))
        OD = OD[mask]
        dictionary = spams.trainDL(OD.T, K=2, lambda1=lamda, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False).T
        if dictionary[0, 0] < dictionary[1, 0]:
            dictionary = dictionary[[1, 0], :]
        dictionary = normalize_rows(dictionary)
        return dictionary


    def __init__(self):
        self.stain_matrix_target = None

    def save(self, file):
        np.savez_compressed(file, STM=self.stain_matrix_target)

    def load(self, file):
        self.stain_matrix_target = np.load(file)['STM']

    def fit(self, target):
        target = standardize_brightness(target)
        self.stain_matrix_target = self._get_stain_matrix(target)


    def target_stains(self):
        return OD_to_RGB(self.stain_matrix_target)


    def apply(self, I):
        I = standardize_brightness(I)
        stain_matrix_source = self._get_stain_matrix(I)
        source_concentrations = get_concentrations(I, stain_matrix_source)
        return (255 * np.exp(-1 * np.dot(source_concentrations,
                                         self.stain_matrix_target).reshape(I.shape))).astype(np.uint8)


    def H(self, I):
        I = standardize_brightness(I)
        h, w, c = I.shape
        stain_matrix_source = self._get_stain_matrix(I)
        source_concentrations = get_concentrations(I, stain_matrix_source)
        _H = source_concentrations[:, 0].reshape(h, w)
        _H = np.exp(-1 * _H)
        return _H

## StainNormalizerFactory
class StainNormalizerFactory(object):
    @staticmethod
    def getMacenkoNormalizer():
        return MacenkoNormalizer()

    @staticmethod
    def getReinhardNormalizer():
        return ReinhardNormalizer()

    @staticmethod
    def getVahadaneNormalizer():
        pass
## end StainNormalizerFactory

### local functions
def standardize_brightness(I):
    """

    :param I:
    :return:
    """
    p = np.percentile(I, 90)
    return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)


def remove_zeros(I):
    """
    Remove zeros, replace with 1's.
    :param I: uint8 array
    :return:
    """
    mask = (I == 0)
    I[mask] = 1
    return I


def RGB_to_OD(I):
    """
    Convert from RGB to optical density
    :param I:
    :return:
    """
    I = remove_zeros(I)
    return -1 * np.log(I / 255)


def OD_to_RGB(OD):
    """
    Convert from optical density to RGB
    :param OD:
    :return:
    """
    return (255 * np.exp(-1 * OD)).astype(np.uint8)


def normalize_rows(A):
    """
    Normalize rows of an array
    :param A:
    :return:
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


def notwhite_mask(I, thresh=0.8):
    """
    Get a binary mask where true denotes 'not white'
    :param I:
    :param thresh:
    :return:
    """
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0
    return L < thresh


def sign(x):
    """
    Returns the sign of x
    :param x:
    :return:
    """
    if x > 0:
        return +1
    elif x < 0:
        return -1
    elif x == 0:
        return 0


def get_concentrations(I, stain_matrix, lamda=0.01):
    """
    Get concentrations, a npix x 2 matrix
    :param I:
    :param stain_matrix: a 2x3 stain matrix
    :return:
    """
    OD = RGB_to_OD(I).reshape((-1, 3))
    return spams.lasso(OD.T, D=stain_matrix.T, mode=2, lambda1=lamda, pos=True).toarray().T
