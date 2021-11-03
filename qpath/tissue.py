# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

# TISSUE: methods for tissue segmentation, stain normalization etc

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = 0.1

__all__ = []

import warnings
import numpy as np
from . import Error
import mahotas as mh
import cv2
from skimage.util import img_as_bool, img_as_ubyte
import skimage.morphology as skm
from sklearn.cluster import MiniBatchKMeans
from .color import R_, G_, B_
from .stain import rgb2he
from skimage import color
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
from scipy.optimize import fmin_slsqp
from scipy import signal


def detect_foreground(img, method='fast-he', **kwargs):
    """Detects the foreground (tissue) parts in a whole H&E image
    slide and returns a mask for them (0: background).

    Args:
        img (ndarray H x W x 3): the image as an numpy array
        method (str): which method to use. Allowed values:
            'fast-he': uses OpenCV and simply processes the G-plane for
                detecting the tissue regions in H&E images
            'simple-he': automatically detects the threshold in G-plane using
                clustering and some ad-hoc rules; for foreground detection
                in H&E slides
            'simple': assumes a bimodal distribution in L-plane (from Lab space)
                and tries to segment the foreground by choosing a robust
                threshold.
            'htk': uses HistomicsTK's detection method. See
                https://digitalslidearchive.github.io/HistomicsTK/histomicstk.segmentation.html
            'fesi': inspired by Bug et al., "Foreground Extraction for Histopathological Whole Slide Imaging",
                Bildverarbeitung fur die Medizin 2015, pp 419-424
        kwargs: a dictionary with optional arguments, with specific
            values for each method:
            -For 'fast-he':
                g_th: threshold to use for G plane foreground separation,
                    default: 220
                ker_size: kernel size of the structuring element in closing
                    operations; default: 33
                min_area: area of the smallest object to keep; default: 150
            -For 'simple-he':
                g_th: threshold to use for G plane foreground separation. If
                    None or 0, a value will be estimated from image;
                    default: None
                min_area: area of the smallest object to keep; default: 150
            -For 'simple':
                th: threshold to use for L plane foreground separation. If
                    None or 0, a value will be estimated from image;
                    default: None
                min_area: area of the smallest object to keep; default: 150
                background_is_white: if True, the object is considered to be
                    darker than the background, otherwise the vice-versa; default: True
            -For 'htk':
                bandwidth : double, optional
                    Bandwidth for kernel density estimation - used for smoothing the
                    grayscale histogram. Default value = 2.
                bgnd_std : double, optional
                    Standard deviation of background gaussian to be used if
                    estimation fails. Default value = 2.5.
                tissue_std: double, optional
                    Standard deviation of tissue gaussian to be used if estimation fails.
                    Default value = 30.
                min_peak_width: double, optional
                    Minimum peak width for finding peaks in KDE histogram. Used to
                    initialize curve fitting process. Default value = 10.
                max_peak_width: double, optional
                    Maximum peak width for finding peaks in KDE histogram. Used to
                    initialize curve fitting process. Default value = 25.
                fraction: double, optional
                    Fraction of pixels to sample for building foreground/background
                    model. Default value = 0.10.
                min_tissue_prob : double, optional
                    Minimum probability to qualify as tissue pixel. Default value = 0.05.
            -For 'fesi':
                laplace_ker: int, optional
                    kernel size for Laplacian
                gauss_ker: int, optional
                    kernel size for Gaussian
                gauss_sigma: double, optional
                    sigma for Gaussian filter
                morph_open_ker: int, optional
                    kernel size for morphological opening
                morph_open_iter: int, optional
                    number of iterations for morphological opening
                morph_blur: int, optional
                    size of the kernel for morphological blurring
                min_area: int, optional
                    area of the smallest object to keep; default: 150

    Returns:
          a pair (mask, value) where mask is a binary mask for the foreground (labeled with '1')
          and value is None (for 'fast-he' and 'htk') of the threshold (for 'simple' and 'simple-he'), respectively
    """

    methods = {
        'fast-he': _he_fast_foreground_detection,
        'simple-he': _he_simple_foreground_detection,
        'simple': _simple_foreground_detection,
        'fesi': _fesi_foreground_detection,
        'htk': _htk_foreground_detection
    }

    method = method.lower()
    if method not in methods:
        raise Error('Unknown method ' + method)

    res = methods[method](img, **kwargs)

    return res


def _he_fast_foreground_detection(img, **kwargs):
    """Fast and simple foreground detection in H&E slides based on morphological
    operations in G(reen) plane.

    Args:
        img (ndarray H x W x 3): the image as an numpy array
        kwargs: a dictionary with optional arguments
            g_th: threshold to use for G plane foreground separation. If
                    None or 0, a value will be estimated from image;
                    default: None
            ker_size: kernel size of the structuring element in closing
                operations; default: 33
            min_area: area of the smallest object to keep; default: 150
    """
    _g_th     = kwargs.get('g_th', 220)
    _ker_size = kwargs.get('ker_size', 33)
    _min_area = kwargs.get('min_area', 150)

    mask = img[:,:,1].copy()
    mask[mask > _g_th] = 0
    mask[mask > 0] = np.iinfo(mask.dtype).max

    mask = mh.close_holes(mask)
    kr = cv2.getStructuringElement(cv2.MORPH_RECT, (int(_ker_size), int(_ker_size)))
    mask = cv2.erode(img_as_ubyte(mask), kr)
    with warnings.catch_warnings():  # avoid "Possible precision loss when converting from uint8 to bool"
        warnings.simplefilter("ignore")
        mask = skm.remove_small_objects(img_as_bool(mask), min_size=_min_area, in_place=True)

    return mask, None


def _he_simple_foreground_detection(img, **kwargs):
    """Fast and simple foreground detection in H&E slides based on morphological
    operations in G(reen) plane.

    Args:
        img (ndarray H x W x 3): the image as an numpy array
        kwargs: a dictionary with optional arguments
            g_th: threshold to use for G plane foreground separation. If
                 None or 0, a value will be estimated from image;
                 default: None
            min_area: area of the smallest object to keep; default: 150
    """
    _g_th     = kwargs.get('g_th', 0)
    _min_area = kwargs.get('min_area', 150)

    if _g_th is None or _g_th == 0:
        # Apply vector quantization to remove the "white" background - work in the
        # green channel:
        vq = MiniBatchKMeans(n_clusters=2)
        _g_th = int(np.round(0.95 * np.max(vq.fit(G_(img).reshape((-1, 1)))
                                           .cluster_centers_.squeeze())))

    mask = img[:,:,1] < _g_th  # G-plane

    skm.binary_closing(mask, skm.disk(3), out=mask)
    mask = img_as_bool(mask)
    mask = skm.remove_small_objects(mask, min_size=_min_area, in_place=True)

    # Some hand-picked rules:
    # -at least 5% H and E
    # -at most 50% background
    # for a region to be considered tissue

    h, e, b = rgb2he(img)

    mask &= (h > np.percentile(h, 5)) | (e > np.percentile(e, 5))
    mask &= (b < np.percentile(b, 50))  # at most at 50% of "other components"

    mask = mh.close_holes(mask)

    return img_as_bool(mask), _g_th


def _simple_foreground_detection(img, **kwargs):
    """Fast and simple foreground detection in grey-scale space based on morphological
    operations.

    Args:
        img (ndarray H x W x 3): the image as an numpy array
        kwargs: a dictionary with optional arguments
            th: threshold to use; if None or 0, a value will be estimated
                from the image; default: None
            min_area: area of the smallest object to keep; default: 150
            background_is_white: whether the background is white (and the object darker),
                or vice-versa. default: True
    """
    _th     = kwargs.get('th', 0)
    _min_area = kwargs.get('min_area', 150)
    _bkg_is_white = kwargs.get('background_is_white', True)

    if img.ndim > 2:
        img = color.rgb2lab(img)[..., 0]

    if _th is None or _th == 0:
        # Apply vector quantization to remove the background
        vq = MiniBatchKMeans(n_clusters=2)
        _th = int(np.round(0.95 * np.max(vq.fit(img.reshape((-1, 1)))
                                           .cluster_centers_.squeeze())))

    if _bkg_is_white:
        mask = img < _th
    else:
        mask = img > _th

    skm.binary_closing(mask, skm.disk(3), out=mask)
    mask = img_as_bool(mask)
    mask = skm.remove_small_objects(mask, min_size=_min_area, in_place=True)

    mask = mh.close_holes(mask)
    mask = mh.morph.open(mask, mh.disk(5))

    return img_as_bool(mask), _th


def _fesi_foreground_detection(img: np.ndarray, **kwargs):
    """Fast and simple foreground detection in grey-scale space based on morphological
        operations.

        Args:
            img (ndarray H x W x 3): the image as an numpy array
            kwargs: a dictionary with optional arguments
                laplace_ker: kernel size for Laplacian
                gauss_ker: kernel size for Gaussian
                gauss_sigma: sigma for Gaussian filter
                morph_open_ker: kernel size for morphological opening
                morph_open_iter: number of iterations for morphological opening
                morph_blur: size of the kernel for morphological blurring
                min_area: area of the smallest object to keep; default: 150
    """
    _laplace_ker = kwargs.get('laplace_ker', 5)
    _gauss_ker = kwargs.get('gauss_ker', 15)
    _gauss_sigma = kwargs.get('gauss_sigma', 4.0)
    _morph_open_ker = kwargs.get('morph_open_ker', 9)
    _morph_open_iter = kwargs.get('morph_open_iter', 7)
    _morph_blur = kwargs.get('morph_blur', 15)
    _min_area = kwargs.get('min_area', 150)

    imgg = 0.299 * R_(img) + 0.587 * G_(img) + 0.144 * B_(img)
    imglp = cv2.Laplacian(imgg, cv2.CV_16S, ksize=_laplace_ker)
    imglp = cv2.convertScaleAbs(imglp)
    imggs = cv2.GaussianBlur(imglp, (_gauss_ker, _gauss_ker), _gauss_sigma)
    _, imgmk = cv2.threshold(imggs, imggs.mean(), 255, cv2.THRESH_BINARY)
    imgmk = cv2.medianBlur(imgmk, _morph_blur)
    imgmk = cv2.morphologyEx(imgmk, cv2.MORPH_OPEN,
                             (_morph_open_ker, _morph_open_ker),
                             imgmk,
                             (-1, -1),
                             _morph_open_iter)

    mask = img_as_bool(imgmk)
    mask = skm.remove_small_objects(mask, min_size=_min_area, in_place=True)
    mask = mh.close_holes(mask)
    mask = mh.morph.erode(mask, mh.disk(5))
    mask = mh.morph.open(mask, mh.disk(5))

    return img_as_bool(mask), None


def _htk_foreground_detection(im_rgb: np.ndarray, **kwargs):
    """Foreground detection based on Gaussian mixture model. This function is copied
    from HistomicsTK: https://github.com/DigitalSlideArchive/HistomicsTK

    Args:
        im_rgb : array_like
            An RGB image of type unsigned char.
        kwargs: a dictionary with additional parameters:
            bandwidth : double, optional
                Bandwidth for kernel density estimation - used for smoothing the
                grayscale histogram. Default value = 2.
            bgnd_std : double, optional
                Standard deviation of background gaussian to be used if
                estimation fails. Default value = 2.5.
            tissue_std: double, optional
                Standard deviation of tissue gaussian to be used if estimation fails.
                Default value = 30.
            min_peak_width: double, optional
                Minimum peak width for finding peaks in KDE histogram. Used to
                initialize curve fitting process. Default value = 10.
            max_peak_width: double, optional
                Maximum peak width for finding peaks in KDE histogram. Used to
                initialize curve fitting process. Default value = 25.
            fraction: double, optional
                Fraction of pixels to sample for building foreground/background
                model. Default value = 0.10.
            min_tissue_prob : double, optional
                Minimum probability to qualify as tissue pixel. Default value = 0.05.

    Returns
        im_mask : array_like
            A binarized version of input image where foreground (tissue) has value '1'.
    """
    bandwidth = kwargs.get('bandwidth', 2)
    bgnd_std = kwargs.get('bgnd_std', 2.5)
    tissue_std = kwargs.get('tissue_std', 30)
    min_peak_width = kwargs.get('min_peak_width', 10)
    max_peak_width = kwargs.get('max_peak_width', 25)
    fraction = kwargs.get('fraction', 0.10)
    min_tissue_prob = kwargs.get('min_tissue_prob', 0.05)

    # convert image to grayscale, flatten and sample
    im_rgb = 255 * color.rgb2gray(im_rgb)
    im_rgb = im_rgb.astype(np.uint8)
    num_samples = np.int(fraction * im_rgb.size)
    sI = np.random.choice(im_rgb.flatten(), num_samples)[:, np.newaxis]

    # kernel-density smoothed histogram
    KDE = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(sI)
    xHist = np.linspace(0, 255, 256)[:, np.newaxis]
    yHist = np.exp(KDE.score_samples(xHist))[:, np.newaxis]
    yHist = yHist / sum(yHist)

    # flip smoothed y-histogram so that background mode is on the left side
    yHist = np.flipud(yHist)

    # identify initial mean parameters for gaussian mixture distribution
    # take highest peak among remaining peaks as background
    Peaks = signal.find_peaks_cwt(yHist.flatten(),
                                  np.arange(min_peak_width, max_peak_width))
    BGPeak = Peaks[0]
    if len(Peaks) > 1:
        TissuePeak = Peaks[yHist[Peaks[1:]].argmax() + 1]
    else:  # no peak found - take initial guess at 2/3 distance from origin
        TissuePeak = np.asscalar(xHist[int(np.round(0.66*xHist.size))])

    # analyze background peak to estimate variance parameter via FWHM
    BGScale = estimate_variance(xHist, yHist, BGPeak)
    if BGScale == -1:
        BGScale = bgnd_std

    # analyze tissue peak to estimate variance parameter via FWHM
    TissueScale = estimate_variance(xHist, yHist, TissuePeak)
    if TissueScale == -1:
        TissueScale = tissue_std

    # solve for mixing parameter
    Mix = yHist[BGPeak] * (BGScale * (2 * np.pi)**0.5)

    # flatten kernel-smoothed histogram and corresponding x values for
    # optimization
    xHist = xHist.flatten()
    yHist = yHist.flatten()

    # define gaussian mixture model
    def gaussian_mixture(x, mu1, mu2, sigma1, sigma2, p):
        rv1 = norm(loc=mu1, scale=sigma1)
        rv2 = norm(loc=mu2, scale=sigma2)
        return p * rv1.pdf(x) + (1 - p) * rv2.pdf(x)

    # define gaussian mixture model residuals
    def gaussian_residuals(Parameters, y, x):
        mu1, mu2, sigma1, sigma2, p = Parameters
        yhat = gaussian_mixture(x, mu1, mu2, sigma1, sigma2, p)
        return sum((y - yhat) ** 2)

    # fit Gaussian mixture model and unpack results
    Parameters = fmin_slsqp(gaussian_residuals,
                            [BGPeak, TissuePeak, BGScale, TissueScale, Mix],
                            args=(yHist, xHist),
                            bounds=[(0, 255), (0, 255),
                                    (np.spacing(1), 10),
                                    (np.spacing(1), 50), (0, 1)],
                            iprint=0)

    muBackground = Parameters[0]
    muTissue = Parameters[1]
    sigmaBackground = Parameters[2]
    sigmaTissue = Parameters[3]
    p = Parameters[4]

    # create mask based on Gaussian mixture model
    Background = norm(loc=muBackground, scale=sigmaBackground)
    Tissue = norm(loc=muTissue, scale=sigmaTissue)
    pBackground = p * Background.pdf(xHist)
    pTissue = (1 - p) * Tissue.pdf(xHist)

    # identify maximum likelihood threshold
    Difference = pTissue - pBackground
    Candidates = np.nonzero(Difference >= 0)[0]
    Filtered = np.nonzero(xHist[Candidates] > muBackground)
    ML = xHist[Candidates[Filtered[0]][0]]

    # identify limits for tissue model (MinProb, 1-MinProb)
    Endpoints = np.asarray(Tissue.interval(1 - min_tissue_prob / 2))

    # invert threshold and tissue mean
    ML = 255 - ML
    muTissue = 255 - muTissue
    Endpoints = np.sort(255 - Endpoints)

    # generate mask
    im_mask = (im_rgb <= ML) & (im_rgb >= Endpoints[0]) & \
              (im_rgb <= Endpoints[1])
    im_mask = im_mask.astype(np.uint8)

    return im_mask, None



def estimate_variance(x, y, peak):
    """Estimates variance of a peak in a histogram using the FWHM of an
    approximate normal distribution.
    This function is copied
    from HistomicsTK: https://github.com/DigitalSlideArchive/HistomicsTK

    Starting from a user-supplied peak and histogram, this method traces down
    each side of the peak to estimate the full-width-half-maximum (FWHM) and
    variance of the peak. If tracing fails on either side, the FWHM is
    estimated as twice the HWHM.
    Parameters
    ----------
    x : array_like
        vector of x-histogram locations.
    y : array_like
        vector of y-histogram locations.
    peak : double
        index of peak in y to estimate variance of
    Returns
    -------
    scale : double
        Standard deviation of normal distribution approximating peak. Value is
        -1 if fitting process fails.
    See Also
    --------
    SimpleMask

    """

    # analyze peak to estimate variance parameter via FWHM
    peak = int(peak)
    Left = peak
    scale = 0

    while y[Left] > y[peak] / 2 and Left >= 0:
        Left -= 1
        if Left == -1:
            break
    Right = peak
    while y[Right] > y[peak] / 2 and Right < y.size:
        Right += 1
        if Right == y.size:
            break
    if Left != -1 and Right != y.size:
        LeftSlope = y[Left + 1] - y[Left] / (x[Left + 1] - x[Left])
        Left = (y[peak] / 2 - y[Left]) / LeftSlope + x[Left]
        RightSlope = y[Right] - y[Right - 1] / (x[Right] - x[Right - 1])
        Right = (y[peak] / 2 - y[Right]) / RightSlope + x[Right]
        scale = (Right - Left) / 2.355
    if Left == -1:
        if Right == y.size:
            scale = -1
        else:
            RightSlope = y[Right] - y[Right - 1] / (x[Right] - x[Right - 1])
            Right = (y[peak] / 2 - y[Right]) / RightSlope + x[Right]
            scale = 2 * (Right - x[peak]) / 2.355
    if Right == y.size:
        if Left == -1:
            scale = -1
        else:
            LeftSlope = y[Left + 1] - y[Left] / (x[Left + 1] - x[Left])
            Left = (y[peak] / 2 - y[Left]) / LeftSlope + x[Left]
            scale = 2 * (x[peak] - Left) / 2.355

    return scale
