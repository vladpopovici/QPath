# -*- coding: utf-8 -*-

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = 0.1

"""
QPATH.BASE: Core classes and functions.

Defines exception classes and other basic classes.
"""

__all__ = ['WSIInfo', 'NumpyImage', 'MRI', 'SlidingWindowSampler', 'RandomWindowSampler']

import zarr
import pathlib
import numpy as np
from abc import ABC, abstractmethod
from . import Error
import shapely.geometry as shg
import shapely.affinity as sha
from .utils import geom2xy
from .mask import add_region, apply_mask
from math import log2


#####
class WSIInfo(object):
    """Hold some basic info about a WSI.

    Args:
        path (str): full path to root of imported WSI file. This is the folder
            containing a 'pyramid' ZARR group with datasets for each level. Normally,
            for an imported slide <SLIDE_NAME>, the canonical path should be
            .../<SLIDE_NAME>/slide/pyramid.zarr.

    Attributes:
        path (str): full path to WSI file
        info (dict): a dictionary containing WSI properties
            Example:
            {'background': 'FFFFFF',
             'objective_power': 20,
             'pyramid': [{'downsample_factor': 1, 'height': 111388, 'level': 0, 'width': 75829},
                         {'downsample_factor': 2, 'height': 55694, 'level': 1, 'width': 37914},
                         {'downsample_factor': 4, 'height': 27847, 'level': 2, 'width': 18957},
                         {'downsample_factor': 8, 'height': 13923, 'level': 3, 'width': 9478},
                         {'downsample_factor': 16, 'height': 6961, 'level': 4, 'width': 4739},
                         {'downsample_factor': 32, 'height': 3480, 'level': 5, 'width': 2369},
                         {'downsample_factor': 64, 'height': 1740, 'level': 6, 'width': 1184},
                         {'downsample_factor': 128, 'height': 870, 'level': 7, 'width': 592},
                         {'downsample_factor': 256, 'height': 435, 'level': 8, 'width': 296},
                         {'downsample_factor': 512, 'height': 217, 'level': 9, 'width': 148}],
            'resolution_units': 'microns',
            'resolution_x_level_0': 0.23387573964497,
            'resolution_y_level_0': 0.234330708661417, 'vendor': 'mirax'}
    """

    path = None  # Image path
    info = {}  # Info
    _pyramid_levels = None  # Convenient access to pyramid levels [0,]->width, [1,]->height


    def __init__(self, path):
        self.path = pathlib.Path(path)

        with zarr.open(self.path, mode='r') as z:
            self.info = z.attrs['metadata']
            self.info['pyramid'] = z.attrs['pyramid']

        self._pyramid_levels = np.zeros((2, len(self.info['pyramid'])), dtype=int)
        for p in self.info['pyramid']:
            self._pyramid_levels[:, p['level']] = [p['width'], p['height']]

        return


    def level_count(self) -> int:
        """Return the number of levels in the multi-resolution pyramid."""
        return self._pyramid_levels.shape[1]


    def downsample_factor(self, level:int) -> int:
        """Return the downsampling factor (relative to level 0) for a given level."""
        if level < 0 or level >= self.level_count():
            return -1
        for p in self.info['pyramid']:
            if p['level'] == level:
                return p['downsample_factor']


    def get_native_magnification(self) -> float:
        """Return the original magnification for the scan."""
        return self.info['objective_power']


    def get_level_for_magnification(self, mag: float, eps=1e-6) -> int:
        """Returns the level in the image pyramid that corresponds the given magnification.

        Args:
            mag (float): magnification
            eps (float): accepted error when approximating the level

        Returns:
            level (int) or -1 if no suitable level was found
        """
        if mag > self.info['objective_power'] or mag < 2.0**(1-self.level_count()) * self.info['objective_power']:
            return -1

        lx = log2(self.info['objective_power'] / mag)
        k = np.where(np.isclose(lx, range(0, self.level_count()), atol=eps))[0]
        if len(k) > 0:
            return k[0]   # first index matching
        else:
            return -1   # no match close enough


    def get_magnification_for_level(self, level: int) -> float:
        """Returns the magnification (objective power) for a given level.

        Args:
            level (int): level in the pyramidal image

        Returns:
            magnification (float)
            If the level is out of bounds, returns -1.0
        """
        if level < 0 or level >= self.level_count():
            return -1.0
        if level == 0:
            return self.info['objective_power']

        return 2.0**(-level) * self.info['objective_power']


    def get_extent_at_level(self, level: int) -> (int, int):
        """Returns width and height of the image at a desired level.

        Args:
            level (int): level in the pyramidal image

        Returns:
            (width, height) of the level
        """
        if level < 0 or level >= self.level_count():
            return None, None
        return tuple(self._pyramid_levels[:, level])


#####
class NumpyImage:
    """This is barely a namespace for collecting a number of useful
    functions that are applied to images stored as Numpy arrays.
    Usually, such an image -either single channel or 3(4) channels -
    is stored as a H x W (x C) array, with H (height) rows and W (width)
    columns. C=3 or 4.
    """

    @staticmethod
    def width(img):
        img: np.ndarray
        return img.shape[1]

    @staticmethod
    def height(img):
        img: np.ndarray
        return img.shape[0]

    @staticmethod
    def nchannels(img):
        img: np.ndarray
        if img.ndim > 2:
            return img.shape[2]
        else:
            return 1

    @staticmethod
    def is_empty(img, empty_level: float=0) -> bool:
        """Is the image empty?

        Args:
            img (numpy.ndarray): image
            empty_level (int/numeric): if the sum of pixels is at most this
                value, the image is considered empty.

        Returns:
            bool
        """

        return img.sum() <= empty_level

    @staticmethod
    def is_almost_white(img, almost_white_level: float=254) -> bool:
        """Is the image almost white?

        Args:
            img (numpy.ndarray): image
            almost_white_level (int/numeric): if the average intensity per channel
        is above the given level, decide "almost white" image.

        Returns:
            bool
        """

        return img.mean() >= almost_white_level


#####
class MRI(object):
    """MultiResolution Image - a simple and convenient interface to access pixels from a
    pyramidal image. The image is supposed to by stored in ZARR format and to have an
    attributre 'pyramid' describing the properties of the pyramid levels. There is no
    information related to resolutions etc, these are charaterstic for a slide image -
    see WSIInfo.

    Args:
        path (str): folder with a ZARR store providing the levels (indexed 0, 1, ...)
        of the pyramid

    Attributes:
        _path (Path)
        _pyramid (dict)
    """
    _path = None
    _pyramid = None
    _pyramid_levels = None

    def __init__(self, path: str):
        self._path = pathlib.Path(path)

        with zarr.open(self.path, mode='r') as z:
            self._pyramid = z.attrs['pyramid']

        self._pyramid_levels = np.zeros((2, len(self._pyramid)), dtype=int)
        self._downsample_factors = np.zeros((len(self._pyramid)), dtype=int)
        for p in self._pyramid:
            self._pyramid_levels[:, p['level']] = [p['width'], p['height']]
            self._downsample_factors[p['level']] = p['downsample_factor']


    @property
    def path(self) -> pathlib.Path:
        return self._path

    @property
    def widths(self) -> np.array:
        # All widths for the pyramid levels
        return self._pyramid_levels[0,:]

    @property
    def heights(self) -> np.array:
        # All heights for the pyramid levels
        return self._pyramid_levels[1,:]

    def extent(self, level:int=0) -> (int, int):
        # width, height for a given level
        return tuple(self._pyramid_levels[:, level])

    @property
    def nlevels(self) -> int:
        return self._pyramid_levels.shape[1]

    def between_level_scaling_factor(self, from_level:int, to_level:int) -> float:
        """Return the scaling factor for converting coordinates (magnification)
        between two levels in the MRI.

        Args:
            from_level (int): original level
            to_level (int): destination level

        Returns:
            float
        """
        f = self._downsample_factors[from_level] / self._downsample_factors[to_level]

        return f

    def convert_px(self, point, from_level, to_level):
        """Convert pixel coordinates of a point from <from_level> to
        <to_level>

        Args:
            point (tuple): (x,y) coordinates in <from_level> plane
            from_level (int): original image level
            to_level (int): destination level

        Returns:
            x, y (float): new coodinates - no rounding is applied
        """
        if from_level == to_level:
            return point  # no conversion is necessary
        x, y = point
        f = self.between_level_scaling_factor(from_level, to_level)
        x *= f
        y *= f

        return x, y


    def get_region_px(self, x0: int, y0: int,
                      width: int, height: int,
                      level: int=0, as_type=np.uint8) -> np.array:
        """Read a region from the image source. The region is specified in
            pixel coordinates.

            Args:
                x0, y0 (long): top left corner of the region (in pixels, at the specified
                level)
                width, height (long): width and height (in pixels) of the region.
                level (int): the magnification level to read from
                as_type: type of the pixels (default numpy.uint8)

            Returns:
                a numpy.ndarray
        """

        if level < 0 or level >= self.nlevels:
            raise Error("requested level does not exist")

        # check bounds:
        if x0 >= self.widths[level] or y0 >= self.heights[level] or \
                x0 + width > self.widths[level] or \
                y0 + height > self.heights[level]:
            raise Error("region out of layer's extent")

        with zarr.open_group(self.path, mode='r') as zarr_root:
            img = np.array(zarr_root[str(level)][y0:y0+height, x0:x0+width, :], dtype=as_type)

        return img


    def get_plane(self, level: int = 0, as_type=np.uint8) -> np.array:
        """Read a whole plane from the image pyramid and return it as a Numpy array.

        Args:
            level (int): pyramid level to read
            as_type: type of the pixels (default numpy.uint8)

        Returns:
            a numpy.ndarray
        """
        if level < 0 or level >= self.nlevels:
            raise Error("requested level does not exist")

        with zarr.open_group(self.path, mode='r') as zarr_root:
            img = np.array(zarr_root[str(level)][...], dtype=as_type)

        return img


    def get_polygonal_region_px(self, contour: shg.Polygon, level: int,
                                border: int=0, as_type=np.uint8) -> np.ndarray:
        """Returns a rectangular view of the image source that minimally covers a closed
        contour (polygon). All pixels outside the contour are set to 0.

        Args:
            contour (shapely.geometry.Polygon): a closed polygonal line given in
                terms of its vertices. The contour's coordinates are supposed to be
                precomputed and to be represented in pixel units at the desired level.
            level (int): image pyramid level
            border (int): if > 0, take this many extra pixels in the rectangular
                region (up to the limits on the image size)
            as_type: pixel type for the returned image (array)

        Returns:
            a numpy.ndarray
        """
        x0, y0, x1, y1 = [int(_z) for _z in contour.bounds]
        x0, y0 = max(0, x0-border), max(0, y0-border)
        x1, y1 = min(x1+border, self.extent(level)[0]), \
                 min(y1+border, self.extent(level)[1])
        # Shift the annotation such that (0,0) will correspond to (x0, y0)
        contour = sha.translate(contour, -x0, -y0)

        # Read the corresponding region
        img = self.get_region_px(x0, y0, x1-x0, y1-y0, level, as_type=np.uint8)

        # Prepare mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        add_region(mask, geom2xy(contour))

        # Apply mask
        img = apply_mask(img, mask)

        return img

##


class WindowSampler(ABC):
    """
    Defines an interface for an image sampler that returns rectangular
    regions from the image.
    """

    @abstractmethod
    def reset(self):
        """Reset the explore, next call to next() will start from the
        initial conditions.
        """
        pass

    @abstractmethod
    def last(self):
        """Go to last position and return it."""
        pass

    @abstractmethod
    def next(self):
        """Go to next position."""
        pass

    @abstractmethod
    def prev(self):
        """Go to previous position."""
        pass

    @abstractmethod
    def here(self):
        """Returns current position, does not change it."""
        pass

    @abstractmethod
    def total_steps(self):
        """Returns the total number of steps to iterate over all positions
        in the image, according to the specific schedule.
        """
        pass

    @staticmethod
    def _corners_to_poly(x0, y0, x1, y1):
        """Returns a Shapely Polygon with all four vertices of the window
        defined by (x0,y0) -> (x1,y1).
        """
        return shg.Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])

    @staticmethod
    def _check_window(x0: int, y0: int, x1: int, y1: int,
                      width: int, height: int, clip: bool = True) -> tuple:
        """Checks whether the coordinates of the window are valid and, eventually
        (only if clip is True), truncates the window to fit the image extent
        given by width and height.

        Args:
            x0, y0 : int
                Top-left corner of the window.
            x1, y1 : int
                Bottom-right corner of the window.
            width, height : int
                Image shape.
            clip : bool
                Whether the window should be clipped to image boundary.

        Return:
            a tuple (x0, y0, x1, y1) of window vertices or None if the
            window is not valid (e.g. negative coordinates).
        """
        if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
            return None
        if x0 >= width or y0 >= height:
            return None

        if clip:
            x1 = min(x1, width)
            y1 = min(y1, height)

        return x0, y0, x1, y1

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __prev__(self):
        return self.prev()


class SlidingWindowSampler(WindowSampler):
    """
    A sliding window image sampler. It returns successively the coordinates
    of the sliding window as a tuple (x0, y0, x1, y1).

    Args:
        image_shape : tuple (img_width, img_height)
            Image shape (img.shape).
        w_size : tuple (width, height)
            Window size as a pair of width and height values.
        start : tuple (x0, y0)
            Top left corner of the first window. Defaults to (0,0).
        step : tuple (x_step, y_step)
            Step size for the sliding window, as a pair of horizontal
            and vertical steps. Defaults to (1,1).
        poly : shapely.geometry.Polygon
            (if not None) Defines the region within which the windows will be generated.
        nv_inside : int
            number of corners/vertices of the the window required to be inside the
            polygon defining the region. This relaxes the constraint that whole window
            must lie within the polygon. Must be between 1 and 4._
    """

    def __init__(self, image_shape: tuple, w_size: tuple,
                 start: tuple = (0, 0), step=(1, 1), poly: shg.Polygon = None,
                 nv_inside: int = 4):
        self._image_shape = image_shape
        self._w_size = w_size
        self._start = start
        self._step = step
        self._k = 0
        self._poly = poly

        nv_inside = max(1, min(nv_inside, 4))  # >=1 && <=4

        img_w, img_h = image_shape

        if w_size[0] < 2 or w_size[1] < 2:
            raise ValueError('Window size too small.')

        if img_w < start[0] + w_size[0] or img_h < start[1] + w_size[1]:
            raise ValueError('Start position and/or window size out of image.')

        x, y = np.meshgrid(np.arange(start[0], img_w - w_size[0] + 1, step[0]),
                           np.arange(start[1], img_h - w_size[1] + 1, step[1]))

        tmp_top_left_corners = [p for p in zip(x.reshape((-1,)).tolist(),
                                               y.reshape((-1,)).tolist())]

        if self._poly is None:
            self._top_left_corners = tmp_top_left_corners
        else:
            # need to filter out regions outside the Polygon
            self._top_left_corners = []
            for x0, y0 in tmp_top_left_corners:
                x1 = x0 + self._w_size[0]
                y1 = y0 + self._w_size[1]
                t = WindowSampler._check_window(x0, y0, x1, y1,
                                                img_w, img_h, clip=True)
                if t is None:
                    continue
                x0, y0, x1, y1 = t
                w = [int(shg.Point(p).within(self._poly)) for p in [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]]
                if np.array(w).sum() >= nv_inside:
                    self._top_left_corners.append((x0, y0))

        return

    def total_steps(self):
        return len(self._top_left_corners)

    def reset(self):
        self._k = 0

    def here(self):
        if 0 <= self._k < self.total_steps():
            x0, y0 = self._top_left_corners[self._k]
            x1 = min(x0 + self._w_size[0], self._image_shape[0])
            y1 = min(y0 + self._w_size[1], self._image_shape[1])

            return x0, y0, x1, y1
        raise RuntimeError("Position outside bounds")

    def last(self):
        if self.total_steps() > 0:
            self._k = self.total_steps() - 1
            x0, y0, x1, y1 = self.here()
            return x0, y0, x1, y1
        else:
            raise RuntimeError("Empty iterator")

    def next(self):
        if self._k < self.total_steps():
            x0, y0, x1, y1 = self.here()
            self._k += 1
            return x0, y0, x1, y1
        else:
            raise StopIteration()

    def prev(self):
        if self._k >= 1:
            self._k -= 1
            x0, y0, x1, y1 = self.here()
            return x0, y0, x1, y1
        else:
            raise StopIteration()


class RandomWindowSampler(WindowSampler):
    """
    A random window image sampler. It returns a sequence of random window coordinates
    (x0, y0, x1, y1) within the image.

    Args:
        image_shape : tuple (img_width, img_height)
            Image shape (img.shape).
        w_size : tuple (width, height)
            Window size as a pair of width and height values.
        n : int
            Number of windows to return.
        poly : shapely.geometry.Polygon
            (if not None) Defines the region within which the windows will be generated.
        rng_seed : int or None
            random number generator seed for initialization in a known state. If None,
            the seed is set by the system.
        nv_inside : int
            number of corners/vertices of the the window required to be inside the
            polygon defining the region. This relaxes the constraint that whole window
            must lie within the polygon. Must be between 1 and 4.
    """

    def __init__(self, image_shape: tuple, w_size: tuple, n: int,
                 poly: shg.Polygon = None, rng_seed: int = None, nv_inside: int = 4):
        self._image_shape = image_shape
        self._w_size = w_size
        self._poly = poly
        self._n = n
        self._k = 0
        self._rng_seed = rng_seed
        self._rng = np.random.default_rng(rng_seed)

        nv_inside = max(1, min(nv_inside, 4))  # >=1 && <=4
        img_w, img_h = image_shape

        if w_size[0] < 2 or w_size[1] < 2:
            raise ValueError('Window size too small.')

        if img_w < w_size[0] or img_h < w_size[1]:
            raise ValueError('Window size larger than image.')

        if self._poly is None:
            self._top_left_corners = []
            k = 0
            while k < self._n:
                x0 = self._rng.integers(low=0, high=img_w - self._w_size[0], size=1)[0]
                y0 = self._rng.integers(low=0, high=img_h - self._w_size[1], size=1)[0]
                x1 = x0 + self._w_size[0]
                y1 = y0 + self._w_size[1]
                t = WindowSampler._check_window(x0, y0, x1, y1, img_w, img_h, clip=True)
                if t is None:
                    continue
                x0, y0, x1, y1 = t
                # finally, a valid window
                self._top_left_corners.append((x0, y0))
                k += 1
        else:
            # need to filter out regions outside the Polygon
            k = 0
            self._top_left_corners = []
            while k < self._n:
                x0 = self._rng.integers(low=0, high=img_w - self._w_size[0], size=1)[0]
                y0 = self._rng.integers(low=0, high=img_h - self._w_size[1], size=1)[0]
                x1 = x0 + self._w_size[0]
                y1 = y0 + self._w_size[1]
                t = WindowSampler._check_window(x0, y0, x1, y1, img_w, img_h, clip=True)
                if t is None:
                    continue
                x0, y0, x1, y1 = t
                w = [int(shg.Point(p).within(self._poly)) for p in [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]]
                if np.array(w).sum() >= nv_inside:
                    self._top_left_corners.append((x0, y0))
                    k += 1
        return

    def total_steps(self):
        return self._n

    def reset(self):
        self._k = 0

    def here(self):
        if 0 <= self._k < self.total_steps():
            x0, y0 = self._top_left_corners[self._k]
            # bounds where checked in the constructor
            x1 = x0 + self._w_size[0]
            y1 = y0 + self._w_size[1]

            return x0, y0, x1, y1

        raise RuntimeError("Position outside bounds")

    def last(self):
        if self.total_steps() > 0:
            self._k = self.total_steps() - 1
            x0, y0, x1, y1 = self.here()
            return x0, y0, x1, y1
        else:
            raise RuntimeError("Empty iterator")

    def next(self):
        if self._k < self.total_steps():
            x0, y0, x1, y1 = self.here()
            self._k += 1
            return x0, y0, x1, y1
        else:
            raise StopIteration()

    def prev(self):
        if self._k >= 1:
            self._k -= 1
            x0, y0, x1, y1 = self.here()
            return x0, y0, x1, y1
        else:
            raise StopIteration()
##

#
# #####
# class TiledImage(object):
#     """A tiled image, loading regions on demand.
#
#     """
#     _meta_file   = None
#     _meta        = None
#     _root_folder = None
#
#     def __init__(self, meta_file: str):
#         """Initializes a TiledImage by reading the configuration from a file.
#
#         Args:
#             meta_file (str): the file with meta-info for the image. This should be
#                 located at the root of the folder hierarchy.
#         """
#         self._root_folder, self._meta_file = os.path.split(meta_file)
#         with open(meta_file, 'r') as fp:
#             self._meta = json.load(fp)
#
#     @property
#     def root_folder(self):
#         return self._root_folder
#
#     @property
#     def height(self):
#         return np.long(self._meta['level_image_height'])
#
#     @property
#     def width(self):
#         return np.long(self._meta['level_image_width'])
#
#     @property
#     def level(self):
#         return int(self._meta['level'])
#
#     @property
#     def tile_count_horizontal(self):
#         return int(self._meta['n_tiles_horiz'])
#
#     @property
#     def tile_count_vertical(self):
#         return int(self._meta['n_tiles_vert'])
#
#     @property
#     def tile_count(self):
#         return self.tile_count_horizontal * self.tile_count_vertical
#
#     @property
#     def tile_width(self):
#         return int(self._meta['tile_width'])
#
#     @property
#     def tile_height(self):
#         return int(self._meta['tile_height'])
#
#     @property
#     def n_channels(self):
#         return int(self._meta['n_channels'])
#
#     @staticmethod
#     def tile_id(i: int, j: int) -> str:
#         return 'tile_' + str(i) + '_' + str(j)
#
#     def get_nonempty_tiles(self) -> list:
#         nh, nv = self.tile_count_horizontal, self.tile_count_vertical
#         tiles = [TiledImage.tile_id(i,j) for i in range(nv) for j in range(nh) \
#                  if self._meta[TiledImage.tile_id(i,j)]['is_empty'] == 'False']
#         return tiles
#
#     def get_tile(self, i, j, skip_empty=True):
#         """Return the (i,j)-th tile.
#         Args:
#             i, j (int): tile coordinates
#             skip_empty (bool): if true and if the file corresponding to the
#                 tile (i,j) does not exist, assume the tile is empty and
#                 return a 0-filled array.
#
#         Returns:
#             numpy.ndarray
#         """
#         tile_id = 'tile_' + str(i) + '_' + str(j)
#         if (skip_empty and not os.path.exists(self.root_folder + '/' + self._meta['tile_' + str(i) + '_' + str(j)]['name'])) or \
#             self._meta[tile_id]["is_empty"] == "True":
#             img = np.zeros((self._meta[tile_id]['height'], self._meta[tile_id]['width'],
#                             self._meta[tile_id]['n_channels']), dtype=np.uitn8)
#         else:
#             img = imread(self.root_folder + os.path.pathsep + 'level_{:d}'.format(self._meta['level']) + \
#                                   os.path.sep + self._meta[tile_id]['name'])
#
#         return img
#
#     def is_tile_empty(self, i: int, j: int) -> bool:
#         tile_id = 'tile_' + str(i) + '_' + str(j)
#         return self._meta[tile_id]["is_empty"] == "True"
#
#     def get_image(self):
#         """Return the whole image, by loading all tiles."""
#         return TiledImage.load_tiled_image(self._meta)
#
#     def get_tile_coverage(self, x, y, width, height):
#         """Return the list of indices (i,j) of the tiles covering a given
#         rectangular region.
#
#         Args:
#             x, y (long): top-left corner coordinates (column, row)
#             width, height (long): region extent
#
#         Returns:
#             list of pairs: [(i,j), ...] corresponding to tiles_i_j covering
#              the region
#         """
#         x, y, width, height = [np.long(_z) for _z in [x, y, width, height]]
#         if not (0 <= x < self.width):
#             raise Error('x out of bounds')
#         if not (0 <= y < self.height):
#             raise Error('y out of bounds')
#         if x + width > self.width or y + height > self.height:
#             raise Error('region too large for the image')
#
#         # Find the tiles covering the requested reqion
#         start_i = np.int(np.floor(y / self.tile_height))
#         start_j = np.int(np.floor(x / self.tile_width))
#         end_i = np.int(np.floor((y + height) / self.tile_height) + \
#                 (1 if (y + height) % self.tile_height != 0 else 0))
#         end_j = np.int(np.floor((x + width) / self.tile_width) + \
#                 (1 if (x + width) % self.tile_width != 0 else 0))
#
#         ij = [(i, j) for i in np.arange(start_i, end_i) for j in np.arange(start_j, end_j)]
#
#         return ij
#
#     def get_region_px(self, x, y, width, height):
#         """Return an arbitrary region within a tiled image.
#         Args:
#             x, y (long): top-left corner coordinates (column, row)
#             width, height (long): region extent
#
#         Returns:
#             numpy.ndarray
#         """
#         x, y, width, height = [np.long(_z) for _z in [x, y, width, height]]
#         if not (0 <= x < self.width):
#             raise Error('x out of bounds')
#         if not (0 <= y < self.height):
#             raise Error('y out of bounds')
#         if x + width > self.width or y + height > self.height:
#             raise Error('region too large for the image')
#
#         # Algo:
#         # -find the tiles to load
#         # -load all the tiles
#         # -adjust, if needed, the starting and ending points of the
#         #  region
#         # This is not optimal from a memory usage perspective, but
#         # it's simpler.
#
#         # Find the tiles covering the requested reqion
#         start_i = np.int(np.floor(y / self.tile_height))
#         start_j = np.int(np.floor(x / self.tile_width))
#         end_i = np.int(np.floor((y + height) / self.tile_height) + \
#                 (1 if (y + height) % self.tile_height != 0 else 0))
#         end_j = np.int(np.floor((x + width) / self.tile_width) + \
#                 (1 if (x + width) % self.tile_width != 0 else 0))
#
#         # Load the tiles start_i:end_i, start_j:end_j
#         tile = self.get_tile(start_i, start_j)
#         nchannels = 1 if tile.ndim == 2 else 3
#         if nchannels == 1:
#             img = np.zeros((self.tile_height * (end_i - start_i),
#                             self.tile_width * (end_j - start_j)), dtype=np.uint8)
#         else:
#             img = np.zeros((self.tile_height * (end_i - start_i),
#                             self.tile_width * (end_j - start_j),
#                             tile.shape[2]), dtype=np.uint8)
#
#         if nchannels == 1:
#             for i in range(start_i, end_i):
#                 for j in range(start_i, end_i):
#                     tile = self.get_tile(i, j)
#
#                     # last tile in row and last row of tiles might have non-standard
#                     # dimensions, so better use the actual tile shape in computing the
#                     # end point:
#                     img[(i-start_i)*self.tile_height:(i-start_i)*self.tile_height + tile.shape[0],
#                         (j-start_j)*self.tile_width:(j-start_j)*self.tile_width + tile.shape[1]] = tile
#         else:
#             for i in range(start_i, end_i):
#                 for j in range(start_j, end_j):
#                     tile = self.get_tile(i, j)
#
#                     # last tile in row and last row of tiles might have non-standard
#                     # dimensions, so better use the actual tile shape in computing the
#                     # end point:
#                     img[(i-start_i)*self.tile_height:(i-start_i)*self.tile_height + tile.shape[0],
#                         (j-start_j)*self.tile_width:(j-start_j)*self.tile_width + tile.shape[1], :] = tile
#
#         # Adjust image to the requested region:
#         if nchannels == 1:
#             res = img[y - start_i*self.tile_height : y + height - start_i*self.tile_height,
#                   x - start_j * self.tile_width : x + width - start_j * self.tile_width].copy()
#         else:
#             res = img[y - start_i*self.tile_height : y + height - start_i*self.tile_height,
#                   x - start_j * self.tile_width : x + width - start_j * self.tile_width, :].copy()
#
#         return res
#
#
#     def load_tiled_image(self):
#         """Load a tiled image. All the information about the tile geometry and tile paths is
#          supposed to be already stored in the object itself.
#
#          The meta info contains:
#                     level_image_width
#                     level_image_height
#                     level_image_nchannels
#                     n_tiles_horiz
#                     n_tiles_vert
#                 and for each tile, an entry as
#                     'tile_i_j' which is a dict with keys:
#                     i
#                     j
#                     name
#                     x
#                     y
#
#         Canonical usage:
#             img = TiledImage("path/to/meta_data_file.json").load_tiled_image()
#
#         Returns:
#             a numpy.ndarray
#         """
#         img_w, img_h = self.width, self.height
#
#         nh, nv = self.tile_count_horizontal, self.tile_count_vertical
#
#         img = np.zeros((img_h, img_w, self.n_channels), dtype=np.uint8)
#
#         for i in range(nv):
#             for j in range(nh):
#                 tile_id = 'tile_'+str(i)+'_'+str(j)
#                 if self._meta[tile_id]['is_empty'] == 'False':
#                     # not empty, need to read the tile
#                     tile = imread(self.root_folder + os.path.sep + \
#                                   'level_{:d}'.format(self._meta['level']) + os.path.sep + \
#                                   self._meta[tile_id]['name']).astype(np.uint8)
#                     # the tile might not have the regular default shape, so it's better to use the
#                     # tile's shape than 'tile_width' and 'tile_height'
#                     x, y = np.long(self._meta[tile_id]['x']), np.long(self._meta[tile_id]['y'])
#                     img[x:x+tile.width, y:y+tile.height, :] = tile
#
#         return img
#
#
#     @staticmethod
#     def save_tiled_image(img: np.ndarray, root: str, level: int, tile_geom: tuple, img_type: str="jpeg",
#                          skip_empty: bool=True, empty_level: float=0):
#         """Save an image as a collection of tiles. This is a static method since the required meta information
#         about the hierarchy is not yet know (is computed here) and, hence, the object could not have been
#         initialized.
#
#         The image is split into a set of fixed-sized (with the exception of right-most and
#         bottom-most) tiles.
#
#         *WARNING*: any existing tiles in the path root/level will be deleted!
#
#         Args:
#             img (numpy array): an image (RGB)
#             root (string): root folder of the image storing hierarchy. The tiles will be
#                 stored into root/level_xx folder
#             level (int): the magnification level
#             tile_geom (tuple): (width, height) of the tile
#             img_type (string, optional): file type for the tiles
#             skip_empty (bool): if true, do not save images for empty tiles (those for which
#                 the sum of pixels (intensity or color) is <= empty_level).
#             empty_level (int): if the sum of pixels is at most this value, the image/tile is
#                 considered empty
#         Returns:
#             a TiledImage object
#         """
#         assert(img.ndim == 2 or (img.ndim == 3 and img.shape[2] <= 3))
#
#         n_channels = NumpyImage.nchannels(img)
#
#         tg = (min(tile_geom[0], img.shape[1]), min(tile_geom[1], img.shape[0]))
#         nh = int(floor(img.shape[1] / tg[0])) + (1 if img.shape[1] % tg[0] != 0 else 0)
#         nv = int(floor(img.shape[0] / tg[1])) + (1 if img.shape[0] % tg[1] != 0 else 0)
#
#         tile_meta = dict({'level': level,
#                           'level_image_width': img.shape[1],
#                           'level_image_height': img.shape[0],
#                           'level_image_nchannels': 1 if img.ndim == 2 else img.shape[2],
#                           'n_tiles_horiz': nh,
#                           'n_tiles_vert': nv,
#                           'tile_width': tg[0],
#                           'tile_height': tg[1],
#                           'n_channels': n_channels})
#
#         dst_path = root + os.path.sep + 'level_{:d}'.format(level)
#
#         if os.path.exists(dst_path):
#             shutil.rmtree(dst_path)
#         os.mkdir(dst_path)
#
#         for i in range(nv):
#             for j in range(nh):
#                 i0, j0 = i * tg[1], j * tg[0]
#                 i1, j1 = min((i + 1) * tg[1], img.shape[0]), min((j + 1) * tg[0], img.shape[1])
#                 if n_channels == 1:
#                     im_sub = img[i0:i1, j0:j1]
#                 else:
#                     im_sub = img[i0:i1, j0:j1, :]
#                 tile_id = 'tile_' + str(i) + '_' + str(j)
#                 tile_meta[tile_id] = dict(
#                     {'name': tile_id + '.' + img_type,
#                      'i': i, 'j': j,
#                      'x': j0, 'y': i0,
#                      'width': (j1-j0), 'height': (i1-i0)})
#                 tile_meta[tile_id]['is_empty'] = str(NumpyImage.is_empty(im_sub, empty_level))
#                 if skip_empty and NumpyImage.is_empty(im_sub, empty_level):
#                     continue
#                 # effectively save the tile
#                 imsave(dst_path + os.path.sep + tile_meta[tile_id]['name'], im_sub, quality=100)
#
#         with open(dst_path + os.path.sep + 'meta.json', 'w') as fp:
#             json.dump(tile_meta, fp, separators=(',', ':'), indent='  ', sort_keys=True)
#
#         return TiledImage(meta_file=dst_path + os.path.sep + 'meta.json')
#
#
# ###
# class TiledArrayH5(TiledImage):
#     """
#     This class considers the tiles stored as HDF5 arrays in individual
#     files (*.h5). Additionally, the arrays are all stored under a uniform
#     path (within .h5 file) to be provided via constructor. The meta file
#     to be provided should be structured as for TileImage objects.
#     """
#     _h5_path         = None
#     _first_non_empty = None  # the first non empty tile to test a few things (no. channels, h5path,...)
#
#     def __init__(self, meta_file:str, h5_path: str):
#         super(TiledArrayH5, self).__init__(meta_file)
#         self._h5_path = h5_path
#
#         for i in range(self.tile_count_vertical):
#             for j in range(self.tile_count_horizontal):
#                 tile_id = 'tile_{:d}_{:d}'.format(i, j)
#                 if self._meta[tile_id]['is_empty'] == "False":
#                     self._first_non_empty = (i, j)
#                     break
#             if self._first_non_empty is not None:
#                 break
#
#         if self._first_non_empty is None:
#             raise RuntimeWarning('Could not find non-empty tiles.')
#
#         i, j = self._first_non_empty
#         tile_id = 'tile_{:d}_{:d}'.format(i, j)
#         with h5.File(os.path.join(self.root_folder, tile_id + '.h5'), 'r') as f:
#             if self._h5_path not in f:
#                 raise RuntimeError('Cannot find the specified path within HDF5 files.')
#             self._meta['ndim'] = f[self._h5_path].ndim
#             if f[self._h5_path].ndim == 2:
#                 self._meta['n_channels'] = 1
#             else:
#                 self._meta['n_channels'] = f[self._h5_path].shape[2]
#             self._tile_shape = f[self._h5_path].shape
#             self._meta['dtype'] = f[self._h5_path].dtype
#
#         return
#
#
#     def get_tile(self, i, j, skip_empty=True):
#         """Return the (i,j)-th tile.
#         Args:
#             i, j (int): tile coordinates
#             skip_empty (bool): if true and if the file corresponding to the
#                 tile (i,j) does not exist, assume the tile is empty and
#                 return a 0-filled array.
#
#         Returns:
#             numpy.ndarray
#         """
#         tile_id = 'tile_{:d}_{:d}'.format(i, j)
#         tile_file = os.path.join(self.root_folder, tile_id+'.h5')
#         if self._meta['ndim'] == 2:
#             arr = np.zeros((self._meta[tile_id]['height'], self._meta[tile_id]['width']), dtype=self._meta['dtype'])
#         else:
#             arr = np.zeros((self._meta[tile_id]['height'], self._meta[tile_id]['width'],
#                             self._meta['n_channels']), dtype=self._meta['dtype'])
#
#         if os.path.exists(tile_file) and self._meta[tile_id]["is_empty"] == "False":
#             with h5.File(tile_file, 'r') as f:
#                 f[self._h5_path].read_direct(arr)
#
#         return arr.squeeze()
#
#
#     def load_tiled_array(self, squeeze=True):
#         """
#         Load the whole array from tiles. Empty tiles are filled with 0s.
#         :return:
#             numpy.ndarray
#         """
#         if self._meta['ndim'] == 2:
#             arr = np.zeros((self.height, self.width), dtype=self._meta['dtype'])
#         else:
#             arr = np.zeros((self.height, self.width, self._meta['n_channels']),
#                            dtype=self._meta['dtype'])
#
#         for i in range(self.tile_count_vertical):
#             for j in range(self.tile_count_horizontal):
#                 tile_id = 'tile_{:d}_{:d}'.format(i, j)
#                 if self._meta[tile_id]['is_empty'] == 'False':
#                     tile_file = os.path.join(self.root_folder, tile_id + '.h5')
#                     if os.path.exists(tile_file):
#                         with h5.File(tile_file, 'r') as f:
#                             x, y = np.long(self._meta[tile_id]['x']), np.long(self._meta[tile_id]['y'])
#                             w, h = np.long(self._meta[tile_id]['width']), np.long(self._meta[tile_id]['height'])
#                             f[self._h5_path].read_direct(arr, np.s_[0:h, 0:w, ...], np.s_[y:y+h, x:x+w, ...])
#
#         return arr.squeeze() if squeeze else arr
#
#
#     def load_tiled_image(self):
#         return self.load_tiled_array()
#
#
#     def save_tiled_image(img: np.ndarray, root: str, level: int, tile_geom: tuple, img_type: str="jpeg",
#                          skip_empty: bool=True, empty_level: float=0):
#         raise RuntimeError("Not implemented")
#
#
#     def get_region_px(self, x, y, width, height):
#         raise RuntimeError("Not implemented")
#
#
# ###
# class TiledArrayNPY(TiledImage):
#     """
#     This class considers the tiles stored as Numpy arrays in individual
#     files (*.npy or *.npz), a single array per file. The meta file
#     to be provided should be structured as for TileImage objects.
#     """
#     _first_non_empty = None  # the first non empty tile to test a few things (no. channels, h5path,...)
#     _tile_ext        = None  # the extension of the files (.npy or .npz)
#
#     def __init__(self, meta_file:str, ext='.npy'):
#         super(TiledArrayNPY, self).__init__(meta_file)
#         self._tile_ext = ext
#
#         for i in range(self.tile_count_vertical):
#             for j in range(self.tile_count_horizontal):
#                 tile_id = 'tile_{:d}_{:d}'.format(i, j)
#                 if self._meta[tile_id]['is_empty'] == "False":
#                     self._first_non_empty = (i, j)
#                     break
#             if self._first_non_empty is not None:
#                 break
#
#         if self._first_non_empty is None:
#             raise RuntimeWarning('Could not find non-empty tiles.')
#
#         i, j = self._first_non_empty
#         tile_id = 'tile_{:d}_{:d}'.format(i, j)
#         tmp = np.load(os.path.join(self.root_folder, tile_id + self._tile_ext)).squeeze()
#         self._meta['ndim'] = tmp.ndim
#         if tmp.ndim == 2:
#             self._meta['n_channels'] = 1
#         else:
#             self._meta['n_channels'] = tmp.shape[2]
#         self._tile_shape = tmp.shape
#         self._meta['dtype'] = tmp.dtype
#
#         return
#
#
#     def get_tile(self, i, j, skip_empty=True):
#         """Return the (i,j)-th tile.
#         Args:
#             i, j (int): tile coordinates
#             skip_empty (bool): if true and if the file corresponding to the
#                 tile (i,j) does not exist, assume the tile is empty and
#                 return a 0-filled array.
#
#         Returns:
#             numpy.ndarray
#         """
#         tile_id = 'tile_{:d}_{:d}'.format(i, j)
#         tile_file = os.path.join(self.root_folder, tile_id+self._tile_ext)
#         if self._meta['ndim'] == 2:
#             arr = np.zeros((self._meta[tile_id]['height'], self._meta[tile_id]['width']), dtype=self._meta['dtype'])
#         else:
#             arr = np.zeros((self._meta[tile_id]['height'], self._meta[tile_id]['width'],
#                             self._meta['n_channels']), dtype=self._meta['dtype'])
#
#         if os.path.exists(tile_file) and self._meta[tile_id]["is_empty"] == "False":
#             # silently turns missing tiles into 0s
#             arr = np.load(tile_file).squeeze()
#
#         return arr
#
#
#     def load_tiled_array(self, squeeze=True):
#         """
#         Load the whole array from tiles. Empty tiles are filled with 0s.
#         :return:
#             numpy.ndarray
#         """
#         if self._meta['ndim'] == 2:
#             arr = np.zeros((self.height, self.width), dtype=self._meta['dtype'])
#         else:
#             arr = np.zeros((self.height, self.width, self._meta['n_channels']),
#                            dtype=self._meta['dtype'])
#
#         for i in range(self.tile_count_vertical):
#             for j in range(self.tile_count_horizontal):
#                 tile_id = 'tile_{:d}_{:d}'.format(i, j)
#                 x, y = np.long(self._meta[tile_id]['x']), np.long(self._meta[tile_id]['y'])
#                 w, h = np.long(self._meta[tile_id]['width']), np.long(self._meta[tile_id]['height'])
#                 arr[np.s_[y:y+h, x:x+w, ...]] = self.get_tile(i, j)
#
#         return arr.squeeze() if squeeze else arr
#
#
#     def load_tiled_image(self):
#         return self.load_tiled_array()
#
#
#     def save_tiled_image(img: np.ndarray, root: str, level: int, tile_geom: tuple, img_type: str="jpeg",
#                          skip_empty: bool=True, empty_level: float=0):
#         raise RuntimeError("Not implemented")
#
#
#     def get_region_px(self, x, y, width, height):
#         raise RuntimeError("Not implemented")
