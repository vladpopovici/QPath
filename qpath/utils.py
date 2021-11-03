# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = 0.1

#
# QPATH.UTILS: handy functions
#

__all__ = []

import numpy as np
import shapely.geometry
import simplejson as json
import pyvips

from . import Error


def geom2xy(geom: shapely.geometry, as_type=None) -> np.array:
    """Return the coordinates of a 2D geometrical object as a numpy array (N x 2).

    :param geom: shapely.geometry
        a 2D geometrical object

    :return:
        numpy.array
    """

    if as_type is None:
        z = np.array(geom.array_interface_base['data'])
    else:
        z = np.array(geom.array_interface_base['data'], dtype=as_type)
    n = z.size // 2

    return z.reshape((n, 2))
##


class NumpyJSONEncoder(json.JSONEncoder):
    """Provides an encoder for Numpy types for serialization."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)
##


def np2vips(img: np.array) -> pyvips.Image:
    """Converts a NumPy image (3d array) to VIPS Image."""
    dtype_to_format = {
        'uint8': 'uchar',
        'int8': 'char',
        'uint16': 'ushort',
        'int16': 'short',
        'uint32': 'uint',
        'int32': 'int',
        'float32': 'float',
        'float64': 'double',
        'complex64': 'complex',
        'complex128': 'dpcomplex',
    }

    if img.ndim > 3:
        raise Error("the image may have at most 3 dimensions")
    if img.ndim == 3:
        height, width, bands = img.shape[:3]
    else:
        height, width, bands = img.shape[:2], 1

    linear = img.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                      dtype_to_format[str(img.dtype)])

    return vi
##


def write_pyramidal_tiff(img: np.array, file_name: str) -> None:
    """Write a Numpy array as a pyramidal tiled TIFF file.

    :param: img (np.array)
        the image
    :param: file_name (str)
        file to write to
    """
    v_img = np2vips(img)
    v_img.write_to_file(file_name, pyramid=True, tile=True, compression="jpeg")

    return
##