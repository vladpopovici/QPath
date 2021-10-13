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
