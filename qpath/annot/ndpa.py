# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = 0.1

#
# QPATH.ANNOT.NDPA: functions for handling Hamamatsu's annotation files (.NDPA)
#

__all__ = ['ndpa2xy', 'ndpa_read_single', 'ndpa_read']

import numpy as np
from ..base import WSIInfo
import xml.etree.ElementTree as ET

##-
def ndpa2xy(ndpa_annot: AnnotationObject, level: int, wsi_source: WSIInfo) -> AnnotationObject:
    """Convert a set of points from Hamamatsu's annotation file (.ndpa)
    to (x,y) image coordinates. This function accesses the low-level
    meta information about a WSI.

    Args:
        ndpa_annot (AnnotationObject): an annotation read from NDPA file
        level (int): magnification level (0: minimum magnification, 1: double of
            the minimum magnification, etc.)
        wsi_source (qpath.core.WSIInfo): an image source object

    Returns:
        an AnnotationObject of the same type as the input object, but with
        the coordinates in the image space.
    """

    raise RuntimeError('Not implemented')
#
#     if wsi_source.info['vendor'] != 'hamamatsu':
#         raise Error('vendor mismatch')
#     if level > wsi_source.level_count():
#         raise Error('level out of bounds')
#
#     d = 2**level
#     xy_coords = list()
#     for p in ndpa_annot.xy:
#         x, y = p
#         x -= wsi_source.info['x_offset']
#         y -= wsi_source.info['y_offset']
#         x /= (1000.0 * wsi_source.info['x_mpp'])
#         y /= (1000.0 * wsi_source.info['y_mpp'])
#         x = np.int64((x + wsi_source.info['levels'][0]['x_size'] / 2) / d)  # in pixels, relative to UL corner
#         y = np.int64((y + wsi_source.info['levels'][0]['y_size'] / 2) / d)
#
#         xy_coords.append([x, y])
#
#     out = ndpa_annot.duplicate()
#     out.xy = np.array(xy_coords, dtype=np.int64)
#     out._coordinate_system = AnnotationObject.COORDINATE_SYSTEM_IMAGE
#     out._coordinate_system_magnification = wsi_source.info['objective'] / wsi_source.info['levels'][level]['downsample_factor']
#
#     if np.any(out.xy < 0):
#         raise Error('negative coordinates')
#
#     return out
# ##-


##-
def ndpa_read_single(ndpa_file, ann_title):
    """Read a single annotation object from the NDPA file. Note that an
    annotation object may actually have several components, each with the
    same title. All these components are collected in a list of lists.

    Args:
        ndpa_file (str): filename
        ann_title (str): name of the annotation object

    Returns:
        -a list of lists [(x,y),...] with the coordinates of annotation
        points in slide coordinate system
        -None if the annotation object was not found

    See also:
        ndap_read
    """
    raise RuntimeError('Not implemented')
#     xml_file = ET.parse(ndpa_file)
#     xml_root = xml_file.getroot()
#
#     xy_coords = []
#
#     for ann in list(xml_root):
#         name = ann.find('title').text
#         if not name == ann_title:
#             continue
#         p = ann.find('annotation')
#         if p is None:
#             continue
#
#         p = p.find('pointlist')
#         if p is None:
#             continue
#
#         xy_coords.append([(long(pts.find('x').text), long(pts.find('y').text)) for pts in list(p)])
#
#
#     if len(xy_coords) == 0:
#         return None
#     else:
#         if len(xy_coords) > 1:
#             annot = []
#             for xy in xy_coords:
#                 xy = np.array(xy, dtype=np.floa64)
#                 if np.all(xy[0,] == xy[-1,]):
#                     # actually, it's a polygon
#                     annot.append(Polygon(xy, name=name))
#                 else:
#                     # it's a set of points
#                     annot.append(PointSet(xy, name=name))
#             return annot
#         else:
#             xy = np.array(xy_coords[0], dtype=np.float64)
#             if np.all(xy[0,] == xy[-1,]):
#                 # actually, it's a polygon
#                 return Polygon(xy, name=name)
#             else:
#                 # it's a set of points
#                 return PointSet(xy, name=name)
#
#     return None
##-


##-
def ndpa_read(ndpa_file, force_closed=False):
    """Read all annotations.

    Args:
        ndpa_file (str): annotation file name
        force_closed (bool): should polygonal lines be forced to form a closed
            contour (i.e. first and last points are identical)

    Returns:
        -a dictionary with keys corresponding to annotation object
        names and with values the corresponding lists of points

    See also:
        ndpa_read_single
    """

    raise RuntimeError('Not implemented')

    xml_file = ET.parse(ndpa_file)
    xml_root = xml_file.getroot()

    annot = dict()

    for ann in list(xml_root):
        name = ann.find('title').text
        annot[name] = []

        p = ann.find('annotation')
        if p is None:
            continue

        p = p.find('pointlist')
        if p is None:
            continue

        coords = np.array([(np.long(pts.find('x').text), np.long(pts.find('y').text))
                           for pts in list(p)])
        if np.all(coords[0,] == coords[-1,]):
            # actually, it's a polygon
            annot[name].append(Polygon(coords, name=name))
        else:
            # it's a set of points, but may be forced to Polygon
            if force_closed:
                coords = np.vstack((coords, coords[0,]))
                annot[name].append(Polygon(coords, name=name))
            else:
                annot[name].append(PointSet(coords, name=name))

    return annot
##-
