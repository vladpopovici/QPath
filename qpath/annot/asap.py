# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = 0.1

#
# QPATH.ANNOT.ASAP: functions for handling ASAP's annotation files
# (https://github.com/computationalpathologygroup/ASAP)
#

__all__ = ['annotation_to_asap']


import numpy as np
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from xml.dom import minidom

from . import Annotation


# def asap2xy(annot: AnnotationObject, level: int, wsi_source: WSIInfo)-> AnnotationObject:
#     """Convert a set of points from ASAP coordinates to (x,y) image coordinates as the
#     specified magnification level.

#     Args:
#         annot (AnnotationObject): an annotation read from XML file (ASAP format)
#         level (int): magnification level (0: maximum magnification, 1: half of the maximum
#             magnification, etc.)
#         wsi_source (qpath.core.WSIInfo): an image source object

#     Returns:
#         an AnnotationObject of the same type as the input object, but with the coordinates
#         in the image space, at the desired magnification.
#     """

#     # ASAP uses the image coordinates at maximum magnification. The only transformation
#     # needed is scaling, in case level != 0.

#     if level < 0:
#         raise RuntimeError("Negative level!")
#     if level == 0:
#         return annot

#     res = annot.copy()
#     res.scale(2.0**(-level))

#     return res
# ##-


# def asap_read_single():
#     pass


# def asap_read(asap_file:str):
#     tree = ET.parse(asap_file)
#     root = tree.getroot()

#     if root.tag != "ASAP_Annotations":
#         raise RuntimeWarning("Not an ASAP annotation!")
#         return None

#     return


def annotation_to_asap(annot: Annotation, asap_file: str, magnification: float = -1.0) -> bool:
    """Save an <Annotation> in an XML file compatible with ASAP software.

    :param annot: (Annotation) an annotation object
    :param asap_file: (str) name of the file for saving the annotation
    :param magnification: (float) if positive then the desired magnification for the saved
        annotation (all is scaled by a (magnification/annot.magnification) factor)

    :return: True if everything is OK
    """

    # scale the annotation for the target:
    if magnification > 0.0:
        annot.resize(magnification / annot._magnification)

    asap = Element('ASAP_Annotations')
    grps = SubElement(asap, 'AnnotationGroups')
    grp1 = SubElement(grps, 'Group',
                      {'Name': annot.name, 'PartOfGroup': 'None', 'Color': '#64FE2E'})
    _ = SubElement(grp1, 'Attribute')

    grps = SubElement(asap, 'Annotations')
    for a in annot._annots:
        if a._annotation_type == "DOT":
            el = SubElement(grps, 'Annotation',
                            {'Name': a.name, 'Type': 'Dot', 'PartOfGroup': annot.name, 'Color': '#F4FA58'})
            c = SubElement(el, 'Coordinates')
            _ = SubElement(c, 'Coordinate',
                           {'Order': '0', 'X': str(a.x()), 'Y': str(a.y())})
        elif a._annotation_type == "POLYGON":
            xy = a.xy()
            el = SubElement(grps, 'Annotation',
                            {'Name': a.name, 'Type': 'Polygon', 'PartOfGroup': annot.name, 'Color': '#F4FA58'})
            c = SubElement(el, 'Coordinates')
            for k in np.arange(xy.shape[0]):
                _ = SubElement(c, 'Coordinate',
                               {'Order': str(k), 'X': str(xy[k, 0]), 'Y': str(xy[k, 1])})
        elif a._annotation_type == "POINTSET":
            pass
        else:
            raise RuntimeError("unknown annotation type " + a._annotation_type)

    with open(asap_file, 'w') as out:
        xmlstr = minidom.parseString(ElementTree.tostring(asap, encoding='unicode')).toprettyxml(indent='  ')
        out.write(xmlstr)

    return
