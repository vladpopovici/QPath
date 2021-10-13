# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = 0.1

## This module handles annotations for own algorithms as well as several
## import/export formats (HistomicsTK, Hamamatsu, ASAP, etc).


__all__ = ['AnnotationObject', 'Dot', 'Polygon', 'PointSet', 'Annotation']

from abc import ABC, abstractmethod

import shapely.geometry as shg
import shapely.affinity as sha
import shapely.ops as sho
import geojson as gj

import numpy as np
import collections

from .. import Error

COORDINATE_SYSTEM_WSI = 'wsi'
COORDINATE_SYSTEM_IMAGE = 'image'


##-
class AnnotationObject(ABC):
    """Define the AnnotationObject minimal interface. This class is made
    abstract to force more meaningful names (e.g. Dot, Polygon, etc.) in
    subclasses."""

    def __init__(self):
        # main geometrical object describing the annotation:
        self.geom = shg.base.BaseGeometry()

        self._name = None
        self._annotation_type = None

    @abstractmethod
    def duplicate(self):
        pass

    def __str__(self):
        """Return a string representation of the object."""
        return str(self.type) + " <" + str(self.name) + ">: " + str(self.geom)

    def bounding_box(self):
        """Compute the bounding box of the object."""
        return self.geom.bounds

    def translate(self, x_off, y_off=None):
        """Translate the object by a vector [x_off, y_off], i.e.
        the new coordinates will be x' = x + x_off, y' = y + y_off.
        If y_off is None, then the same value as in x_off will be
        used.

        :param x_off: (double) shift in thr X-direction
        :param y_off: (double) shift in the Y-direction; if None,
            y_off == x_off
        """
        if y_off is None:
            y_off = x_off
        self.geom = sha.translate(self.geom, x_off, y_off, zoff=0.0)

        return

    def scale(self, x_scale, y_scale=None, origin='center'):
        """Scale the object by a specified factor with respect to a specified
        origin of the transformation. See shapely.geometry.scale() for details.

        :param x_scale: (double) X-scale factor
        :param y_scale: (double) Y-scale factor; if None, y_scale == x_scale
        :param origin: reference point for scaling. Default: "center" (of the
            object). Alternatives: "centroid" or a shapely.geometry.Point object
            for arbitrary origin.
        """
        if y_scale is None:
            y_scale = x_scale
        self.geom = sha.scale(self.geom, xfact=x_scale, yfact=y_scale, zfact=1, origin=origin)

        return

    def resize(self, factor: float) -> None:
        """Resize an object with the specified factor. This is equivalent to
        scaling with the origin set to (0,0) and same factor for both x and y
        coordinates.

        :param factor: (float) resizig factor.
        """
        self.scale(factor, origin=shg.Point((0.0, 0.0)))

        return

    def affine(self, M):
        """Apply an affine transformation to all points of the annotation.

        If M is the affine transformation matrix, the new coordinates
        (x', y') of a point (x, y) will be computed as

        x' = M[1,1] x + M[1,2] y + M[1,3]
        y' = M[2,1] x + M[2,2] y + M[2,3]

        In other words, if P is the 3 x n matrix of n points,
        P = [x; y; 1]
        then the new matrix Q is given by
        Q = M * P

        :param M: numpy array [2 x 3]

        :return:
            nothing
        """

        self.geom = sha.affine_transform(self.geom, [M[0, 0], M[0, 1], M[1, 0], M[1, 1], M[0, 2], M[1, 2]])

        return

    def x(self):
        """Return the x coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        raise NotImplementedError

    def y(self):
        """Return the y coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        raise NotImplementedError

    def xy(self) -> np.array:
        """Return the xy-coordinates as a numpy.array"""
        raise NotImplementedError

    def size(self) -> int:
        """Return the number of points defining the object."""
        raise NotImplementedError

    def asdict(self) -> dict:
        """Return a dictionary representation of the object."""
        raise NotImplementedError

    def fromdict(self, d: dict) -> None:
        """Intialize the objct from a dictionary."""
        raise NotImplementedError

    def asGeoJSON(self) -> dict:
        """Return a dictionary compatible with GeoJSON specifications."""
        return gj.Feature(geometry=shg.mapping(self.geom),
                          properties=dict(object_type="annotation",
                                          annotation_type=self._annotation_type,
                                          name=self._name)
                          )

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize the object from a dictionary compatible with GeoJSON specifications."""
        """This is a basic function - further tests should be implemented for particular object
        types."""

        self.geom = shg.shape(d["geometry"])
        try:
            self._name = d["properties"]["name"]
        except KeyError:
            pass

    @property
    def name(self):
        """Return the name of the annotation object."""
        return self._name

    @property
    def type(self):
        """Return the annotation type as a string."""
        return self._annotation_type


##-


##-
class Dot(AnnotationObject):
    """Dot: a single position in the image."""

    def __init__(self, coords=[0.0, 0.0], name=None):
        """Initialize a DOT annotation, i.e. a single point in plane.

        Args:
            coords (list or vector or tuple): the (x,y) coordinates of the point
            name (str): the name of the annotation
        """
        self._annotation_type = "DOT"
        self._name = "DOT"

        if not isinstance(coords, collections.Iterable):
            raise Error('coords parameter cannot be interpreted as a 2D vector')

        if name is not None:
            self._name = name

        self.geom = shg.Point(coords)

        return

    def duplicate(self):
        return Dot(np.array(self.geom.array_interface_base['data']), name=self.name)

    def x(self):
        """Return the x coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        return self.geom.array_interface_base['data'][0]

    def y(self):
        """Return the y coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        return self.geom.array_interface_base['data'][1]

    def xy(self) -> np.array:
        return np.array(self.geom.array_interface_base['data']).reshape((1, 2))

    def size(self) -> int:
        """Return the number of points defining the object."""
        return 1

    def asdict(self) -> dict:
        """Return a dictionary representation of the object."""
        d = {
            "annotation_type": self._annotation_type,
            "name": self._name,
            "x": self.x(),
            "y": self.y()
        }

        return d

    def fromdict(self, d: dict) -> None:
        """Intialize the objct from a dictionary."""

        self._annotation_type = d["annotation_type"]
        self._name = d["name"]
        self.geom = shg.Point((d["x"], d["y"]))

        return

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize the object from a dictionary compatible with GeoJSON specifications."""
        if d["geometry"]["type"].lower() != "point":
            raise RuntimeError("Need a Point feature! Got: " + str(d))

        super().fromGeoJSON(d)
        self._annotation_type = "DOT"

        return


##-


##-
class PointSet(AnnotationObject):
    """PointSet: an ordered collection of points."""

    def __init__(self, coords, name=None):
        """Initialize a POINTSET annotation, i.e. a collection
         of points in plane.

        Args:
            coords (list or tuple): coordinates of the points as in [(x0,y0), (x1,y1), ...]
            name (str): the name of the annotation
        """

        self._annotation_type = "POINTSET"
        self._name = "POINTS"

        if name is not None:
            self._name = name

        # check whether coords is iterable and build the coords from it:
        if not isinstance(coords, collections.Iterable):
            raise Error('coords parameter cannot be interpreted as a 2D array')

        self.geom = shg.MultiPoint(coords)

        return

    def duplicate(self):
        return PointSet(np.array(self.geom.array_interface_base['data']), name=self.name)

    def x(self):
        """Return the x coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        return self.geom.array_interface_base['data'][0::2]

    def y(self):
        """Return the y coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        return self.geom.array_interface_base['data'][1::2]

    def xy(self) -> np.array:
        """Return the xy-coordinates as a numpy.array"""
        return np.array(self.geom.array_interface_base['data']).reshape((self.size(), 2))

    def size(self) -> int:
        """Return the number of points defining the object."""
        return np.array(self.geom.array_interface_base['data']).size // 2

    def asdict(self) -> dict:
        """Return a dictionary representation of the object."""
        d = {
            "annotation_type": self._annotation_type,
            "name": self._name,
            "x": self.x(),
            "y": self.y()
        }

        return d

    def fromdict(self, d: dict) -> None:
        """Intialize the objct from a dictionary."""

        self._annotation_type = d["annotation_type"]
        self._name = d["name"]
        self.geom = shg.MultiPoint(zip(d["x"], d["y"]))

        return

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize the object from a dictionary compatible with GeoJSON specifications."""
        if d["geometry"]["type"].lower() != "multipoint":
            raise RuntimeError("Need a MultiPoint feature! Got: " + str(d))

        super().fromGeoJSON(d)
        self._annotation_type = "POINTSET"

        return


##-


class PolyLine(AnnotationObject):
    """PolyLine: polygonal line (a sequence of segments)"""

    def __init__(self, coords, name=None):
        """Initialize a POLYLINE object.

        Args:
            coords (list or tuple): coordinates of the points [(x0,y0), (x1,y1), ...]
                defining the segments (x0,y0)->(x1,y1); (x1,y1)->(x2,y2),...
            name (str): the name of the annotation
        """
        self._annotation_type = "POLYLINE"
        self._name = name if not None else "POLYLINE"

        # check whether x is iterable and build the coords from it:
        if not isinstance(coords, collections.Iterable):
            raise Error('coords parameter cannot be interpreted as a 2D array')

        self.geom = shg.LineString(coords)

        return

    def duplicate(self):
        return PolyLine(np.array(self.geom.array_interface_base['data']), name=self.name)

    def x(self):
        """Return the x coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        return self.geom.array_interface_base['data'][0::2]

    def y(self):
        """Return the y coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        return self.geom.array_interface_base['data'][1::2]

    def xy(self) -> np.array:
        """Return the xy-coordinates as a numpy.array"""
        return np.array(self.geom.array_interface_base['data']).reshape((self.size(), 2))

    def size(self) -> int:
        """Return the number of points defining the object."""
        return np.array(self.geom.array_interface_base['data']).size // 2

    def asdict(self) -> dict:
        """Return a dictionary representation of the object."""
        d = {
            "annotation_type": self._annotation_type,
            "name": self._name,
            "x": self.x(),
            "y": self.y()
        }

        return d

    def fromdict(self, d: dict) -> None:
        """Intialize the objct from a dictionary."""

        self._annotation_type = "POLYLINE"
        self._name = d["name"]
        self.geom = shg.LineString(zip(d["x"], d["y"]))

        return

    def asGeoJSON(self) -> dict:
        """Return a dictionary compatible with GeoJSON specifications."""
        return gj.Feature(geometry=gj.LineString(zip(self.x(), self.y())),
                          properties=dict(object_type="annotation",
                                          annotation_type=self._annotation_type,
                                          name=self._name)
                          )

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize the object from a dictionary compatible with GeoJSON specifications."""
        if d["geometry"]["type"].lower() != "linestring":
            raise RuntimeError("Need a LineString feature! Got: " + str(d))

        super().fromGeoJSON(d)
        self._annotation_type = "POLYLINE"

        return


##-

##-
class Polygon(AnnotationObject):
    """Polygon: an ordered collection of points where the first and
    last points coincide."""

    def __init__(self, coords, name=None):
        """Initialize a POINTSET annotation, i.e. a collection
         of points in plane.

        Args:
            coords (list or tuple): coordinates of the points as in [(x0,y0), (x1,y1), ...]
            name (str): the name of the annotation
        """

        self._annotation_type = "POLYGON"
        self._name = name if not None else "POLYGON"

        #        if name is not None:
        #            self._name = name

        # check whether x is iterable and build the coords from it:
        if not isinstance(coords, collections.Iterable):
            raise Error('coords parameter cannot be interpreted as a 2D array')

        self.geom = shg.Polygon(coords)

        return

    def duplicate(self):
        return Polygon(np.array(self.geom.array_interface_base['data']), name=self.name)

    def x(self):
        """Return the x coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        return self.geom.array_interface_base['data'][0::2]

    def y(self):
        """Return the y coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        return self.geom.array_interface_base['data'][1::2]

    def xy(self) -> np.array:
        """Return the xy-coordinates as a numpy.array"""
        return np.array(self.geom.array_interface_base['data']).reshape((self.size(), 2))

    def size(self) -> int:
        """Return the number of points defining the object."""
        return np.array(self.geom.array_interface_base['data']).size // 2

    def asdict(self) -> dict:
        """Return a dictionary representation of the object."""
        d = {
            "annotation_type": self._annotation_type,
            "name": self._name,
            "x": self.x(),
            "y": self.y()
        }

        return d

    def fromdict(self, d: dict) -> None:
        """Intialize the objct from a dictionary."""

        self._annotation_type = d["annotation_type"]
        self._name = d["name"]
        self.geom = shg.Polygon(zip(d["x"], d["y"]))

        return

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize the object from a dictionary compatible with GeoJSON specifications."""
        if d["geometry"]["type"].lower() != "polygon":
            raise RuntimeError("Need a Polygon feature! Got: " + str(d))

        super().fromGeoJSON(d)
        self._annotation_type = "POLYGON"

        return


##-


def createEmptyAnnotationObject(annot_type: str) -> AnnotationObject:
    """Function to create an empty annotation object of a desired type.

    Args:
        annot_type (str):
            type of the annotation object:
            DOT/POINT
            POINTSET
            POLYLINE/LINESTRING
            POLYGON

    """

    obj = None
    if annot_type.upper() == 'DOT' or annot_type.upper() == 'POINT':
        obj = Dot(coords=[0, 0])
    elif annot_type.upper() == 'POINTSET':
        obj = PointSet([[0, 0]])
    elif annot_type.upper() == 'LINESTRING' or annot_type.upper() == 'POLYLINE':
        obj = PolyLine([[0, 0], [1, 1], [2, 2]])
    elif annot_type.upper() == 'POLYGON':
        obj = Polygon([[0, 0], [1, 1], [2, 2]])
    else:
        raise RuntimeError("unknown annotation type: " + annot_type)
    return obj


##-
class Annotation(object):
    """
    An annotation is a list of AnnotationObjects extracted from a region of interest (ROI).
    The coordinates are in base image coordinates (highest resolution) and relative to ROI (for
    the annotation objects). ROI coordinates are relative to base image.
    """

    def __init__(self, name: str, image_shape: dict, magnification: float) -> None:
        """Initialize an Annotation for a slide.

        :param name: (str) name of the annotation
        :param image_shape: (dict) shape of the image corresponding to the annotation
            {'width':..., 'height':...}
        :param magnification: (float) slide magnification for the image
        """
        self._name = name
        self._image_shape = dict(width=0, height=0)
        self._annots = []
        self._magnification = magnification

        if 'width' not in image_shape or 'height' not in image_shape:
            raise RuntimeError('Invalid shape specification (<width> or <height> key missing)')

        self._image_shape = image_shape

        return

    def add_annotation_object(self, a: AnnotationObject) -> None:
        self._annots.append(a)

    def add_annotations(self, a: list) -> None:
        self._annots.extend(a)

    def get_base_image_shape(self) -> dict:
        return self._image_shape

    @property
    def name(self):
        """Return the name of the annotation object."""
        return self._name

    @property
    def type(self):
        """Return the annotation type as a string."""
        return 'Annotation'

    def get_magnification(self) -> float:
        return self._magnificaiton

    def resize(self, factor: float) -> None:
        self._magnification *= factor
        self._image_shape['width'] *= factor
        self._image_shape['height'] *= factor

        for a in self._annots:
            a.resize(factor)

        return

    def set_magnification(self, magnification: float) -> None:
        """Scales the annotation to the desired magnification.

        :param magnfication: (float) target magnification
        """
        if magnification != self._magnification:
            f = magnification / self._magnification
            self.resize(f)
            self._magnification = magnification

        return

    def asdict(self) -> dict:
        d = {'name': self._name,
             'image_shape': self._image_shape,
             'magnification': self._magnification,
             'annotations': [a.asdict() for a in self._annots]
             }

        return d

    def fromdict(self, d: dict) -> None:
        self._name = d['name']
        self._image_shape = d['image_shape']
        self._magnification = d['magnification']

        self._annots.clear()
        for a in d['annotations']:
            obj = createEmptyAnnotationObject(a['annotation_type'])
            obj.fromdict(a)
            self.add_annotation_object(obj)

        return

    def asGeoJSON(self) -> dict:
        """Creates a dictionary compliant with GeoJSON specifications."""

        # GeoJSON does not allow for FeatureCollection properties, therefore
        # we save magnification and image extent as properties of individual
        # features/annotation objects.

        all_annots = []
        for a in self._annots:
            b = a.asGeoJSON()
            b["properties"]["magnification"] = self._magnification
            b["properties"]["image_shape"] = self._image_shape
            all_annots.append(b)

        return gj.FeatureCollection(all_annots)

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize an annotation from a dictionary compatible with GeoJSON specifications."""
        if d["type"].lower() != "featurecollection":
            raise RuntimeError("Need a FeatureCollection as annotation! Got: " + d["type"])

        self._annots.clear()
        mg, im_shape = None, None
        for a in d["features"]:
            obj = createEmptyAnnotationObject(a["geometry"]["type"])
            obj.fromGeoJSON(a)
            self.add_annotation_object(obj)
            if mg is None and "properties" in a:
                mg = a["properties"]["magnification"]
            if im_shape is None and "properties" in a:
                im_shape = a["properties"]["image_shape"]
        self._magnification = mg
        self._image_shape = im_shape

        return
##-

