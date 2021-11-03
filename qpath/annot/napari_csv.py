# -*- coding: utf-8 -*-

# Save annotations to Napari's CSV format.


#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################
from datetime import datetime

_time = datetime.now()

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = "1.0"

from . import Annotation
from pathlib import Path
import csv
import numpy as np


def annotation_to_napari(annot: Annotation, csv_file: str, magnification: float = -1.0) -> None:
    """Save an <Annotation> in CSV files following Napari's specifications. Note that, since Points
        require a different format from other shapes, a separate file (with suffix '_points') will
        be created for storing all points in the annotation.

        :param annot: (Annotation) an annotation object
        :param asap_file: (str) name of the file for saving the annotation
        :param magnification: (float) if positive then the desired magnification for the saved
            annotation (all is scaled by a (magnification/annot.magnification) factor)

        :return: True if everything is OK
        """

    # scale the annotation for the target:
    if magnification > 0.0:
        annot.resize(magnification / annot._magnification)

    points_file = Path(csv_file).with_name(Path(csv_file).stem + '_points.csv')
    shapes_file = Path(csv_file)

    points_lines = []
    points_idx = 0
    shapes_lines = []
    shapes_idx = 0

    for a in annot._annots:
        if a._annotation_type == "DOT":
            points_lines.append([points_idx, a.y(), a.x()])
            points_idx += 1
        elif a._annotation_type == "POLYGON":
            xy = a.xy()
            for k in np.arange(xy.shape[0]):
                shapes_lines.append([shapes_idx, 'polygon', k, xy[k, 1], xy[k, 0]])
            shapes_idx += 1
        elif a._annotation_type == "POINTSET":
            xy = a.xy()
            for k in np.arange(xy.shape[0]):
                shapes_lines.append([shapes_idx, 'path', k, xy[k, 1], xy[k, 0]])
            shapes_idx += 1
        else:
            raise RuntimeError("unknown annotation type " + a._annotation_type)

    if points_idx > 0:
        # more than the header
        with open(points_file, 'w') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONE)
            writer.writerow(['index', 'axis-0', 'axis-1'])
            writer.writerows(points_lines)

    if shapes_idx > 0:
        # more than the header
        with open(shapes_file, 'w') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONE)
            writer.writerow(['index', 'shape-type', 'vertex-index', 'axis-0', 'axis-1'])
            writer.writerows(shapes_lines)

    return