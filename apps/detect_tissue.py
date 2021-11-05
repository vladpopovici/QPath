# -*- coding: utf-8 -*-

# Detect tissue regions in a whole slide image.


#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################
from datetime import datetime
import hashlib

_time = datetime.now()
__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = "1.0"
__description__ = {
    'name': 'detect_tissue',
    'unique_id' : hashlib.md5(str.encode('detect_tissue' + __version__)).hexdigest(),
    'version': __version__,
    'timestamp': _time.isoformat(),
    'input': [None],
    'output': [None],
    'params': dict()
}

from tinydb import TinyDB, Query
import simplejson as json
import geojson as gjson
import configargparse as opt
import numpy as np
from pathlib import Path

from shapely.affinity import translate

from qpath.base import WSIInfo, MRI
from qpath.annot import Annotation
from qpath.mask import mask_to_external_contours
from qpath.tissue import detect_foreground
from qpath.utils import NumpyJSONEncoder

# minimum object sizes (areas, in px^2) for different magnifications to be considered as "interesting"
min_obj_size = {'0.3125': 1500, '1.25': 50000, '2.5': 100000, '5.0': 500000}
WORK_MAG_1 = 0.3125
WORK_MAG_2 = 2.5


def main():
    p = opt.ArgumentParser(description="Detect tissue regions in a whole slide image.")
    p.add_argument("--mri_path", action="store", help="root folder for the multiresolution image (ZARR format)",
                   required=True)
    p.add_argument("--out", action="store",
                   help="JSON file for storing the resulting annotation (will be saved to ../annot/ relative to ZARR path)",
                   required=True)
    p.add_argument("--annotation_name", action="store", help="name of the resulting annotation",
                   default="tissue", required=False)
    p.add_argument("--min_area", action="store", type=int, default=None,
                   help="minimum area of a tissue region", required=False)
    p.add_argument("--he", action="store_true", help="use H&E-specific method for detecting the objects")
    p.add_argument("--track_processing", action="store_true",
                   help="should this action be stored in the <-RUN-detect_tissue.json> file for the slide?")

    args = p.parse_args()

    if args.min_area is None:
        args.min_area = min_obj_size[str(WORK_MAG_2)]
    else:
        min_obj_size[str(WORK_MAG_2)] = args.min_area

    in_path = Path(args.mri_path).expanduser().absolute()
    out_path = (in_path.parent.parent / 'annot').expanduser().absolute()
    __description__['params'] = vars(args)
    __description__['input'] = [str(in_path)]
    __description__['output'] = [str(out_path / args.out)]


    if args.track_processing:
        (out_path.parent / '.run').mkdir(exist_ok=True)
        with open(out_path.parent / '.run' / 'run-detect_tissue.json', 'w') as f:
            json.dump(__description__, f, indent=2)

    # print(__description__)

    wsi = WSIInfo(in_path)
    img_src = MRI(in_path)

    # use a two pass strategy: first detect a bounding box, then zoom-in and
    # detect the final mask
    level = wsi.get_level_for_magnification(WORK_MAG_1)
    img = img_src.get_plane(level=level)
    mask, _ = detect_foreground(img, method='fesi', min_area=min_obj_size[str(WORK_MAG_1)])
    contours = mask_to_external_contours(mask, approx_factor=0.0001)

    # find the bounding box of the contours:
    xmin, ymin = img.shape[:2]
    xmax, ymax = 0, 0
    for c in contours:
        minx, miny, maxx, maxy = c.geom.bounds
        xmin = min(xmin, minx)
        ymin = min(ymin, miny)
        xmax = max(xmax, maxx)
        ymax = max(ymax, maxy)

    # some free space around the ROI and rescale to new magnification level:
    f = WORK_MAG_2 / WORK_MAG_1
    xmin = int(f * max(0, xmin - 5))
    ymin = int(f * max(0, ymin - 5))
    xmax = int(f * min(img.shape[1] - 1, xmax + 5))
    ymax = int(f * min(img.shape[0] - 1, ymax + 5))

    # print("ROI @{}x: {},{} -> {},{}".format(WORK_MAG_2, xmin, ymin, xmax, ymax))
    level = wsi.get_level_for_magnification(WORK_MAG_2)
    img = img_src.get_region_px(xmin, ymin,
                                width=xmax - xmin, height=ymax - ymin,
                                level=level, as_type=np.uint8)
    # print("Image size 2: {}x{}".format(img.shape[0], img.shape[1]))

    if args.he:
        mask, _ = detect_foreground(img, method='simple-he', min_area=min_obj_size[str(WORK_MAG_2)])
    else:
        mask, _ = detect_foreground(img, method='fesi',
                                    laplace_ker=15, gauss_ker=17, gauss_sigma=25.0,
                                    morph_open_ker=5, morph_open_iter=7, morph_blur=17,
                                    min_area=min_obj_size[str(WORK_MAG_2)])

    contours = mask_to_external_contours(mask,
                                         approx_factor=0.00005,
                                         min_area=min_obj_size[str(WORK_MAG_2)])

    # don't forget to shift detections by (xmin, ymin) to obtain coords in original space for
    # this magnification level...
    for c in contours:
        c.geom = translate(c.geom, xoff=xmin, yoff=ymin)
        c._name = "tissue"

    # ...and get image extent at working magnification
    img_shape = img_src.extent(level)
    annot = Annotation(name=args.annotation_name,
                       image_shape={'height': int(img_shape[1]), 'width': int(img_shape[0])},
                       magnification=WORK_MAG_2)
    annot.add_annotations(contours)

    # get back to native magnification...
    annot.set_magnification(wsi.get_native_magnification())
    # ...and correct the image extent (due to rounding it may be off by a few pixels), since
    # we actually know it:
    img_shape = img_src.extent(0)
    annot._image_shape = dict(width=img_shape[0], height=img_shape[1])

    with open(out_path / args.out , 'w') as f:
        gjson.dump(annot.asGeoJSON(), f, cls=NumpyJSONEncoder)

    annot_idx = out_path.parent / '.annot_idx.json'
    with TinyDB(annot_idx) as db:
        q = Query()
        r = db.search(q.unique_id == __description__['unique_id'])
        if len(r) == 0:
            # empty DB or no such record
            db.insert({'unique_id' : __description__['unique_id'],
                       'annotator': __description__['name'], 'parameters': __description__['params']})
        else:
            db.update({'annotator': __description__['name'], 'parameters': __description__['params']},
                      q.unique_id == __description__['unique_id'])

    return
##


if __name__ == '__main__':
    main()
