# -*- coding: utf-8 -*-

# Detect tissue regions in a whole slide image.


#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################
from datetime import datetime

_time = datetime.now()

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = "1.0"
__description__ = {
    'name': 'detect_tissue',
    'version': __version__,
    'timestamp': _time.isoformat(),
    'input': [None],
    'output': [None],
    'params': dict()
}

import simplejson as json
import geojson as gjson
import configargparse as opt
import os.path

from shapely.affinity import translate
from shapely.geometry import Polygon

from qpath.base import WSIInfo, MRI
from qpath.annot import Annotation
from qpath.annot.asap import annotation_to_asap
from qpath.mask import mask_to_external_contours
from qpath.tissue import detect_foreground
from qpath.color import RGBA2RGB


# minimum object sizes (areas, in px^2) for different magnifications to be considered as "interesting"
min_obj_size = {'0.3125': 1500, '1.25': 50000, '2.5': 100000, '5.0': 500000}
WORK_MAG_1 = 0.3125
WORK_MAG_2 = 2.5


def main():
    p = opt.ArgumentParser(description="Detect tissue regions in a whole slide image.")
    p.add_argument("--wsi", action="store", help="whole slide image to process", required=True)
    p.add_argument("--out", action="store", help="JSON database for storing (adding) the resulting annotation to",
                   required=True)
    p.add_argument("--annotation_name", action="store", help="name of the resulting annotation",
                   default="tissue", required=False)
    p.add_argument("--simple_json", action="store_true",
                   help="save annotation in a simplified JSON format instead of GeoJSON",
                   required=False)
    p.add_argument("--asap_xml", action="store", default=None, required=False,
                   help="if provided, save the annotation to ASAP XML file as well")
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

    __description__['params'] = vars(args)

    input_path = os.path.abspath(os.path.dirname(args.wsi))
    input_file = os.path.basename(args.wsi)
    __description__['input'] = [input_path + '/' + input_file]

    output_path = os.path.abspath(os.path.dirname(args.out))
    output_file = os.path.basename(args.out)
    __description__['output'] = [output_path + '/' + output_file]

    if args.track_processing:
        input_file_noext = os.path.splitext(input_file)[0]
        with open(output_path + '/' + input_file_noext + '-RUN-detect_tissue.json', 'w') as f:
            json.dump(__description__, f, indent=2)

    # print(__description__)

    wsi = WSIInfo(args.wsi)
    img_src = MRI(wsi)

    # use a two pass strategy: first detect a bounding box, then zoom-in and
    # detect the final mask
    level = wsi.get_level_for_magnification(WORK_MAG_1)
    img = img_src.get_region_px(0, 0, level)
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
    img = img_src.getRegion(xmin, ymin,
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

    # print(annot.asdict())
    with open(args.out, 'w') as f:
        if args.simple_json:
            json.dump(annot.asdict(), f, indent=None)
        else:
            gjson.dump(annot.asGeoJSON(), f)

    # print(annot.asdict())
    if args.asap_xml is not None:
        annotation_to_asap(annot, args.asap_xml)

    return


if __name__ == '__main__':
    main()
