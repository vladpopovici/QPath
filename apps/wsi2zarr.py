# -*- coding: utf-8 -*-

# Converts a whole slide image to ZARR (OME) format.

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
    'name': 'wsi_2zarr',
    'version': __version__,
    'timestamp': _time.isoformat(),
    'input': [None],
    'output': [None],
    'params': dict()
}

import simplejson as json
import configargparse as opt
import pathlib
import openslide as osl
import zarr
import numpy as np
from tqdm import tqdm, trange


def main():
    p = opt.ArgumentParser(description="Convert an image from WSI format to ZARR (pseudo-OME).")
    p.add_argument("--input", action="store", help="whole slide image to process", required=True)
    p.add_argument("--output", action="store", help="root folder for the result (i.e. path without SLIDE name - see Conventions.md)",
                   required=True)
    p.add_argument("--new_slide_name", action="store", help="use the new slide name insted of the one infer from <input>",
                   default=None, required=False)
    p.add_argument("--autocrop", action="store_true",
                   help="""try to crop the image to the bounding box of the tissue (if OpenSlide provides one!)"""
                        """If <autocrop> is provided, <crop> is overlooked.""")
    p.add_argument("--crop", action="store", help="region to crop (x0, y0, width, height in level-0 coordinates)",
                   nargs=4, type=int, required=False, default=None)
    p.add_argument("--band_size", action="store", type=int, default=1528,
                   help="the slide image is read/written in bands of at most <band_size> height", required=False)
    p.add_argument("--track_processing", action="store_true",
                   help="should this action be stored in the ./run/run-wsi_2zarr.json file for the slide?")

    args = p.parse_args()

    __description__['params'] = vars(args)
    __description__['input'] = [args.input]
    __description__['output'] = [args.output]

    in_path = pathlib.Path(args.input)
    slide_name = in_path.name if args.new_slide_name is None else args.new_slide_name
    out_path = pathlib.Path(args.output) / slide_name

    if not out_path.exists():
        out_path.mkdir()

    (out_path/'.run').mkdir(exist_ok=True)
    if args.track_processing:
        with open(out_path/'.run'/'run-wsi2zarr.json', 'w') as f:
            json.dump(__description__, f, indent=2)

    # modified from https://github.com/sofroniewn/image-demos/blob/4ddcfc23980e37fbe5eda8150c14af8220369f24/helpers/make_2D_zarr_pathology.py

    band_size = args.band_size

    with osl.OpenSlide(str(in_path)) as wsi:
        orig_metadata = dict(wsi.properties)

        background_1channel = int(orig_metadata['openslide.background-color'][:2], 16)

        # roi: (x, y, width, height)
        base_roi = (0, 0, wsi.dimensions[0], wsi.dimensions[1])
        if args.autocrop:
            try:
                base_roi = (int(orig_metadata['openslide.bounds-x']),
                            int(orig_metadata['openslide.bounds-y']),
                            int(orig_metadata['openslide.bounds-width']),
                            int(orig_metadata['openslide.bounds-height']))
            except KeyError:
                print("WARN: no bounding box found, continuing with full image")
        elif args.crop is not None:
            base_roi = tuple(args.crop)

        # print("Base ROI: ", base_roi)
        n_pyr_levels = int(orig_metadata['openslide.level-count'])
        pyramid = []
        with zarr.open_group(str(out_path/'slide/pyramid.zarr'), mode='w') as root:
            for i in trange(n_pyr_levels, desc='Pyramid'):
                if args.autocrop or args.crop is not None:
                    roi = tuple([int(2 ** (-i) * r) for r in base_roi])
                else:
                    roi = (0, 0, wsi.level_dimensions[i][0], wsi.level_dimensions[i][1])
                # orig_width, orig_height = wsi.level_dimensions[i]
                # shape = (orig_height, orig_width, 4)
                # print("Current ROI: ", roi)
                pyramid.append({
                    'level': i,
                    'width': roi[2],
                    'height': roi[3],
                    'downsample_factor': 2**i
                })
                shape = (roi[3], roi[2], 3)
                grp = root.create_dataset(str(i),
                                          shape=shape,
                                          chunks=(4096, 4096, None),
                                          dtype='uint8')
                # n_bands = orig_height // band_size
                n_bands = roi[3] // band_size
                incomplete_band = roi[3] % band_size
                for j in trange(n_bands, desc='Level {}'.format(i)):  # by horizontal bands
                    image = np.array(wsi.read_region(
                        (base_roi[0], base_roi[1] + j * band_size * (2 ** i)),  # top-left (IN LEVEL-0 coords!)
                        i,  # magnification level/pyramid level
                        (roi[2], band_size)  # width, height of the region to read
                    ))
                    for k in range(3):
                        image[image[..., 3] == 0, k] = background_1channel

                    grp[j * band_size:(j + 1) * band_size] = image[..., :-1]

                if incomplete_band > 0:
                    image = np.array(wsi.read_region(
                        (base_roi[0], base_roi[1] + n_bands * band_size * (2 ** i)),
                        i,
                        (roi[2], incomplete_band)
                    ))
                    for k in range(3):
                        image[image[..., 3] == 0, k] = background_1channel
                    grp[n_bands * band_size: n_bands * band_size + incomplete_band] = image[..., :-1]

            root.attrs['orig_metadata'] = orig_metadata
            root.attrs['orig_crop_box'] = {'x': base_roi[0], 'y': base_roi[1], 'width': base_roi[2], 'height': base_roi[3]}
            root.attrs['metadata'] = {
                'vendor': orig_metadata['openslide.vendor'],
                'objective_power': int(orig_metadata['openslide.objective-power']),
                'resolution_x_level_0': float(orig_metadata['openslide.mpp-x']),
                'resolution_y_level_0': float(orig_metadata['openslide.mpp-y']),
                'resolution_units': 'microns',
                'background': orig_metadata['openslide.background-color']
            }
            root.attrs['pyramid'] = pyramid
            root.attrs['pyramid_desc'] = 'imported slide pyramid'

            # prepare the rest of the structure
            with open(out_path/'slide'/'.metadata', 'w') as f:
                json.dump(root.attrs.asdict(), f)
    (out_path/'annot').mkdir(exist_ok=True)
    (out_path/'.annot_idx.json').touch(exist_ok=True)

    return


if __name__ == "__main__":
    main()
