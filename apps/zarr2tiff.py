# -*- coding: utf-8 -*-

# Converts a ZARR array into a pyramidal TIFF

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

import configargparse as opt
from pathlib import Path
from qpath.utils import write_pyramidal_tiff
from qpath.base import MRI


def main():
    p = opt.ArgumentParser(description="Convert a ZARR file (image) into a pyramidal TIFF.")
    p.add_argument("--mri_path", action="store", help="root folder for the multiresolution image (ZARR format)",
                   required=True)
    p.add_argument("--out", action="store", help="output TIFF file", required=True)
    p.add_argument("--track_processing", action="store_true", help="disabled for this app")

    args = p.parse_args()

    in_file = Path(args.mri_path)
    out_file = Path(args.out).with_suffix('.tiff')

    mri = MRI(in_file)
    img = mri.get_plane(0)  # highest resolution
    write_pyramidal_tiff(img, str(out_file))

    return


if __name__ == '__main__':
    main()