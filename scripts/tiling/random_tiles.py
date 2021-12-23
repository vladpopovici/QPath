# RANDOM_TILES: for each image in the set, generate a series of random tiles and
# save them in a file (to be used for training models).

from qpath.base import MRI, RandomWindowSampler
from qpath.annot import Annotation
from pathlib import Path
import geojson
import zarr
import numpy as np
import tqdm


PATH_ARCHIVE = Path("/teradata/data/COAD/colobiome_histopath/")
PATH_TILES = Path("/bigdata/colobiome/train/tiles")
N_SLIDES = 5 #600  # use first N_SLIDES only
N_TILES  = 100 #1000  # generate at most N_TILES frome each slide
PYR_LEVEL = 1   # pyramid level to use (0: max res (20x in our case), 1: half the max res, etc)
MAX_MAGNIF = 20 # corresponding to level 0
TILE_SHAPE = (256, 256)  # tile shape (width, height)

EFFECTIVE_MAGNIF = MAX_MAGNIF / 2**PYR_LEVEL
DST_ZARR = f"tiles_{EFFECTIVE_MAGNIF}x_{TILE_SHAPE[0]}_by_{TILE_SHAPE[1]}.zarr"
DST_ZARR = "test-tiles_10.0x_256_by_256.zarr"

first_time = True
slides = sorted([s for s in PATH_ARCHIVE.glob("SB*") if s.is_dir()])[0:N_SLIDES]
slides_for_tiles_idx = []
last_idx = 0
for slide in tqdm.tqdm(slides):
    annot = Annotation("annotation", {'width':0, 'height': 0}, 20.0)  # dummy object
    d = dict()
    try:
        with open(slide / "annot/tissue_contour.geojson", 'r') as f:
            d = geojson.load(f)
        annot.fromGeoJSON(d)
        annot.set_magnification(EFFECTIVE_MAGNIF)  # bring the magnification to the desired level

        # find the largest tissue area:
        max_area = 0
        for a in [_z for _z in annot._annots if _z.type == "POLYGON"]:
            if a.geom.area > max_area:
                tissue_region = a.geom

        wsi = MRI(slide / "slide/pyramid.zarr")

        # initialize window sampler:
        wnd = RandomWindowSampler(wsi.extent(PYR_LEVEL), w_size=TILE_SHAPE, n=N_TILES,
                                  poly=tissue_region, nv_inside=4)

        all_tiles = []
        all_tiles_coords = []
        for w in wnd:
            x0, y0 = w[:2]
            tile = wsi.get_region_px(x0, y0, TILE_SHAPE[0], TILE_SHAPE[1], level=PYR_LEVEL)
            all_tiles.append(tile)
            all_tiles_coords.append(w)

        slides_for_tiles_idx.append([last_idx, last_idx + len(all_tiles)])
        last_idx += len(all_tiles)
    except:
        continue

    if first_time:
        zroot = zarr.group(PATH_TILES / DST_ZARR)
        _ = zroot.create_dataset('tiles', data=np.array(all_tiles),
                                 chunks=(10*N_TILES, TILE_SHAPE[0], TILE_SHAPE[1], 3))
        _ = zroot.create_dataset('coords', data=np.array(all_tiles_coords),
                                 chunks=(10*N_TILES, ))
        first_time = False
    else:
        zroot = zarr.open_group(PATH_TILES / DST_ZARR, mode='a')
        zroot['/tiles'].append(np.array(all_tiles))
        zroot['/coords'].append(np.array(all_tiles_coords))

# finally save the info about source images and corresponding tile
# index ranges:
zroot = zarr.group(PATH_TILES / DST_ZARR)
_ = zroot.create_dataset('/source', data=[str(s) for s in slides], dtype=str)
_ = zroot.create_dataset('/source_idx', data=np.array(slides_for_tiles_idx), dtype=int)



