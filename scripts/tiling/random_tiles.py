# RANDOM_TILES: for each image in the set, generate a series of random tiles and
# save them in a file (to be used for training models).

from qpath.base import MRI, RandomWindowSampler
from qpath.annot import Annotation
from pathlib import Path
import geojson

PATH_ARCHIVE = Path("/teradata/data/COAD/colobiome_histopath/")
PATH_TILES = Path("/bigdata/colobiome/train/tiles")
N_SLIDES = 400  # use first N_SLIDES only
N_TILES  = 500  # generate at most N_TILES frome each slide
PYR_LEVEL = 1   # pyramid level to use (0: max res (20x in our case), 1: half the max res, etc)
MAX_MAGNIF = 20 # corresponding to level 0
TILE_SHAPE = (256, 256)  # tile shape (width, height)


slides = sorted([s for s in PATH_ARCHIVE.glob("SB*") if s.is_dir()])[0:N_SLIDES]
for slide in slides:
    annot = Annotation("annotation", {'width':0, 'height': 0}, 20.0)  # dummy object
    d = dict()
    with open(slide / "annot/tissue_contour.geojson", 'r') as f:
        d = geojson.load(f)
    annot.fromGeoJSON(d)
    annot.set_magnification(MAX_MAGNIF / 2**PYR_LEVEL)  # bring the magnification to the desired level

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


