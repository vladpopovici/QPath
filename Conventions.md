#Conventions
The _QPath_ package/system is made up of two components: a computational pathology library
(i.e. a Python module __qpath__) which provides the core functionality and can be easily used
from elsewhere as well, and a set of applications/tools that implement a number of processing
pipelines and other utilities to be used together as an integrated environment for computational
pathology. The second component relies on __qpath__ and other libraries, while __qpath__ is
independent of it.

To make the tools work smoothly together, a number of conventions/common practices have been
adopted. Here we describe the main points:

## Image format(s)
The virtual slides come in a variety of formats (usually proprietary) 
and open source formats emerge (N5, OME-ZARR, etc) as well. In _QPath_ we use a single internal
format based on [ZARR](https://zarr.readthedocs.io/en/stable/) 
and loosely compatible with OME-ZARR (it's missing OME-XML part). The ZARR
store contains the whole image pyramid and can be visualized in [Napari](https://napari.org)
(drag and drop the "pyramid" folder onto Napari).

Therefore, all virtual slides are imported into ZARR (apps/wsi2zarr.py) with as much as possible
of the original metadata saved into ZARR attributes. We also provide a tool for converting
the ZARR data into a pyramidal BigTIFF to be use elsewhere. However, all processing is done
on the ZARR data.

## Image data and annotations
A slide has initially only image data and (instrument) metadata associated with. As it is 
processed and anlyzed, the image data may be enriched (e.g. construct a version with stain
normalized) or annotations can be created (either as images or as geometrical shapes).
Since we try to keep all relevant information for a slide together, we adopt the following
storage convention for imported slides. Let SLIDE_NAME be the name (with extension droped) of the
virtual slide that is imported. The the following hierarchy is developed (over time, as
new processing steps are applied):

```
SLIDE_NAME -
      + slide -
             + pyramid.zarr -            <- original image/slide
                         + 0 -...
                         + 1 -...
                          ...
               .metadata                 <- metadata extracted from WSI
             + pyramid_1.zarr -          <- some new pyramid           
                         + 0 -...
                         + 1 -...
                          ...
             ...
      + tiles -                          <- holds various tiles generated from image
                 ...
      + annot -
              + annot 1
              ...
      .annot_idx.json                    <- list of available annotations
      
      + .run - ...                       <- folder with traces of programs run on this data
```

- each pyramid is a ZARR store containing the layers of the pyramid. 
- _tiles_ contain image patches generated from a 
pyramid (normally at a predefined level), along with their coordinates, and are used for
training models, for example.
- _annot_ is a folder containing annotations, including masks (which are themselves ZARR
stores containing pyramids)
- .annot_idx.json_ is a registry (index) of available annotations. Each tool producing
an annotation may either overwrite a previous entry or add a new one, and it's responsible
for updating the registry.

