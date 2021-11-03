#Annotations

Annotations are a central part of the pathology image analysis. In general, there
are two main sources for annotations:

- a pathologist (or anyone understanding the images) may mark regions and points
in whole slide images. These may later serve as examples for training models for
image segmentation/classification etc.
- computational tools/applications. Many of these results can be seen as annotations.
For example, detecting the cells in a tissue or the tumor region result in lists of
points and polygons. While the lists can be digests by other tools, for visualization
purposes they need to be saved in a format compatible with the visualizer.

In _QPath_ we aim at provinding means to
- import annotations from [QuPath](https://qupath.github.io/) 
(GEOJson format), Hamamatsu (NDPA), [ASAP](https://github.com/computationalpathologygroup/ASAP) (xml),
[Cytomine](https://cytomine.com/) and [OMERO](https://www.openmicroscopy.org/omero/);
- export annotations to GEOJson ([QuPath](https://qupath.github.io/) compatible), 
XML ([ASAP](https://github.com/computationalpathologygroup/ASAP) compatible), 
[Cytomine](https://cytomine.com/),
[OMERO](https://www.openmicroscopy.org/omero/) and CSV ([Napari](https://napari.org) compatible)

The annotations themselves are mainly of two types:

- geometrical shapes (points, circles, polygons, etc), or
- masks - usually single channel images (arrays) that indicate some regions.