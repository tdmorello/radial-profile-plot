
import numpy as np
from rasterio.features import rasterize
from shapely.affinity import translate
from shapely.geometry import Polygon


def reposition_shape(shape, center):
    return translate(shape, -shape.centroid.x + center, -shape.centroid.y + center)


class Patch:
    """A class to hold an image patch and its corresponding object mask
    """
    def __init__(self,
                 img: np.ndarray,
                 polygon: Polygon,
                 id: int = None):
        # assert isinstance(polygon, Polygon)

        self._img = img
        self._polygon = polygon
        self._id = id

    @property
    def size(self):
        return self._img.shape[0]

    @property
    def id(self):
        return self._id

    @property
    def polygon(self):
        return self._polygon

    def get_image(self):
        return self._img

    def get_outline(self):
        shape = reposition_shape(self._polygon, self.size/2)
        return shape.exterior.xy

    def get_mask(self):
        shape = reposition_shape(self._polygon, self.size/2)
        return rasterize([shape], (self.size, self.size))

    def get_masked_image(self):
        img = self.get_image()
        mask = self.get_mask()
        if len(img.shape) == 3:
            mask = np.repeat(mask[..., np.newaxis], img.shape[2], axis=2)
        return np.ma.masked_array(img, mask, fill_value=0)


def main():
    import geojson
    import matplotlib.pyplot as plt
    from shapely.geometry import shape as shapely_shape

    from bioformatsreader import BioFormatsReader

    patchsize = 500
    json_fname = r'data/b32f-sag_L-lxn_pv-w03_2-027-cla-63x-a_overview-core_lxn_detections.json'  # noqa
    wsi_fname = "data/b32f-sag_L-lxn_pv-w03_2-027-cla-63x-a_overview.czi"  # noqa

    reader = BioFormatsReader(wsi_fname)

    with open(json_fname) as f:
        allobjects = geojson.load(f)

    allshapes = [shapely_shape(obj["geometry"]) for obj in allobjects]

    # select a random shape from the list
    idx = np.random.randint(0, high=len(allshapes) - 1)
    shape = allshapes[idx]

    # Get coordinates for image
    X, Y = [int(coord - patchsize / 2) for coord in shape.centroid.xy]
    # Y = int(shape.centroid.y - patchsize / 2)
    W, H = patchsize, patchsize
    XYWH = [X, Y, W, H]

    img = reader.read(XYWH=XYWH)
    patch = Patch(img, shape)  # type: ignore
    plt.imshow(patch.get_masked_image().filled())
    # plt.imshow(patch.get_image())
    # plt.plot(*patch.get_outline(), c='red')


if __name__ == '__main__':
    main()
