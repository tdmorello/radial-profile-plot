
from rasterio.features import rasterize
from shapely.affinity import translate
from shapely.geometry import Polygon
import numpy as np


def reposition_shape(shape, center):
    return translate(shape, -shape.centroid.x + center, -shape.centroid.y + center)  # noqa


class Patch:
    """A class to hold an image patch and its corresponding object mask
    """
    def __init__(self, img: np.array, shape: Polygon, id: int = None):
        self.img = img
        self.shape = shape
        self.id = id

    @property
    def size(self):
        return self.img.shape[0]

    @property
    def mask(self):
        shape = reposition_shape(self.shape, self.size/2)
        return 1 - rasterize([shape], (self.size, self.size))

    def get_outline(self):
        shape = reposition_shape(self.shape, self.size/2)
        return shape.exterior.xy

    def get_masked_image(self):
        img = self.img
        mask = self.mask
        return np.ma.masked_array(img, mask, fill_value=0)


def main():
    import matplotlib.pyplot as plt
    import geojson
    from shapely.geometry import shape as shapely_shape
    from utils import BioFormatsReader

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
    X = int(shape.centroid.x - patchsize / 2)
    Y = int(shape.centroid.y - patchsize / 2)
    W, H = patchsize, patchsize
    XYWH = [X, Y, W, H]

    img = reader.read_image(0, XYWH=XYWH)
    patch = Patch(img, shape)
    # plt.imshow(patch.masked_image().filled())
    x, y = patch.get_outline()
    plt.imshow(patch.img)
    plt.plot(*patch.get_outline(), c='red')


if __name__ == '__main__':
    main()
