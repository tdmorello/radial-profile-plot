
import numpy as np
from skimage.morphology import binary_dilation, disk
from scipy import ndimage


# TODO normalizing function


def get_radial_profile(img: np.array,
                       mask: np.array,
                       dilation: int,
                       bins: int) -> np.array:
    """Create radial profile curve for an image

    Args:
        img (np.array): single or multichannel image array
        mask (np.array): mask over central object
        dilation (int): amount (in pixels) to expand mask
        bins (int): number of bins for radial measurements

    Returns:
        np.array: array of profiles
    """
    # dilate cell mask
    dil_mask = binary_dilation(mask, disk(dilation))
    # get distance map
    dist_map = ndimage.distance_transform_edt(dil_mask)
    # get max value from distance map
    dist_map_max = np.max(dist_map)
    # get N concentric rings from the distance map
    levels = np.linspace(0, dist_map_max, bins)

    # make sure image has 3 dimensions
    if len(img.shape) == 2:
        img = np.expand_dims(img, 2)
    # create placeholder array for profiles
    profiles = np.zeros((img.shape[2], bins - 1))

    # iterate over each image channel
    for ch in range(img.shape[2]):
        values = np.zeros(bins - 1)
        # iterate over areas between rings, from in to out
        for i in range(bins - 1):
            # create mask outside space between rings
            area_mask = (dist_map > levels[i]) == (dist_map > levels[i + 1])
            # find area of the space between rings
            area = (area_mask == False).sum()  # noqa
            # create a mask over the image channel
            img_masked = np.ma.masked_array(img[..., ch], area_mask)
            # add value to profile
            values[i] = img_masked.sum() / area

        # add profile to channels
        profiles[ch] = values

    return profiles


def main():
    from skimage.io import imread
    import matplotlib.pyplot as plt

    cell_red = imread('data/surround_cell_RhRX.tif')
    cell_green = imread('data/surround_cell_AF488.tif')
    cell_blue = imread('data/surround_cell_DAPI.tif')
    cell_mask = imread('data/surround_cell_mask.tif')

    img = np.zeros((*cell_red.shape, 3))
    img[..., 0] = cell_red
    img[..., 1] = cell_green
    img[..., 2] = cell_blue

    fig, ax = plt.subplots(1, 1)
    profiles = get_radial_profile(img, cell_mask, 30, 40)
    for prof in profiles:
        ax.plot(np.flip(prof))

    plt.show()


if __name__ == '__main__':
    main()
