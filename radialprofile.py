import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

from scipy import ndimage
from skimage.measure import find_contours
from skimage.morphology import binary_dilation, disk
from skimage.io import imread
from skimage.exposure import equalize_hist
from skimage.color import gray2rgb

# from typing import List


def get_distance_map(mask):
    return ndimage.distance_transform_edt(mask)


def apply_color_lut(image, color: str, do_equalize_hist=True):
    multiplier = {'red': [1, 0, 0], 'green': [0, 1, 0], 'blue': [0, 0, 1]}
    if do_equalize_hist:
        image = (equalize_hist(image) * 200).astype(np.uint8)
    return gray2rgb(image) * multiplier[color]


# def apply_color_lut(image: np.array, color: np.array, equalize_histogram: bool = True):
#     if equalize_histogram:
#         image = (equalize_hist(image) * 200).astype(np.uint8)
#     return gray2rgb(image) * color


def normalize_values(values, areas):
    return (values / areas) / (values / areas).mean()


def mask_outside_contours(distance_map, outer_level, inner_level):
    return (distance_map > outer_level) == (distance_map > inner_level)


def get_intensity_between_rings(image, distance_map, levels):
    values = np.empty(len(levels) - 1)
    areas = np.empty(len(levels) - 1)
    # Get values from areas between rings
    for i in range(len(levels) - 1):
        mask = area_between_contours(distance_map, levels[i], levels[i + 1])
        area = (mask == False).sum()  # noqa
        image_masked = np.ma.masked_array(image, mask, fill_value=0)
        values[i] = image_masked.sum()
        areas[i] = area
    return values, areas


class RadialProfile:
    def __init__(self,
                 images,
                 image_params,
                 mask,
                 dilate: int = 30,
                 bins: int = 25):
        self.images = images
        self.titles = image_params['titles']
        self.colors = image_params['colors']
        self.mask = mask
        self.dilate = dilate
        self.bins = bins
        self.dilated_mask = binary_dilation(mask, disk(dilate))
        self.distance_map = get_distance_map(self.dilated_mask)
        self.dist_map_max = np.max(self.distance_map)

    def get_profile(self):
        images = self.images
        bins = self.bins

        distance_map = self.distance_map
        dist_map_max = self.dist_map_max

        isovalues = np.linspace(0, dist_map_max, bins)

        # Initiate values and areas arrays
        values = np.empty(bins - 1)
        areas = np.empty(bins - 1)

        all_values = []
        all_areas = []
        for image in images:
            values, areas = get_intensity_between_rings(image, distance_map, isovalues)
            all_values.append(values)
            all_areas.append(areas)

        return all_values, all_areas

    def plot(self):
        images = self.images
        titles = self.titles
        colors = self.colors
        distance_map = self.distance_map
        bins = self.bins
        dist_map_mask = self.dist_map_max

        isovalues = np.linspace(0, dist_map_mask, bins)
        rings = [find_contours(distance_map, val) for val in isovalues]

        # Grid spec layout
        fig = plt.figure(constrained_layout=False)
        spec = gridspec.GridSpec(ncols=4, nrows=3, figure=fig, wspace=0)
        ax_plot = fig.add_subplot(spec[:, :-1])
        axs_patch = [fig.add_subplot(spec[i, -1]) for i in range(3)]

        cell_boundary_line = int(len(rings) / 2)

        # Show plot
        values, areas = self.get_profile()
        for val, area, ttl, col in zip(values, areas, titles, colors):
            normalized = normalize_values(val, area)
            ax_plot.plot(np.flip(normalized), c=col, label=ttl)

        xy = (cell_boundary_line, 0.05)
        # Shift the text over just a bit
        xaxis_range = np.ptp(ax_plot.get_xlim())
        xytext = (cell_boundary_line + (xaxis_range * 0.1), 0.08)

        ax_plot.axvline(cell_boundary_line, linewidth=1,
                        linestyle='--', color='black')
        ax_plot.annotate('cell boundary',
                         xy=xy, xycoords=("data", "axes fraction"),
                         xytext=xytext, textcoords=("data", "axes fraction"),
                         arrowprops=dict(arrowstyle='->'))

        ax_plot.set_title('Radial profile plot')
        ax_plot.set_xlabel('Distance from cell center')
        ax_plot.set_ylabel('Normalized intensity')
        ax_plot.set_xticks([])
        ax_plot.set_yticks([])
        ax_plot.legend()

        # Show patches
        for ax, img, ttl, col in zip(axs_patch, images, titles, colors):
            ax.imshow(apply_color_lut(img, col))

            # Middle ring (corresponding to mask outline before dilation)
            r = rings[int(len(rings) / 2)][0]
            line = ax.plot(r[:, 1], r[:, 0], linewidth=1, c='white', alpha=1)
            line[0].set_dashes((4, 6))
            ax.text(0.02, 0.05, ttl, transform=ax.transAxes, c='white')

        for ax in axs_patch:
            ax.axis('off')

        plt.show()


def main():
    cell_red = imread('data/surround_cell_RhRX.tif')
    cell_green = imread('data/surround_cell_AF488.tif')
    cell_blue = imread('data/surround_cell_DAPI.tif')
    cell_mask = imread('data/surround_cell_mask.tif')

    images = [cell_red, cell_green, cell_blue]
    mask = cell_mask
    titles = ['PV', 'LXN', 'DAPI']
    colors = ['red', 'green', 'blue']

    img_params = {
        'titles': titles,
        'colors': colors
    }

    radial_profile = RadialProfile(images, img_params, mask, 30, 25)
    radial_profile.plot()


if __name__ == '__main__':
    main()
