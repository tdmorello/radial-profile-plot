
# from matplotlib import colors
import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple
from utils import apply_color_lut

# TODO normalizing function
# TODO fit least squares circle to outlines to estimate distance to centroid


def get_radial_profiles(img: np.ndarray,
                        mask: np.ndarray,
                        dilation: int,
                        bins: int):
    """Convenience function for creating radial profile

    Args:
        img (np.array): single or multichannel image array
        mask (np.array): mask over central object
        dilation (int): amount (in pixels) to expand mask
        bins (int): number of bins for radial measurements

    Returns:
        np.array: [description]
    """
    return RadialProfiler(img, mask, dilation, bins).get_profiles()


class RadialProfiler:
    def __init__(self,
                 img: np.ndarray,
                 mask: np.ndarray,
                 dilation: int,
                 bins: int):
        """Create radial profile curve for an image

        Args:
            img (np.array): single or multichannel image array
            mask (np.array): mask over central object
            dilation (int): amount (in pixels) to expand mask
            bins (int): number of bins for radial measurements
        """

        # make sure image has 3 dimensions
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
        self.img = img
        self.mask = mask
        self.dilation = dilation
        self.bins = bins

    @property
    def dilated_mask(self):
        selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*self.dilation-1, 2*self.dilation-1))
        return cv2.dilate(self.mask, selem, iterations=1)

    @property
    def distance_map(self):
        return cv2.distanceTransform(self.dilated_mask, cv2.DIST_L2, 5)

    @property
    def distance_map_max(self):
        return np.max(self.distance_map)

    @property
    def distance_map_cell_min(self):
        masked_distance = np.ma.masked_array(self.distance_map, 1 - self.mask.astype(bool))
        return np.min(masked_distance)

    # @property
    # def levels(self):
    #     return np.linspace(0, self.distance_map_max, self.bins)

    @property
    def levels(self):
        """Find a constant number of bins between dilated mask boundary and cell boundary. Use the calculated step size
        to build rings inside the cell until the center is reached"""
        dist_cell_min = self.distance_map_cell_min
        outer_levels, step = np.linspace(0, dist_cell_min, 15, retstep=True)
        inner_levels = np.arange(dist_cell_min, self.distance_map_max, step)[1:]
        levels = np.concatenate([outer_levels, inner_levels])
        return levels

    @property
    def rings(self):
        thresholds = [thr.astype('uint8') for thr in [self.distance_map > lev for lev in self.levels]]
        contours = [cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] for thr in thresholds]
        # contours = [cont.reshape(cont.shape[0], cont.shape[2]) for cont in contours]
        return contours

    def get_profiles(self, norm_func=None) -> Tuple[np.ndarray, np.ndarray]:
        """Returns profiles and distances from the cell boundary"""
        img = self.img
        dist_map = self.distance_map
        levels = self.levels
        total_bins = levels.size

        profiles = np.zeros((img.shape[2], total_bins - 1))
        # iterate over each image channel
        for ch in range(img.shape[2]):
            values = np.zeros(total_bins - 1)
            # iterate over areas between rings, from in to out
            for i in range(total_bins - 1):
                # create mask outside space between rings
                area_mask = (dist_map > levels[i]) == (dist_map > levels[i + 1])  # noqa
                # find area of the space between rings
                area = (area_mask == False).sum()  # noqa
                # create a mask over the image channel
                img_masked = np.ma.masked_array(img[..., ch], area_mask)
                # add value to profile
                values[i] = img_masked.sum() / area
            # add profile to channels
            profiles[ch] = values

        distances = levels - self.distance_map_cell_min

        return profiles, distances


class RadialProfilePlotter(RadialProfiler):
    def __init__(self,
                 img: np.ndarray,
                 mask: np.ndarray,
                 dilation: int,
                 bins: int,
                 ):
        super().__init__(img, mask, dilation, bins)

    def plot(self,
             colors=['red', 'green', 'blue'],
             titles=['1', '2', '3'],
             show_cell_boundary: bool = True,
             show_fig: bool = True,
             save_fig: bool = False,
             fname=None):

        contours = self.rings

        fig = plt.figure(constrained_layout=False)
        spec = gridspec.GridSpec(ncols=4, nrows=3, figure=fig, wspace=0)
        ax1 = fig.add_subplot(spec[:, :-1])
        ax2 = fig.add_subplot(spec[0, -1])
        ax3 = fig.add_subplot(spec[1, -1])
        ax4 = fig.add_subplot(spec[2, -1])

        ax_plot = ax1
        axs_patch = [ax2, ax3, ax4]

        images = [self.img[..., i] for i in range(self.img.shape[-1])]

        # Plot
        for ax, img, title, color in zip(axs_patch, images, titles, colors):  # noqa
            img = apply_color_lut(img, color)
            ax.imshow(img)

            # Middle contour (corresponding to mask outline before dilation)
            c = contours[int(len(contours) / 2)][0]
            line = ax.plot(c[..., 0][0], c[..., 1][0], linewidth=1, c='white', alpha=1)
            line[0].set_dashes((4, 6))
            ax.text(0.02, 0.05, title, transform=ax.transAxes, c='white')
            ax.axis('off')

        cell_boundary_line = 0

        all_plots = {}
        profiles, distances = self.get_profiles()
        for prof, img, title, color in zip(profiles, images, titles, colors):
            ax_plot.plot(distances[:-1], np.flip(prof), c=color, label=title)
            all_plots[title] = prof

        if show_cell_boundary:
            xy = (cell_boundary_line, 0.05)
            # Shift the text over just a bit
            xaxis_range = np.ptp(ax1.get_xlim())
            xytext = (cell_boundary_line + (xaxis_range * 0.1), 0.08)
            ax_plot.axvline(
                cell_boundary_line,
                linewidth=1,
                linestyle='--',
                color='black')
            ax_plot.annotate(
                'cell boundary',
                xy=xy, xycoords=("data", "axes fraction"),
                xytext=xytext, textcoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle='->'))

        ax_plot.set_title('Radial profile plot')
        ax_plot.set_xlabel('Distance from cell center')
        ax_plot.set_xticks([])
        ax_plot.set_ylabel('Normalized intensity')
        ax_plot.legend()

        if not show_fig:
            plt.close(fig)
        if save_fig:
            fig.savefig(fname)

        return all_plots


def main():
    from skimage.io import imread

    # import matplotlib.pyplot as plt

    cell_red = imread('data/surround_cell_RhRX.tif')
    cell_green = imread('data/surround_cell_AF488.tif')
    cell_blue = imread('data/surround_cell_DAPI.tif')
    cell_mask = imread('data/surround_cell_mask.tif')

    img = np.zeros((*cell_red.shape, 3))
    img[..., 0] = cell_red
    img[..., 1] = cell_green
    img[..., 2] = cell_blue

    # fig, ax = plt.subplots(1, 1)
    # # profiler = RadialProfiler(img, cell_mask, 30, 30)
    # # profiles = profiler.get_profiles()
    # profiles = get_radial_profiles(img, cell_mask, 30, 10)
    # print(profiles)

    # for prof in profiles:
    #     ax.plot(np.flip(prof))

    colors = ['red', 'green', 'blue']
    titles = ['PV', 'LXN', 'DAPI']
    plotter = RadialProfilePlotter(img, cell_mask, 30, 15)
    plotter.plot(colors, titles)
    plt.show()

    # plot = get_radial_profiles(img, cell_mask, 30, 30)


if __name__ == '__main__':
    main()
