
# from matplotlib import colors
import numpy as np
from skimage.morphology import binary_dilation, disk
from scipy import ndimage
from skimage.measure import find_contours
import matplotlib.gridspec as gridspec

from utils import apply_color_lut
import matplotlib.pyplot as plt

# TODO normalizing function


def get_radial_profiles(img: np.array,
                        mask: np.array,
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
                 img: np.array,
                 mask: np.array,
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
        return binary_dilation(self.mask, disk(self.dilation))

    @property
    def distance_map(self):
        return ndimage.distance_transform_edt(self.dilated_mask)

    @property
    def distance_map_max(self):
        return np.max(self.distance_map)

    @property
    def levels(self):
        return np.linspace(0, self.distance_map_max, self.bins)

    @property
    def rings(self):
        return [find_contours(self.distance_map, lev) for lev in self.levels]

    def get_profiles(self) -> np.array:
        img = self.img
        bins = self.bins
        dist_map = self.distance_map
        levels = self.levels

        profiles = np.zeros((img.shape[2], bins - 1))
        # iterate over each image channel
        for ch in range(img.shape[2]):
            values = np.zeros(bins - 1)
            # iterate over areas between rings, from in to out
            for i in range(bins - 1):
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

        return profiles


class RadialProfilePlotter(RadialProfiler):
    def __init__(self,
                 img: np.array,
                 mask: np.array,
                 dilation: int,
                 bins: int,
                 colors,
                 titles,
                 ):
        super().__init__(img, mask, dilation, bins)

        self.colors = colors
        self.titles = titles

    # @property
    # def colors(self):
    #     pass

    # @property
    # def titles(self):
    #     pass

    def plot(self,
             show_cell_boundary: bool = True,
             show_fig: bool = True,
             save_fig: bool = False,
             fname=None):

        contours = self.rings
        titles = self.titles
        colors = self.colors

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
            line = ax.plot(c[:, 1], c[:, 0], linewidth=1, c='white', alpha=1)
            line[0].set_dashes((4, 6))
            ax.text(0.02, 0.05, title, transform=ax.transAxes, c='white')
            ax.axis('off')

        cell_boundary_line = int(len(contours) / 2)

        all_plots = {}
        profiles = self.get_profiles()
        for prof, img, title, color in zip(profiles, images, titles, colors):
            ax_plot.plot(np.flip(prof), c=color, label=title)
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
        # ax_plot.set_ylim(top=1.8)
        # ax_plot.set_yticks([])
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
    # profiles = get_radial_profiles(img, cell_mask, 30, 30)
    # for prof in profiles:
    #     ax.plot(np.flip(prof))

    # plt.show()
    colors = ['red', 'green', 'blue']
    titles = ['1', '2', '3']
    profile_plotter = RadialProfilePlotter(img,
                                           cell_mask,
                                           30,
                                           30,
                                           colors,
                                           titles)
    profile_plotter.plot()


if __name__ == '__main__':
    main()
