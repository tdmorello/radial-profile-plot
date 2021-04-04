
# %%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

from scipy import ndimage
from skimage.measure import find_contours
from skimage.morphology import binary_dilation, disk
from skimage.io import imread
from skimage.exposure import equalize_hist
from skimage.color import gray2rgb

# %%
cell_red = imread('data/surround_cell_RhRX.tif')
cell_green = imread('data/surround_cell_AF488.tif')
cell_blue = imread('data/surround_cell_DAPI.tif')
cell_mask = imread('data/surround_cell_mask.tif')

images = [cell_mask, cell_red, cell_green, cell_blue]
titles = ['Mask', 'PV', 'LXN', 'DAPI']
colors = ['', 'red', 'green', 'blue']

# Dilate mask
selem = disk(30)
dil_mask = binary_dilation(cell_mask, selem)
mask_cont = find_contours(cell_mask)
dil_mask_cont = find_contours(dil_mask)[0]

distance = ndimage.distance_transform_edt(dil_mask)
dist_max = np.max(distance)

levels = np.linspace(0, dist_max, 25)
contours = [find_contours(distance, level) for level in levels]


# create a mask for the zone between contours
def mask_between_contours(distance_map, outer_level, inner_level):
    # Only keep area where outer is True and inner is False
    return (distance_map > outer_level) == (distance_map > inner_level)


def get_radial_profile(img, dist_map, n_levels):
    dist_max = np.max(dist_map)
    levels = np.linspace(0, dist_max, n_levels)
    values = np.empty(n_levels - 1)
    areas = np.empty(n_levels - 1)

    for i in range(len(levels) - 1):
        mask = mask_between_contours(distance, levels[i], levels[i + 1])
        area = (mask == False).sum()  # noqa
        img_masked = np.ma.masked_array(img, mask, fill_value=0)

        values[i] = img_masked.sum()
        areas[i] = area

    return values, areas


def normalize(values, areas):
    return (values / areas) / (values / areas).mean()


# %%
# Grid spec layout
fig = plt.figure(constrained_layout=False)
spec = gridspec.GridSpec(ncols=4, nrows=3, figure=fig, wspace=0)
ax1 = fig.add_subplot(spec[:, :-1])
ax2 = fig.add_subplot(spec[0, -1])
ax3 = fig.add_subplot(spec[1, -1])
ax4 = fig.add_subplot(spec[2, -1])

ax_plot = ax1
axs_patch = [ax2, ax3, ax4]

# Plot
for ax, img, title, color in zip(axs_patch, images[1:], titles[1:], colors[1:]):  # noqa
    multiplier = {
        'red': [1, 0, 0],
        'green': [0, 1, 0],
        'blue': [0, 0, 1]
    }

    img = (equalize_hist(img) * 200).astype(np.uint8)
    img = gray2rgb(img) * multiplier[color]

    ax.imshow(img)

    # Show all contours
    # for j, contour in enumerate(contours[0:-1:5]):
    #     c = contour[0]
    #     line_color = 'red' if (j == 0) else 'yellow'
    #     ax.plot(c[:,1], c[:,0], linewidth=0.5, c=line_color, linestyle='--')

    # Middle contour (corresponding to mask outline before dilation)
    c = contours[int(len(contours) / 2)][0]
    line = ax.plot(c[:, 1], c[:, 0], linewidth=1, c='white', alpha=1)
    line[0].set_dashes((4, 6))
    ax.text(0.02, 0.05, title, transform=ax.transAxes, c='white')
for ax in axs_patch:
    ax.axis('off')


cell_boundary_line = int(len(contours) / 2)

for img, title, color in zip(images[1:], titles[1:], colors[1:]):
    values, areas = get_radial_profile(img, distance, 25)
    normalized = normalize(values, areas)
    ax1.plot(np.flip(normalized), c=color, label=title)


xy = (cell_boundary_line, 0.05)
# Shift the text over just a bit
xaxis_range = np.ptp(ax1.get_xlim())
xytext = (cell_boundary_line + (xaxis_range * 0.1), 0.08)
ax1.axvline(cell_boundary_line,
            linewidth=1,
            linestyle='--',
            color='black')
ax1.annotate('cell boundary',
             xy=xy, xycoords=("data", "axes fraction"),
             xytext=xytext, textcoords=("data", "axes fraction"),
             arrowprops=dict(arrowstyle='->'))

ax1.set_title('Radial profile plot')
ax1.set_xlabel('Distance from cell center')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylabel('Normalized intensity')
ax1.legend()

plt.show()
