"""Command line tool to create radial profile plots"""

# TODO extend capability to binary masks
# TODO add a search area polygon option
# TODO search for cells with greatest difference between 2 µm and 5 µm (measure the average diameter of processes
#   surrounding cell of N neurons)
# bins = avg distance of dilated mask to cell center / desired resolution

import argparse
import logging
import sys
import cv2
from pathlib import Path

import geojson
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import spatial
# from scipy import spatial
# from scipy.optimize.optimize import _prepare_scalar_function
from shapely.geometry import Polygon, shape
from shapely.strtree import STRtree
from tiler import Tiler
from tqdm import tqdm

from bioformatsreader import BioFormatsReader
from patch import Patch
from radial_profiler import RadialProfilePlotter, get_radial_profiles

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger('rasterio').setLevel(logging.ERROR)


parser = argparse.ArgumentParser()
defaults = {'tile_size': 2000, 'patch_size': 200, 'min_hits': 5}
parser.add_argument('image_file', type=str, help='path to image file (read by BioFormats)')
parser.add_argument('detection_file', type=str, help='path to detections JSON file')
parser.add_argument('-t', '--tile_size', type=int, default=defaults['tile_size'],
                    help=f'tile size for iterating over image; default is {defaults["tile_size"]}')
parser.add_argument('-p', '--patch_size', type=int, default=defaults['patch_size'],
                    help=f'patch size to extract from image; default is {defaults["patch_size"]}')
parser.add_argument('-m', '--min_hits', type=int, default=defaults['min_hits'],
                    help=f'minimum number of detections to process a tile; default is {defaults["min_hits"]}')
parser.add_argument('--padding', type=int, help='amount of padding for tiles')
parser.add_argument('--error-bars', type=bool, help='display error bars on graph')


def build_search_box(X, Y, tilesize, paddingsize):
    search_box = Polygon([(X+paddingsize, Y+paddingsize),
                          (X+tilesize-paddingsize, Y+paddingsize),
                          (X+tilesize-paddingsize, Y+tilesize-paddingsize),
                          (X+paddingsize, Y+tilesize-paddingsize)])
    return search_box


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    # args = parser.parse_args(['data/b41f-sag_L-neun_pv-w09_2-cla-63x.czi',
    #                           'data/b41f-sag_L-neun_pv-w09_2-cla-63x-detections_cortex.json'])

    image_file = args.image_file
    detection_file = args.detection_file
    tile_size = args.tile_size
    patch_size = args.patch_size
    min_hits = args.min_hits

    # channel_order = [1, 2, 0]  # G, B, R
    titles = ['NEUN', 'DAPI', 'CALR']
    colors = ['green', 'blue', 'red']

    with open(detection_file) as f:
        allobjects = geojson.load(f)
    logger.debug("Loaded detections")
    allshapes = [shape(obj["geometry"]) for obj in allobjects]
    allcenters = [s.centroid for s in allshapes]
    logger.debug("Converted %d objects", len(allcenters))
    for i in range(len(allshapes)):
        setattr(allcenters[i], 'id', i)
    searchtree = STRtree(allcenters)
    logger.debug("Built search tree")

    # Get reader and metadata
    reader = BioFormatsReader(image_file)

    # Define reader function for tiler
    def reader_func(*args):
        X, Y, W, H = args[0], args[1], args[3], args[4]
        return reader.read(XYWH=(X, Y, W, H))

    sizeX, sizeY, sizeC = [reader.rdr.getSizeX(), reader.rdr.getSizeY(), reader.rdr.getSizeC()]
    paddingsize = patch_size // 2
    logger.debug('Read image')

    profile_plots = {}
    tiler = Tiler(data_shape=(sizeX, sizeY, sizeC),
                  tile_shape=(tile_size, tile_size, sizeC),
                  overlap=paddingsize,
                  mode='constant')

    detection_count = 0
    for tile_idx in tqdm(range(len(tiler)), desc='Iterating tiles', leave=True):
        tile_bbox = tiler.get_tile_bbox_position(tile_idx)
        X, Y = tile_bbox[0][0:2]

        search_box = build_search_box(X, Y, tile_size, paddingsize)

        hits = searchtree.query(search_box)
        if len(hits) < min_hits:
            continue
        logger.debug('Found %d detections', len(hits))
        detection_count += len(hits)

        tile = tiler.get_tile(reader_func, tile_idx)

        # Make a list of ids and polygons (detections)
        ids = [hit.id for hit in hits]
        polygons = [allshapes[id_] for id_ in ids]
        imgs = []
        # Get image patches
        imgs = np.empty((len(hits), patch_size, patch_size, sizeC), dtype=tile.dtype)
        for idx, hit in enumerate(hits):
            # hit_X, hit_Y = hit.coords[:][0]
            c, r = int(hit.x - X), int(hit.y - Y)
            imgs[idx, ...] = tile[r - patch_size // 2: r + patch_size // 2,
                                  c - patch_size // 2: c + patch_size // 2,
                                  ...]

        # Create Patch objects
        patches = [Patch(imgs[idx, ...], polygons[idx], ids[idx]) for idx in range(len(hits))]
        logger.debug('Created %d patches', len(patches))

        # Process this batch
        for p in tqdm(patches, desc='Extracting radial profiles', leave=False):
            # # Keep plots identifiable by their keys
            profiler = RadialProfilePlotter(p.get_image(), p.get_mask(), 30, 50)
            contour = cv2.findContours(profiler.dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0]
            contour = contour.reshape(contour.shape[0], contour.shape[2])
            centroid = np.array([int(patch_size / 2)]*2)
            avg_dist_to_center = spatial.distance.cdist(contour, centroid[np.newaxis]).mean()
            nBins = int(avg_dist_to_center / 1)
            # profiler.bins = nBins
            profs, dists = profiler.get_profiles()
            # profiler.plot()

            # profs, locs = get_radial_profiles(p.get_image(), p.get_mask(), 30, 60)

            # Do some normalizing
            # for i in range(profs.shape[0]):
            #     profs[i] = profs[i] / np.nanmean(profs[i])

            profs_dict = {}
            profs_dict['distance'] = dists[:-1]
            for ch, titl in enumerate(titles):
                profs_dict[titl] = profs[ch]
            profile_plots[p.id] = profs_dict
    logger.info('Processed %d detections', detection_count)

    df_ids = []
    df_plot_keys = []
    df_profiles = []
    df_distances = []
    df_locs = []
    # Format for pandas dataframe
    for id_, profiles in profile_plots.items():
        for key, value in profiles.items():
            df_ids.append(id_)
            df_distances.append(profiles['distance'])
            df_plot_keys.append(key)
            df_profiles.append(value)

    df = pd.DataFrame({'plot_key': df_plot_keys, 'profiles': df_profiles, 'distance': df_distances}, index=df_ids)

    max_arr_length = np.max([prof.shape for prof in df['profiles']])

    # Match lengths of all profile arrays
    profs_extended = []
    for prof in df['profiles']:
        emp = np.empty((max_arr_length, 1))
        emp[:] = np.nan
        emp[0:prof.size][:, 0] = prof
        profs_extended.append(emp)
    df['profiles_ext'] = profs_extended

    fig, ax = plt.subplots()
    for titl, color in zip(titles, colors):
        profs = df[df['plot_key'] == titl]['profiles'].to_numpy()
        dists = df[df['plot_key'] == titl]['distance'].to_numpy()
        
        for i in range(len(profs)):
            profs[i] = np.flip(profs[i])
        for i in range(len(dists)):
            dists[i] = np.flip(dists[i])
        
        # profs = df[df['plot_key'] == titl]['profs_flipped'].to_numpy()
        for dist, prof in zip(dists, profs):
            ax.plot(dist, prof, c=color, alpha=0.2)

        # avg = np.mean(profs, axis=0)
        # X = np.arange(0, avg.size)
        # std_dev = np.std(profs, axis=0)
        # upper_err = avg + std_dev
        # lower_err = avg - std_dev
        # plt.plot(X, avg, c=color)
        # plt.title('Radial Profile Plot \n' + Path(detection_file).stem, fontsize=8)
        # # plt.suptitle()
        plt.xlabel('Distance from cell boundary')
        plt.ylabel('Intensity')
        # # plt.errorbar(X, avg, std_dev, c=color)
        # plt.fill_between(X, upper_err, lower_err, color=color, alpha=0.1)
    # img_fname = Path('/Users/tim/Desktop') / (Path(detection_file).stem + '_plot.png')
    # plt.savefig(img_fname)

    plt.show()
