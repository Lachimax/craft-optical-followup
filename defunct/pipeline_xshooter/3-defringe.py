# Code by Lachlan Marnoch, 2019
from astropy.io import fits
import os
import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile

from craftutils import fits_files as ff
from craftutils import utils as u
from craftutils import params as p
from craftutils import plotting as pl


# https://www.eso.org/sci/facilities/lasilla/instruments/efosc/inst/fringing.html

def main(data_title: str, show: bool = False):
    properties = p.object_params_xshooter(data_title)
    path = properties['data_dir']

    master_path = path + '/1-master_calibs/'
    reduced_path = path + '/2-reduced/'
    defringed_path = path + '/3-defringed/'

    u.mkdir_check(defringed_path)

    # Define fringe measurement points.

    # high_xs = [267, 267, 267, 267, 267, 267, 267, 267]
    # high_ys = [279, 279, 279, 279, 279, 279, 279, 279]
    # low_xs = [266, 266, 266, 267, 267, 270, 274, 273]
    # low_ys = [293, 295, 298, 301, 303, 305, 303, 292]

    high_xs = [219, 380, 426, 515, 156, 495, 310]
    high_ys = [166, 369, 185, 33, 59, 195, 70]
    low_xs = [219, 380, 424, 474, 160, 500, 315]
    low_ys = [120, 342, 213, 39, 34, 160, 35]

    # n_random = 1000
    #
    # high_xs = np.random.random(n_random)
    # high_xs *= 507
    # high_xs += 29
    # high_xs = np.round(high_xs)
    # high_xs = high_xs.astype(int)
    #
    # high_ys = np.random.random(n_random)
    # high_ys *= 200
    # high_ys += 20
    # high_ys = np.round(high_ys)
    # high_ys = high_ys.astype(int)
    #
    # low_xs = np.random.random(n_random)
    # low_xs *= 507
    # low_xs += 29
    # low_xs = np.round(low_xs)
    # low_xs = low_xs.astype(int)
    #
    # low_ys = np.random.random(n_random)
    # low_ys *= 200
    # low_ys += 20
    # low_ys = np.round(low_ys)
    # low_ys = low_ys.astype(int)

    filters = filter(lambda f: os.path.isdir(reduced_path + f), os.listdir(reduced_path))
    for f in filters:
        print('Constructing fringe map for', f)
        filter_path = reduced_path + f + '/'
        defringed_filter_path = defringed_path + f + '/'
        master_filter_path = master_path + f + '/'
        u.mkdir_check(defringed_filter_path)

        files = list(filter(lambda file: file[-5:] == '.fits', os.listdir(filter_path)))
        # Construct fringe map by median-combining science images.
        fringe_map = ff.stack(files, directory=filter_path, output=master_filter_path + 'fringe_map.fits',
                              stack_type='median', inherit=False, show=show, normalise=True)
        fringe_map = fringe_map[0].data
        map_differences = []

        for i in range(len(high_xs)):
            # Take
            high_y = high_ys[i]
            high_x = high_xs[i]
            high_cut = fringe_map[high_y - 1:high_y + 1, high_x - 1:high_x + 1]
            high = np.nanmedian(high_cut)

            low_y = low_ys[i]
            low_x = low_xs[i]
            low_cut = fringe_map[low_y - 1:low_y + 1, low_x - 1:low_x + 1]
            low = np.nanmedian(low_cut)

            map_differences.append(high - low)

        for file in os.listdir(filter_path):
            print(file)
            hdu = fits.open(filter_path + file)
            data = hdu[0].data
            image_differences = []
            factors = []
            for i in range(len(high_xs)):
                high_y = high_ys[i]
                high_x = high_xs[i]
                high_cut = data[high_y - 2:high_y + 2, high_x - 2:high_x + 2]
                high = np.nanmedian(high_cut)

                low_y = low_ys[i]
                low_x = low_xs[i]
                low_cut = data[low_y - 2:low_y + 2, low_x - 2:low_x + 2]
                low = np.nanmedian(low_cut)

                difference = high - low
                image_differences.append(difference)
                factor = difference / map_differences[i]
                factors.append(factor)
            used_factor = np.nanmedian(factors)
            adjusted_map = fringe_map * used_factor
            data = data - adjusted_map
            hdu[0].data = data

            norm = pl.nice_norm(data)
            if show:
                plt.imshow(data, norm=norm, origin='lower')
                plt.show()

            hdu.writeto(defringed_filter_path + file, overwrite=True)

        if show:
            norm = pl.nice_norm(fringe_map)
            plt.imshow(fringe_map, norm=norm, origin='lower')
            plt.scatter(high_xs, high_ys)
            plt.scatter(low_xs, low_ys)
            plt.show()

        copyfile(reduced_path + data_title + ".log", defringed_path + data_title + ".log")
        u.write_log(path=defringed_path + data_title + ".log", action='Edges trimmed using 4-trim.py\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Reduce raw IMACS data.')

    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        type=str)
    parser.add_argument('-show',
                        help='Show stages of reduction as plots?',
                        action='store_true')

    args = parser.parse_args()

    main(data_title=args.op, show=args.show)
