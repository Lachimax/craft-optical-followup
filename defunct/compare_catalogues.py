# Code by Lachlan Marnoch, 2019
# This script trims the edges of the Montage-produced coaddition, in order to remove the regions with sub-nominal
# overlap.

import craftutils.fits_files as ff
import craftutils.params as p
import craftutils.utils as u
import craftutils.plotting as pl

from astropy import table
from astropy.io import fits
from astropy import wcs
import matplotlib.pyplot as plt
import numpy as np
import sys
import shutil
import os
from shutil import copyfile


# TODO: Refactor all script inputs to match argparse inputs, for readability.

def main(image, cat_path_1, cat_path_2, cat_name_1, cat_name_2, offset_x, offset_y, psf):
    if cat_name_1 == 'DES':
        ra_name_1 = 'RA'
        dec_name_1 = 'DEC'
    else:
        ra_name_1 = 'ra'
        dec_name_1 = 'dec'

    if cat_name_2 == 'DES':
        ra_name_2 = 'RA'
        dec_name_2 = 'DEC'
    else:
        ra_name_2 = 'ra'
        dec_name_2 = 'dec'

    if psf:
        names = p.sextractor_names_psf()
    else:
        names = p.sextractor_names()

    if cat_name_1 == 'SExtractor':
        cat_1 = table.Table(np.genfromtxt(cat_path_1, names=names))
    else:
        cat_1 = table.Table()
        cat_1 = cat_1.read(cat_path_1, format='ascii.csv')

    if cat_name_2 == 'SExtractor':
        cat_2 = table.Table(np.genfromtxt(cat_path_2, names=names))
    else:
        cat_2 = table.Table()
        cat_2 = cat_2.read(cat_path_2, format='ascii.csv')

    param_dict = {}

    image = fits.open(image)
    header = image[0].header
    data = image[0].data
    wcs_info = wcs.WCS(header=header)

    cat_1['x'], cat_1['y'] = wcs_info.all_world2pix(cat_1[ra_name_1], cat_1[dec_name_1], 0)
    cat_2['x'], cat_2['y'] = wcs_info.all_world2pix(cat_2[ra_name_2], cat_2[dec_name_2], 0)

    norm = pl.nice_norm(data)
    plt.subplot(projection=wcs_info)
    plt.imshow(data, norm=norm, origin='lower', cmap='viridis')
    plt.scatter(cat_1['x'], cat_1['y'], label=cat_name_1, c='violet')
    plt.scatter(cat_2['x'], cat_2['y'], label=cat_name_2, c='red')
    plt.legend()
    plt.show()

    scale_ra, scale_dec = ff.get_pixel_scale(image)

    ra_corrected = cat_2[ra_name_2] + offset_x * scale_ra
    dec_corrected = cat_2[dec_name_2] + offset_y * scale_ra

    x, y = wcs_info.all_world2pix(ra_corrected, dec_corrected, 0)

    norm = pl.nice_norm(data)
    plt.subplot(projection=wcs_info)
    plt.imshow(data, norm=norm, origin='lower', cmap='viridis')
    plt.scatter(cat_1['x'], cat_1['y'], label=cat_name_1, c='violet')
    plt.scatter(x, y, label=cat_name_2, c='red')
    plt.legend()
    plt.show()
    image.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Display the SExtractor positions of a catalogue.")
    parser.add_argument('--image',
                        type=str)
    parser.add_argument('--cat_path_1',
                        type=str)
    parser.add_argument('--cat_path_2',
                        type=str)
    parser.add_argument('--cat_name_1',
                        type=str)
    parser.add_argument('--cat_name_2',
                        type=str)
    parser.add_argument('--offset_x',
                        type=float,
                        default=0.)
    parser.add_argument('--offset_y',
                        type=float,
                        default=0.)
    parser.add_argument('-psf',
                        action='store_true')

    args = parser.parse_args()

    main(image=args.image,
         cat_path_1=args.cat_path_1,
         cat_path_2=args.cat_path_2,
         cat_name_1=args.cat_name_1,
         cat_name_2=args.cat_name_2,
         offset_x=args.offset_x,
         offset_y=args.offset_y,
         psf=args.psf)
