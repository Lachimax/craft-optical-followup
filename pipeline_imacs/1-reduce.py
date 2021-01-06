# Code by Lachlan Marnoch, 2019
from astropy.nddata import CCDData
from astropy.io import fits
import ccdproc
import os
import shutil
import matplotlib.pyplot as plt
from astropy.visualization import (ImageNormalize, SqrtStretch, ZScaleInterval)
import numpy as np

from craftutils import fits_files as ff
from craftutils import utils as u
from craftutils import params as p


# TODO: Standardise input names, eg --path or --directory?
# TODO: This script is quite slow - lots of expensive saving and loading. If you can work out how to do it more cleanly,
#  maybe by preserving each object as CCDData across the whole script, that would be better.

def main(data_title: str, show: bool = False):
    properties = p.object_params_imacs(data_title)
    path = properties['data_dir']

    raw_path = path + '/0-raw_data_with_calibs/'
    master_path = path + '/1-master_calibs/'
    reduced_path = path + '/2-reduced/'

    u.mkdir_check(raw_path)
    u.mkdir_check(master_path)
    u.mkdir_check(reduced_path)

    for file in os.listdir(path):
        if os.path.isfile(path + file):
            shutil.move(path + file, raw_path + file)

    files = os.listdir(raw_path)

    biases = []
    flats = {}
    science = {}

    airmasses = {}

    print('Creating lists of files.')

    param_dict = {}

    for file in files:
        if file[-5:] == '.fits':
            hdu = fits.open(raw_path + file)
            header = hdu[0].header
            obj = header['OBJECT']
            f = header['FILTER']
            if 'bias' in obj:
                biases.append(file)
            elif 'flat' in obj:
                if f not in flats:
                    flats[f] = []
                flats[f].append(file)
            else:
                if f not in science:
                    science[f] = []
                    airmasses[f] = []
                science[f].append(file)
                airmasses[f].append(header['AIRMASS'])

    # Loop through chip numbers.
    for chip in range(1, 9):
        print(f'Processing biases for chip {chip}.')
        # Get only those bias frames relevant to the current chip.
        biases_chip = list(filter(lambda name: name[-6] == str(chip), biases))
        bias_hdus = []
        for bias in biases_chip:
            # Correct NAXIS1 & 2 in headers; they are too large in the unreduced files, leading to padding by astropy.
            bias = ff.correct_naxis(file=raw_path + bias, x=2048, y=4096, write=False)
            bias_hdus.append(bias)
        # Stack biases.
        print(biases_chip)
        ff.stack(bias_hdus, output=master_path + f'master_bias_c{chip}.fits', stack_type='median', directory=raw_path)
        master_bias = CCDData.read(master_path + f'master_bias_c{chip}.fits', unit='du')

        # Loop through filters.
        for f in science:



            flats_filter = flats[f]
            master_path_filter = master_path + f + '/'
            u.mkdir_check(master_path_filter)

            print(f'Processing flats for chip {chip}, filter {f}.')
            # Get only those flats relevant to the current chip and filter.
            flats_chip = list(filter(lambda name: name[-6] == str(chip), flats_filter))
            flats_chip_ccds = []
            for flat in flats_chip:
                flat_ccd = CCDData.read(raw_path + flat, unit='du')
                # Correct NAXIS1 & 2 in headers; they are too large in the unreduced files, leading to padding by astropy.
                flat_ccd = ff.correct_naxis(file=flat_ccd, x=2048, y=4096, write=False)
                # Subtract master bias from each flat.
                flat_ccd = ccdproc.subtract_bias(ccd=flat_ccd, master=master_bias)
                flats_chip_ccds.append(flat_ccd)
            # Stack debiased flats.
            master_flat = ff.stack(flats_chip_ccds, output=None, stack_type='median')
            master_flat.writeto(master_path_filter + f'master_flat_c{chip}.fits', overwrite=True)
            master_flat = CCDData.read(master_path_filter + f'master_flat_c{chip}.fits', unit='du')

            science_filter = science[f]

            param_dict[f + '_airmass_mean'] = float(np.nanmean(airmasses[f]))
            param_dict[f + '_airmass_err'] = float(2 * np.nanstd(airmasses[f]))
            param_dict[f + '_n_frames'] = len(science_filter)
            param_dict[f + '_n_exposures'] = len(science_filter) / 8

            reduced_path_filter = reduced_path + f + '/'
            u.mkdir_check(reduced_path_filter)
            # Loop through the science images taken on the current chip.
            for image in filter(lambda name: name[-6] == str(chip), science_filter):
                print(f'Reducing {image}.')
                image_ccd = CCDData.read(raw_path + image, unit='du')
                if show:
                    norm = ImageNormalize(image_ccd.data, interval=ZScaleInterval(), stretch=SqrtStretch())
                    plt.imshow(image_ccd.data, origin='lower', norm=norm)
                    plt.title('Unreduced image')
                    plt.show()
                # Correct NAXIS1 & 2 in header; they are too large in the unreduced files, leading to padding by astropy.
                image_ccd = ff.correct_naxis(file=image_ccd, x=2048, y=4096, write=False)
                if show:
                    norm = ImageNormalize(image_ccd.data, interval=ZScaleInterval(), stretch=SqrtStretch())
                    plt.imshow(image_ccd.data, origin='lower', norm=norm)
                    plt.title('After NAXIS correction')
                    plt.show()
                # Subtract master bias from science image.
                image_ccd = ccdproc.subtract_bias(image_ccd, master_bias)
                if show:
                    norm = ImageNormalize(image_ccd.data, interval=ZScaleInterval(), stretch=SqrtStretch())
                    plt.imshow(image_ccd.data, origin='lower', norm=norm)
                    plt.title('After debiasing')
                    plt.show()
                # Divide by master flat.
                image_ccd = ccdproc.flat_correct(image_ccd, master_flat)
                if show:
                    norm = ImageNormalize(image_ccd.data, interval=ZScaleInterval(), stretch=SqrtStretch())
                    plt.imshow(image_ccd.data, origin='lower', norm=norm)
                    plt.title('After flatfielding')
                    plt.show()
                # Convert back to HDU object for saving.
                image_ccd = image_ccd.to_hdu()
                image_ccd.writeto(reduced_path_filter + image, overwrite=True)
    u.write_log(path=reduced_path + data_title + ".log", action=f'Reduced using 1-reduce.py')
    p.add_params(path + '/output_values', param_dict)


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
