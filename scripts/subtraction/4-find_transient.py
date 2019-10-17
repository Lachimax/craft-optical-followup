# Code by Lachlan Marnoch, 2019
from PyCRAFT import params as p
from PyCRAFT import photometry as ph
from PyCRAFT import utils as u
from PyCRAFT import plotting
from PyCRAFT import fits_files as ff

from astropy import table
from astropy.io import fits
from astropy import wcs
from matplotlib import pyplot as plt
import numpy as np
import os


def main(field, subtraction_path, epoch, instrument):
    comparison_name = f'{field}_{epoch}'

    params = p.object_params_frb(field)
    # comparison_params = p.object_params_instrument(comparison_name, instrument)

    subtraction_path = f'{params["data_dir"]}subtraction/{subtraction_path}'

    filters = params['filters']

    burst_ra = params['burst_ra']
    burst_dec = params['burst_dec']
    hg_ra = params['hg_ra']
    hg_dec = params['hg_dec']
    burst_err_a = params['burst_err_stat_a'] / 3600
    burst_err_b = params['burst_err_stat_b'] / 3600
    burst_err_theta = params['burst_err_theta']

    for f in filters:
        print()
        print(f)
        f_0 = f[0]
        destination_path_filter = f'{subtraction_path}{f}/'

        # This relies on there only being one file with these suffixes in each directory, which, if previous scripts
        # ran correctly, should be true.

        comparison_image_file = filter(lambda file: file[-24:] == '_comparison_aligned.fits',
                                       os.listdir(f'{destination_path_filter}')).__next__()
        difference_image_file = filter(lambda file: file[-16:] == '_difference.fits',
                                       os.listdir(f'{destination_path_filter}')).__next__()

        cat_sextractor_path = f'{destination_path_filter}difference.cat'
        comparison_image_path = f'{destination_path_filter}{comparison_image_file}'
        difference_image_path = f'{destination_path_filter}{difference_image_file}'

        cat_sextractor = table.Table(np.genfromtxt(cat_sextractor_path, names=p.sextractor_names()))

        difference_image = fits.open(difference_image_path)
        comparison_image = fits.open(comparison_image_path)

        comparison_header = comparison_image[0].header

        exp_time = comparison_header['EXPTIME']

        comparison_zeropoint, _, airmass, _, comparison_extinction, _ = ph.select_zeropoint(comparison_name, f, instrument=instrument)

        cat_sextractor['mag'], _, _ = ph.magnitude_complete(flux=cat_sextractor['flux_aper'],
                                                            exp_time=exp_time, airmass=airmass,
                                                            zeropoint=comparison_zeropoint,
                                                            ext=comparison_extinction)

        cat_sextractor['mag_auto'], _, _ = ph.magnitude_complete(flux=cat_sextractor['flux_auto'],
                                                                 exp_time=exp_time, airmass=airmass,
                                                                 zeropoint=comparison_zeropoint,
                                                                 ext=comparison_extinction)

        wcs_info = wcs.WCS(comparison_header)
        hg_x, hg_y = wcs_info.all_world2pix(hg_ra, hg_dec, 0)

        # norm = plotting.nice_norm(difference_image[0].data)
        # print('Generating full-image plot...')
        # plt.figure(figsize=(30, 20))
        # plt.imshow(difference_image[0].data, norm=norm, origin='lower')
        # plotting.plot_all_params(image=difference_image, cat=cat_sextractor, show=False)
        # # plt.legend()
        # plt.savefig(f'{destination_path_filter}recovered_all.pdf')
        # plt.close()

        closest_index, distance = u.find_object(burst_ra, burst_dec, cat_sextractor['ra'], cat_sextractor['dec'])

        closest = cat_sextractor[closest_index]
        x = closest['x']
        y = closest['y']

        print(closest)
        print('Magnitude:', closest['mag'])
        print('Threshold:', closest['threshold'])
        print('Flux max:', closest['flux_max'])
        print('RA:', closest['ra'], '; DEC:', closest['dec'])
        print('X:', x, '; Y:', y)
        print('Offset from HG (pixels):', np.sqrt((x - hg_x) ** 2 + (y - hg_y) ** 2))

        x = int(x)
        y = int(y)

        norm = plotting.nice_norm(difference_image[0].data)
        print('Generating full-image plot...')
        plt.figure(figsize=(30, 20))
        plt.imshow(difference_image[0].data, norm=norm, origin='lower')
        print(min(cat_sextractor['ra']), min(cat_sextractor['dec']))
        plotting.plot_all_params(image=difference_image, cat=cat_sextractor, show=False)
        # plt.legend()
        plt.savefig(f'{destination_path_filter}recovered_all.png')
        plt.close()

        print('Generating cutout plots...')

        plt.imshow(difference_image[0].data, norm=norm, origin='lower')
        # plt.legend()
        plotting.plot_gal_params(hdu=difference_image, ras=[burst_ra], decs=[burst_dec], a=[burst_err_a],
                                 b=[burst_err_b], theta=[burst_err_theta], colour='blue', show_centre=True,
                                 label='Burst')
        plt.scatter(hg_x, hg_y, c='orange', label='Galaxy centroid')
        plotting.plot_all_params(image=difference_image, cat=cat_sextractor, show=False)
        plt.xlim(hg_x - 50, hg_x + 50)
        plt.ylim(hg_y - 50, hg_y + 50)
        plt.legend()
        plt.savefig(f'{destination_path_filter}recovered_hg_cutout.png')
        plt.close()

        cutout_source = difference_image[0].data[y - 50:y + 50, x - 50:x + 50]

        plt.imshow(cutout_source, origin='lower')
        plt.savefig(f'{destination_path_filter}nearest_source_cutout.png')
        plt.show()
        plt.close()

        cutout_hg = ff.trim(difference_image, int(hg_x) - 20, int(hg_x) + 20, int(hg_y) - 20, int(hg_y) + 20)
        cutout_hg_data = cutout_hg[0].data

        midpoint = int(cutout_hg_data.shape[0] / 2)
        xx = np.arange(-midpoint, midpoint)
        data_slice = cutout_hg_data[midpoint, :]
        plt.plot(xx, data_slice)
        plt.savefig(f'{destination_path_filter}hg_x_slice.png')
        plt.show()
        plt.close()
        print()
        x_misalignment = np.argmax(data_slice) - np.argmin(data_slice)
        print('x peak - trough:', x_misalignment)

        midpoint = int(cutout_hg_data.shape[0] / 2)
        yy = np.arange(-midpoint, midpoint)
        data_slice = cutout_hg_data[:, midpoint]
        plt.plot(yy, data_slice)
        plt.savefig(f'{destination_path_filter}hg_y_slice.png')
        plt.show()
        plt.close()
        y_misalignment = np.argmax(data_slice) - np.argmin(data_slice)
        print('y peak - trough:', y_misalignment)
        print()

        print('Mean false positive mag:', np.nanmean(cat_sextractor['mag']))
        print('Median false positive mag:', np.nanmedian(cat_sextractor['mag']))

        plt.figure(figsize=(6, 8))

        sextractor_cutout = cat_sextractor[(cat_sextractor['x'] > int(hg_x) - 20) & (cat_sextractor['x'] < int(hg_x) + 20) & (cat_sextractor['y'] > int(hg_y) - 20) & (cat_sextractor['y'] < int(hg_y) + 20)]
        plotting.plot_all_params(image=difference_image, cat=sextractor_cutout, show=False)
        plotting.plot_gal_params(hdu=difference_image, ras=[burst_ra], decs=[burst_dec], a=[burst_err_a],
                                 b=[burst_err_b], theta=[burst_err_theta], colour='blue', show_centre=True,
                                 label='Burst')

        norm = plotting.nice_norm(cutout_hg[0].data)
        plt.imshow(cutout_hg[0].data, norm=norm, origin='lower')
        plt.scatter(hg_x - (int(hg_x) - 20), hg_y - (int(hg_y) - 20), c='orange', label='Galaxy centroid')
        plt.legend()
        plt.savefig(f'{destination_path_filter}residuals_hg.png')
        plt.show()
        plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--field',
                        help='Name of object parameter file without .yaml, eg FRB190102.',
                        type=str)
    parser.add_argument('--subtraction_path',
                        help='Folder to operate on.',
                        type=str)
    parser.add_argument('--epoch',
                        help='Number of comparison epoch to subtract template from.',
                        type=int,
                        default=1)
    parser.add_argument('--instrument',
                        help='Instrument of comparison epoch.',
                        type=str,
                        default='FORS2')
    parser.add_argument('--instrument_template',
                        help='Instrument of template epoch.',
                        type=str,
                        default='FORS2')

    args = parser.parse_args()

    main(field=args.field,
         subtraction_path=args.subtraction_path,
         epoch=args.epoch,
         instrument=args.instrument)
