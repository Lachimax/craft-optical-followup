# Code by Lachlan Marnoch, 2019
from astropy.io import fits
from astropy import table
from astropy import wcs
import astropy.time as time
import numpy as np
from matplotlib import pyplot as plt
import os

from craftutils import params as p
from craftutils import plotting
from craftutils import fits_files as ff
from craftutils import utils as u
from craftutils import photometry as ph


def main(field, subtraction_path, epoch, instrument):
    comparison_name = f'{field}_{epoch}'

    params = p.object_params_frb(field)
    ra_burst = params['burst_ra']
    dec_burst = params['burst_dec']
    burst_err_a = params['burst_err_a']
    burst_err_b = params['burst_err_b']
    burst_err_theta = params['burst_err_theta']
    comparison_params = p.object_params_instrument(comparison_name, instrument)

    subtraction_path = f'{params["data_dir"]}subtraction/{subtraction_path}'

    filters = params['filters']
    for f in filters:
        f_0 = f[0]
        destination_path_filter = f'{subtraction_path}{f}/'

        template_epoch = params['template_epoch']

        comparison_extinction = comparison_params[f_0 + '_ext_up']

        cat_generated_file = filter(lambda file: file[-15:] == '_comparison.csv',
                                    os.listdir(f'{destination_path_filter}')).__next__()
        comparison_image_file = filter(lambda file: file[-24:] == '_comparison_aligned.fits',
                                       os.listdir(f'{destination_path_filter}')).__next__()
        difference_image_file = filter(lambda file: file[-16:] == '_difference.fits',
                                       os.listdir(f'{destination_path_filter}')).__next__()

        cat_generated_path = f'{destination_path_filter}{cat_generated_file}'
        cat_sextractor_path = f'{destination_path_filter}difference.cat'
        comparison_image_path = f'{destination_path_filter}{comparison_image_file}'
        difference_image_path = f'{destination_path_filter}{difference_image_file}'

        cat_generated = table.Table.read(cat_generated_path, format='ascii.csv')
        cat_sextractor = table.Table(np.genfromtxt(cat_sextractor_path, names=p.sextractor_names()))
        difference_image = fits.open(difference_image_path)
        comparison_image = fits.open(comparison_image_path)

        comparison_header = comparison_image[0].header

        exp_time = comparison_header['EXPTIME']

        comparison_zeropoint, _, airmass, _ = ph.select_zeropoint(comparison_name, f, instrument=instrument)

        plotting.plot_all_params(image=difference_image, cat=cat_sextractor, show=False)
        for obj in cat_generated:
            plt.scatter(obj['x_0'], obj['y_0'], c='white')
        plt.show()

        _, pix_scale = ff.get_pixel_scale(comparison_image)

        match_ids_sextractor, match_ids_generated = u.match_cat(x_match=cat_sextractor['ra'],
                                                                y_match=cat_sextractor['dec'],
                                                                x_cat=cat_generated['ra'],
                                                                y_cat=cat_generated['dec'],
                                                                tolerance=3 * pix_scale)

        matches_sextractor = cat_sextractor[match_ids_sextractor]
        matches_generated = cat_generated[match_ids_generated]

        plotting.plot_all_params(image=difference_image, cat=matches_sextractor, show=False)
        # plt.scatter(x_burst, y_burst, c='white')
        plotting.plot_gal_params(hdu=difference_image, ras=[ra_burst], decs=[dec_burst], a=[burst_err_a],
                                 b=[burst_err_b], theta=[burst_err_theta], colour='blue', show_centre=True)
        for obj in matches_generated:
            plt.scatter(obj['x_0'], obj['y_0'], c='white')
        plt.show()

        matches_sextractor['mag_sextractor'], _, _ = ph.magnitude_complete(flux=matches_sextractor['flux_aper'],
                                                                           exp_time=exp_time, airmass=airmass,
                                                                           zeropoint=comparison_zeropoint,
                                                                           ext=comparison_extinction)

        matches = table.hstack([matches_generated, matches_sextractor], table_names=['generated', 'sextracted'])

        matches['delta_mag'] = matches['mag_sextractor'] - matches['mag']

        delta_mag = np.median(matches['delta_mag'])

        matches.write(f'{destination_path_filter}{field}_{epoch}-{template_epoch}_difference_matched_sources.csv',
                      format='ascii.csv', overwrite=True)

        print(f'{len(matches)} matches / {len(cat_generated)} generated = '
              f'{100 * len(matches) / len(cat_generated)} % ')

        print(f'Faintest recovered: {max(matches["mag"])} generated; '
              f'{max(matches_sextractor["mag_sextractor"])} sextracted')

        print('Median delta mag:', delta_mag)

        plt.scatter(matches['mag'], matches['delta_mag'])
        plt.show()

        difference_image.close()


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
