# Code by Lachlan Marnoch, 2019
from astropy.io import fits
from astropy import table
from astropy import wcs

import numpy as np
from matplotlib import pyplot as plt
import os

from PyCRAFT import params as p
from PyCRAFT import plotting
from PyCRAFT import fits_files as ff
from PyCRAFT import utils as u
from PyCRAFT import photometry as ph
from PyCRAFT import astrometry as am


def main(field, subtraction_path, epoch, instrument, instrument_template, show):
    comparison_name = f'{field}_{epoch}'

    params = p.object_params_frb(field)
    burst_ra = params['burst_ra']
    burst_dec = params['burst_dec']
    burst_err_a, burst_err_b, theta = am.calculate_error_ellipse(frb=params, error='systematic')
    hg_ra = params['hg_ra']
    hg_dec = params['hg_dec']

    burst_err_theta = 0.0  # params['burst_err_theta']
    # comparison_params = p.object_params_instrument(comparison_name, instrument)
    # comparison_outputs = p.object_output_params(comparison_name, instrument)

    subtraction_path = f'{params["data_dir"]}subtraction/{subtraction_path}'

    filters = params['filters']

    folders = list(filter(lambda file: os.path.isdir(subtraction_path + file), os.listdir(subtraction_path)))
    folders.sort()

    output_table = table.Table(names=['name'], dtype=[f'S{len(folders[-1])}'])

    first = True

    sextractor_names = p.sextractor_names()

    print(subtraction_path)

    for i, folder in enumerate(folders):
        print()
        print(folder)
        subtraction_path_spec = subtraction_path + folder + '/'
        output_table.add_row()
        output_table['name'][i] = folder

        for f in filters:
            print()
            print(f)

            f_0 = f[0]
            destination_path_filter = f'{subtraction_path_spec}{f}/'

            template_epoch = params['template_epoch_' + instrument_template.lower()]
            template_name = f'{field}_{template_epoch}'

            print(destination_path_filter)
            print('Finding files...')

            cats = list(filter(lambda file: file[-15:] == '_comparison.csv',
                               os.listdir(f'{destination_path_filter}')))
            comparisons = list(filter(lambda file: file[-24:] == '_comparison_aligned.fits',
                                      os.listdir(f'{destination_path_filter}')))
            differences = list(filter(lambda file: file[-16:] == '_difference.fits',
                                      os.listdir(f'{destination_path_filter}')))

            if cats:  # If the generated catalogue was successfully copied.
                print('Loading generated table...')
                cat_generated_file = cats[0]
                cat_generated_path = f'{destination_path_filter}{cat_generated_file}'
                cat_generated = table.Table.read(cat_generated_path, format='ascii.csv')

                if comparisons and differences:  # If the subtraction was successful.

                    comparison_image_file = comparisons[0]
                    difference_image_file = differences[0]

                    print('Loading SExtractor table...')
                    cat_sextractor_path = f'{destination_path_filter}difference.cat'
                    comparison_image_path = f'{destination_path_filter}{comparison_image_file}'
                    difference_image_path = f'{destination_path_filter}{difference_image_file}'

                    cat_sextractor = table.Table(np.genfromtxt(cat_sextractor_path, names=sextractor_names))
                    if len(cat_sextractor) > 0:
                        difference_image = fits.open(difference_image_path)
                        comparison_image = fits.open(comparison_image_path)

                        comparison_header = comparison_image[0].header

                        exp_time = comparison_header['EXPTIME']

                        print('Selecting zeropoint...')

                        comparison_zeropoint, _, airmass, _, comparison_extinction, _ = ph.select_zeropoint(
                            comparison_name,
                            f,
                            instrument=instrument)

                        print('Matching coordinates...')

                        match_ids_generated, match_ids_sextractor, match_distances = u.match_cat(
                            x_cat=cat_sextractor['ra'],
                            y_cat=cat_sextractor[
                                'dec'],
                            x_match=cat_generated[
                                'ra'],
                            y_match=cat_generated[
                                'dec'],
                            tolerance=2 / 3600,
                            world=True,
                            return_dist=True)

                        matches_sextractor = cat_sextractor[match_ids_sextractor]
                        matches_generated = cat_generated[match_ids_generated]

                        w = wcs.WCS(header=difference_image[0].header)
                        hg_x, hg_y = w.all_world2pix(hg_ra, hg_dec, 0)

                        # norm = plotting.nice_norm(difference_image[0].data)
                        # print('Generating full-image plot...')
                        # plt.figure(figsize=(30, 20))
                        # plt.imshow(difference_image[0].data, norm=norm, origin='lower')
                        # plotting.plot_all_params(image=difference_image, cat=cat_sextractor, show=False)
                        # # plt.legend()
                        # plt.savefig(f'{destination_path_filter}recovered_all.png')
                        # if show:
                        #     plt.show()
                        # plt.close()

                        plt.figure(figsize=(6, 8))

                        print('Generating cutout plot...')
                        cutout = ff.trim(difference_image, int(hg_x) - 20, int(hg_x) + 20, int(hg_y) - 20,
                                         int(hg_y) + 20)
                        plotting.plot_all_params(image=difference_image, cat=matches_sextractor, show=False)
                        plotting.plot_gal_params(hdu=difference_image, ras=[burst_ra], decs=[burst_dec],
                                                 a=[burst_err_a],
                                                 b=[burst_err_b], theta=[burst_err_theta], colour='blue',
                                                 show_centre=True,
                                                 label='Burst')

                        norm = plotting.nice_norm(cutout[0].data)
                        plt.imshow(cutout[0].data, norm=norm, origin='lower')
                        plt.scatter(hg_x - (int(hg_x) - 20), hg_y - (int(hg_y) - 20), c='orange',
                                    label='Galaxy centroid')
                        for obj in matches_generated:
                            plt.scatter(obj['x_0'] - (int(hg_x) - 20), obj['y_0'] - (int(hg_y) - 20), c='white',
                                        label='Generated')
                        plt.legend()
                        plt.savefig(f'{destination_path_filter}recovered.png')
                        if show:
                            plt.show()
                        plt.close()

                        matches_sextractor['mag_sextractor'], _, _ = ph.magnitude_complete(
                            flux=matches_sextractor['flux_auto'],
                            exp_time=exp_time, airmass=airmass,
                            zeropoint=comparison_zeropoint,
                            ext=comparison_extinction)

                        matches = table.hstack([matches_generated, matches_sextractor],
                                               table_names=['generated', 'sextracted'])

                        matches['delta_mag'] = matches['mag_sextractor'] - matches['mag']

                        delta_mag = np.median(matches['delta_mag'])

                        matches.write(
                            f'{destination_path_filter}{field}_{epoch}-{template_epoch}_difference_matched_sources.csv',
                            format='ascii.csv', overwrite=True)

                        print(f'{len(matches)} matches / {len(cat_generated)} generated = '
                              f'{100 * len(matches) / len(cat_generated)} % ')

                        # TODO: Will need to rewrite this if you want to insert more than one synthetic.

                        output_table = table_setup(f, first, output_table, cat_generated)

                        output_table[f + '_subtraction_successful'][i] = True
                        if len(matches) > 0:
                            print(f'Faintest recovered: {max(matches["mag"])} generated; '
                                  f'{max(matches_sextractor["mag_sextractor"])} sextracted')

                            print('Median delta mag:', delta_mag)

                            if show:
                                plt.scatter(matches['mag'], matches['delta_mag'])
                                plt.show()

                            arg_faintest = np.argmax(matches["mag"])
                            output_table[f + '_mag_generated'][i] = cat_generated["mag"][arg_faintest]
                            output_table[f + '_mag_recovered'][i] = matches['mag_sextractor'][arg_faintest]
                            output_table[f + '_difference'][i] = delta_mag
                            output_table[f + '_matching_distance_arcsec'][i] = match_distances[arg_faintest] * 3600
                            output_table[f + '_nuclear_offset_pix_x'][i] = matches['x'][arg_faintest] - hg_x
                            output_table[f + '_nuclear_offset_pix_y'][i] = matches['y'][arg_faintest] - hg_y
                            for col in cat_generated.colnames:
                                output_table[f + '_' + col][i] = cat_generated[col][arg_faintest]

                        else:  # If there were no matches
                            output_table[f + '_mag_generated'][i] = cat_generated["mag"][0]
                            output_table[f + '_mag_recovered'][i] = np.nan
                            output_table[f + '_difference'][i] = np.nan
                            output_table[f + '_matching_distance_arcsec'][i] = np.nan
                            output_table[f + '_nuclear_offset_pix_x'][i] = np.nan
                            output_table[f + '_nuclear_offset_pix_y'][i] = np.nan
                            for col in cat_generated.colnames:
                                output_table[f + '_' + col][i] = cat_generated[col][0]
                    else:  # If SExtractor was not successful, probably because of a blank difference image

                        output_table = table_setup(f, first, output_table, cat_generated)
                        output_table[f + '_mag_generated'][i] = cat_generated["mag"][0]
                        output_table[f + '_subtraction_successful'][i] = False
                        output_table[f + '_mag_recovered'][i] = np.nan
                        output_table[f + '_difference'][i] = np.nan
                        output_table[f + '_matching_distance_arcsec'][i] = np.nan
                        output_table[f + '_nuclear_offset_pix_x'][i] = np.nan
                        output_table[f + '_nuclear_offset_pix_y'][i] = np.nan
                        for col in cat_generated.colnames:
                            output_table[f + '_' + col][i] = cat_generated[col][0]

                else:  # If the subtraction was not successful.

                    output_table = table_setup(f, first, output_table, cat_generated)
                    output_table[f + '_mag_generated'][i] = cat_generated["mag"][0]
                    output_table[f + '_subtraction_successful'][i] = False
                    output_table[f + '_mag_recovered'][i] = np.nan
                    output_table[f + '_difference'][i] = np.nan
                    output_table[f + '_matching_distance_arcsec'][i] = np.nan
                    output_table[f + '_nuclear_offset_pix_x'][i] = np.nan
                    output_table[f + '_nuclear_offset_pix_y'][i] = np.nan
                    for col in cat_generated.colnames:
                        output_table[f + '_' + col][i] = cat_generated[col][0]

            else:  # If the generated catalogue was not successfully copied.
                output_table[f + '_mag_generated'][i] = np.nan
                output_table[f + '_subtraction_successful'][i] = True
                output_table[f + '_mag_recovered'][i] = np.nan
                output_table[f + '_mag_generated'][i] = np.nan
                output_table[f + '_difference'][i] = np.nan
                output_table[f + '_matching_distance_arcsec'][i] = np.nan
                output_table[f + '_nuclear_offset_pix_x'][i] = np.nan
                output_table[f + '_nuclear_offset_pix_y'][i] = np.nan

        first = False
        

    plt.close()
    for f in filters:
        plt.scatter(output_table[f + '_mag_generated'], output_table[f + '_mag_recovered'],
                    label=f)
        plt.plot([min(output_table[f + '_mag_generated']), max(output_table[f + '_mag_generated'])],
                 [min(output_table[f + '_mag_generated']), max(output_table[f + '_mag_generated'])], c='red',
                 linestyle=':')
        plt.ylim(20, 30)
        plt.savefig(subtraction_path + 'genvrecovered_' + f + '.png')
        plt.close()

        plt.scatter(output_table[f + '_mag_generated'],
                    output_table[f + '_mag_recovered'] - output_table[f + '_mag_generated'],
                    label=f)
        plt.plot([min(output_table[f + '_mag_generated']), max(output_table[f + '_mag_generated'])],
                 [0, 0], c='red',
                 linestyle=':')
        plt.xlim(20, 30)
        plt.savefig(subtraction_path + 'residuals_' + f + '.png')
        plt.close()

    for f in filters:
        plt.scatter(output_table[f + '_mag_generated'],
                    output_table[f + '_mag_recovered'] - output_table[f + '_mag_generated'],
                    label=f)

        plt.plot([min(output_table[f + '_mag_generated']), max(output_table[f + '_mag_generated'])],
                 [0, 0], c='red',
                 linestyle=':')
    plt.xlim(20, 30)
    plt.ylabel('Recovered magnitudes')
    plt.legend()
    plt.savefig(subtraction_path + 'residuals.png')
    if show:
        plt.show()
    plt.close()

    for f in filters:
        plt.scatter(output_table[f + '_mag_generated'], output_table[f + '_mag_recovered'],
                    label=f)
        plt.plot([min(output_table[f + '_mag_generated']), max(output_table[f + '_mag_generated'])],
                 [min(output_table[f + '_mag_generated']), max(output_table[f + '_mag_generated'])], c='red',
                 linestyle=':')
    plt.xlim(20, 30)
    plt.ylim(20, 30)
    plt.xlabel('Generated magnitudes')
    plt.ylabel('Recovered magnitudes')
    plt.legend()
    plt.savefig(subtraction_path + 'genvrecovered.png')
    if show:
        plt.show()
    plt.close()

    output_table.write(subtraction_path + 'recovery_table.csv', format='ascii.csv', overwrite=True)


def table_setup(f, first, output_table, cat_generated):
    if first:
        column = table.Column(name=f + '_mag_generated', length=len(output_table))
        output_table.add_column(column)
        column = table.Column(name=f + '_mag_recovered', length=len(output_table))
        output_table.add_column(column)
        column = table.Column(name=f + '_subtraction_successful', length=len(output_table))
        output_table.add_column(column)
        column = table.Column(name=f + '_difference', length=len(output_table))
        output_table.add_column(column)
        column = table.Column(name=f + '_matching_distance_arcsec', length=len(output_table))
        output_table.add_column(column)
        column = table.Column(name=f + '_nuclear_offset_pix_x', length=len(output_table))
        output_table.add_column(column)
        column = table.Column(name=f + '_nuclear_offset_pix_y', length=len(output_table))
        output_table.add_column(column)
        for col in cat_generated.colnames:
            if col not in ['type', 'source']:
                dtype = float
            else:
                dtype = 'S20'
            column = table.Column(name=f + '_' + col, length=len(output_table), dtype=dtype)
            output_table.add_column(column)

    return output_table

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
    parser.add_argument('-show',
                        action='store_true')

    args = parser.parse_args()

    main(field=args.field,
         subtraction_path=args.subtraction_path,
         epoch=args.epoch,
         instrument=args.instrument,
         instrument_template=args.instrument_template,
         show=args.show)  # args.show)
