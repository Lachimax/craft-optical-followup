# Code by Lachlan Marnoch, 2019

from PyCRAFT import photometry
from PyCRAFT import params as p
from PyCRAFT import fits_files as ff
from PyCRAFT.utils import mkdir_check

import os
import matplotlib
import astropy.time as time

matplotlib.rcParams.update({'errorbar.capsize': 3})


# TODO: Argparse, scriptify

def main(obj,
         instrument,
         test_name,
         sex_x_col,
         sex_y_col,
         sex_ra_col,
         sex_dec_col,
         sex_flux_col,
         get_sextractor_names,
         mag_tol,
         stars_only,
         star_class_col,
         star_class_tol,
         show_plots,
         mag_range_sex_lower,
         mag_range_sex_upper,
         pix_tol,
         separate_chips,
         write):

    sextractor_names = p.sextractor_names_psf()  # None to auto-detect

    properties = p.object_params_instrument(obj=obj, instrument=instrument)
    outputs = p.object_output_params(obj=obj, instrument=instrument)
    paths = p.object_output_paths(obj=obj, instrument=instrument)

    cat_name = properties['cat_field_name']

    output = properties['data_dir'] + '/9-zeropoint/'
    mkdir_check(output)
    output = output + 'field/'
    mkdir_check(output)

    if instrument not in ['FORS2', 'fors2']:
        separate_chips = False

    for fil in properties['filters']:

        print('Doing filter', fil)

        f_0 = fil[0]
        do_zeropoint = properties[f_0 + '_do_zeropoint_field']

        if f_0 + '_zeropoint_provided' not in outputs and do_zeropoint:

            print('Zeropoint not found, calculating zeropoint...')

            f_up = f_0.upper()
            f_low = f_0.lower()

            output_path = output + f_0

            chip_1_bottom = 740
            chip_2_top = 600

            cat_zeropoint = 0.
            cat_zeropoint_err = 0.

            now = time.Time.now()
            now.format = 'isot'
            test_name = str(now) + '_' + test_name

            mkdir_check(output_path)
            output_path = output_path + '/' + f_0 + '/'
            mkdir_check(output_path)
            output_path = output_path + '/' + test_name + '/'
            mkdir_check(output_path)

            if cat_name == 'DES':
                cat_path = properties['des_cat']
                if cat_path is None:
                    raise ValueError(
                        'No DES catalogue available at the position of ' + obj + '.')
                cat_ra_col = 'RA'
                cat_dec_col = 'DEC'
                cat_mag_col = 'WAVG_MAG_PSF_' + f_up
                # star_class_col = 'CLASS_STAR_G'
                cat_type = 'csv'

            elif cat_name == 'SExtractor':
                cat_path = properties[f_0 + '_cat_calib_path']
                cat_zeropoint = properties[f_0 + '_cat_calib_zeropoint']
                cat_zeropoint_err = properties[f_0 + '_cat_calib_zeropoint_err']
                if cat_path is None:
                    raise ValueError(
                        'No other epoch catalogue available at the position of ' + obj + '.')
                cat_ra_col = 'ra_cat'
                cat_dec_col = 'dec_cat'
                cat_mag_col = 'mag_psf_other'
                cat_type = 'sextractor'
                star_class_col = star_class_col + '_fors'
                sex_x_col = sex_x_col + '_fors'
                sex_y_col = sex_y_col + '_fors'
                cat_name = 'other'

            elif cat_name == 'SDSS':
                cat_path = properties['sdss_cat']
                if cat_path is None:
                    raise ValueError(
                        'No SDSS catalogue available at the position of ' + obj + '.')
                cat_ra_col = 'ra'
                cat_dec_col = 'dec'
                cat_mag_col = 'psfMag_' + f_low
                cat_type = 'csv'

            else:  # elif cat_name == 'SkyMapper'
                cat_fits_path = properties['sm_fits']
                cat_path = properties['sm_cat']
                if cat_fits_path is None:
                    raise ValueError(
                        'No SkyMapper catalogue available at the position of ' + obj + '.')
                cat_ra_col = 'ra_img'
                cat_dec_col = 'decl_img'
                cat_mag_col = 'mag_psf'
                # star_class_col = 'class_star_SkyMapper'
                cat_type = 'csv'

            if not os.path.isdir(output_path):
                os.mkdir(output_path)

            output_path = output_path + test_name + '/'

            if not os.path.isdir(output_path):
                os.mkdir(output_path)

            image_path = paths[f_0 + '_' + properties['subtraction_image']]
            sextractor_path = paths[f_0 + '_cat_path']
            mag_range_lower = properties[f_0 + '_field_mag_range_lower']
            mag_range_upper = properties[f_0 + '_field_mag_range_upper']

            exp_time = ff.get_exp_time(image_path)

            print('SExtractor catalogue path:', sextractor_path)
            print('Image path:', image_path)
            print('Catalogue path:', cat_path)
            print('Output:', output_path + test_name)
            print()

            print(cat_zeropoint)

            if separate_chips:
                # Split based on which CCD chip the object falls upon.
                if not os.path.isdir(output_path + "chip_1"):
                    os.mkdir(output_path + "chip_1")
                if not os.path.isdir(output_path + "chip_2"):
                    os.mkdir(output_path + "chip_2")

                print('Chip 1:')
                up = photometry.determine_zeropoint_sextractor(sextractor_cat_path=sextractor_path,
                                                               image=image_path,
                                                               cat_path=cat_path,
                                                               cat_name=cat_name,
                                                               output_path=output_path + "/chip_1/",
                                                               show=show_plots,
                                                               cat_ra_col=cat_ra_col,
                                                               cat_dec_col=cat_dec_col,
                                                               cat_mag_col=cat_mag_col,
                                                               sex_ra_col=sex_ra_col,
                                                               sex_dec_col=sex_dec_col,
                                                               sex_x_col=sex_x_col,
                                                               sex_y_col=sex_y_col,
                                                               pix_tol=pix_tol,
                                                               mag_tol=mag_tol,
                                                               flux_column=sex_flux_col,
                                                               mag_range_cat_upper=mag_range_upper,
                                                               mag_range_cat_lower=mag_range_lower,
                                                               mag_range_sex_upper=mag_range_sex_upper,
                                                               mag_range_sex_lower=mag_range_sex_lower,
                                                               stars_only=stars_only,
                                                               star_class_tol=star_class_tol,
                                                               star_class_col=star_class_col,
                                                               exp_time=exp_time,
                                                               y_lower=chip_1_bottom,
                                                               get_sextractor_names=get_sextractor_names,
                                                               sextractor_names=sextractor_names,
                                                               cat_type=cat_type,
                                                               cat_zeropoint=cat_zeropoint,
                                                               cat_zeropoint_err=cat_zeropoint_err
                                                               )

                print('Chip 2:')
                down = photometry.determine_zeropoint_sextractor(sextractor_cat_path=sextractor_path,
                                                                 image=image_path,
                                                                 cat_path=cat_path,
                                                                 cat_name=cat_name,
                                                                 output_path=output_path + "/chip_2/",
                                                                 show=show_plots,
                                                                 cat_ra_col=cat_ra_col,
                                                                 cat_dec_col=cat_dec_col,
                                                                 cat_mag_col=cat_mag_col,
                                                                 sex_ra_col=sex_ra_col,
                                                                 sex_dec_col=sex_dec_col,
                                                                 sex_x_col=sex_x_col,
                                                                 sex_y_col=sex_y_col,
                                                                 pix_tol=pix_tol,
                                                                 mag_tol=mag_tol,
                                                                 flux_column=sex_flux_col,
                                                                 mag_range_cat_upper=mag_range_upper,
                                                                 mag_range_cat_lower=mag_range_lower,
                                                                 mag_range_sex_upper=mag_range_sex_upper,
                                                                 mag_range_sex_lower=mag_range_sex_lower,
                                                                 stars_only=stars_only,
                                                                 star_class_tol=star_class_tol,
                                                                 star_class_col=star_class_col,
                                                                 exp_time=exp_time,
                                                                 y_upper=chip_2_top,
                                                                 get_sextractor_names=get_sextractor_names,
                                                                 sextractor_names=sextractor_names,
                                                                 cat_type=cat_type,
                                                                 cat_zeropoint=cat_zeropoint,
                                                                 cat_zeropoint_err=cat_zeropoint_err
                                                                 )
                if write and up is not None:
                    update_dict = {f_0 + '_zeropoint_derived': float(up['zeropoint_sub_outliers']),
                                   f_0 + '_zeropoint_derived_err': float(up['zeropoint_err'])}

                    p.add_output_values(obj=obj, params=update_dict, instrument=instrument)

            else:
                chip = photometry.determine_zeropoint_sextractor(sextractor_cat_path=sextractor_path,
                                                                 image=image_path,
                                                                 cat_path=cat_path,
                                                                 cat_name=cat_name,
                                                                 output_path=output_path + "/",
                                                                 show=show_plots,
                                                                 cat_ra_col=cat_ra_col,
                                                                 cat_dec_col=cat_dec_col,
                                                                 cat_mag_col=cat_mag_col,
                                                                 sex_ra_col=sex_ra_col,
                                                                 sex_dec_col=sex_dec_col,
                                                                 sex_x_col=sex_x_col,
                                                                 sex_y_col=sex_y_col,
                                                                 pix_tol=pix_tol,
                                                                 mag_tol=mag_tol,
                                                                 flux_column=sex_flux_col,
                                                                 mag_range_cat_upper=mag_range_upper,
                                                                 mag_range_cat_lower=mag_range_lower,
                                                                 mag_range_sex_upper=mag_range_sex_upper,
                                                                 mag_range_sex_lower=mag_range_sex_lower,
                                                                 stars_only=stars_only,
                                                                 star_class_tol=star_class_tol,
                                                                 star_class_col=star_class_col,
                                                                 exp_time=exp_time,
                                                                 y_lower=0,
                                                                 get_sextractor_names=get_sextractor_names,
                                                                 sextractor_names=sextractor_names,
                                                                 cat_type=cat_type,
                                                                 cat_zeropoint=cat_zeropoint,
                                                                 cat_zeropoint_err=cat_zeropoint_err
                                                                 )
                if write and chip is not None:
                    update_dict = {f_0 + '_zeropoint_derived': float(chip['zeropoint_sub_outliers']),
                                   f_0 + '_zeropoint_derived_err': float(chip['zeropoint_err'])}

                    p.add_output_values(obj=obj, params=update_dict, instrument=instrument)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Use the PSF-fitted magnitudes to extract a zeropoint.')

    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        type=str)
    parser.add_argument('--instrument',
                        help='Name of instrument.',
                        default='FORS2',
                        type=str)
    parser.add_argument('--test_name',
                        help='Name of test, and of the folder within "output" where the data will be written',
                        type=str,
                        default='')
    parser.add_argument('--sextractor_x_column',
                        help='Name of SExtractor column containing x pixel coordinate.',
                        default='x_psf',
                        type=str)
    parser.add_argument('--sextractor_y_column',
                        help='Name of SExtractor column containing y pixel coordinate.',
                        default='y_psf',
                        type=str)
    parser.add_argument('--sextractor_ra_column',
                        help='Name of SExtractor column containing RA coordinate.',
                        default='ra_psf',
                        type=str)
    parser.add_argument('--sextractor_dec_column',
                        help='Name of SExtractor column containing DEC coordinate.',
                        default='dec_psf',
                        type=str)
    parser.add_argument('--sextractor_flux_column',
                        help='Name of SExtractor column containing flux.',
                        default='flux_psf',
                        type=str)
    parser.add_argument('--get_sextractor_names',
                        help='Read column headers from SExtractor file.',
                        action='store_true')
    parser.add_argument('-not_stars_only',
                        help='Don\'t only use stars for zeropoint appraisal.',
                        action='store_false')
    parser.add_argument('--star_class_column',
                        help='Name of SExtractor column containing star classification parameter.',
                        default='class_star',
                        type=str)
    parser.add_argument('--star_class_tolerance',
                        help='Minimum value of star_class to be included in appraisal.',
                        default=0.95,
                        type=float)
    parser.add_argument('-show_plots',
                        help='Show plots onscreen? (Warning: may be tedious)',
                        action='store_true')
    parser.add_argument('--mag_tolerance',
                        help='Tolerance for magnitude scatter for removing outliers from linear fit.',
                        default=0.1,
                        type=float)
    parser.add_argument('--sextractor_mag_range_upper',
                        help='Upper SExtractor magnitude cutoff for zeropoint determination',
                        default=100.,
                        type=float)
    parser.add_argument('--sextractor_mag_range_lower',
                        help='Lower SExtractor magnitude cutoff for zeropoint determination',
                        default=-100.,
                        type=float)
    parser.add_argument('--pixel_tolerance',
                        help='Search tolerance for matching objects, in FORS2 (g) pixels.',
                        default=10.,
                        type=float)
    parser.add_argument('-no_separate_chips',
                        help='Do not calculate zeropoints separately for each chip.',
                        action='store_false')
    parser.add_argument('-write',
                        help='Write to relevant .yaml file.',
                        action='store_true')

    args = parser.parse_args()
    main(obj=args.op,
         test_name=args.test_name,
         sex_x_col=args.sextractor_x_column,
         sex_y_col=args.sextractor_y_column,
         sex_ra_col=args.sextractor_ra_column,
         sex_dec_col=args.sextractor_dec_column,
         sex_flux_col=args.sextractor_flux_column,
         get_sextractor_names=args.get_sextractor_names,
         mag_tol=args.mag_tolerance,
         stars_only=args.not_stars_only,
         star_class_col=args.star_class_column,
         star_class_tol=args.star_class_tolerance,
         show_plots=args.show_plots,
         mag_range_sex_lower=args.sextractor_mag_range_lower,
         mag_range_sex_upper=args.sextractor_mag_range_upper,
         pix_tol=args.pixel_tolerance,
         separate_chips=args.no_separate_chips,
         write=args.write,
         instrument=args.instrument
         )
