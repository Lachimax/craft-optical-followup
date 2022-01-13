# Code by Lachlan Marnoch, 2019

from craftutils import photometry
from craftutils import params as p
from craftutils import fits_files as ff
from craftutils.utils import mkdir_check

import os
import matplotlib
import astropy.time as time

matplotlib.rcParams.update({'errorbar.capsize': 3})


# TODO: Argparse, scriptify

def main(obj,
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
    print(separate_chips)

    sextractor_names = p.sextractor_names_psf()  # None to auto-detect

    properties = p.object_params_des(obj=obj)
    outputs = p.object_output_params(obj=obj, instrument='des')
    paths = p.object_output_paths(obj=obj)

    cat_name = 'DES'
    cat_path = properties['data_dir'] + 'des_objects.csv'
    output = properties['data_dir'] + '/3-zeropoint/'
    mkdir_check(output)
    output = output + 'field/'
    mkdir_check(output)

    for fil in properties['filters']:

        f_0 = fil[0]

        f_up = f_0.upper()

        output_path = output + f_0

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

        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        output_path = output_path + test_name + '/'

        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        image_path = properties['data_dir'] + '2-sextractor/' + f_0 + '_cutout.fits'
        sextractor_path = properties['data_dir'] + '2-sextractor/' + f_0 + '_psf-fit.cat'
        # We override here because there's no need to separate the chips if we're using the DES image.
        separate_chips = False
        # pix_tol = 1.
        mag_range_lower = 16
        mag_range_upper = 25

        exp_time = 1.

        print('SExtractor catalogue path:', sextractor_path)
        print('Image path:', image_path)
        print('Catalogue path:', cat_path)
        print('Output:', output_path + test_name)
        print()

        print(cat_zeropoint)

        up = photometry.determine_zeropoint_sextractor(sextractor_cat=sextractor_path,
                                                       image=image_path,
                                                       image_name='DES',
                                                       cat_path=cat_path,
                                                       cat_name=cat_name,
                                                       output_path=output_path + "/chip_1/",
                                                       show=show_plots,
                                                       cat_ra_col='RA',
                                                       cat_dec_col='DEC',
                                                       cat_mag_col='WAVG_MAG_PSF_' + f_up,
                                                       sex_ra_col=sex_ra_col,
                                                       sex_dec_col=sex_dec_col,
                                                       sex_x_col=sex_x_col,
                                                       sex_y_col=sex_y_col,
                                                       dist_tol=pix_tol,
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
                                                       get_sextractor_names=get_sextractor_names,
                                                       sextractor_names=sextractor_names,
                                                       cat_type='csv',
                                                       cat_zeropoint=cat_zeropoint,
                                                       cat_zeropoint_err=cat_zeropoint_err,
                                                       )

        if write:
            update_dict = {f_0 + '_zeropoint_derived': float(up['zeropoint_sub_outliers']),
                           f_0 + '_zeropoint_derived_err': float(up['zeropoint_err'])}

            p.add_output_values(obj=obj, params=update_dict, instrument='des')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Use the PSF-fitted magnitudes to extract a zeropoint.')

    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
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
    parser.add_argument('--stars_only',
                        help='Only use stars for zeropoint appraisal.',
                        default=True,
                        type=bool)
    parser.add_argument('--star_class_column',
                        help='Name of SExtractor column containing star classification parameter.',
                        default='class_star',
                        type=str)
    parser.add_argument('--star_class_tolerance',
                        help='Minimum value of star_class to be included in appraisal.',
                        default=0.98,
                        type=float)
    parser.add_argument('-no_show_plots',
                        help='Show plots onscreen? (Warning: may be tedious)',
                        action='store_false')
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
                        help='Search tolerance for matching objects, in DES pixels.',
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
         stars_only=args.stars_only,
         star_class_col=args.star_class_column,
         star_class_tol=args.star_class_tolerance,
         show_plots=args.no_show_plots,
         mag_range_sex_lower=args.sextractor_mag_range_lower,
         mag_range_sex_upper=args.sextractor_mag_range_upper,
         pix_tol=args.pixel_tolerance,
         separate_chips=args.no_separate_chips,
         write=args.write,
         )
