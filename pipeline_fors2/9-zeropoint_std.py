# Code by Lachlan Marnoch, 2019

from craftutils import photometry
from craftutils import params as p
from craftutils import fits_files as ff
from craftutils.utils import mkdir_check
from craftutils.retrieve import update_std_photometry, cat_columns

import os
import matplotlib
import astropy.time as time
import numpy as np

matplotlib.rcParams.update({'errorbar.capsize': 3})


def main(obj,
         test_name,
         sex_x_col,
         sex_y_col,
         sex_ra_col,
         sex_dec_col,
         sex_flux_col,
         star_class_col,
         stars_only,
         show_plots,
         mag_range_sex_lower,
         mag_range_sex_upper,
         pix_tol):
    print("\nExecuting Python script pipeline_fors2/9-zeropoint_std.py, with:")
    print(f"\tepoch {obj}")
    print()

    properties = p.object_params_fors2(obj)
    outputs = p.object_output_params(obj=obj, instrument='fors2')

    proj_paths = p.config

    output = properties['data_dir'] + '9-zeropoint/'
    mkdir_check(output)
    output_std = output + 'std/'
    mkdir_check(output_std)
    output_std = output + 'field/'
    mkdir_check(output_std)

    std_path = properties['data_dir'] + 'calibration/std_star/'

    filters = outputs['filters']

    std_field_path = proj_paths['top_data_dir'] + "std_fields/"
    std_fields = list(filter(lambda path: os.path.isdir(path), os.listdir(std_field_path)))
    print('Standard fields with available catalogues:')
    print(std_fields)
    print()

    for fil in filters:
        print(fil)

        zeropoints = {}
        field_names = []
        zeropoints_err = []
        airmasses = []

        f = fil[0]

        fil_path = std_path + fil + '/'

        fields = filter(lambda d: os.path.isdir(fil_path + d), os.listdir(fil_path))

        output_path_fil_std = output_std + '/' + fil + '/'
        mkdir_check(output_path_fil_std)

        # Obtain zeropoints from FRB field, if data is available.

        # Obtain zeropoints from available standard fields and available data.

        zeropoints["standard_field"] = {}

        for field in fields:

            zeropoints["standard_field"][field] = {}

            ra = float(field[field.find("RA") + 2:field.find("_")])
            dec = float(field[field.find("DEC") + 3:])

            print("Looking for photometry data in field " + field + ":")
            mkdir_check(std_field_path + field)
            field_path = fil_path + field + '/'
            output_path = output_path_fil_std + field + '/'

            std_cat_path = std_field_path + field + '/'

            std_properties = p.load_params(field_path + 'params.yaml')
            use_sex_star_class = std_properties['use_sex_star_class']

            # star_class_col = 'class_star'

            # Cycle through the three catalogues used to determine zeropoint, in order of preference.
            cat_names = ['DES', 'SDSS', 'SkyMapper']

            for cat_i, cat_name in cat_names:

                # Check for photometry on-disk in the relevant catalogue; if none present, attempt to retrieve from
                # online archive.
                print(f"In {cat_name}:")

                cat_path = f"{std_cat_path}{cat_name}/{cat_name}.csv"

                if not (os.path.isdir(std_cat_path + cat_name)) or os.path.isfile(cat_path):
                    print("None found on disk. Attempting retrieval from archive...")
                    if update_std_photometry(ra=ra, dec=dec, cat=cat_name) is None:
                        print("\t\tNo data found in archive.")
                        zeropoints["standard_field"][field][cat_name] = None
                        continue

                column_names = cat_columns(cat=cat_name, f=f)
                cat_ra_col = column_names['ra']
                cat_dec_col = column_names['dec']
                cat_mag_col = column_names['mag_psf']
                if not use_sex_star_class:
                    star_class_col = column_names['class_star']
                else:
                    star_class_col = 'class_star'

                cat_type = 'csv'

                sextractor_path = field_path + 'sextractor/_psf-fit.cat'
                image_path = field_path + '3-trimmed/standard_trimmed_img_up.fits'
                star_class_tol = std_properties['star_class_tol']

                now = time.Time.now()
                now.format = 'isot'
                test_name = str(now) + '_' + test_name

                mkdir_check(properties['data_dir'] + '/analysis/zeropoint/')
                mkdir_check(output_path)

                exp_time = ff.get_exp_time(image_path)

                print('SExtractor catalogue path:', sextractor_path)
                print('Image path:', image_path)
                print('Catalogue name:', cat_name)
                print('Catalogue path:', cat_path)
                print('Class star column:', star_class_col)
                print('Output:', output_path + test_name)
                print('Exposure time:', exp_time)
                print("Use sextractor class star:", use_sex_star_class)

                params = photometry.determine_zeropoint_sextractor(
                    sextractor_cat_path=sextractor_path,
                    image=image_path,
                    cat_path=cat_path,
                    cat_name=cat_name,
                    output_path=output_path,
                    show=show_plots,
                    cat_ra_col=cat_ra_col,
                    cat_dec_col=cat_dec_col,
                    cat_mag_col=cat_mag_col,
                    sex_ra_col=sex_ra_col,
                    sex_dec_col=sex_dec_col,
                    sex_x_col=sex_x_col,
                    sex_y_col=sex_y_col,
                    pix_tol=pix_tol,
                    flux_column=sex_flux_col,
                    mag_range_sex_upper=mag_range_sex_upper,
                    mag_range_sex_lower=mag_range_sex_lower,
                    stars_only=stars_only,
                    star_class_tol=star_class_tol,
                    star_class_col=star_class_col,
                    exp_time=exp_time,
                    y_lower=0,
                    cat_type=cat_type,
                )

                zeropoints["standard_field"][field][cat_name] = params

            update_dict = {'zeropoints': zeropoints["standard_field"][field]}

            p.add_params(file=field_path + 'output_values.yaml', params=update_dict)

        if len(zeropoints) == 0:
            print('No standard-field zeropoint could be determined for this observation.')
        else:
            best_arg = int(np.argmin(zeropoints_err))
            output_dict = {f + '_zeropoint_std': zeropoints[best_arg],
                           f + '_zeropoint_std_err': zeropoints_err[best_arg],
                           f + '_zeropoint_std_cat': cat_names[best_arg],
                           f + '_zeropoint_std_field': field_names[best_arg],
                           f + '_airmass_std': airmasses[best_arg]}
            p.add_output_values(obj=obj, instrument='FORS2', params=output_dict)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Use the PSF-fitted magnitudes to extract a zeropoint.')

    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        default='FRB181112_1',
                        type=str)
    parser.add_argument('--test_name',
                        help='Name of test, and of the folder within "output" where the data will be written',
                        default='test',
                        type=str)
    parser.add_argument('--sextractor_x_column',
                        help='Name of SExtractor column containing x pixel coordinate.',
                        default='XPSF_IMAGE',
                        type=str)
    parser.add_argument('--sextractor_y_column',
                        help='Name of SExtractor column containing y pixel coordinate.',
                        default='YPSF_IMAGE',
                        type=str)
    parser.add_argument('--sextractor_ra_column',
                        help='Name of SExtractor column containing RA coordinate.',
                        default='ALPHAPSF_SKY',
                        type=str)
    parser.add_argument('--sextractor_dec_column',
                        help='Name of SExtractor column containing DEC coordinate.',
                        default='DELTAPSF_SKY',
                        type=str)
    parser.add_argument('--sextractor_flux_column',
                        help='Name of SExtractor column containing flux.',
                        default='FLUX_PSF',
                        type=str)
    parser.add_argument('--catalogue_path',
                        help='Path to catalogue for comparison.',
                        type=str)
    parser.add_argument('-not_stars_only',
                        help='Only use stars for zeropoint appraisal.',
                        action='store_false')
    parser.add_argument('-show',
                        help='Show plots onscreen? (Warning: may be tedious)',
                        action='store_true', )
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
    parser.add_argument('--star_class_column',
                        help='Name of SExtractor column containing star classification parameter.',
                        default='CLASS_STAR',
                        type=str)

    args = parser.parse_args()
    main(obj=args.op,
         test_name=args.test_name,
         sex_x_col=args.sextractor_x_column,
         sex_y_col=args.sextractor_y_column,
         sex_ra_col=args.sextractor_ra_column,
         sex_dec_col=args.sextractor_dec_column,
         sex_flux_col=args.sextractor_flux_column,
         star_class_col=args.star_class_column,
         stars_only=args.not_stars_only,
         show_plots=args.show,
         mag_range_sex_lower=args.sextractor_mag_range_lower,
         mag_range_sex_upper=args.sextractor_mag_range_upper,
         pix_tol=args.pixel_tolerance,
         )
