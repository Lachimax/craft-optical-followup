# Code by Lachlan Marnoch, 2019

from craftutils import photometry
from craftutils import params
from craftutils import fits_files as ff

import os
import matplotlib

matplotlib.rcParams.update({'errorbar.capsize': 3})


# TODO: Argparse, scriptify

def main(test_name,
         output_path,
         cat_path,
         image_path,
         sextractor_path,
         sex_x_col,
         sex_y_col,
         sex_ra_col,
         sex_dec_col,
         sex_flux_col,
         sextractor_names,
         get_sextractor_names,
         y_exclusion,
         cat_name,
         mag_tol,
         stars_only,
         star_class_col,
         star_class_tol,
         show_plots,
         mag_range_lower,
         mag_range_upper,
         mag_range_sex_lower,
         mag_range_sex_upper,
         pix_tol,
         cat_zeropoint,
         cat_zeropoint_err):

    if sextractor_names is not None:
        sextractor_names = params.load_params(f'param/{sextractor_names}')  # None to auto-detect
    else:
        get_sextractor_names = True

    print('SExtractor catalogue path:', sextractor_path)
    print('Image path:', image_path)
    print('Catalogue path:', cat_path)
    print('Output:', output_path + test_name)
    print()

    if cat_name == 'DES':
        cat_ra_col = 'RA'
        cat_dec_col = 'DEC'
        cat_mag_col = 'MAG_APER_4_G'
        # star_class_col = 'CLASS_STAR_G'
        cat_type = 'csv'

    elif cat_name == 'SDSS':
        cat_ra_col = 'ra'
        cat_dec_col = 'dec'
        cat_mag_col = 'fiberMag_g'
        # star_class_col = 'probPSF_g'
        cat_type = 'csv'

    else:  # elif cat_name == 'SExtractor'
        cat_ra_col = 'ra_cat'
        cat_dec_col = 'dec_cat'
        cat_mag_col = 'mag_aper_other'
        cat_type = 'sextractor'
        cat_name = 'other'
        sex_x_col = 'x_fors'
        sex_y_col = 'y_fors'
        star_class_col = 'class_star_fors'

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    exp_time = ff.get_exp_time(image_path)

    photometry.determine_zeropoint_sextractor(sextractor_cat_path=sextractor_path,
                                              image=image_path,
                                              cat_path=cat_path,
                                              cat_name=cat_name,
                                              output_path=output_path + test_name + "/",
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
                                              y_lower=y_exclusion,
                                              get_sextractor_names=get_sextractor_names,
                                              sextractor_names=sextractor_names,
                                              cat_type=cat_type,
                                              cat_zeropoint=cat_zeropoint,
                                              cat_zeropoint_err=cat_zeropoint_err
                                              )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Use the PSF-fitted magnitudes to extract a zeropoint.')

    # parser.add_argument('--op',
                        # help='Name of object parameter file without .yaml, eg FRB180924_1')
    parser.add_argument('--test_name',
                        help='Name of test, and of the folder within "output" where the data will be written')
    parser.add_argument('--output',
                        help='Path to test output folder',
                        default="/home/lachlan/Data/test/")
    parser.add_argument('--image_path',
                        help='The path of the FITS file on which SExtractor was operated.')
    parser.add_argument('--sextractor_path',
                        help='Path to the SExtractor output catalogue.')
    parser.add_argument('--sextractor_x_column',
                        help='Name of SExtractor column containing x pixel coordinate.',
                        default='x')
    parser.add_argument('--sextractor_y_column',
                        help='Name of SExtractor column containing y pixel coordinate.',
                        default='y')
    parser.add_argument('--sextractor_ra_column',
                        help='Name of SExtractor column containing RA coordinate.',
                        default='ra')
    parser.add_argument('--sextractor_dec_column',
                        help='Name of SExtractor column containing DEC coordinate.',
                        default='dec')
    parser.add_argument('--sextractor_flux_column',
                        help='Name of SExtractor column containing flux.',
                        default='flux_psf')
    parser.add_argument('--get_sextractor_names',
                        help='Read column headers from SExtractor file.',
                        default=False)
    parser.add_argument('--sextractor_names',
                        help='Name for the yaml file for the column headers from SExtractor.',
                        default='sextractor_names_psf')
    parser.add_argument('--catalogue_path',
                        help='Path to catalogue for comparison.')
    parser.add_argument('--y_exclusion',
                        help='Lower edge of the field to be considered; ie, exclude all objects below this',
                        default=740)
    parser.add_argument('--catalogue_name',
                        help='Name of catalogue type used, allowed values are SDSS, DES, SExtractor.',
                        default='DES')
    parser.add_argument('--stars_only',
                        help='Only use stars for zeropoint appraisal.',
                        default=True)
    parser.add_argument('--star_class_column',
                        help='Name of SExtractor column containing star classification parameter.',
                        default='class_star')
    parser.add_argument('--star_class_tolerance',
                        help='Minimum value of star_class to be included in appraisal.',
                        default=0.95)
    parser.add_argument('--show_plots',
                        help='Show plots onscreen? (Warning: may be tedious)',
                        default=True)
    parser.add_argument('--mag_tolerance',
                        help='Tolerance for magnitude scatter for removing outliers from linear fit.',
                        default=0.1)
    parser.add_argument('--mag_range_upper',
                        help='Upper catalogue magnitude cutoff for zeropoint determination',
                        default=25.0)
    parser.add_argument('--mag_range_lower',
                        help='Lower catalogue magnitude cutoff for zeropoint determination',
                        default=20.)
    parser.add_argument('--sextractor_mag_range_upper',
                        help='Upper SExtractor magnitude cutoff for zeropoint determination',
                        default=100.)
    parser.add_argument('--sextractor_mag_range_lower',
                        help='Lower SExtractor magnitude cutoff for zeropoint determination',
                        default=-100.)
    parser.add_argument('--pixel_tolerance',
                        help='Search tolerance for matching objects, in FORS2 (g) pixels.',
                        default=10.)
    parser.add_argument('--cat_zeropoint',
                        help='Zeropoint to add to the magnitudes found in the catalogue. Should normally be zero, if '
                             'the catalogue is calibrated.',
                        default=0.)
    parser.add_argument('--cat_zeropoint_err',
                        help='Error in the above.',
                        default=0.)

    args = parser.parse_args()
    main(test_name=args.test_name,
         output_path=args.output,
         cat_path=args.catalogue_path,
         image_path=args.image_path,
         sextractor_path=args.sextractor_path,
         sex_x_col=args.sextractor_x_column,
         sex_y_col=args.sextractor_y_column,
         sex_ra_col=args.sextractor_ra_column,
         sex_dec_col=args.sextractor_dec_column,
         sex_flux_col=args.sextractor_flux_column,
         sextractor_names=args.sextractor_names,
         get_sextractor_names=args.get_sextractor_names,
         y_exclusion=args.y_exclusion,
         cat_name=args.catalogue_name,
         mag_tol=args.mag_tolerance,
         stars_only=args.stars_only,
         star_class_col=args.star_class_column,
         star_class_tol=args.star_class_tolerance,
         show_plots=args.show_plots,
         mag_range_lower=args.mag_range_lower,
         mag_range_upper=args.mag_range_upper,
         mag_range_sex_lower=args.sextractor_mag_range_lower,
         mag_range_sex_upper=args.sextractor_mag_range_upper,
         pix_tol=args.pixel_tolerance,
         cat_zeropoint=args.cat_zeropoint,
         cat_zeropoint_err=args.cat_zeropoint_err
         )
