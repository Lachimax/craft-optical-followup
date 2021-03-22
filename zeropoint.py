# Code by Lachlan Marnoch, 2019-2021

from craftutils import photometry
from craftutils import params as p
from craftutils import fits_files as ff
from craftutils.utils import mkdir_check, mkdir_check_nested
from craftutils.retrieve import update_frb_sdss_photometry, update_frb_des_photometry

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
         stars_only,
         star_class_col,
         star_class_tol,
         show_plots,
         mag_range_sex_lower,
         mag_range_sex_upper,
         pix_tol,
         separate_chips,
         write):
    print()
    print(f"Running script zeropoint.py, with obj {obj}")
    print(f"\ttest_name {test_name}")
    print()

    photometry.zeropoint_science_field(epoch=obj,
                                       instrument=instrument,
                                       test_name=test_name,
                                       sex_x_col=sex_x_col,
                                       sex_y_col=sex_y_col,
                                       sex_ra_col=sex_ra_col,
                                       sex_dec_col=sex_dec_col,
                                       sex_flux_col=sex_flux_col,
                                       stars_only=stars_only,
                                       star_class_col=star_class_col,
                                       star_class_tol=star_class_tol,
                                       show_plots=show_plots,
                                       mag_range_sex_lower=mag_range_sex_lower,
                                       mag_range_sex_upper=mag_range_sex_upper,
                                       pix_tol=pix_tol,
                                       separate_chips=separate_chips,
                                       write=write,
                                       )


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
    parser.add_argument('-not_stars_only',
                        help='Don\'t only use stars for zeropoint appraisal.',
                        action='store_false')
    parser.add_argument('--star_class_column',
                        help='Name of SExtractor column containing star classification parameter.',
                        default='CLASS_STAR',
                        type=str)
    parser.add_argument('--star_class_tolerance',
                        help='Minimum value of star_class to be included in appraisal.',
                        default=0.95,
                        type=float)
    parser.add_argument('-show_plots',
                        help='Show plots onscreen? (Warning: may be tedious)',
                        action='store_true')
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
