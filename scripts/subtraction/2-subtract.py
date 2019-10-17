# Code by Lachlan Marnoch, 2019
from PyCRAFT import fits_files as ff
from PyCRAFT import params as p
from PyCRAFT import utils as u
from PyCRAFT import photometry as ph

import shutil
import os
import math
from astropy.io import fits


# TODO: Document!!!!!!!!!

def main(field, destination, epoch, force_subtract_better_seeing, subtraction_type, template_instrument):
    if destination[-1] != '/':
        destination = destination + '/'

    params = p.object_params_frb(field)
    destination_path = f'{params["data_dir"]}subtraction/{destination}/'
    u.mkdir_check(destination_path)

    params = p.object_params_frb(field)
    template_epoch = params['template_epoch_' + template_instrument.lower()]

    comparison_title = f'{field}_{epoch}'
    template_title = f'{field}_{template_epoch}'

    filters = params['filters']

    for f in filters:
        f_0 = f[0]

        destination_path_filter = f'{destination_path}{f}/'
        u.mkdir_check(destination_path_filter)

        template_params = p.load_params(destination_path_filter + f'{template_title}_template_output_values.yaml')
        comparison_params = p.load_params(destination_path_filter + f'{comparison_title}_comparison_output_values.yaml')

        if f_0 + '_fwhm_arcsec' in template_params:
            fwhm_template = template_params[f_0 + '_fwhm_arcsec']
        elif f_0.lower() + '_fwhm_arcsec' in template_params:
            fwhm_template = template_params[f_0.lower() + '_fwhm_arcsec']
        else:
            raise ValueError(f_0 + '_fwhm_arcsec or ' + f_0.lower() + '_fwhm_arcsec not found for template image.')

        if f_0 + '_fwhm_arcsec' in comparison_params:
            fwhm_comparison = comparison_params[f_0 + '_fwhm_arcsec']
        elif f_0.lower() + '_fwhm_arcsec' in comparison_params:
            fwhm_comparison = comparison_params[f_0.lower() + '_fwhm_arcsec']
        else:
            raise ValueError(f_0 + '_fwhm_arcsec or ' + f_0.lower() + '_fwhm_arcsec not found for comparison image.')

        template = destination_path_filter + f'{template_title}_template_aligned.fits'
        comparison = destination_path_filter + f'{comparison_title}_comparison_aligned.fits'

        _, difference = ph.subtract(template_origin=template,
                                    comparison_origin=comparison,
                                    output=destination_path_filter,
                                    template_fwhm=fwhm_template,
                                    comparison_fwhm=fwhm_comparison,
                                    force_subtract_better_seeing=force_subtract_better_seeing,
                                    comparison_title=comparison_title,
                                    template_title=template_title,
                                    field=field, comparison_epoch=epoch, template_epoch=template_epoch)


        # p.add_output_path(obj=comparison_title, key=f_0 + '_pre_subtraction_image_' + subtraction_type,
        #                   path=destination_path_filter + f'{comparison_title}_comparison_tweaked.fits')
        # p.add_output_path(obj=comparison_title, key=f_0 + '_pre_subtraction_template_' + subtraction_type,
        #                   path=destination_path_filter + f'{template_title}_template.fits')
        # p.add_output_path(obj=comparison_title, key=f_0 + '_post_subtraction_image_' + subtraction_type,
        #                   path=difference)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--field',
                        help='Name of object parameter file without .yaml, eg FRB190102.',
                        type=str)
    parser.add_argument('--destination',
                        help='Folder to copy to and operate on.',
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
    parser.add_argument('--type',
                        help='Specify synthetic insertion file to use. If unspecified, uses the unmodified file.',
                        type=str,
                        default='normal')
    parser.add_argument('-force_subtract_better_seeing',
                        help='Force the better-seeing file to be the one subtracted.',
                        action='store_true')

    args = parser.parse_args()
    main(field=args.field,
         destination=args.destination,
         epoch=args.epoch,
         force_subtract_better_seeing=args.force_subtract_better_seeing,
         subtraction_type=args.type,
         template_instrument=args.instrument_template)
