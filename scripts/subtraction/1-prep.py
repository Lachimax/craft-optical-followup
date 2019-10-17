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


def main(field, destination, epoch, instrument, template_instrument, comparison_type):
    if destination[-1] != '/':
        destination = destination + '/'

    p.refresh_params_frbs()
    types = ['normal', 'synth_random', 'synth_frb']
    if comparison_type not in types:
        raise ValueError(comparison_type + ' is not a valid synthetic argument; choose from ' + str(types))
    if comparison_type == 'normal':
        comparison_type = ''
    else:
        comparison_type = '_' + comparison_type

    params = p.object_params_frb(field)
    u.mkdir_check(f'{params["data_dir"]}subtraction/')
    destination_path = f'{params["data_dir"]}subtraction/{destination}/'
    u.mkdir_check(destination_path)

    comparison_title = f'{field}_{epoch}'

    comparison_paths = p.object_output_paths(obj=comparison_title, instrument=instrument)
    comparison_params = p.object_params_instrument(obj=comparison_title, instrument=instrument)

    params = p.object_params_frb(field)
    template_epoch = params['template_epoch_' + template_instrument.lower()]

    template_title = f'{field}_{template_epoch}'
    template_paths = p.object_output_paths(obj=template_title, instrument=template_instrument)
    template_outputs = p.object_output_params(obj=template_title, instrument=template_instrument)
    template_params = p.object_params_instrument(obj=template_title, instrument=template_instrument)

    filters = params['filters']
    for f in filters:

        values = {}

        f_0 = f[0]
        destination_path_filter = f'{destination_path}{f}/'
        u.mkdir_check(destination_path_filter)

        # COMPARISON IMAGE:

        comparison_image_name = comparison_params['subtraction_image'] + comparison_type

        # Get path to comparison image from parameter .yaml file
        if f'{f_0}_{comparison_image_name}' in comparison_paths:
            comparison_origin = comparison_paths[f'{f_0}_{comparison_image_name}']
        elif f'{f_0.lower()}_{comparison_image_name}' in comparison_paths:
            comparison_origin = comparison_paths[f'{f_0.lower()}_{comparison_image_name}']
        else:
            raise ValueError(f'{f_0.lower()}_{comparison_image_name} not found in {comparison_title} paths')

        comparison_destination = f'{comparison_title}_comparison.fits'

        if comparison_type != '':
            shutil.copyfile(comparison_origin.replace('.fits', '.csv'),
                            destination_path_filter + comparison_destination.replace('.fits', '.csv'))

        print('Copying comparison image')
        print('From:')
        print('\t', comparison_origin)

        print('To:')
        print(f'\t {destination_path}{f}/{comparison_destination}')
        shutil.copy(comparison_params['data_dir'] + 'output_values.yaml',
                    f'{destination_path}{f}/{comparison_title}_comparison_output_values.yaml')
        shutil.copy(comparison_params['data_dir'] + 'output_values.json',
                    f'{destination_path}{f}/{comparison_title}_comparison_output_values.json')
        shutil.copy(comparison_origin, f'{destination_path}{f}/{comparison_destination}')
        values['comparison_file'] = comparison_origin

        # TEMPLATE IMAGE

        if template_instrument != 'FORS2' and template_instrument != 'XSHOOTER':
            f_0 = f_0.lower()

        template_image_name = template_params['subtraction_image'] + comparison_type
        if f'{f_0}_{template_image_name}' in template_paths:
            template_origin = template_paths[f'{f_0}_{template_image_name}']
        elif f'{f_0.lower()}_{template_image_name}' in template_paths:
            template_origin = template_paths[f'{f_0.lower()}_{template_image_name}']
        else:
            raise ValueError(f'{f_0.lower()}_{template_image_name} not found in {template_title} paths')
        fwhm_template = template_outputs[f_0 + '_fwhm_pix']
        template_destination = f'{template_title}_template.fits'

        print('Copying template')
        print('From:')
        print('\t', template_origin)
        print('To:')
        print(f'\t {destination_path}{f}/{template_destination}')
        shutil.copy(template_params['data_dir'] + 'output_values.yaml',
                    f'{destination_path}{f}/{template_title}_template_output_values.yaml')
        shutil.copy(template_params['data_dir'] + 'output_values.json',
                    f'{destination_path}{f}/{template_title}_template_output_values.json')
        shutil.copy(template_origin, f'{destination_path}{f}/{template_destination}')
        values['template_file'] = comparison_origin

        p.add_params(f'{destination_path}{f}/output_values.yaml', values)


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
                        default=None)

    args = parser.parse_args()
    main(field=args.field,
         destination=args.destination,
         epoch=args.epoch,
         instrument=args.instrument,
         template_instrument=args.instrument_template,
         comparison_type=args.type)
