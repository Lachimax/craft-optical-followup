# Code by Lachlan Marnoch, 2019
from craftutils import fits_files as ff
from craftutils import params as p
from craftutils import utils as u
from craftutils import photometry as ph

import shutil
import os
import math
from astropy.io import fits


# TODO: Document!!!!!!!!!


def main(field, destination, epoch, instrument, template_instrument, comparison_type):
    if destination[-1] != '/':
        destination = destination + '/'


    # p.refresh_params_frbs()
    # types = ['multi_frb_range', 'multi_sn_models', 'multi_sn_random', 'multi_sn_random_ia', 'multi_sn_random_ib']
    # if comparison_type not in types:
    #     raise ValueError(comparison_type + ' is not a valid multi-synthetic argument; choose from ' + str(types))
    comparison_type = '_' + comparison_type
    type_suffix = comparison_type[7:]

    params = p.object_params_frb(field)
    u.mkdir_check(f'{params["data_dir"]}subtraction/')
    destination_path = f'{params["data_dir"]}subtraction/{destination}/'
    u.mkdir_check(destination_path)

    comparison_title = f'{field}_{epoch}'

    comparison_paths = p.object_output_paths(obj=comparison_title, instrument=instrument)

    comparison_origin_top = comparison_paths['subtraction_image_synth_' + type_suffix]
    comparison_tests = os.listdir(comparison_origin_top)

    comparison_params = p.object_params_instrument(obj=comparison_title, instrument=instrument)

    params = p.object_params_frb(field)
    template_epoch = params['template_epoch_' + template_instrument.lower()]

    template_title = f'{field}_{template_epoch}'
    template_paths = p.object_output_paths(obj=template_title, instrument=template_instrument)
    template_outputs = p.object_output_params(obj=template_title, instrument=template_instrument)
    template_params = p.object_params_instrument(obj=template_title, instrument=template_instrument)

    filters = params['filters']

    for test in comparison_tests:

        destination_test = destination_path + test + '/'
        origin_test = comparison_origin_top + test + '/'
        u.mkdir_check(destination_test)

        test_files = os.listdir(origin_test)

        for f in filters:

            for file in filter(lambda fil: fil[:2] != f + '_', os.listdir(origin_test)):
                shutil.copyfile(origin_test + file, destination_test + file)

            values = {}

            f_0 = f[0]
            destination_path_filter = f'{destination_test}{f}/'
            u.mkdir_check(destination_path_filter)

            # COMPARISON IMAGE:

            # Get path to comparison image from parameter .yaml file

            comparison_origin = origin_test + filter(lambda file: file[0] == f_0 and file[-5:] == '.fits',
                                                     test_files).__next__()

            comparison_destination = f'{comparison_title}_comparison.fits'

            shutil.copyfile(comparison_origin, destination_path_filter + comparison_destination)
            shutil.copyfile(comparison_origin.replace('.fits', '.csv'),
                            destination_path_filter + comparison_destination.replace('.fits', '.csv'))

            print('Copying comparison image from:')
            print('\t', comparison_origin)

            print('To:')
            print(f'\t {destination_test}{f}/{comparison_destination}')
            shutil.copy(comparison_params['data_dir'] + 'output_values.yaml',
                        f'{destination_test}{f}/{comparison_title}_comparison_output_values.yaml')
            shutil.copy(comparison_params['data_dir'] + 'output_values.json',
                        f'{destination_test}{f}/{comparison_title}_comparison_output_values.json')
            shutil.copy(comparison_origin, f'{destination_test}{f}/{comparison_destination}')
            values['comparison_file'] = comparison_origin

            if template_instrument != 'FORS2':

                # TEMPLATE IMAGE

                if template_instrument != 'XSHOOTER':
                    f_0 = f_0.lower()

                template_image_name = template_params['subtraction_image']
                if f'{f_0}_{template_image_name}' in template_paths:
                    template_origin = template_paths[f'{f_0}_{template_image_name}']
                elif f'{f_0.lower()}_{template_image_name}' in template_paths:
                    template_origin = template_paths[f'{f_0.lower()}_{template_image_name}']
                else:
                    raise ValueError(f'{f_0.lower()}_{template_image_name} not found in {template_title} paths')
                fwhm_template = template_outputs[f_0 + '_fwhm_pix']
                template_destination = f'{template_title}_template.fits'

                print('Copying template from:')
                print('\t', template_origin)
                print('To:')
                print(f'\t {destination_test}{f}/{template_destination}')
                shutil.copy(template_params['data_dir'] + 'output_values.yaml',
                            f'{destination_test}{f}/{template_title}_template_output_values.yaml')
                shutil.copy(template_params['data_dir'] + 'output_values.json',
                            f'{destination_test}{f}/{template_title}_template_output_values.json')
                shutil.copy(template_origin, f'{destination_test}{f}/{template_destination}')
                values['template_file'] = comparison_origin

                p.add_params(f'{destination_test}{f}/output_values.yaml', values)


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
