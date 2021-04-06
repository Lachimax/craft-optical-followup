# Code by Lachlan Marnoch, 2019-2021

import ruamel.yaml as yaml
import csv
import json
from typing import Union
import os
import numpy as np
from datetime import date

import astropy.table as tbl

from craftutils import utils as u


def tabulate_output_values(path: str, output: str = None):
    path = u.check_trailing_slash(path=path)
    outputs = []
    for file in filter(lambda filename: 'output_values.yaml' in filename, os.listdir(path)):
        output_values = load_params(file=path + file)
        output_values["filename"] = file
        outputs.append(output_values)

    outputs = tbl.Table(outputs)

    if output is not None:
        output = u.sanitise_file_ext(filename=output, ext='.csv')
        outputs.write(output)
        outputs.sort(keys="filename")

    return outputs


def check_for_config():
    p = load_params('param/config.yaml')
    if p is None:
        print("No config.yaml file found.")
    else:
        for param in p:
            p[param] = u.check_trailing_slash(p[param])
        save_params('param/config.yaml', p)
        yaml_to_json('param/config.yaml')
    return p


def load_params(file: str, quiet: bool = False):
    file = u.sanitise_file_ext(file, '.yaml')

    if not quiet:
        print('Loading parameter file from ' + str(file))

    if os.path.isfile(file):
        with open(file) as f:
            p = yaml.safe_load(f)
    else:
        p = None
        if not quiet:
            print('No parameter file found at', str(file) + ', returning None.')

    return p


def save_params(file: str, dictionary: dict, quiet: bool = False):
    file = u.sanitise_file_ext(filename=file, ext=".yaml")

    if not quiet:
        print('Saving parameter file to ' + str(file))

    with open(file, 'w') as f:
        yaml.dump(dictionary, f)


def yaml_to_json(yaml_file: str, output: str = None, quiet: bool = False):
    yaml_file = u.sanitise_file_ext(yaml_file, '.yaml')
    if output is not None:
        output = u.sanitise_file_ext(output, '.json')
    elif output is None:
        output = yaml_file.replace('.yaml', '.json')

    p = load_params(file=yaml_file, quiet=quiet)

    if not quiet:
        print('Saving parameter file to ' + output)

    for param in p:
        if type(p[param]) is date:
            p[param] = str(p[param])

    with open(output, 'w') as fj:
        json.dump(p, fj)

    return p


config = check_for_config()
param_path = u.check_trailing_slash(config['param_dir'])


def path_or_params_obj(obj: Union[dict, str], instrument: str = 'FORS2', quiet: bool = False):
    if type(obj) is str:
        return obj, object_params_instrument(obj, instrument=instrument, quiet=quiet)
    elif type(obj) is dict:
        params = obj
        obj = params['data_title']  # TODO: This is broken since you removed data_title from epoch params.
        return obj, params


def change_yaml_param(file: str = 'project', param: str = None, value=None, update_json=False, quiet: bool = False):
    if not quiet:
        print(f'Setting {param} in file {file} to {value}.')
    if file[-5:] != '.yaml':
        file = file + '.yaml'
    with open(file) as f:
        p = yaml.safe_load(f)
        if param is not None:
            p[param] = value
    with open(file, 'w') as f:
        yaml.dump(p, f)

    if update_json:
        yaml_to_json(file)
        with open(file.replace('.yaml', '.json'), 'w'):
            json.dump(p, f)

    return p


def add_params(file: str, params: dict, quiet: bool = False):
    file = u.sanitise_file_ext(file, '.yaml')
    if os.path.isfile(file):
        param_dict = load_params(file)
    else:
        param_dict = {}
    param_dict.update(params)
    save_params(file, param_dict, quiet=quiet)
    yaml_to_json(file, quiet=quiet)


def add_config_param(params: dict, quiet=False):
    add_params(file="param/config.yaml", params=params, quiet=quiet)
    params.config = check_for_config()


def add_frb_param(obj: str, params: dict, quiet=False):
    add_params(file=param_path + "FRBs/" + obj + ".yaml", params=params, quiet=quiet)


def add_epoch_param(obj: str, params: dict, instrument: str = 'FORS2', quiet=False):
    add_params(file=param_path + "epochs_" + instrument.lower() + "/" + obj + ".yaml", params=params, quiet=quiet)


def add_output_path(obj: str, key: str, path: str, instrument='fors2', quiet: bool = False):
    instrument = instrument.lower()
    key = u.remove_trailing_slash(key)
    if not quiet:
        print(f"Writing new path for {key}: {path}")
    p = object_params_instrument(obj=obj, instrument=instrument, quiet=quiet)
    add_params(file=p['data_dir'] + 'output_paths', params={key: path}, quiet=quiet)


def add_output_value(obj: str, key: str, value: str, instrument='fors2', quiet: bool = False):
    instrument = instrument.lower()
    p = object_params_instrument(obj=obj, instrument=instrument, quiet=quiet)
    add_params(file=p['data_dir'] + 'output_values', params={key: value}, quiet=quiet)


def add_output_value_frb(obj: str, key: str, value: str, quiet: bool = False):
    p = object_params_frb(obj=obj, quiet=quiet)
    add_params(file=p['data_dir'] + 'output_values', params={key: value}, quiet=quiet)


def add_output_values(obj: str, params: dict, instrument='fors2', quiet: bool = False):
    instrument = instrument.lower()
    p = object_params_instrument(obj=obj, instrument=instrument, quiet=quiet)
    add_params(file=p['data_dir'] + 'output_values', params=params, quiet=quiet)


def add_output_values_frb(obj: str, params: dict, quiet: bool = False):
    p = object_params_frb(obj=obj, quiet=quiet)
    add_params(file=p['data_dir'] + 'output_values', params=params, quiet=quiet)


def apertures_fors(quiet: bool = False):
    return load_params(param_path + '/aperture_diameters_fors2', quiet=quiet)


def apertures_des(quiet: bool = False):
    return load_params(param_path + 'aperture_diameters_des', quiet=quiet)


def sextractor_names(quiet: bool = False):
    return load_params(param_path + 'sextractor_names', quiet=quiet)


def sextractor_names_psf(quiet: bool = False):
    return load_params(param_path + 'sextractor_names_psf', quiet=quiet)


def sncosmo_models(quiet: bool = False):
    return load_params(param_path + 'sncosmo_models', quiet=quiet)


def plotting_params(quiet: bool = False):
    return load_params(param_path + 'plotting', quiet=quiet)


def ingest_filter_properties(path: str, instrument: str, update: bool = False, quiet: bool = False):
    """
    Imports a dataset from http://archive.eso.org/bin/qc1_cgi?action=qc1_browse_table&table=fors2_photometry into a
    filter properties .yaml file within this project.
    :param path:
    :param instrument:
    :return:
    """
    data = tbl.Table.read(path, format='ascii')
    name = data['filter_name'][0]
    if sum(data['filter_name'] != name) > 0:
        raise ValueError('This file contains data for more than one filter.')
    if name == 'R_SPEC':
        name = 'R_SPECIAL'

    params = filter_params(f=name, instrument=instrument, quiet=quiet)
    if params is None:
        params = new_filter_params(quiet=quiet)
        params['name'] = name
        params['instrument'] = instrument

    params['mjd'] = u.numpy_to_list(data['mjd_obs'])
    params['date'] = u.numpy_to_list(data['civil_date'])
    params['zeropoint'] = u.numpy_to_list(data['zeropoint'])
    params['zeropoint_err'] = u.numpy_to_list(data['zeropoint_err'])
    params['colour_term'] = u.numpy_to_list(data['colour_term'])
    params['colour_term_err'] = u.numpy_to_list(data['colour_term_err'])
    params['extinction'] = u.numpy_to_list(data['extinction'])
    params['extinction_err'] = u.numpy_to_list(data['extinction_err'])
    if update:
        params['calib_last_updated'] = str(date.today())
    save_params(file=param_path + f'filters/{instrument}-{name}', dictionary=params)


def ingest_filter_transmission(path: str, f: str, instrument: str, filter_only: bool = True, lambda_eff: float = None,
                               fwhm: float = None, source: float = None, unit: str = 'nm', percentage: bool = False,
                               quiet: bool = False):
    units = ['nm', 'Angstrom']
    if unit not in units:
        raise ValueError('Units must be one of ', units)

    params = filter_params(f=f, instrument=instrument, quiet=quiet)
    if params is None:
        params = new_filter_params(quiet=quiet)
        params['name'] = f
        params['instrument'] = instrument

    if lambda_eff is not None:
        params['lambda_eff'] = lambda_eff
        # TODO: If None, measure?
    if fwhm is not None:
        params['fwhm'] = fwhm
        # TODO: If None, measure?
    if source is not None:
        if filter_only:
            params['source_filter_only'] = source
        else:
            params['source'] = source

    data = np.genfromtxt(path)
    wavelengths = data[:, 0]
    transmissions = data[:, 1]

    if percentage:
        transmissions /= 100
    if unit == 'nm':
        wavelengths *= 10

    # Make sure the wavelengths increase instead of decrease; this assumes that the wavelengths are at least in order.
    if wavelengths[0] > wavelengths[-1]:
        wavelengths = np.flip(wavelengths)
        transmissions = np.flip(transmissions)

    if filter_only:
        params['wavelengths_filter_only'] = wavelengths.tolist()
        params['transmissions_filter_only'] = transmissions.tolist()
    else:
        params['wavelengths'] = wavelengths.tolist()
        params['transmissions'] = transmissions.tolist()

    save_params(file=param_path + f'filters/{instrument}-{f}', dictionary=params, quiet=quiet)


def ingest_filter_set(path: str, instrument: str, filter_only: bool = True, source: float = None,
                      unit: str = 'Angstrom',
                      percentage: bool = False, lambda_name='LAMBDA', quiet: bool = False):
    units = ['nm', 'Angstrom']
    if unit not in units:
        raise ValueError('Units must be one of ', units)

    data = tbl.Table.read(path, format='ascii')
    wavelengths = data[lambda_name]
    if unit == 'nm':
        wavelengths *= 10
    for f in data.colnames:
        if f != lambda_name:
            params = filter_params(f=f, instrument=instrument, quiet=quiet)
            transmissions = data[f]

            if params is None:
                params = new_filter_params(quiet=quiet)
            params['name'] = f
            if source is not None:
                if filter_only:
                    params['source_filter_only'] = source
                else:
                    params['source'] = source
            if percentage:
                transmissions /= 100
            # Make sure the wavelengths increase instead of decrease; this assumes that the wavelengths are at least in
            # order.
            if wavelengths[0] > wavelengths[-1]:
                wavelengths = np.flip(wavelengths)
                transmissions = np.flip(transmissions)
            if filter_only:
                params['wavelengths_filter_only'] = wavelengths.tolist()
                params['transmissions_filter_only'] = transmissions.tolist()
            else:
                params['wavelengths'] = wavelengths.tolist()
                params['transmissions'] = transmissions.tolist()

            save_params(file=param_path + f'filters/{instrument}-{f}', dictionary=params, quiet=quiet)
    refresh_params_filters(quiet=quiet)


def new_filter_params(quiet: bool = False):
    return load_params(param_path + 'filters/filter_template.yaml', quiet=quiet)


def filter_params(f: str, instrument: str = 'FORS2', quiet: bool = False):
    return load_params(param_path + f'filters/{instrument}-{f}', quiet=quiet)


def instrument_all_filters(instrument: str = 'FORS2', quiet: bool = False):
    # refresh_params_filters()
    filters = {}
    directory = param_path + 'filters/'
    for file in filter(lambda f: instrument in f and f[-5:] == '.yaml', os.listdir(directory)):
        params = load_params(param_path + f'filters/{file}', quiet=quiet)
        filters[params['name']] = params
    return filters


def instrument_filters_single_param(param: str, instrument: str = 'FORS2', sort_value: bool = False,
                                    quiet: bool = False):
    filters = instrument_all_filters(instrument=instrument, quiet=quiet)
    filter_names = list(filters.keys())

    param_dict = {}
    for f in filter_names:
        f_params = filters[f]
        param_dict[f] = f_params[param]

    if sort_value:
        param_dict = u.sort_dict_by_value(param_dict)

    return param_dict


def object_params_instrument(obj: str, instrument: str, quiet: bool = False):
    instrument = instrument.lower()
    return load_params(param_path + f'epochs_{instrument}/{obj}', quiet=quiet)


def object_params_fors2(obj: str, quiet: bool = False):
    return load_params(param_path + 'epochs_fors2/' + obj, quiet=quiet)


def object_params_xshooter(obj: str, quiet: bool = False):
    return load_params(param_path + 'epochs_xshooter/' + obj, quiet=quiet)


def object_params_imacs(obj: str, quiet: bool = False):
    return load_params(param_path + 'epochs_imacs/' + obj, quiet=quiet)


def object_params_des(obj: str, quiet: bool = False):
    return load_params(param_path + 'epochs_des/' + obj, quiet=quiet)


def object_params_sdss(obj: str, quiet: bool = False):
    return load_params(param_path + 'epochs_sdss/' + obj, quiet=quiet)


def object_params_frb(obj: str, quiet: bool = False):
    return load_params(param_path + 'FRBs/' + obj, quiet=quiet)


def object_params_all_epochs(obj: str, instrument: str = 'FORS2', quiet: bool = False):
    properties = {}
    directory = param_path + 'epochs_' + instrument.lower() + '/'
    for file in os.listdir(directory):
        if file[-5:] == '.yaml':
            if obj + '_' in file:
                properties[file[-6:-5]] = load_params(directory + file, quiet=quiet)

    return properties


def params_all_epochs(instrument: str = 'FORS2', quiet: bool = False):
    properties = {}
    directory = param_path + 'epochs_' + instrument.lower() + '/'
    for file in os.listdir(directory):
        if file[-5:] == '.yaml' and 'template' not in file:
            properties[file[:-5]] = load_params(directory + file, quiet=quiet)

    return properties


def single_param_all_epochs(obj: str, param: str, instrument: str = 'FORS2', quiet: bool = False):
    if obj == 'all':
        properties = params_all_epochs(instrument=instrument, quiet=quiet)
    else:
        properties = object_params_all_epochs(obj=obj, instrument=instrument, quiet=quiet)
    output = {}
    for epoch in properties:
        output[epoch] = properties[epoch][param]
    return output


def frb_output_params(obj: str, quiet: bool = False):
    """

    :param obj: The FRB title, FRBXXXXXX
    :return:
    """
    p = object_params_frb(obj=obj, quiet=quiet)
    if p is None:
        return None
    else:
        return load_params(p['data_dir'] + 'output_values', quiet=quiet)


# TODO: Object should be epoch, almost everywhere.

def object_output_params(obj: str, instrument: str = 'FORS2', quiet: bool = False):
    """

    :param obj: The epoch title, FRBXXXXXX_X; ie, the title of the epoch parameter file without .yaml
    :param instrument: Instrument on which the data was taken.
    :return:
    """
    p = object_params_instrument(obj=obj, instrument=instrument, quiet=quiet)
    if p is None:
        return None
    else:
        return load_params(p['data_dir'] + 'output_values', quiet=quiet)


def purge_output_param(key: str, instrument: str = 'FORS2', quiet: bool = False):
    paths = single_param_all_epochs(obj='all', param='data_dir', instrument=instrument)

    for epoch in paths:
        path = paths[epoch]
        p = load_params(path + 'output_values.yaml', quiet=quiet)
        p.pop(key, None)
        save_params(path + 'output_values.yaml', dictionary=p, quiet=quiet)


def object_output_paths(obj: str, instrument: str = 'FORS2', quiet: bool = False):
    p = object_params_instrument(obj=obj, instrument=instrument, quiet=quiet)
    if p is None:
        return None
    else:
        return load_params(p['data_dir'] + 'output_paths', quiet=quiet)


def std_output_params(obj: str, quiet: bool = False):
    p = object_params_fors2(obj=obj, quiet=quiet)
    if p is None:
        return None
    else:
        return load_params(p['data_dir'] + f'/calibration/std_star/output_values', quiet=quiet)


def refresh_params_frbs(quiet: bool = False):
    refresh_params_folder('FRBs', template='FRB_template', quiet=quiet)


def refresh_params_des(quiet: bool = False):
    refresh_params_folder('epochs_des', template='FRB_des_epoch_template', quiet=quiet)


def refresh_params_sdss(quiet: bool = False):
    refresh_params_folder('epochs_sdss', template='FRB_sdss_epoch_template', quiet=quiet)


def refresh_params_fors2(quiet: bool = False):
    refresh_params_folder('epochs_fors2', template='FRB_fors2_epoch_template', quiet=quiet)


def refresh_params_xshooter(quiet: bool = False):
    refresh_params_folder('epochs_xshooter', template='FRB_xshooter_epoch_template', quiet=quiet)


def refresh_params_imacs(quiet: bool = False):
    refresh_params_folder('epochs_imacs', template='FRB_imacs_epoch_template', quiet=quiet)


def refresh_params_filters(quiet: bool = False):
    refresh_params_folder('filters', template='filter_template', quiet=quiet)


def refresh_params_all(quiet=False):
    refresh_params_fors2(quiet=quiet)
    refresh_params_xshooter(quiet=quiet)
    refresh_params_imacs(quiet=quiet)
    refresh_params_des(quiet=quiet)
    refresh_params_frbs(quiet=quiet)


def refresh_params_folder(folder: str, template: str, quiet: bool = False):
    template = u.sanitise_file_ext(template, '.yaml')
    user_dir = f"{param_path}/{folder}/"
    proj_dir = f"param/{folder}/"
    # Get template file from within this project; use to update param files in param directory as specified in
    # config.yaml
    if not quiet:
        print(f'Loading template from {proj_dir}/{template}')
    template_params = load_params(proj_dir + template, quiet=quiet)

    per_filter = False
    if 'imacs' in folder:
        imacs = True
    else:
        imacs = False

    paths = [user_dir]
    if user_dir != proj_dir:
        paths.append(proj_dir)

    for path in paths:
        files = filter(lambda x: x[-5:] == '.yaml' and x != template, os.listdir(path))

        for file in files:
            file_params = load_params(path + file, quiet=quiet)
            if not quiet:
                print(path + file)
            if 'filters' in file_params:
                per_filter = True
                filter_trunc = []
                filters = file_params['filters']
                for f in filters:
                    if imacs:
                        filter_trunc.append(f[-1:] + '_')
                    else:
                        if len(f) > 1:
                            filter_trunc.append(f[:2])
                        else:
                            filter_trunc.append(f + '_')

            # Use template to insert missing parameters.
            for param in template_params:
                # Apply filter-specific parameters to all filters listed in 'filters'.
                if per_filter and param[:2] == 'f_':
                    for f in filter_trunc:
                        param_true = param.replace('f_', f, 1)
                        if param_true not in file_params:
                            file_params[param_true] = template_params[param]
                elif param not in file_params:
                    file_params[param] = template_params[param]
            # Use template to remove extraneous parameters.
            new_file_params = {}
            for param in file_params:
                # Apply filter-specific parameters to all filters listed in 'filters'.
                if per_filter and param[:2] in filter_trunc and param.replace(param[:2], 'f_', 1) in template_params:
                    new_file_params[param] = file_params[param]
                elif param in template_params and param[:2] != 'f_':
                    new_file_params[param] = file_params[param]
            # Write to .yaml file
            save_params(path + '/' + file, new_file_params, quiet=quiet)
            # Convert to json
            yaml_to_json(path + '/' + file, quiet=quiet)


def sanitise_wavelengths(quiet: bool = False):
    files = filter(lambda x: x[-5:] == '.yaml', os.listdir(param_path + 'filters/'))
    for file in files:
        file_params = load_params(param_path + 'filters/' + file, quiet=quiet)
        wavelengths = file_params['wavelengths']
        transmissions = file_params['transmissions']
        if len(wavelengths) > 0 and wavelengths[0] > wavelengths[-1]:
            wavelengths.reverse()
            transmissions.reverse()
        file_params['wavelengths'] = wavelengths
        file_params['transmissions'] = transmissions

        wavelengths = file_params['wavelengths_filter_only']
        transmissions = file_params['transmissions_filter_only']
        if len(wavelengths) > 0 and wavelengths[0] > wavelengths[-1]:
            wavelengths.reverse()
            transmissions.reverse()
        file_params['wavelengths_filter_only'] = wavelengths
        file_params['transmissions_filter_only'] = transmissions

        save_params(param_path + 'filters/' + file, file_params, quiet=quiet)
    refresh_params_filters()


def convert_to_angstrom(quiet: bool = False):
    files = filter(lambda x: x[-5:] == '.yaml', os.listdir(param_path + 'filters/'))
    for file in files:
        file_params = load_params(param_path + 'filters/' + file, quiet=quiet)

        wavelengths = np.array(file_params['wavelengths'])
        wavelengths *= 10
        file_params['wavelengths'] = wavelengths.tolist()

        wavelengths = np.array(file_params['wavelengths_filter_only'])
        wavelengths *= 10
        file_params['wavelengths_filter_only'] = wavelengths.tolist()

        save_params(param_path + 'filters/' + file, file_params, quiet=quiet)
    refresh_params_filters(quiet=quiet)


def trim_transmission_curves(f: str, instrument: str, lambda_min: float, lambda_max: float, quiet: bool = False):
    file_params = filter_params(f=f, instrument=instrument, quiet=quiet)

    wavelengths = file_params['wavelengths']
    if len(wavelengths) > 0:
        transmissions = file_params['transmissions']
        arg_lambda_min, _ = u.find_nearest(np.array(wavelengths), lambda_min, sorted=True)
        arg_lambda_max, _ = u.find_nearest(np.array(wavelengths), lambda_max, sorted=True)
        arg_lambda_max += 1
        file_params['transmissions'] = transmissions[arg_lambda_min:arg_lambda_max]
        file_params['wavelengths'] = wavelengths[arg_lambda_min:arg_lambda_max]

    wavelengths = file_params['wavelengths_filter_only']
    if len(wavelengths) > 0:
        transmissions = file_params['transmissions_filter_only']
        arg_lambda_min, _ = u.find_nearest(np.array(wavelengths), lambda_min, sorted=True)
        arg_lambda_max, _ = u.find_nearest(np.array(wavelengths), lambda_max, sorted=True)
        arg_lambda_max += 1
        file_params['transmissions_filter_only'] = transmissions[arg_lambda_min:arg_lambda_max]
        file_params['wavelengths_filter_only'] = wavelengths[arg_lambda_min:arg_lambda_max]

    save_params(param_path + f'filters/{instrument}-{f}.yaml', file_params, quiet=quiet)


def keys():
    with open(param_path + "keys.json") as fp:
        file = json.load(fp)
    return file


# def change_param_name(folder):


if __name__ == '__main__':
    refresh_params_all()
