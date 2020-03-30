import ruamel.yaml as yaml
import json
from typing import Union
import os
from PyCRAFT.utils import sanitise_file_ext, find_nearest, numpy_to_list
import numpy as np
import astropy.table as tbl


def path_or_params_obj(obj: Union[dict, str], instrument: str = 'FORS2'):
    if type(obj) is str:
        return obj, object_params_instrument(obj, instrument=instrument)
    elif type(obj) is dict:
        params = obj
        obj = params['data_title']
        return obj, params


def change_yaml_param(file: str = 'project', param: str = None, value=None, update_json=False):
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


def yaml_to_json(yaml_file: str, output: str = None):
    yaml_file = sanitise_file_ext(yaml_file, '.yaml')
    if output is not None:
        output = sanitise_file_ext(output, '.json')
    elif output is None:
        output = yaml_file.replace('.yaml', '.json')

    p = load_params(file=yaml_file)

    print('Saving parameter file to ' + output)

    with open(output, 'w') as fj:
        json.dump(p, fj)

    return p


def load_params(file: str):
    file = sanitise_file_ext(file, '.yaml')

    print('Loading parameter file from ' + str(file))

    if os.path.isfile(file):
        with open(file) as f:
            p = yaml.safe_load(f)
    else:
        p = None
        print('No parameter file found at', str(file) + ', returning None.')

    return p


def add_params(file: str, params: dict):
    file = sanitise_file_ext(file, '.yaml')
    if os.path.isfile(file):
        param_dict = load_params(file)
    else:
        param_dict = {}
    param_dict.update(params)
    save_params(file, param_dict)
    yaml_to_json(file)


def add_output_path(obj: str, key: str, path: str, instrument='fors2'):
    instrument = instrument.lower()
    p = object_params_instrument(obj=obj, instrument=instrument)
    add_params(file=p['data_dir'] + 'output_paths', params={key: path})


def add_output_value(obj: str, key: str, value: str, instrument='fors2'):
    instrument = instrument.lower()
    p = object_params_instrument(obj=obj, instrument=instrument)
    add_params(file=p['data_dir'] + 'output_values', params={key: value})


def add_output_values(obj: str, params: dict, instrument='fors2'):
    instrument = instrument.lower()
    p = object_params_instrument(obj=obj, instrument=instrument)
    add_params(file=p['data_dir'] + 'output_values', params=params)


def apertures_fors():
    return load_params('param/aperture_diameters_fors2')


def apertures_des():
    return load_params('param/aperture_diameters_des')


def sextractor_names():
    return load_params('param/sextractor_names')


def sextractor_names_psf():
    return load_params('param/sextractor_names_psf')


def sncosmo_models():
    return load_params('param/sncosmo_models')


def plotting_params():
    return load_params('param/plotting')


def ingest_filter_properties(path: str, instrument: str):
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

    params = filter_params(f=name, instrument=instrument)
    if params is None:
        params = new_filter_params()
        params['name'] = name
        params['instrument'] = instrument

    params['mjd'] = numpy_to_list(data['mjd_obs'])
    params['date'] = numpy_to_list(data['civil_date'])
    params['zeropoint'] = numpy_to_list(data['zeropoint'])
    params['zeropoint_err'] = numpy_to_list(data['zeropoint_err'])
    params['colour_term'] = numpy_to_list(data['colour_term'])
    params['colour_term_err'] = numpy_to_list(data['colour_term_err'])
    params['extinction'] = numpy_to_list(data['extinction'])
    params['extinction_err'] = numpy_to_list(data['extinction_err'])

    save_params(file=f'param/filters/{instrument}-{name}', dictionary=params)


def ingest_filter_transmission(path: str, f: str, instrument: str, filter_only: bool = True, lambda_eff: float = None,
                               fwhm: float = None, source: float = None, unit: str = 'nm', percentage: bool = False):
    units = ['nm', 'Angstrom']
    if unit not in units:
        raise ValueError('Units must be one of ', units)

    params = filter_params(f=f, instrument=instrument)
    if params is None:
        params = new_filter_params()
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

    save_params(file=f'param/filters/{instrument}-{f}', dictionary=params)


def ingest_filter_set(path: str, instrument: str, filter_only: bool = True, source: float = None,
                      unit: str = 'Angstrom',
                      percentage: bool = False, lambda_name='LAMBDA'):
    units = ['nm', 'Angstrom']
    if unit not in units:
        raise ValueError('Units must be one of ', units)

    data = tbl.Table.read(path, format='ascii')
    wavelengths = data[lambda_name]
    if unit == 'nm':
        wavelengths *= 10
    for f in data.colnames:
        if f != lambda_name:
            params = filter_params(f=f, instrument=instrument)
            transmissions = data[f]

            if params is None:
                params = new_filter_params()
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

            save_params(file=f'param/filters/{instrument}-{f}', dictionary=params)
    refresh_params_filters()


def new_filter_params():
    return load_params('param/filters/filter_template.yaml')


def filter_params(f: str, instrument: str = 'FORS2'):
    return load_params(f'param/filters/{instrument}-{f}')


def instrument_all_filters(instrument: str = 'FORS2'):
    # refresh_params_filters()
    filters = {}
    directory = 'param/filters/'
    for file in filter(lambda f: instrument in f and f[-5:] == '.yaml', os.listdir(directory)):
        params = load_params(f'param/filters/{file}')
        filters[params['name']] = params
    return filters


def project_params(project: str):
    return load_params(f'param/project/{project}.yaml')


def object_params_instrument(obj: str, instrument: str):
    instrument = instrument.lower()
    return load_params(f'param/epochs_{instrument}/{obj}')


def object_params_fors2(obj: str):
    return load_params('param/epochs_fors2/' + obj)


def object_params_xshooter(obj: str):
    return load_params('param/epochs_xshooter/' + obj)


def object_params_imacs(obj: str):
    return load_params('param/epochs_imacs/' + obj)


def object_params_des(obj: str):
    return load_params('param/epochs_des/' + obj)


def object_params_sdss(obj: str):
    return load_params('param/epochs_sdss/' + obj)


def object_params_frb(obj: str):
    return load_params('param/FRBs/' + obj)


def object_params_all_epochs(obj: str, instrument: str = 'FORS2'):
    properties = {}
    directory = 'param/epochs_' + instrument.lower() + '/'
    for file in os.listdir(directory):
        if file[-5:] == '.yaml':
            if obj + '_' in file:
                properties[file[-6:-5]] = load_params(directory + file)

    return properties


def params_all_epochs(instrument: str = 'FORS2'):
    properties = {}
    directory = 'param/epochs_' + instrument.lower() + '/'
    for file in os.listdir(directory):
        if file[-5:] == '.yaml' and 'template' not in file:
            properties[file[:-5]] = load_params(directory + file)

    return properties


def single_param_all_epochs(obj: str, param: str, instrument: str = 'FORS2'):
    if obj == 'all':
        properties = params_all_epochs(instrument=instrument)
    else:
        properties = object_params_all_epochs(obj=obj, instrument=instrument)
    output = {}
    for epoch in properties:
        output[epoch] = properties[epoch][param]
    return output


def object_output_params(obj: str, instrument: str = 'FORS2'):
    p = object_params_instrument(obj=obj, instrument=instrument)
    if p is None:
        return None
    else:
        return load_params(p['data_dir'] + 'output_values')


def purge_output_param(key: str, instrument: str = 'FORS2'):
    paths = single_param_all_epochs(obj='all', param='data_dir', instrument='FORS2')

    for epoch in paths:
        path = paths[epoch]
        p = load_params(path + 'output_values.yaml')
        p.pop(key, None)
        save_params(path + 'output_values.yaml', dictionary=p)


def object_output_paths(obj: str, instrument: str = 'FORS2'):
    p = object_params_instrument(obj=obj, instrument=instrument)
    if p is None:
        return None
    else:
        return load_params(p['data_dir'] + 'output_paths')


def std_output_params(obj: str):
    p = object_params_fors2(obj=obj)
    if p is None:
        return None
    else:
        return load_params(p['data_dir'] + f'/calibration/std_star/output_values')


def save_params(file, dictionary):
    if file[-5:] != '.yaml':
        file = file + '.yaml'

    print('Saving parameter file to ' + str(file))

    with open(file, 'w') as f:
        yaml.dump(dictionary, f)


def refresh_params_frbs():
    refresh_params_folder('FRBs', template='FRB_template')


def refresh_params_des():
    refresh_params_folder('epochs_des', template='FRB_des_epoch_template')


def refresh_params_sdss():
    refresh_params_folder('epochs_sdss', template='FRB_sdss_epoch_template')


def refresh_params_fors2():
    refresh_params_folder('epochs_fors2', template='FRB_fors2_epoch_template')


def refresh_params_xshooter():
    refresh_params_folder('epochs_xshooter', template='FRB_xshooter_epoch_template')


def refresh_params_imacs():
    refresh_params_folder('epochs_imacs', template='FRB_imacs_epoch_template')


def refresh_params_project():
    refresh_params_folder('project', template='project_template')


def refresh_params_filters():
    refresh_params_folder('filters', template='filter_template')


def refresh_params_all():
    refresh_params_fors2()
    refresh_params_xshooter()
    refresh_params_imacs()
    refresh_params_des()
    refresh_params_frbs()
    refresh_params_project()


def refresh_params_folder(folder: str, template: str):
    template = sanitise_file_ext(template, '.yaml')
    files = filter(lambda x: x[-5:] == '.yaml' and x != template, os.listdir('param/' + folder + '/'))
    template_params = load_params('param/' + folder + '/' + template)

    per_filter = False
    if 'imacs' in folder:
        imacs = True
    else:
        imacs = False

    for file in files:
        file_params = load_params('param/' + folder + '/' + file)
        print('param/' + folder + '/' + file)
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
        save_params('param/' + folder + '/' + file, new_file_params)
        p = yaml_to_json('param/' + folder + '/' + file)
        save_params('param/' + folder + '/' + file, p)


def sanitise_wavelengths():
    files = filter(lambda x: x[-5:] == '.yaml', os.listdir('param/filters/'))
    for file in files:
        file_params = load_params('param/filters/' + file)
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

        save_params('param/filters/' + file, file_params)
    refresh_params_filters()


def convert_to_angstrom():
    files = filter(lambda x: x[-5:] == '.yaml', os.listdir('param/filters/'))
    for file in files:
        file_params = load_params('param/filters/' + file)

        wavelengths = np.array(file_params['wavelengths'])
        wavelengths *= 10
        file_params['wavelengths'] = wavelengths.tolist()

        wavelengths = np.array(file_params['wavelengths_filter_only'])
        wavelengths *= 10
        file_params['wavelengths_filter_only'] = wavelengths.tolist()

        save_params('param/filters/' + file, file_params)
    refresh_params_filters()


def trim_transmission_curves(f: str, instrument: str, lambda_min: float, lambda_max: float):
    file_params = filter_params(f=f, instrument=instrument)

    wavelengths = file_params['wavelengths']
    if len(wavelengths) > 0:
        transmissions = file_params['transmissions']
        arg_lambda_min, _ = find_nearest(np.array(wavelengths), lambda_min, sorted=True)
        arg_lambda_max, _ = find_nearest(np.array(wavelengths), lambda_max, sorted=True)
        arg_lambda_max += 1
        file_params['transmissions'] = transmissions[arg_lambda_min:arg_lambda_max]
        file_params['wavelengths'] = wavelengths[arg_lambda_min:arg_lambda_max]

    wavelengths = file_params['wavelengths_filter_only']
    if len(wavelengths) > 0:
        transmissions = file_params['transmissions_filter_only']
        arg_lambda_min, _ = find_nearest(np.array(wavelengths), lambda_min, sorted=True)
        arg_lambda_max, _ = find_nearest(np.array(wavelengths), lambda_max, sorted=True)
        arg_lambda_max += 1
        file_params['transmissions_filter_only'] = transmissions[arg_lambda_min:arg_lambda_max]
        file_params['wavelengths_filter_only'] = wavelengths[arg_lambda_min:arg_lambda_max]

    save_params(f'param/filters/{instrument}-{f}.yaml', file_params)


# def change_param_name(folder):


if __name__ == '__main__':
    refresh_params_all()
