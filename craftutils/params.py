# Code by Lachlan Marnoch, 2019-2021

import json
import os
import shutil
import warnings
from datetime import date
from typing import Union
from shutil import copy

import astropy.io.misc.yaml as yaml
import astropy.units as units
import numpy as np
import pkg_resources
from astropy.table import Table, QTable

from craftutils import utils as u

__all__ = []

yaml.AstropyDumper.ignore_aliases = lambda *args: True

instruments_imaging = [
    "vlt-fors2",
    "vlt-xshooter",
    "vlt-hawki",
    "gs-aoi",
    "hst-wfc3_ir",
    "hst-wfc3_uvis2",
    "mgb-imacs",
    "panstarrs1",
    "decam"
]
instruments_spectroscopy = ["vlt-fors2", "vlt-xshooter"]
surveys = ["panstarrs1"]


def serialise_attributes(dumper, data):
    dict_representation = data.__dict__
    node = dumper.represent_dict(dict_representation)
    return node


def tabulate_output_values(path: str, output: str = None):
    path = u.check_trailing_slash(path=path)
    outputs = []
    for file in filter(lambda filename: 'output_values.yaml' in filename, os.listdir(path)):
        output_values = load_params(file=path + file)
        output_values["filename"] = file
        outputs.append(output_values)

    outputs = Table(outputs)

    if output is not None:
        output = u.sanitise_file_ext(filename=output, ext='.csv')
        outputs.write(output)
        outputs.sort(keys="filename")

    return outputs


home_path = os.path.expanduser("~")
config_dir = os.path.join(home_path, ".craftutils")
config_file = os.path.join(config_dir, "config.yaml")


@u.export
def check_for_config():
    u.mkdir_check(config_dir)
    config_template = {
        'eso_install_dir': None,
        'esoreflex_input_dir': None,
        'esoreflex_output_dir': None,
        'furby_dir': None,
        'param_dir': param_dir_project,
        'top_data_dir': os.path.join(os.path.expanduser("~"), "Data"),
        'table_dir': None,
        'publications_output_dir': os.path.join(os.path.expanduser("~"), "Data", "publications")
    }
    config_dict = load_params(config_file)
    if config_dict is None:
        # Copy template config file from project directory.
        # try:
        shutil.copy(
            pkg_resources.resource_filename(
                __name__,
                os.path.join("param", "config_template.yaml")
            ),
            config_file
        )
        print(f"No config file was detected at {config_file}.")
        print(f"A fresh config file will been created at '{config_file}'.")
        # print("I will now ask you for some directories that will go into this file, but it may be edited at any time.")
        # top_data_dir = u.user_input(
        #     "\nPlease enter a directory in which to store all data products of this package "
        #     "(This may require a large amount of space.). If the directory does not exist, "
        #     "I will attempt to create it.",
        #     default=config_template["top_data_dir"]
        # )
        # os.makedirs(top_data_dir, exist_ok=True)
        # config_dict["top_data_dir"] = top_data_dir
        #
        # param_dir_n = u.user_input(
        #     "\nEnter a directory in which parameter files will be written and loaded from; leave as default to use this"
        #     "repository's param directory.",
        #     default=config_template["param_dir"]
        # )
        # os.makedirs(param_dir_n, exist_ok=True)
        # config_dict["param_dir"] = param_dir_n
        # if param_dir_n != config_template["param_dir"]:
        #     print("Copying included param files to param directory.")
        #     shutil.copytree(config_template["param_dir"], param_dir_n)

        # if u.select_yn(message="Do you have ESO Reflex installed?"):
        #     if os.path.isdir()

        config_dict = load_params(config_file)

    for param in config_template:
        if param not in config_dict:
            config_dict[param] = config_template[param]

    for path_name in config_dict:
        path = config_dict[path_name]
        if path is not None:
            config_dict[path_name] = os.path.abspath(path)
    else:
        for param in config_dict:
            if config_dict[param] is not None:
                config_dict[param] = u.check_trailing_slash(config_dict[param])
        save_params(config_file, config_dict)
        yaml_to_json(config_file)
    return config_dict


def load_params(file: str):
    file = u.sanitise_file_ext(file, '.yaml')

    u.debug_print(2, 'Loading parameter file from ' + str(file))

    if os.path.isfile(file):
        with open(file) as f:
            p = yaml.load(f)
    else:
        p = None
        u.debug_print(1, 'No parameter file found at', str(file) + ', returning None.')
    return p


def check_abs_path(path: str, root: str = "top_data_dir"):
    if not os.path.isabs(path):
        path = os.path.join(config[root], str(path))
    return path


def sanitise_yaml_dict(dictionary: dict):
    for key in dictionary:
        if isinstance(dictionary[key], np.str_):
            dictionary[key] = str(dictionary[key])
    return dictionary


def save_params(file: str, dictionary: dict):
    file = u.sanitise_file_ext(filename=file, ext=".yaml")

    u.debug_print(1, 'Saving parameter file to ' + str(file))
    u.debug_print(2, "params.save_params: dictionary ==", dictionary)

    if u.debug_level > 2:
        for key in dictionary:
            print(key, type(dictionary[key]), dictionary[key])

    file_backup = file.replace(".yaml", "_backup.yaml")
    if os.path.exists(file):
        shutil.copy(file, file_backup)

    from yaml.representer import RepresenterError
    try:
        with open(file, 'w') as f:
            yaml.dump(dictionary, f)
    except RepresenterError as err:
        if os.path.exists(file_backup):
            shutil.copy(file_backup, file)
        raise err


def select_coords(dictionary):
    if "ra" in dictionary:
        ra_key = "ra"
    elif "alpha" in dictionary:
        ra_key = "alpha"
    else:
        raise ValueError("No valid Right Ascension found in dictionary.")

    if "hms" in dictionary[ra_key] and dictionary[ra_key]["hms"] not in [None, 0]:
        ra = dictionary[ra_key]["hms"]
    elif "decimal" in dictionary[ra_key] and dictionary[ra_key]["decimal"] not in [None, 0]:
        ra = f"{dictionary[ra_key]['decimal']}d"
    else:
        raise ValueError("No valid Right Ascension found in dictionary.")

    if "dec" in dictionary:
        dec_key = "dec"
    elif "delta" in dictionary:
        dec_key = "delta"
    else:
        raise ValueError("No valid Declination found in dictionary.")

    if "dms" in dictionary[dec_key] and dictionary[dec_key]["dms"] not in [None, 0]:
        dec = dictionary[dec_key]["dms"]
    elif "decimal" in dictionary[dec_key] and dictionary[dec_key]["decimal"] not in [None, 0]:
        dec = f"{dictionary[dec_key]['decimal']}d"
    else:
        raise ValueError("No valid Declination found in dictionary.")

    return ra, dec


def yaml_to_json(yaml_file: str, output: str = None):
    yaml_file = u.sanitise_file_ext(yaml_file, '.yaml')
    if output is not None:
        output = u.sanitise_file_ext(output, '.json')
    elif output is None:
        output = yaml_file.replace('.yaml', '.json')

    p = load_params(file=yaml_file)

    u.debug_print(1, 'Saving parameter file to ' + output)

    for param in p:
        if type(p[param]) is date:
            p[param] = str(p[param])

    with open(output, 'w') as fj:
        json.dump(p, fj)

    return p


def get_project_path():
    return os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


# Here we set up the various directories used by the pipeline.
config = check_for_config()
project_dir = get_project_path()
param_dir_project = os.path.join(project_dir, "craftutils", "param")
param_dir = config['param_dir']


def write_config():
    for param in config:
        if config[param] is not None:
            config[param] = u.check_trailing_slash(config[param])
    save_params(config_file, config)
    yaml_to_json(config_file)


@u.export
def set_param_dir(path: str, write: bool = True):
    config["param_dir"] = path
    global param_dir
    param_dir = path
    u.mkdir_check_nested(path, remove_last=False)
    u.mkdir_check(os.path.join(path, "fields"))
    u.mkdir_check(os.path.join(path, "instruments"))
    u.mkdir_check(os.path.join(path, "surveys"))
    key_path = os.path.join(path, 'keys.json')
    if not os.path.isfile(key_path):
        copy(os.path.join(param_dir_project, "keys.json"), key_path)
    if write:
        write_config()


if param_dir is None:
    set_param_dir(param_dir_project, write=False)

furby_path = None
if "furby_dir" in config:
    furby_path = config["furby_dir"]

data_dir = config["top_data_dir"]
if data_dir is None:
    warnings.warn(
        f"data_dir has not been set in config file. Set it with craftutils.params.set_data_dir() or "
        f"by editing the config file at {config_file}"
    )


def set_eso_user():
    if not config["eso_install_dir"]:
        eso_install = u.user_input(
            message="Please enter the directory in which esoreflex is installed.",
            input_type=str
        )
        if eso_install.endswith("/"):
            eso_install = eso_install[:-1]
        if not os.path.basename(eso_install) == "install":
            eso_install = os.path.join(eso_install, "install")
        set_eso_install_dir(eso_install, write=False)
    eso_root = config["eso_install_dir"]
    if not config["esoreflex_input_dir"]:
        set_esoreflex_input_dir(
            path=u.user_input(
                message="Please enter the directory in which esoreflex looks for input.",
                input_type=str,
                default=os.path.join(eso_root, "data_wkf", "reflex_input")
            ),
            write=False
        )
    if not config["esoreflex_output_dir"]:
        set_esoreflex_output_dir(
            path=u.user_input(
                message="Please enter the directory in which esoreflex writes output. It is named 'reflex_data' by default",
                input_type=str,
                default=os.path.join(os.path.expanduser("~"), "reflex_data")
            ),
            write=False
        )
    write_config()


def set_eso_install_dir(path: str, write: bool = True):
    set_config_path(key="eso_install_dir", path=path, write=write)


def set_esoreflex_output_dir(path: str, write: bool = True):
    set_config_path(key="esoreflex_output_dir", path=path, write=write)


def set_esoreflex_input_dir(path: str, write: bool = True):
    set_config_path(key="esoreflex_input_dir", path=path, write=write)


@u.export
def set_data_dir(path: str, write: bool = True):
    set_config_path(key="top_data_dir", path=path, write=write)
    global data_dir
    data_dir = path
    u.mkdir_check_nested(path, remove_last=False)


def set_table_dir(path: str, write: bool = True):
    set_config_path(key="table_dir", path=path, write=write)
    u.mkdir_check_nested(path, remove_last=False)


def set_config_path(key: str, path: str, write: bool = True):
    config[key] = path
    if write:
        write_config()


def get_project_git_hash(short: bool = False):
    global project_dir
    return u.get_git_hash(directory=project_dir, short=short)


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


def add_params(file: str, params: dict, skip_json: bool = False):
    file = u.sanitise_file_ext(file, '.yaml')
    if os.path.isfile(file):
        param_dict = load_params(file)
    else:
        param_dict = {}
    param_dict.update(params)
    save_params(file, param_dict)
    if not skip_json:
        yaml_to_json(file)


def add_config_param(params: dict):
    add_params(file="param/config.yaml", params=params)
    params.config = check_for_config()


def add_frb_param(obj: str, params: dict, quiet=False):
    add_params(file=param_dir + "FRBs/" + obj + ".yaml", params=params)


def add_epoch_param(obj: str, params: dict, instrument: str = 'FORS2', quiet=False):
    add_params(file=param_dir + "epochs_" + instrument.lower() + "/" + obj + ".yaml", params=params, quiet=quiet)


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


def apertures_fors():
    return load_params(param_dir + '/aperture_diameters_fors2')


def apertures_des():
    return load_params(param_dir + 'aperture_diameters_des')


def sextractor_names():
    return load_params(param_dir + 'sextractor_names')


def sextractor_names_psf():
    return load_params(param_dir + 'sextractor_names_psf')


def sncosmo_models():
    return load_params(param_dir + 'sncosmo_models')


def plotting_params():
    return load_params(param_dir + 'plotting')


def ingest_eso_filter_properties(path: str, instrument: str, update: bool = False, quiet: bool = False):
    """
    Imports a dataset from http://archive.eso.org/bin/qc1_cgi?action=qc1_browse_table&table=fors2_photometry into a
    filter properties .yaml file within this project.
    :param path:
    :param instrument:
    :return:
    """
    data = Table.read(path, format='ascii')
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
    save_params(file=param_dir + f'filters/{instrument}-{name}', dictionary=params)


def ingest_filter_transmission(
        path: str,
        fil_name: str,
        instrument: str,
        instrument_response: bool = False,
        atmosphere: bool = False,
        lambda_eff: units.Quantity = None,
        fwhm: float = None,
        source: str = None,
        wavelength_unit: units.Unit = units.Angstrom,
        percentage: bool = False,
        quiet: bool = False
):
    """

    :param path:
    :param fil_name:
    :param instrument:
    :param instrument_response: Filter curve includes instrument response
    :param atmosphere: Filter curve includes atmospheric transmission
    :param lambda_eff:
    :param fwhm:
    :param source:
    :param wavelength_unit:
    :param percentage:
    :param quiet:
    :return:
    """

    if not wavelength_unit.is_equivalent(units.Angstrom):
        raise units.UnitTypeError(f"Wavelength units must be of type length, not {wavelength_unit}")

    type_str = "_filter"
    if instrument_response:
        type_str += "_instrument"
    if atmosphere:
        type_str += "_atmosphere"

    params = filter_params(f=fil_name, instrument=instrument, quiet=quiet)
    if params is None:
        params = new_filter_params(quiet=quiet)
        params['name'] = fil_name
        params['instrument'] = instrument

    if lambda_eff is not None:
        lambda_eff = u.check_quantity(lambda_eff, unit=units.Angstrom, convert=True)
        params['lambda_eff'] = lambda_eff
        # TODO: If None, measure?
    if fwhm is not None:
        params['fwhm'] = fwhm
        # TODO: If None, measure?
    if source is not None:
        params[f'source{type_str}'] = source

    tbl = QTable.read(path, format="ascii")
    tbl["col1"].name = "wavelength"
    tbl["wavelength"] *= wavelength_unit
    tbl["wavelength"] = tbl["wavelength"].to("Angstrom")

    tbl["col2"].name = "transmission"

    if percentage:
        tbl["transmission"] /= 100

    tbl.sort("wavelength")

    params[f'wavelengths{type_str}'] = tbl["wavelength"].value.tolist()
    params[f'transmissions{type_str}'] = tbl["transmission"].value.tolist()

    save_params(file=os.path.join(param_dir, 'filters', f'{instrument}-{fil_name}'), dictionary=params, quiet=quiet)


def ingest_filter_set(path: str, instrument: str,
                      instrument_response: bool = False, atmosphere: bool = False,
                      source: str = None,
                      wavelength_unit: units.Unit = None,
                      percentage: bool = False, lambda_name='LAMBDA', quiet: bool = False):
    """

    :param path:
    :param instrument:
    :param instrument_response: Filter curve includes instrument response
    :param atmosphere: Filter curve includes atmospheric transmission
    :param source:
    :param wavelength_unit:
    :param percentage:
    :param lambda_name:
    :param quiet:
    :return:
    """

    if not wavelength_unit.is_equivalent(units.Angstrom):
        raise units.UnitTypeError(f"Wavelength units must be of type length, not {wavelength_unit}")

    type_str = "_filter"
    if instrument_response:
        type_str += "_instrument"
    if atmosphere:
        type_str += "_atmosphere"

    data = QTable.read(path, format='ascii')
    data.sort("col1")
    wavelengths = data["col1"] * wavelength_unit
    wavelengths = wavelengths.to("Angstrom")
    for f in data.colnames:
        if f != lambda_name:
            params = filter_params(f=f, instrument=instrument, quiet=quiet)
            transmissions = data[f]

            if params is None:
                params = new_filter_params(quiet=quiet)
            params['name'] = f
            if source is not None:
                params[f'source{type_str}'] = source
            if percentage:
                transmissions /= 100
            params[f'wavelengths{type_str}'] = wavelengths
            params[f'transmissions{type_str}'] = transmissions

            save_params(file=param_dir + f'filters/{instrument}-{f}', dictionary=params, quiet=quiet)
    refresh_params_filters(quiet=quiet)


def new_filter_params(quiet: bool = False):
    return load_params(
        os.path.join(project_dir, 'param', 'filters', 'filter_template.yaml'),
        quiet=quiet)


def filter_params(f: str, instrument: str = 'FORS2', quiet: bool = False):
    return load_params(os.path.join(param_dir, 'filters', f'{instrument}-{f}'), quiet=quiet)


def instrument_all_filters(instrument: str = 'FORS2', quiet: bool = False):
    # refresh_params_filters()
    filters = {}
    directory = param_dir + 'filters/'
    for file in filter(lambda f: instrument in f and f[-5:] == '.yaml', os.listdir(directory)):
        params = load_params(param_dir + f'filters/{file}', quiet=quiet)
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


def object_params_instrument(obj: str, instrument: str):
    instrument = instrument.lower()
    return load_params(os.path.join(param_dir, f'epochs_{instrument}', obj))


def object_params_fors2(obj: str):
    return load_params(param_dir + 'epochs_fors2/' + obj)


def object_params_xshooter(obj: str):
    return load_params(param_dir + 'epochs_xshooter/' + obj)


def object_params_imacs(obj: str):
    return load_params(param_dir + 'epochs_imacs/' + obj)


def object_params_des(obj: str):
    return load_params(param_dir + 'epochs_des/' + obj)


def object_params_sdss(obj: str):
    return load_params(param_dir + 'epochs_sdss/' + obj)


def object_params_frb(obj: str):
    return load_params(param_dir + 'FRBs/' + obj)


def object_params_all_epochs(obj: str, instrument: str = 'FORS2'):
    properties = {}
    directory = param_dir + 'epochs_' + instrument.lower() + '/'
    for file in os.listdir(directory):
        if file[-5:] == '.yaml':
            if obj + '_' in file:
                properties[file[-6:-5]] = load_params(directory + file)

    return properties


def params_all_epochs(instrument: str = 'FORS2', quiet: bool = False):
    properties = {}
    directory = param_dir + 'epochs_' + instrument.lower() + '/'
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
        return load_params(os.path.join(p['data_dir'], 'output_values'), quiet=quiet)


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
    user_dir = f"{param_dir}/{folder}/"
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
    files = filter(lambda x: x[-5:] == '.yaml', os.listdir(param_dir + 'filters/'))
    for file in files:
        file_params = load_params(param_dir + 'filters/' + file, quiet=quiet)
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

        save_params(param_dir + 'filters/' + file, file_params, quiet=quiet)
    refresh_params_filters()


def convert_to_angstrom(quiet: bool = False):
    files = filter(lambda x: x[-5:] == '.yaml', os.listdir(param_dir + 'filters/'))
    for file in files:
        file_params = load_params(param_dir + 'filters/' + file, quiet=quiet)

        wavelengths = np.array(file_params['wavelengths'])
        wavelengths *= 10
        file_params['wavelengths'] = wavelengths.tolist()

        wavelengths = np.array(file_params['wavelengths_filter_only'])
        wavelengths *= 10
        file_params['wavelengths_filter_only'] = wavelengths.tolist()

        save_params(param_dir + 'filters/' + file, file_params, quiet=quiet)
    refresh_params_filters(quiet=quiet)


def trim_transmission_curves(f: str, instrument: str, lambda_min: float, lambda_max: float, quiet: bool = False):
    file_params = filter_params(f=f, instrument=instrument, quiet=quiet)

    wavelengths = file_params['wavelengths']
    if len(wavelengths) > 0:
        transmissions = file_params['transmissions']
        arg_lambda_min, _ = u.find_nearest(np.array(wavelengths), lambda_min)
        arg_lambda_max, _ = u.find_nearest(np.array(wavelengths), lambda_max)
        arg_lambda_max += 1
        file_params['transmissions'] = transmissions[arg_lambda_min:arg_lambda_max]
        file_params['wavelengths'] = wavelengths[arg_lambda_min:arg_lambda_max]

    wavelengths = file_params['wavelengths_filter_only']
    if len(wavelengths) > 0:
        transmissions = file_params['transmissions_filter_only']
        arg_lambda_min, _ = u.find_nearest(np.array(wavelengths), lambda_min)
        arg_lambda_max, _ = u.find_nearest(np.array(wavelengths), lambda_max)
        arg_lambda_max += 1
        file_params['transmissions_filter_only'] = transmissions[arg_lambda_min:arg_lambda_max]
        file_params['wavelengths_filter_only'] = wavelengths[arg_lambda_min:arg_lambda_max]

    save_params(param_dir + f'filters/{instrument}-{f}.yaml', file_params, quiet=quiet)


def keys():
    """
    Returns the contents of keys.json as a dict.
    :return:
    """
    key_path = os.path.join(param_dir, "keys.json")
    if os.path.isfile(key_path):
        return load_json(key_path)
    else:
        raise FileNotFoundError(
            f"keys.json does not exist at param_path={param_dir}. "
            f"Please make a copy from {os.path.join(get_project_path(), 'param', 'keys.json')}")


def load_json(path: str):
    with open(path) as fp:
        file = json.load(fp)
    return file


def path_to_config_sextractor_config_pre_psfex():
    return os.path.join(path_to_config_psfex(), "pre-psfex.sex")


def path_to_config_sextractor_failed_psfex_config():
    return os.path.join(path_to_source_extractor(), "failed-psf-fit.sex")


def path_to_config_sextractor_failed_psfex_param():
    return os.path.join(path_to_source_extractor(), "failed-psf-fit.param")


def path_to_config_sextractor_config():
    return os.path.join(path_to_config_psfex(), "psf-fit.sex")


def path_to_config_galfit():
    return os.path.join(project_dir, "craftutils", "param", "galfit", "galfit.feedme")


def path_to_config_sextractor_param_pre_psfex():
    return os.path.join(path_to_config_psfex(), "pre-psfex.param")


def path_to_config_sextractor_param():
    return os.path.join(path_to_config_psfex(), "psf-fit.param")


def path_to_config_psfex():
    return os.path.join(project_dir, "craftutils", "param", "psfex")


def path_to_source_extractor():
    return os.path.join(project_dir, "craftutils", "param", "sextractor")


def params_init(
        param_file: Union[str, dict],
        name: str = None
):
    if type(param_file) is str:
        # Load params from .yaml at path.
        param_file = u.sanitise_file_ext(filename=param_file, ext="yaml")
        param_dict = load_params(file=param_file)
        if param_dict is None:
            return None, param_file, None  # raise FileNotFoundError(f"No parameter file found at {param_file}.")
        if name is None:
            name = u.get_filename(path=param_file, include_ext=False)
        param_dict["param_path"] = param_file
    else:
        param_dict = param_file
        if name is None:
            name = param_dict["name"]
        param_file = param_dict["param_path"]

    return name, param_file, param_dict


def load_output_file(obj):
    if obj.data_path is not None and obj.name is not None:
        obj.output_file = os.path.join(obj.data_path, f"{obj.name}_outputs.yaml")
        outputs = load_params(file=obj.output_file)
        if outputs is not None:
            if "paths" in outputs:
                obj.paths.update(outputs["paths"])
        return outputs
    else:
        raise ValueError("Insufficient information to find output file; data_path and name must be set.")


def update_output_file(obj):
    if obj.output_file is not None:
        param_dict = load_params(obj.output_file)
        if param_dict is None:
            param_dict = {}
        # For each of these, check if None first.
        param_dict.update(obj._output_dict())
        save_params(dictionary=param_dict, file=obj.output_file)
    else:
        raise ValueError("Output could not be saved to file due to lack of valid output path.")


def join_data_dir(path: str):
    """
    If the path is relative, joins it to data_dir; otherwise returns path unchanged.

    :param path:
    :return:
    """
    if not os.path.isabs(path):
        path = os.path.join(data_dir, path)
    return os.path.abspath(path)


def split_data_dir(path: str):
    """
    If a path is inside data_dir, turns it into a relative path (relative to data_dir); else returns the path unchanged.

    :param path: Must be an absolute path.
    :return:
    """
    if not os.path.isabs(path):
        raise ValueError(f"path {path} is not absolute.")
    path = os.path.abspath(path)
    if path.startswith(data_dir):
        return path.replace(data_dir, "")
    else:
        return path


# def change_param_name(folder):

def status_file(path: str):
    status = load_params(path)
    if status is None:
        status = {
            "complete": [],
            "failed": [],
        }
    if "failed" not in status or not isinstance(status["failed"], dict):
        status["failed"] = {}
    if "complete" not in status or not isinstance(status["complete"], dict):
        status["complete"] = {}
    return status


if __name__ == '__main__':
    refresh_params_all()
