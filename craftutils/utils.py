# Code by Lachlan Marnoch, 2019-2021

import math
import os
import sys
from typing import List, Union
from datetime import datetime as dt

import numpy as np

import astropy.table as table
import astropy.io.fits as fits
import astropy.units as units
from astropy.coordinates import SkyCoord
from astropy.time import Time


# TODO: Arrange these into some kind of logical order.
# TODO: Also comment.

def write_list_to_file(path: str, file: list):
    # Delete file, to be rewritten.
    rm_check(path)
    # Write file to disk.
    print(f"Writing pypeit file to {path}")
    with open(path, 'w') as pypeit_file:
        pypeit_file.writelines(file)


def relevant_timescale(time: units.Quantity):
    if not time.unit.is_equivalent(units.second):
        raise ValueError(f"{time} is not a time.")
    microseconds = time.to(units.us)
    if microseconds < 1000 * units.us:
        return microseconds
    milliseconds = time.to(units.ms)
    if milliseconds < 1000 * units.ms:
        return milliseconds
    seconds = time.to(units.second)
    if seconds < 60 * units.second:
        return seconds
    minutes = time.to(units.minute)
    if minutes < 60 * units.minute:
        return minutes
    hours = time.to(units.hour)
    if hours < 24 * units.hour:
        return hours
    days = time.to(units.day)
    if days < 7 * units.day:
        return days
    weeks = time.to(units.week)
    if weeks < 52.2 * units.week:
        return weeks
    years = time.to(units.year)
    return years


def traverse_dict(dictionary: dict, function, keys: list = None):
    if keys is None:
        keys = []
    for key in dictionary:
        keys_this = keys.copy()
        keys_this.append(key)
        value = dictionary[key]
        if type(value) is dict:
            traverse_dict(value, function=function, keys=keys_this)
        else:
            function(keys_this, value)


def get_filename(path: str, include_ext: True):
    # Split the path into file and path.
    filename = os.path.split(path)[-1]
    if not include_ext:
        # Remove file extension.
        filename = os.path.splitext(filename)[0]
    return filename


def check_key(key: str, dictionary: dict, na_values: Union[tuple, list] = (None)):
    """
    Returns true if a key is present in a dictionary AND the value of the key is not in na_values (just None by default)
    :param key: The key to check for.
    :param dictionary: The dictionary to check the key for.
    :param na_values: Values not allowed.
    :return: bool
    """
    return key in dictionary and dictionary[key] not in na_values


def check_dict(key: str, dictionary: dict, na_values: Union[tuple, list] = (None), fail_val=None):
    """

    :param key:
    :param dictionary:
    :param na_values:
    :param fail_val: The value to return if the key is not
    :return:
    """
    if check_key(key=key, dictionary=dictionary, na_values=na_values):
        return dictionary[key]


def check_quantity(number: Union[float, int, units.Quantity], unit: units.Unit, allow_mismatch: bool = True):
    if type(number) is not units.Quantity:
        number *= unit
    elif number.unit != unit:
        if not allow_mismatch:
            raise ValueError(f"This is already a Quantity, but with units {number.unit}; units {unit} were specified.")
        elif not (number.unit.is_equivalent(unit)):
            raise ValueError(
                f"This number is already a Quantity, but with incompatible units ({number.unit}); units {unit} were specified.")
    return number


def dequantify(number: Union[float, int, units.Quantity]):
    if type(number) is units.Quantity:
        return number.value
    else:
        return number


def sort_dict_by_value(dictionary: dict):
    sorted_keys = sorted(dictionary, key=dictionary.get)
    sorted_dict = {}
    for f in sorted_keys:
        sorted_dict[f] = dictionary[f]
    return sorted_dict


def check_trailing_slash(path: str):
    """
    Adds a slash to the end of a string if it is missing.
    :param path:
    :return:
    """

    if not path.endswith("/"):
        path += "/"
    return path


def remove_trailing_slash(path: str):
    """
    Adds a slash to the end of a string if it is missing.
    :param path:
    :return:
    """

    if path.endswith("/"):
        path = path[:-1]
    return path


def size_from_ang_size_distance(theta: float, ang_size_distance: float):
    """
    Simply calculates the projected distance scale in an image from a given angular size distance (might I recommend
    the calculator at http://www.astro.ucla.edu/~wright/CosmoCalc.html) and sky angle.
    :param theta: In degrees.
    :param ang_size_distance: In parsecs.
    :return: Projected distance in parsecs.
    """
    # Convert to radians
    theta *= np.pi / 180
    # Project.
    projected_distance = theta * ang_size_distance

    return projected_distance


def latex_sanitise(string: str):
    """
    Special thanks to https://stackoverflow.com/questions/2627135/how-do-i-sanitize-latex-input
    :param string:
    :return:
    """
    string = string.replace('\\', r'\textbackslash{}')
    string = string.replace(r'_', r'\_')
    return string


def rm_check(path):
    """
    Checks if a file exists; if so, deletes it.
    :param path:
    :return:
    """
    if os.path.isfile(path):
        os.remove(path)


def mkdir_check(path: str):
    """
    Checks if a directory exists; if not, creates it.
    :param path:
    :return:
    """
    if not os.path.isdir(path):
        os.mkdir(path)


# TODO: Make this system independent.
def mkdir_check_nested(path: str):
    """
    Does mkdir_check, but for all parent directories of the given path.
    :param path:
    :return:
    """
    i = 1
    while i < len(path):
        if path[i] == "/":
            if i + 1 == len(path) or path[i + 1] != "/":
                subpath = path[0:i]
                mkdir_check(subpath)
        i += 1


def fwhm_to_std(fwhm: float):
    return fwhm / 2.355


def std_to_fwhm(std: float):
    return std * 2.355


def directory_of(path: str):
    """
    Given the path to a file, removes the file and returns the directory containing it.
    :param path:
    :return:
    """
    file = ''
    while len(path) > 0 and path[-1] != '/':
        file = path[-1] + file
        path = path[:-1]

    return path, file


def uncertainty_product(value, *args: tuple):
    """
    Each arg should be a tuple, in which the first entry is the measurement and the second entry is the uncertainty in
    that measurement. These may be in the form of numpy arrays or table columns.
    """
    variance_pre = 0.
    for measurement, uncertainty in args:
        if hasattr(measurement, "__len__"):
            if isinstance(measurement, units.Quantity):
                measurement[measurement == 0.0] = sys.float_info.min * measurement.unit
            else:
                measurement[measurement == 0.0] = sys.float_info.min
        elif measurement == 0.0:
            measurement = sys.float_info.min
        variance_pre = variance_pre + (uncertainty / measurement) ** 2
    sigma_pre = np.sqrt(variance_pre)
    sigma = np.abs(value) * sigma_pre
    return sigma


def uncertainty_sum(*args):
    variance = 0.
    for measurement, uncertainty in args:
        variance += uncertainty ** 2
    sigma = np.sqrt(variance)
    return sigma


def error_product(value, measurements, errors):
    """
    Produces the absolute uncertainty of a value calculated as a product or as a quotient.
    :param value: The final calculated value.
    :param measurements: An array of the measurements used to calculate the value.
    :param errors: An array of the respective errors of the measurements. Expected to be in the same order as
        measurements.
    :return:
    """

    measurements = np.array(measurements)
    errors = np.array(errors)
    print("VALUE:", value)
    print("UNCERTAINTIES:", errors)
    print("MEASUREMENTS:", measurements)
    return value * np.sum(np.abs(errors[measurements != 0.] / measurements[measurements != 0.]))


def error_func(arg, err, func=lambda x: np.log10(x), absolute=False):
    """

    :param arg:
    :param err: The error in the argument of the function.
    :param func:
    :param absolute:
    :return:
    """
    measurement = func(arg)
    print("\narg", arg)
    print("\nmeasurement", measurement)
    # One of these should come out negative - that becomes the minus error, and the positive the plus error.
    error_plus = func(arg + err) - measurement
    error_minus = func(arg - err) - measurement

    error_plus_actual = []
    error_minus_actual = []
    print("\nerror_plus", error_plus)
    try:
        for i, _ in enumerate(error_plus):
            error_plus_actual.append(np.max([error_plus[i], error_minus[i]]))
            error_minus_actual.append(np.min([error_plus[i], error_minus[i]]))
    except TypeError:
        error_plus_actual.append(np.max([error_plus, error_minus]))
        error_minus_actual.append(np.min([error_plus, error_minus]))

    if absolute:
        return measurement + np.array([0., error_plus_actual, error_minus_actual])
    else:
        return np.array([measurement, error_plus_actual, error_minus_actual])


def error_func_percent(arg, err, func=lambda x: np.log10(x)):
    measurement, error_plus, error_minus = error_func(arg=arg, err=err, func=func, absolute=False)
    return np.array([error_plus / measurement, error_minus / measurement])


def get_column_names(path, delimiter=','):
    with open(path) as f:
        names = f.readline().split(delimiter)
    print(names)
    return names


def get_column_names_sextractor(path):
    columns = []
    with open(path) as f:
        line = f.readline()
        while line[0] == '#':
            line_list = line.split(" ")
            while "" in line_list:
                line_list.remove("")
            columns.append(line_list[2])
            line = f.readline()
    print(columns)
    return columns


def mean_squared_error(model_values, obs_values, weights=None, quiet=True):
    """
    Weighting from https://stats.stackexchange.com/questions/230517/weighted-root-mean-square-error
    :param model_values:
    :param obs_values:
    :param weights:
    :param quiet:
    :return:
    """
    if len(model_values) != len(obs_values):
        raise ValueError("Arrays must be the same length.")
    n = len(model_values)
    if not quiet:
        print("n:", n)
    if weights is None:
        weights = 1
    else:
        weights = weights / np.linalg.norm(weights, ord=1)
        n = 1
        if not quiet:
            print("Normalised weights:", np.sum(weights))

    return (1 / n) * np.sum(weights * (obs_values - model_values) ** 2)


def root_mean_squared_error(model_values, obs_values, weights=None, quiet=True):
    mse = mean_squared_error(model_values=model_values, obs_values=obs_values, weights=weights, quiet=quiet)
    if not quiet:
        print("MSE:", mse)
    return np.sqrt(mse)


def mode(lst: 'list'):
    return max(set(lst), key=list.count)


def write_params(path: 'str', params: 'dict'):
    with open(path, 'a') as file:
        for param in params:
            file.writelines(param + "=" + str(params[param]) + '\n')


def write_log(path: str, action: str, date: str = None):
    """
    Writes a line to a log file.
    :param path: Path to the log file. The file will be created if it does not exist.
    :param action: Line to write to log file, written below the date.
    :param date: String to write as date. If left empty, present date will be written automatically.
    :return:
    """
    if date is None:
        date = dt.now().strftime('%Y-%m-%dT%H:%M:%S')
    try:
        with open(path, 'a') as log:
            log.write('\n' + date)
            log.write('\n' + action + '\n')
    except ValueError:
        print('Log writing failed - skipping.')


def first_file(path: "str", ext: 'str' = None):
    if path[-1] != "/":
        path = path + "/"
    files = os.listdir(path)
    files.sort()
    # TODO: only return files with desired extension
    return path + files[0]


def find_object(x, y, x_search, y_search, world=False):
    # TODO: Throw error here if search arrays not same length
    """
    Returns closest match to given coordinates from the given search space.
    :param x: x-coordinate to find
    :param y: y-coordinate to find
    :param x_search: array to search for x-coordinate
    :param y_search: array to search for y-coordinate
    :return: id (int), distance (float)
    """

    # TODO: This could be done better with SkyCoord
    if world:
        distances = np.sqrt(((x_search - x) * np.cos(y)) ** 2 + (y_search - y) ** 2)
    else:
        distances = np.sqrt((x_search - x) ** 2 + (y_search - y) ** 2)

    match_id = np.argmin(distances)

    return match_id, distances[match_id]


def match_cat(x_match, y_match, x_cat, y_cat, tolerance=np.inf, world=False, return_dist=False):
    """
    Matches a set of objects against a catalogue using their positions.
    Provides a list of matching ids
    :param x_match: Array of x-coordinates to find in the catalogue. Can be pixel coordinates or RA/DEC.
    :param y_match: Array of y-coordinates to find in the catalogue. Can be pixel coordinates or RA/DEC.
    :param x_cat: Array of catalogue x-coordinates. Can be pixel coordinates or RA/DEC.
    :param y_cat: Array of catalogue y-coordinates. Can be pixel coordinates or RA/DEC.
    :param tolerance: Matches are discarded if the distance is greater than tolerance. If tolerance is None, finds
    nearest match no matter how far away it is.
    :return: tuple of arrays containing: 0. the match indices in the search array and 1. the match indices in the
    catalogue.
    """

    if len(x_match) != len(y_match):
        raise ValueError('x_match and y_match must be the same length.')
    if len(x_cat) != len(y_cat):
        raise ValueError('x_cat and y_cat must be the same length.')

    matches_search = []
    matches_cat = []
    match_dists = []
    print(len(x_match), len(x_cat))
    for i in range(len(x_match)):

        match_id, match_dist = find_object(x=x_match[i], y=y_match[i], x_search=x_cat, y_search=y_cat, world=world)
        if match_dist < tolerance:
            matches_cat.append(match_id)
            matches_search.append(i)
            if return_dist:
                match_dists.append(match_dist)

    if return_dist:
        return matches_search, matches_cat, match_dists
    else:
        return matches_search, matches_cat


def match_both(table_1, table_2, cat, table_name_1: 'str' = '1', table_name_2: 'str' = '2', cat_name: 'str' = 'cat',
               x_col: 'str' = 'ra',
               y_col: 'str' = 'dec', x_cat_col='ra', y_cat_col='dec',
               tolerance=(1. / 3600.)):
    """
    For when you want to match multiple sets of objects against the same catalogue and get a list of objects that appear
    in all of them.
    :param table_list:
    :param cat:
    :param table_names:
    :param cat_names:
    :param x_col:
    :param y_col:
    :param x_cat_col:
    :param y_cat_col:
    :param tolerance:
    :return:
    """

    # Find matches, in each table, against the catalogue.
    match_ids_1, match_ids_cat_1 = match_cat(x_match=table_1[x_col], y_match=table_1[y_col], x_cat=cat[x_cat_col],
                                             y_cat=cat[y_cat_col], tolerance=tolerance)
    match_ids_2, match_ids_cat_2 = match_cat(x_match=table_2[x_col], y_match=table_2[y_col], x_cat=cat[x_cat_col],
                                             y_cat=cat[y_cat_col], tolerance=tolerance)

    # Keep only matches that are in both tables.

    match_ids_both_1 = []
    match_ids_both_2 = []
    match_ids_both_cat = []

    for i, n in enumerate(match_ids_cat_1):
        if n in match_ids_cat_2:
            match_ids_both_cat.append(n)
            match_ids_both_1.append(match_ids_1[i])
            match_ids_both_2.append(match_ids_2[match_ids_cat_2.index(n)])

    matches_both_cat = cat[match_ids_both_cat]
    matches_both_1 = table_1[match_ids_both_1]
    matches_both_2 = table_2[match_ids_both_2]

    # Consolidate tables.

    return table.hstack([matches_both_cat, matches_both_1, matches_both_2],
                        table_names=[cat_name, table_name_1, table_name_2])


def in_all(item, list_of_lists: 'list'):
    """
    Returns a list of indices if the given item is in all of the lists in list_of_lists; False if not.
    :param item: The item to check for.
    :param list_of_lists: a list of lists through which to check.
    :return:
    """
    indices = []
    for l in list_of_lists:
        if item not in l:
            return False
        else:
            indices.append(l.index(item))

    return indices


def numpy_to_list(arr):
    ls = []
    for i in arr:
        ls.append(float(i))
    return ls


def find_nearest(array, value, sorted: bool = False):
    """
    Thanks to this thread: https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array/2566508
    :param array:
    :param value:
    :return:
    """
    if not sorted:
        array.sort()

    if value < array[0]:
        return 0, array[0]
    elif value > array[-1]:
        return len(array) - 1, array[-1]

    idx = np.searchsorted(array, value, side="left")
    if idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx]):
        return idx - 1, array[idx - 1]
    else:
        return idx, array[idx]


def round_to_sig_fig(x: float, n: int):
    """
    https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    :param x: Number to round.
    :param n: Number of significant figures to round to.
    :return:
    """

    return round(x, (n - 1) - int(np.floor(np.log10(abs(x)))))


def wcs_as_deg(ra: str, dec: str):
    """
    Using the same syntax as astropy.coordinates.SkyCoord, converts a string of WCS coordinates into degrees.
    :param ra:
    :param dec:
    :return:
    """

    c = SkyCoord(ra=ra, dec=dec)
    return c.ra.deg, c.dec.deg


def sanitise_file_ext(filename: str, ext: str):
    """
    Checks if the filename has the desired extension; adds it if not and returns the filename.
    :param filename: The filename.
    :param ext: The extension, eg '.fits'
    :return:
    """
    if ext[0] != '.':
        ext = '.' + ext
    length = len(ext)
    if filename[-length:] != ext:
        filename = filename + ext

    return filename


def join_csv(filenames: [str], output: str):
    n = len(filenames)
    tables = []
    for file in filenames:
        if file[-4:] == '.csv':
            tables.append(table.Table.read(file, format='ascii.csv'))
        elif file[-5:] == '.fits':
            tables.append(table.Table(fits.open(file)[1].data))
        else:
            raise ValueError('File format not recognised.')

    output_tbl = table.hstack(tables)
    for col in output_tbl.colnames:
        if col[-2:] == '_1':
            col_new = col[:-2]
            column = output_tbl[col]
            output_tbl.remove_column(col)
            output_tbl[col_new] = column
            for i in range(2, n + 1):
                output_tbl.remove_column(col_new + '_' + str(i))

    print('Writing to', output)
    output_tbl.write(output, format='ascii.csv', overwrite=True)


def extract_xml_param(tag: str, xml_str: str):
    """
    Finds and extracts a single tagged value from an XML file. This will be the first instance of that tag in the file.
    :param tag: The tag of the value.
    :param xml_str: XML-formatted string containing the desired value.
    :return:
    """
    print("Tag:", tag)
    value = xml_str[xml_str.find(f"<{tag}>") + len(tag) + 2:xml_str.find(f"</{tag}>")]
    print("Value:", value)
    return value


def unit_str_to_float(string: str):
    """
    Turns a single string value, with format <number> (<units>), into a float and also returns the units.
    :param string:
    :return:
    """
    print("String:", string)
    i = string.find("(")
    units = string[i + 1:string.find(")")]
    print("Units:", units)
    value = float(string[:string.find("(")])
    print("Value:", value)
    return value, units


def option(options: list, default: str = None):
    for i, opt in enumerate(options):
        print(i, opt)

    selection = None
    picked = None

    while selection is None or picked is None:
        selection = input()
        if selection == "" and default is not None:
            selection = default
        try:
            selection = int(selection)
        except ValueError:
            print("Invalid response. Please enter an integer.")
            continue
        try:
            picked = options[selection]
        except IndexError:
            print(f"Response is not in provided options. Please select an integer from 0 to {len(options) - 1}")
    print(f"You have selected {selection}: {picked}")
    return selection, picked


def enter_time(message: str):
    date = None
    while date is None:
        date = input(message + "\n")
        print()
        try:
            date = Time(date)
        except ValueError:
            print("Date format not recognised. Try again:")
    return date


def select_option(message: str, options: Union[List[str], dict], default: Union[str, int] = None, sort: bool = False):
    """
    Options can be a list of strings, or a dict in which the keys are the options to be printed and the values are the
    represented options; that is, the returned object will be the value represented by the selected key.
    :param message:
    :param options:
    :param default:
    :param sort:
    :return:
    """
    if type(default) is str:
        default = options.index(default)
    if default is not None:
        message += f" [default: {default} {options[default]}]"
    print()
    print(message)

    dictionary = False
    if type(options) is dict:
        dictionary = True
        options_list = []
        for opt in options:
            options_list.append(opt)
        if sort:
            options_list.sort()
    else:
        options_list = options

    if sort:
        options_list.sort()
    selection, picked = option(options=options_list, default=default)
    if dictionary:
        return selection, options[picked]
    else:
        return selection, picked


def select_yn(message: str, default: Union[str, bool] = None):
    message += " (y/n) "
    positive = ['y', 'yes', '1', 'true', 't']
    negative = ['n', 'no', '0', 'false', 'f']
    if default in positive or default is True:
        positive.append("")
        message += f"[default: y]"
    elif default in negative or default is False:
        negative.append("")
        message += f"[default: n]"
    elif default is not None:
        print("Warning: default not recognised. No default value will be used.")
    print(message)
    inp = None
    while inp is None:
        inp = input().lower()
        if inp not in positive and inp not in negative:
            print("Input not recognised. Try again:")
    if inp in positive:
        print("You have selected 'yes'.")
        return True
    else:
        print("You have selected 'no'.")
        return False


def user_input(message: str, typ: type = str, default=None):
    inp = None
    if default is not None:
        if type(default) is not typ:
            try:
                default = typ(default)

            except ValueError:
                print(f"Default ({default}) could not be cast to {typ}. Proceeding without default value.")

        message += f" [default: {default}]"

    print(message)
    while type(inp) is not typ:
        inp = input()
        if inp == "":
            inp = default
        if type(inp) is not typ:
            try:
                inp = typ(inp)
            except ValueError:
                print(f"Could not cast {inp} to {typ}. Try again:")
    print(f"You have entered {inp}.")
    return inp


def scan_nested_dict(dictionary: dict, keys: list):
    value = dictionary
    for key in keys:
        value = value[key]
    return value


def get_scope(lines: list, levels: list):
    print("==============================")
    print(lines)
    print(levels)
    if len(lines) == 1:
        return lines[0]
    this_dict = {}
    this_level = levels[0]
    for i, line in enumerate(lines):
        print()
        if levels[i] == this_level:
            scope_end = 1
            while scope_end < len(levels) and levels[scope_end] >= levels[0]:
                scope_end += 1
            this_dict[line] = get_scope(lines=lines[1:scope_end], levels=levels[1:scope_end])

    return this_dict


def get_pypeit_param_levels(lines: list):
    levels = []
    last_non_zero = 0
    for i, line in enumerate(lines):
        level = line.count("[")
        if level == 0:
            level = levels[last_non_zero] + 1
        else:
            last_non_zero = i
        levels.append(level)
    return levels


def get_pypeit_user_params(file: Union[list, str]):
    if isinstance(file, str):
        with open(file) as f:
            file = f.readlines()

    p_start = file.index("# User-defined execution parameters\n") + 1
    p_end = p_start + 1
    while file[p_end] != "\n":
        p_end += 1

    param_dict = {}
    level = 0
    level_list = []
    level_dict = param_dict
    i = p_start

    levels = get_pypeit_param_levels(lines=file[p_start:p_end])

    for i in range(p_start, p_end):
        line = file[i]
        scope_start = i
        scope_end = i + 1
        while levels[scope_end] > levels[scope_start]:
            scope_end += 1

    while i < p_end:
        pass

        # print("==================================================")
        # line = file[i]
        # previous_level = level
        # level = line.count("[")
        # level_list = level_list[:level]
        # if "#" in line:
        #     line = line[:line.find("#")]
        # line = line.replace(" ", "").replace("\t", "").replace("[", "").replace("]", "").replace("\n", "")
        # if level > previous_level:
        #     level_dict[line] = {}
        #     level_list.append(line)
        # else:
        #     if level == 0:
        #         line, value = line.split("=")
        #         level_dict[line] = value
        #         level = previous_level
        #     else:
        #         level_dict[line] = {}
        #         level_list.append(line)
        # print("line", line)
        # print("level", level, "prevous level", previous_level)
        # print("i", i)
        # print(level_list)
        # print("param_dict")
        # print_nested_dict(param_dict)
        # print("level_dict")
        # print_nested_dict(level_dict)
        #
        # level_dict = scan_nested_dict(dictionary=param_dict, keys=level_list)
        # i += 1
    return param_dict


def print_nested_dict(dictionary, level: int = 0):
    if not isinstance(dictionary, dict):
        raise TypeError("dictionary must be dict")
    for key in dictionary:
        print(level * "\t", key + ":")
        if isinstance(dictionary[key], dict):
            print_nested_dict(dictionary[key], level + 1)
        else:
            print((level + 1) * "\t", dictionary[key])
