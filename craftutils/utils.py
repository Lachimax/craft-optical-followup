# Code by Lachlan Marnoch, 2019-2022

import math
import os
import shutil
import sys
from typing import List, Union, Tuple, Iterable
from datetime import datetime as dt
import subprocess

import numpy as np

import astropy.table as table
import astropy.io.fits as fits
import astropy.units as units
from astropy.coordinates import SkyCoord
from astropy.time import Time

# TODO: Arrange these into some kind of logical order.
# TODO: Also comment.

debug_level = 0


def pad_zeroes(n: int, length: int = 2):
    n_str = str(n)
    while len(n_str) < length:
        n_str = "0" + n_str
    return n_str


def get_git_hash(directory: str, short: bool = False):
    """
    Gets the git version hash.
    Special thanks to: https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
    :return:
    """
    current_dir = os.getcwd()
    os.chdir(directory)
    args = ['git', 'rev-parse', 'HEAD']
    if short:
        args.insert(2, "--short")
    try:
        githash = subprocess.check_output(args).decode('ascii').strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        githash = None
    os.chdir(current_dir)
    return githash


def frame_from_centre(frame, x, y, data):
    left = x - frame
    right = x + frame
    bottom = y - frame
    top = y + frame
    return check_margins(data=data, left=left, right=right, bottom=bottom, top=top)


def check_margins(data, left=None, right=None, bottom=None, top=None, margins: tuple = None):
    """

    :param data:
    :param left:
    :param right:
    :param bottom:
    :param top:
    :param margins: In the order left, right, bottom, top
    :return:
    """
    shape = data.shape

    if margins is not None:
        left, right, bottom, top = margins

    if left is None or left < 0.:
        left = 0
    else:
        left = int(dequantify(left))

    if right is None or right < 0.:
        right = shape[1]
    else:
        right = int(dequantify(right))

    if bottom is None or bottom < 0.:
        bottom = 0
    else:
        bottom = int(dequantify(bottom))

    if top is None or top < 0.:
        top = shape[0]
    else:
        top = int(dequantify(top))

    if right < left:
        raise ValueError('Improper inputs; right is smaller than left.')
    if top < bottom:
        raise ValueError('Improper inputs; top is smaller than bottom.')

    return left, right, bottom, top


def trim_image(
        data,
        left=None,
        right=None,
        bottom=None,
        top=None,
        margins: tuple = None,
        return_margins: bool = False
):
    """

    :param data:
    :param left:
    :param right:
    :param bottom:
    :param top:
    :param margins:
    :return:
    """
    left, right, bottom, top = margins = check_margins(
        data=data,
        left=left, right=right, bottom=bottom, top=top,
        margins=margins
    )
    debug_print(2, "fits_files.trim(): left ==", left, "right ==", right, "bottom ==", bottom, "top ==", top)
    trimmed = data[bottom:top + 1, left:right + 1]
    if return_margins:
        return trimmed, margins
    else:
        return trimmed


def check_iterable(obj):
    try:
        len(obj)
    except TypeError:
        obj = [obj]
    return obj


def theta_range(theta: units.Quantity):
    theta = check_iterable(theta.copy())
    theta = units.Quantity(theta)

    theta[theta > 90 * units.deg] -= 180 * units.deg
    theta[theta < -90 * units.deg] += 180 * units.deg

    return theta


def sanitise_endianness(array: np.ndarray):
    """
    If the data is big endian, swap the byte order to make it little endian. Special thanks to this link:
    https://stackoverflow.com/questions/60161759/valueerror-big-endian-buffer-not-supported-on-little-endian-compiler
    :return: A little-endian version of the input array.
    """
    if array.dtype.byteorder == '>':
        array = array.byteswap().newbyteorder()
    return array


def debug_print(level: int = 1, *args):
    if debug_level >= level:
        print(*args)


def path_or_table(tbl: Union[str, table.QTable, table.Table], load_qtable: bool = True, fmt: str = "ascii.ecsv"):
    if isinstance(tbl, str):
        if load_qtable:
            tbl = table.QTable.read(tbl, format=fmt)
        else:
            tbl = table.Table.read(tbl, format=fmt)
    elif not isinstance(tbl, table.Table):
        raise TypeError(f"tbl must be a string or an astropy Table, not {type(tbl)}")
    return tbl


def write_list_to_file(path: str, file: list):
    # Delete file, to be rewritten.
    rm_check(path)
    # Write file to disk.
    print(f"Writing file to {path}")
    with open(path, 'w') as file_stream:
        file_stream.writelines(file)


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


def check_quantity(
        number: Union[float, int, units.Quantity],
        unit: units.Unit,
        allow_mismatch: bool = True,
        enforce_equivalency: bool = True,
        convert: bool = False
):
    """
    If the passed number is not a Quantity, turns it into one with the passed unit. If it is already a Quantity,
    checks the unit; if the unit is compatible with the passed unit, the quantity is returned unchanged (unless convert
    is True).

    :param number: Quantity (or not) to check.
    :param unit: Unit to check for.
    :param allow_mismatch: If False, even compatible units will not be allowed.
    :param convert: If True, convert compatible Quantity to units unit.
    :return:
    """
    if number is None:
        return None
    if not isinstance(number, units.Quantity):  # and number is not None:
        number *= unit
    elif number.unit != unit:
        if not allow_mismatch:
            raise units.UnitsError(
                f"This is already a Quantity, but with units {number.unit}; units {unit} were specified.")
        elif enforce_equivalency and not (number.unit.is_equivalent(unit)):
            raise units.UnitsError(
                f"This number is already a Quantity, but with incompatible units ({number.unit}); units {unit} were specified.")
        elif convert:
            number = number.to(unit)
    return number


def dequantify(number: Union[float, int, units.Quantity], unit: units.Unit = None):
    """
    Removes the unit from an astropy Quantity, or returns the number unchanged if it is not a Quantity.
    If a unit is provided, and number is a Quantity, an attempt will be made to convert the number to that unit before
    returning the value.
    :param number:
    :param unit:
    :return:
    """
    if isinstance(number, units.Quantity):
        if unit is not None:
            number = check_quantity(number=number, unit=unit, convert=True)
        return number.value
    else:
        return number


def sort_dict_by_value(dictionary: dict):
    sorted_keys = sorted(dictionary, key=dictionary.get)
    sorted_dict = {}
    for f in sorted_keys:
        sorted_dict[f] = dictionary[f]
    return sorted_dict


def is_path_absolute(path: str):
    """
    Returns True if path begins with / and False otherwise.
    :param path:
    :return:
    """
    return path[0] == "/"


def make_absolute_path(higher_path: str, path: str):
    """
    If a path is not absolute, makes it into one within higher_path. Higher_path should be absolute.
    :param path:
    :param higher_path:
    :return:
    """
    if not is_path_absolute(higher_path):
        raise ValueError(f"higher_path {higher_path} is not absolute.")
    if not is_path_absolute(path):
        path = os.path.join(higher_path, path)
    return path


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


def world_angle_se_to_pu(
        theta: Union[units.Quantity, float],
        rot_angle: Union[units.Quantity, float] = 0 * units.deg
):
    """
    Converts a Source Extractor world angle to a photutils (relative to image) angle.
    :param theta: world angle, in degrees.
    :param rot_angle: rotation angle of the image.
    :return:
    """
    theta = check_quantity(theta, units.deg)
    rot_angle = check_quantity(rot_angle, units.deg)
    return theta.to(units.radian).value + rot_angle.to(units.radian).value


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


def rmtree_check(path):
    """
    Checks if a directory exists, and removes it if so. USE WITH CAUTION; WILL DELETE ENTIRE TREE WITHOUT WARNING.
    :param path:
    :return:
    """
    if os.path.isdir(path):
        shutil.rmtree(path)


def mkdir_check(*paths: str):
    """
    Checks if a directory exists; if not, creates it.
    :param paths:
    :return:
    """
    for path in paths:
        if not os.path.isdir(path):
            debug_print(2, f"Making directory {path}")
            os.mkdir(path)
        else:
            debug_print(2, f"Directory {path} already exists, doing nothing.")


def mkdir_check_nested(path: str, remove_last: bool = True):
    """
    Does mkdir_check, but for all parent directories of the given path.
    :param path:
    :return:
    """
    path_orig = path
    levels = []
    while len(path) > 1:
        path, end = os.path.split(path)
        levels.append(end)
    levels.append(path)
    levels.reverse()
    if remove_last:
        levels.pop()
    debug_print(2, "utils.mkdir_check_nested(): levels ==", levels)
    mkdir_check_args(*levels)
    # mkdir_check(path_orig)


def move_check(origin: str, destination: str):
    if os.path.exists(origin):
        mkdir_check_nested(destination)
        shutil.move(origin, destination)


def mkdir_check_args(*args: str):
    path = ""
    for arg in args:
        path = os.path.join(path, arg)
        mkdir_check(path)
    return path


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

        debug_print(2, "uncertainty_product(): uncertainty, measurement ==", uncertainty, measurement)
        variance_pre += (uncertainty / measurement) ** 2
    sigma_pre = np.sqrt(variance_pre)
    sigma = np.abs(value) * sigma_pre
    return sigma


def uncertainty_sum(*args):
    variance = 0.
    for uncertainty in args:
        variance += uncertainty ** 2
    sigma = np.sqrt(variance)
    return sigma


def uncertainty_log10(arg: float, uncertainty_arg: float, a: float = 1.):
    """
    Calculates standard uncertainty for function of the form a * log10(arg)
    :return:
    """
    return np.abs(a * uncertainty_arg / (arg * np.log(10)))


def uncertainty_func(arg, err, func=lambda x: np.log10(x), absolute=False):
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


def uncertainty_func_percent(arg, err, func=lambda x: np.log10(x)):
    measurement, error_plus, error_minus = uncertainty_func(arg=arg, err=err, func=func, absolute=False)
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


def std_err_slope(
        y_model: np.ndarray,
        y_obs: np.ndarray,
        x_obs: np.ndarray,
        y_weights: np.ndarray = None,
        x_weights: np.ndarray = None,
        dof_correction: int = 2,
):
    """
    https://sites.chem.utoronto.ca/chemistry/coursenotes/analsci/stats/ErrRegr.html
    https://www.statology.org/standard-error-of-regression-slope/
    :param y_model:
    :param y_obs:
    :param y_weights:
    :return:
    """

    s_regression = root_mean_squared_error(
        model_values=y_model,
        obs_values=y_obs,
        weights=y_weights,
        dof_correction=dof_correction
    )

    if x_weights is None:
        x_weights = 1
    else:
        x_weights = x_weights / np.linalg.norm(x_weights, ord=1)
        n = 1

    x_mean = np.nanmean(x_obs)
    s = s_regression / np.sqrt(np.nansum(x_weights * (x_obs - x_mean)) ** 2)
    return s


def std_err_intercept(
        y_model: np.ndarray,
        y_obs: np.ndarray,
        x_obs: np.ndarray,
        y_weights: np.ndarray = None,
        x_weights: np.ndarray = None,
        dof_correction: int = 2,
):
    """
    https://sites.chem.utoronto.ca/chemistry/coursenotes/analsci/stats/ErrRegr.html
    :return:
    """
    s_regression = root_mean_squared_error(
        model_values=y_model,
        obs_values=y_obs,
        weights=y_weights,
        dof_correction=dof_correction
    )

    n = len(x_obs)
    x_mean = np.nanmean(x_obs)

    if x_weights is None:
        x_weights = 1
    else:
        x_weights = x_weights / np.linalg.norm(x_weights, ord=1)
        n = 1

    s = s_regression * np.sqrt(np.nansum(x_weights * x_obs ** 2) / (n * np.nansum((x_obs - x_mean) ** 2)))
    return s


def mean_squared_error(model_values, obs_values, weights=None, dof_correction: int = 2):
    """
    Weighting from https://stats.stackexchange.com/questions/230517/weighted-root-mean-square-error
    :param model_values:
    :param obs_values:
    :param weights:
    :param quiet:
    :param dof_correction: see https://sites.chem.utoronto.ca/chemistry/coursenotes/analsci/stats/ErrRegr.html
    :return:
    """
    if len(model_values) != len(obs_values):
        raise ValueError("Arrays must be the same length.")
    n = len(model_values)
    if weights is None:
        weights = 1
    else:
        weights = weights / np.linalg.norm(weights, ord=1)
        n = 1
        dof_correction = 0

    return (1 / (n - dof_correction)) * np.nansum(weights * (obs_values - model_values) ** 2)


def root_mean_squared_error(
        model_values: np.ndarray,
        obs_values: np.ndarray,
        weights: np.ndarray = None,
        dof_correction: int = 2
):
    mse = mean_squared_error(
        model_values=model_values,
        obs_values=obs_values,
        weights=weights,
        dof_correction=dof_correction
    )
    return np.sqrt(mse)


def detect_problem_table(tbl: table.Table, fmt: str = "ecsv"):
    for i, row in enumerate(tbl):
        tbl_this = tbl[:i + 1]
        try:
            writepath = os.path.join(os.path.expanduser("~"), f"test.{fmt}")
            tbl_this.write(writepath, overwrite=True, format=fmt)
            os.remove(writepath)
        except NotImplementedError:
            print("Problem row:")
            print(i, row)
            return i, row
        except ValueError:
            print("Problem row:")
            print(i, row)
            return i, row


def mode(lst: list):
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
    """
    Returns closest match to given coordinates from the given search space.
    :param x: x-coordinate to find
    :param y: y-coordinate to find
    :param x_search: array to search for x-coordinate
    :param y_search: array to search for y-coordinate
    :return: id (int), distance (float)
    """
    if len(x_search) != len(y_search):
        raise ValueError('x_search and y_search must be the same length.')
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


def round_to_sig_fig(x: float, n: int) -> float:
    """
    https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    :param x: Number to round.
    :param n: Number of significant figures to round to.
    :return: Rounded number
    """

    return round(x, (n - 1) - int(np.floor(np.log10(abs(x)))))


def round_decimals_up(number: float, decimals: int = 2):
    """
    Returns a value rounded up to a specific number of decimal places.
    Taken from https://kodify.net/python/math/round-decimals/#round-decimal-places-up-in-python
    """

    if decimals < 0:
        raise ValueError(f"decimal places has to be 0 or more (received {decimals})")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


def uncertainty_string(
        value: float,
        uncertainty: float,
        n_digits_err: int = 1,
        unit: units.Unit = None,
        brackets: bool = True,
        limit_val: int = None,
        limit_type: str = "upper",
        nan_string: str = "--"
):
    limit_vals = (limit_val, -99, -999)
    value = dequantify(value, unit)
    uncertainty = dequantify(uncertainty, unit)
    if limit_type == "upper":
        limit_char = "<"
    else:
        limit_char = ">"

    # If we have an upper limit, set uncertainty to blank
    if uncertainty in limit_vals:
        uncertainty = nan_string
        precision = 1
    else:
        precision = np.log10(uncertainty)
        if precision < 0:
            precision = int(-precision + n_digits_err)
        else:
            precision = n_digits_err
        uncertainty = round_decimals_up(uncertainty, abs(precision))

    if value in limit_vals:
        value = nan_string
    else:
        value = np.round(value, precision)

    if uncertainty == nan_string:
        if value != nan_string:
            this_str = f"${limit_char} {value}$"
        else:
            this_str = "--"
    else:
        val_rnd = str(value)
        while len(val_rnd[val_rnd.find("."):]) < precision + 1:
            val_rnd += "0"

        if brackets:
            uncertainty_digit = str(uncertainty)[-n_digits_err:]
            this_str = f"${val_rnd}({uncertainty_digit})$"
        else:
            this_str = f"${val_rnd} \\pm {uncertainty}$"

    return this_str, value, uncertainty


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


def select_option(message: str,
                  options: Union[List[str], dict],
                  default: Union[str, int] = None,
                  sort: bool = False) -> tuple:
    """
    Options can be a list of strings, or a dict in which the keys are the options to be printed and the values are the
    represented options. The returned object is a tuple, with the first entry being the number given by the user and
    the second entry being the corresponding option. If a dict is passed to options, the second tuple entry will be the
    dict value.
    :param message: Message to display before options.
    :param options: Options to display.
    :param default: Option to return if no user input is given.
    :param sort: Sort options?
    :return: Tuple containing (user input, selection)
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


def select_yn_exit(message: str):
    options = ["No", "Yes", "Exit"]
    opt, _ = select_option(message=message, options=options)
    if opt == 0:
        return False
    if opt == 1:
        return True
    if opt == 2:
        exit(0)


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


def bucket_mode(data: np.ndarray, precision: int):
    """
    With help from https://www.statology.org/numpy-mode/
    :param data:
    :param precision:
    :return:
    """
    vals, counts = np.unique(np.round(data, precision), return_counts=True)
    return vals[counts == np.max(counts)]


def scan_nested_dict(dictionary: dict, keys: list):
    value = dictionary
    for key in keys:
        value = value[key]
    return value


def print_nested_dict(dictionary, level: int = 0):
    if not isinstance(dictionary, dict):
        raise TypeError("dictionary must be dict")
    for key in dictionary:
        print(level * "\t", key + ":")
        if isinstance(dictionary[key], dict):
            print_nested_dict(dictionary[key], level + 1)
        else:
            print((level + 1) * "\t", dictionary[key])


def system_command(
        command: str, arguments: Union[str, list] = None,
        suppress_print: bool = False,
        error_on_exit_code: bool = True,
        force_single_dash: bool = False,
        *flags,
        **params
):
    if command in [""]:
        raise ValueError("Empty command.")
    if " " in command:
        raise ValueError("Command contains spaces.")
    sys_str = command
    if arguments is not None:
        if isinstance(arguments, str):
            arguments = [arguments]
        for argument in arguments:
            sys_str += f" {argument}"
    for param in params:
        debug_print(2, "utils.system_command(): flag ==", param, "len", len(param))
        if len(param) == 1 or force_single_dash:
            sys_str += f" -{param} {params[param]}"
        elif len(param) > 1:
            sys_str += f" --{param} {params[param]}"
    for flag in flags:
        debug_print(2, "utils.system_command(): flag ==", flag, "len", len(flag))
        if len(flag) == 1:
            sys_str += f" -{flag}"
        elif len(flag) > 1:
            sys_str += f" --{flag}"

    return system_command_verbose(command=sys_str, suppress_print=suppress_print, error_on_exit_code=error_on_exit_code)


def system_command_verbose(
        command: str,
        suppress_print: bool = False,
        error_on_exit_code: bool = True,
        suppress_path: bool = False
):
    if not suppress_print:
        print()
        if not suppress_path:
            print("In:", os.getcwd())
        print("Executing:")
        print(command)
        print()
    result = os.system(command)
    if result != 0 and error_on_exit_code:
        raise SystemError(f"System command failed with exit code {result}")
    if not suppress_print:
        print()
        print("Finished:")
        print(command)
        print("With code", result)
        print()
    return result


def system_package_version(package_name: str):
    verstring = subprocess.getoutput(f"dpkg -s {package_name} | grep -i version")
    if verstring.startswith("Version: "):
        return verstring[9:]
    else:
        return None


def classify_spread_model(
        cat: table.Table,
        cutoffs: Tuple[float, float, float, float] = (-0.005, 0.005, 0.003, 0.003),
        sm_col: str = "SPREAD_MODEL",
        sm_err_col: str = "SPREADERR_MODEL",
        class_flag_col: str = "CLASS_FLAG"
):
    if sm_col not in cat.colnames:
        print(sm_col, "not found in catalogue columns.")
        return None
    cat[class_flag_col] = (
            ((cat[sm_col] + 3 * cat[sm_err_col]) > cutoffs[1]).astype(int)
            + ((cat[sm_col] + cat[sm_err_col]) > cutoffs[2]).astype(int)
            + ((cat[sm_col] - cat[sm_err_col]) > cutoffs[3]).astype(int)
    )

    cat[class_flag_col][(cat[sm_col] + cat[sm_col]) < cutoffs[0]] = -1

    return cat


def trim_to_class(
        cat: table.Table,
        allowed: Iterable = [0],
        modify: bool = True,
        classify_kwargs: dict = {},
):
    cat = classify_spread_model(cat, **classify_kwargs)
    if cat is None:
        return None
    if "class_flag_col" in classify_kwargs:
        star_class_col = classify_kwargs["class_flag_col"]
    else:
        star_class_col = "CLASS_FLAG"
    good = []
    for row in cat:
        good.append(row[star_class_col] in allowed)
    if modify:
        cat = cat[good]
        return cat
    else:
        return good
