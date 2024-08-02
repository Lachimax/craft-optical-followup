# Code by Lachlan Marnoch, 2019-2022
import numbers

import datetime
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
from astropy.coordinates import SkyCoord, Longitude, Latitude
from astropy.time import Time

# TODO: Arrange these into some kind of logical order.
# TODO: Also comment.

debug_level = 0


def export(obj):
    """
    A function used for decorating those objects which may be exported from a file.
    Taken from `a solution posted by mhostetter <https://github.com/jbms/sphinx-immaterial/issues/152>`_

    :param obj: The object to be added to __all__
    :return:
    """
    # Determine the private module that defined the object
    module = sys.modules[obj.__module__]

    # Set the object's module to the package name. This way the REPL will display the object
    # as craftutils.obj and not craftutils._private_module.obj
    obj.__module__ = "craftutils"

    # Append this object to the private module's "all" list
    public_members = getattr(module, "__all__", [])
    public_members.append(obj.__name__)
    setattr(module, "__all__", public_members)

    return obj


@export
def pad_zeroes(n: int, length: int = 2):
    """

    :param n:
    :param length:
    :return:
    """
    n_str = str(n)
    while len(n_str) < length:
        n_str = "0" + n_str
    return n_str


def check_time(time, fmt: str = None):
    if isinstance(time, datetime.date):
        time = str(time)
    time = Time(time, format=fmt)
    return time


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
    """
    Given the coordinates for a centre and the padding around that centre, generates the x coordinate for the left and
    right and the y coordinate for the bottom and top of the described rectangular cutout of the data.
    :param frame:
    :param x:
    :param y:
    :param data:
    :return: (x_left, x_right, y_bottom, y_top)
    """

    if x is None:
        n_y, n_x = data.shape
        x = int(dequantify(n_x / 2))
    if y is None:
        n_y, n_x = data.shape
        y = int(dequantify(n_y / 2))

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
    :param margins: In the order x_left, x_right, y_bottom, y_top
    :return: (x_left, x_right, y_bottom, y_top)
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
    if not is_iterable(obj):
        obj = [obj]
    return obj


def is_iterable(obj):
    try:
        len(obj)
        return True
    except TypeError:
        return False


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


def check_dict(
        key: str,
        dictionary: dict,
        na_values: Union[tuple, list] = (None),
        fail_val=None
):
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
        unit: Union[str, units.Unit],
        allow_mismatch: bool = True,
        enforce_equivalency: bool = True,
        convert: bool = False,
        equivalencies: Union[List[Tuple], units.Equivalency] = (),
):
    """
    If the passed number is not a Quantity, turns it into one with the passed unit. If it is already a Quantity,
    checks the unit; if the unit is compatible with the passed unit, the quantity is returned unchanged (unless convert
    is True).

    :param number: value (or not) to check.
    :param unit: Unit to check for.
    :param enforce_equivalency: If `True`, and if `allow_mismatch` is True, a `units.UnitsError` will be raised if the
        `number` has units that are not equivalent to `unit`.
        That is, set this (and `allow_mismatch`) to `True` if you want to ensure `number` has the same
        dimensionality as `unit`, but not necessarily the same units. Savvy?
    :param convert: If `True`, convert compatible `Quantity` to units `unit`.
    :param allow_mismatch: If False, even compatible but mismatched units will not be allowed; ie, the unit of the
        quantity must match the one specified in the "unit" parameter.
    :return: number as Quantity with specified unit.
    """
    if number is None:
        return None
    if not isinstance(number, units.Quantity):  # and number is not None:
        number *= unit
    elif number.unit != unit:
        if not allow_mismatch:
            raise units.UnitsError(
                f"This is already a Quantity, but with units {number.unit}; units {unit} were specified.")
        elif enforce_equivalency and not (number.unit.is_equivalent(unit)) and not equivalencies:
            raise units.UnitsError(
                f"This number is already a Quantity, but with incompatible units ({number.unit}); units {unit} were specified. equivalencies ==",
                equivalencies)
        elif convert:
            number = number.to(unit, equivalencies=equivalencies)
    return number


def dequantify(
        number: Union[float, int, units.Quantity],
        unit: units.Unit = None,
        equivalencies: Union[List[Tuple], units.Equivalency, Tuple] = (),
) -> float:
    """
    Removes the unit from an astropy Quantity, or returns the number unchanged if it is not a Quantity.
    If a unit is provided, and number is a Quantity, an attempt will be made to convert the number to that unit before
    returning the value.
    :param number: value to strip.
    :param unit: unit to check for.
    :param equivalencies: List of Equivalency objects to pass to to() function for conversion.
    :return: number that has been stripped of its units, if present.
    """
    if isinstance(number, units.Quantity):
        if unit is not None:
            number = check_quantity(number=number, unit=unit, convert=True, equivalencies=equivalencies)
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
    :param paths: each argument is a path to check and create.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


def mkdir_check_nested(
        path: str,
        remove_last: bool = True
) -> str:
    """
    Does mkdir_check, but for all parent directories of the given path.
    That is, for all of the levels of the given path, a directory will be created if it doesn't exist.
    :param path: path to check and create
    :param remove_last: If True, does not create the given path itself, only its parent directories.
        Useful if the path will in fact be that of a file that you just want to create a directory for.
    :return:
    """

    if remove_last:
        path, end = os.path.split(path)
    os.makedirs(path, exist_ok=True)
    return path


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
    """Each arg should be a tuple, in which the first entry is the measurement and the second entry is the uncertainty in
    that measurement. These may be in the form of numpy arrays or table columns.
    """
    if None in args:
        raise TypeError("A 'None' has been passed as an arg.")
    if value is None:
        raise TypeError("value is None.")
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
        try:
            variance_pre += (uncertainty / measurement) ** 2
        except units.UnitConversionError:
            raise units.UnitConversionError(
                f"uncertainty {uncertainty} and measurement {measurement} have units that do not match.")
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


def uncertainty_func(arg, err, func=np.log10):
    """

    :param arg:
    :param err: The error in the argument of the function.
    :param func:
    :param absolute:
    :return:
    """
    measurement = func(arg)
    # One of these should come out negative - that becomes the minus error, and the positive the plus error.
    error_plus = func(arg + err) - measurement
    error_minus = func(arg - err) - measurement

    error_plus = np.abs(error_plus)
    error_minus = np.abs(error_minus)

    return measurement, error_plus, error_minus


def uncertainty_func_percent(arg, err, func=np.log10):
    measurement, error_plus, error_minus = uncertainty_func(arg=arg, err=err, func=func, absolute=False)
    return np.array([error_plus / measurement, error_minus / measurement])


def uncertainty_sin(theta, sigma_theta, a=1., b=1.):
    return np.abs(a * b * np.cos(theta) * sigma_theta)


def uncertainty_cos(theta, sigma_theta, a=1., b=1.):
    return np.abs(a * b * np.sin(theta) * sigma_theta)


def uncertainty_power(x, power, sigma_x, a=1.):
    f = a * x ** power
    return np.abs(f * power * sigma_x / x)


def great_circle_dist(ra_1, dec_1, ra_2, dec_2):
    delta_ra = ra_2 - ra_1
    term_1 = np.sin(dec_1) * np.sin(dec_2)
    term_2 = np.cos(dec_1) * np.cos(dec_2) * np.cos(delta_ra)
    x = term_1 + term_2
    s = np.arccos(x).to("arcsec")
    return s



def inclination(
        axis_ratio: float,
        q_0: Union[float, str] = 0.2,
        uncos: bool = True
) -> units.Quantity:
    """Using the power of geometry, loosely estimates the inclination angle of a disk galaxy.

    :param axis_ratio: Axis ratio b/a of the galaxy.
    :param q_0: Axis ratio if viewed fully edge-on.
    :return: Inclination angle in degrees.
    """
    if uncos:
        return (np.arccos(np.sqrt((axis_ratio ** 2 - q_0 ** 2) / (1 - q_0 ** 2))) * units.rad).to(units.deg)
    else:
        return np.sqrt((axis_ratio ** 2 - q_0 ** 2) / (1 - q_0 ** 2))

def deprojected_offset(
        object_coord: SkyCoord,
        galaxy_coord: SkyCoord,
        position_angle: Union[units.Quantity, float],
        inc: Union[units.Quantity, float]
):
    """

    :param object_coord:
    :param galaxy_coord:
    :param position_angle:
    :param inc:
    :return:
    """

    position_angle = check_quantity(position_angle, units.deg)
    inc = check_quantity(inc, units.deg)

    x_frb = (object_coord.ra - galaxy_coord.ra) * np.cos(galaxy_coord.dec)
    y_frb = object_coord.dec - galaxy_coord.dec

    u = x_frb * np.sin(position_angle) + y_frb * np.cos(position_angle)
    v = x_frb * np.cos(position_angle) - y_frb * np.sin(position_angle)
    return np.sqrt(u ** 2 + (v / np.cos(inc)) ** 2).to("arcsec")

def get_column_names(path, delimiter=','):
    with open(path) as f:
        names = f.readline().split(delimiter)
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


def detect_problem_row(
        tbl: table.Table,
        fmt: str = "ascii.ecsv",
        remove_output: bool = True
) -> Tuple[int, table.Row]:
    """
    This function iterates through the rows of an astropy Table and attempts to write each one to disk;
    if it fails on one, the row's index and the row itself are returned.

    :param tbl: The Table (or subclass) to check.
    :param fmt: The format to attempt to write; different formats may have trouble with different data types.
    :param remove_output: auto-delete output tables.
    :return: index, row
    """
    i = None
    row = None
    for i, row in enumerate(tbl):
        print(f"Writing up to row {i}")
        tbl_this = tbl[:i + 1]
        try:
            writepath = os.path.join(os.path.expanduser("~"), f"test.{fmt}")
            tbl_this.write(writepath, overwrite=True, format=fmt)
            if remove_output:
                os.remove(writepath)
        except NotImplementedError:
            print("Problem row (NotImplementedError):")
            print(i, row)
            _problem_row(i, row, tbl)
            return i, row
        except ValueError:
            print("Problem row (ValueError): row", i)
            print(row)
            _problem_row(i, row, tbl)
            return i, row


def _problem_row(i, row, tbl):
    problem_values = {}
    j = i - 1
    other_row = tbl[j]
    print(tbl[[i, j]])
    for col in tbl.colnames:
        if type(other_row[col]) is not type(row[col]):
            problem_values[col] = (row[col], other_row[col])
    print(len(problem_values))
    for name, (val, other_val) in problem_values.items():
        print(name, val, type(val), other_val, type(other_val))


def detect_problem_column(
        tbl: table.Table,
        fmt: str = "ascii.ecsv",
        remove_output: bool = True
):
    print(type(tbl))
    colnames = tbl.colnames
    for i, col in enumerate(colnames):
        colnames_trunc = colnames[:i]
        tbl_this = tbl[colnames_trunc]
        try:
            writepath = os.path.join(os.path.expanduser("~"), f"test.{fmt}")
            tbl_this.write(writepath, overwrite=True, format=fmt)
            if remove_output:
                os.remove(writepath)
        except NotImplementedError:
            print(tbl_this)
            print("Problem column (NotImplementedError):")
            print(tbl[col])
            return col, tbl[col]
        except ValueError:
            print(tbl_this)
            print("Problem column (ValueError):")
            print(tbl[col])
            return col, tbl[col]
    return None, None


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


def find_nearest(array, value):
    """
    :param array:
    :param value:
    :return:
    """
    idx = np.nanargmin(np.abs(array - value))
    return idx, array[idx]


def round_to_sig_fig(x: float, n: int) -> float:
    """
    https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    :param x: Number to round.
    :param n: Number of significant figures to round to.
    :return: Rounded number
    """
    print(x, type(x))
    return round(x, (n - 1) - int(np.floor(np.log10(abs(x)))))


def round_decimals_up(number: float, decimals: int = 2):
    """
    Returns a value rounded up to a specific number of decimal places.
    Taken from https://kodify.net/python/math/round-decimals/#round-decimal-places-up-in-python
    """

    if decimals < 0:
        raise ValueError(f"decimal places has to be 0 or more (received {decimals})")
    elif decimals == 0:
        print(f"\t\t{decimals=}, returning {math.ceil(number)=}")
        return math.ceil(number)
    elif number > 1:
        print(f"\t\t{number=}>1, returning {np.round(number, decimals)=}")
        return np.round(number, decimals)
    else:
        factor = 10 ** decimals
        print(f"\t\t{number=}, factor={10 ** decimals=}, returning {math.ceil(number * factor) / factor=}")
        return math.ceil(number * factor) / factor


def uncertainty_string(
        value: Union[float, units.Quantity],
        uncertainty: Union[float, units.Quantity],
        n_digits_err: int = 1,
        n_digits_no_err: int = 1,
        n_digits_lim: int = None,
        unit: units.Unit = None,
        brackets: bool = True,
        limit_val: int = None,
        limit_type: str = "upper",
        nan_string: str = "--",
        include_uncertainty: bool = True
):
    limit_vals = (limit_val, -99, -999, -999.)
    value = float(dequantify(value, unit))
    if value in limit_vals or np.ma.is_masked(value) or np.isnan(value):
        return nan_string, value, uncertainty
    if np.ma.is_masked(uncertainty):
        uncertainty = 0.

    uncertainty = float(dequantify(uncertainty, unit))
    if limit_type == "upper":
        limit_char = "<"
    else:
        limit_char = ">"

    value_str = str(value)
    uncertainty_str = str(abs(uncertainty))

    # Find the decimal point in the value.
    v_point = value_str.find(".")
    # If there isn't one, there's an imaginary one at the end of the string.
    if v_point == -1:
        v_point = len(value_str)

    if uncertainty in limit_vals:
        if n_digits_lim:
            # Account for the decimal point
            if v_point < n_digits_lim:
                n_digits_lim += 1
                x = 0
            else:
                x = v_point - n_digits_lim
            value_str = value_str[:n_digits_lim] + "0" * x

        return f"${limit_char} {value_str}$", value, uncertainty

    def deal_with_e(string, val):
        if "e" in string:
            if abs(val) < 1:
                m = -int(np.log10(val) - 1)
            else:
                m = 10
            return f"{val:.{m}f}"
        else:
            return string

    uncertainty_str = deal_with_e(uncertainty_str, uncertainty)
    value_str = deal_with_e(value_str, value)
    if uncertainty == 0.:
        if isinstance(n_digits_no_err, int):
            value_str = f"${np.round(float(value_str), n_digits_err)}$"
        return value_str, value, uncertainty

    # Find the decimal point in the uncertainty.
    u_point = uncertainty_str.find(".")
    if u_point == -1:
        u_point = len(uncertainty_str)
    # If the uncertainty is less than 1, we iterate along the string starting at the decimal place until we find a non-zero character.
    if uncertainty < 1.:
        i = u_point + 1
        while uncertainty_str[i] == "0":
            i += 1
        # x is the number of digits after the decimal point to show.
        x = i - u_point + n_digits_err
        # Round appropriately
        uncertainty_rnd = np.round(uncertainty, x - 1)
        uncertainty_str = deal_with_e(str(uncertainty_rnd), uncertainty_rnd)[:u_point + x]

        value_rnd = np.round(value, x - 1)
        value_str = str(value_rnd)[:v_point + x]
        # uncertainty_str = str(uncertainty_rnd)[:u_point + x]

        while len(uncertainty_str) < i + n_digits_err:
            uncertainty_str += "0"
        u_dp = len(uncertainty_str) - u_point
        v_dp = len(value_str) - v_point
        while v_dp < u_dp:
            value_str += "0"
            v_dp = len(value_str) - v_point

    else:
        # Here x is the number of digits before the decimal point to set to zero.
        if u_point < n_digits_err:
            # Account for the decimal point
            n_digits_err += 1
        x = u_point - n_digits_err
        uncertainty_rnd = np.round(uncertainty, -x)
        value_rnd = np.round(value, -x)
        value_str = str(value_rnd)[:v_point - x] + "0" * x
        uncertainty_str = str(uncertainty_rnd)[:n_digits_err] + "0" * x

    if not include_uncertainty:
        value_str = value_str
    elif brackets:
        if uncertainty < 1.:
            uncertainty_str = uncertainty_str[-n_digits_err:]
        else:
            x = u_point - n_digits_err
            uncertainty_str = uncertainty_str[:n_digits_err] + "0" * x
        value_str = f"${value_str}({uncertainty_str})$"
    else:
        value_str = f"${value_str} \\pm {uncertainty_str}$"

    return value_str, value_rnd, uncertainty_rnd


def uncertainty_str_coord(
        coord: SkyCoord,
        uncertainty_ra: Union[float, units.Quantity],
        uncertainty_dec: Union[float, units.Quantity],
        n_digits_err: int = 2,
        brackets: bool = True,
        ra_err_seconds: bool = False
):
    print("=" * 100)
    print(coord)
    print(coord.to_string("hmsdms"))
    print(uncertainty_ra)
    uncertainty_ra_s = Longitude(uncertainty_ra).hms.s
    ra_s_str, ra_s_rounded, ra_s_unc_rounded = uncertainty_string(
        value=coord.ra.hms.s,
        uncertainty=uncertainty_ra_s,
        n_digits_err=n_digits_err,
        brackets=brackets
    )
    ra_arcsec_str, ra_arcsec_rounded, ra_arcsec_unc_rounded = uncertainty_string(
        value=coord.ra.to("arcsec"),
        uncertainty=uncertainty_ra.to("arcsec"),
        n_digits_err=n_digits_err,
        brackets=brackets
    )
    ra_s_str = ra_s_str.replace("$", "")
    ra_str = coord.ra.to_string("h", format="latex")
    s_i = ra_str.find(r"{m}}") + 4
    s_2 = ra_str[s_i:]
    e_i = s_2.find("^") + s_i
    ra_replace = ra_str[s_i:e_i]
    if ra_err_seconds:
        ra_uncertainty_str = ra_str.replace(ra_replace, ra_s_str)
    else:

        ra_uncertainty_str = f"{ra_str.replace(ra_replace, str(ra_s_rounded))[:-1]} \pm ({ra_arcsec_unc_rounded}" + r"^{\prime\prime})$"
    dec = coord.dec
    dec_s_str, dec_rounded, dec_unc_rounded = uncertainty_string(
        value=abs(coord.dec.dms.s),
        uncertainty=uncertainty_dec,
        n_digits_err=n_digits_err,
        brackets=brackets
    )
    dec_s_str = dec_s_str.replace("$", "")

    dec_str = dec.to_string(format="latex")
    s_i = dec_str.find("\prime") + 6
    s_2 = dec_str[s_i:]
    e_i = s_2.find("{}^") + s_i
    dec_replace = dec_str[s_i:e_i]
    dec_uncertainty_str = dec_str.replace(dec_replace, dec_s_str)

    return ra_uncertainty_str, dec_uncertainty_str


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


def option(
        options: list,
        default: str = None,
        allow_text_entry: bool = True
):
    for i, opt in enumerate(options):
        print(i, opt)

    selection = None
    picked = None

    while selection is None or picked is None:
        selection = input()
        if selection == "" and default is not None:
            selection = default
        if selection.isnumeric():
            selection = int(selection)
        # The user may type their option instead of making a numeric selection.
        elif allow_text_entry and selection in options:
            picked = selection
            selection = options.index(picked)
            return selection, picked
        else:
            if allow_text_entry:
                print("Invalid response. Please enter an integer, or type a valid option.")
            else:
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


def select_option(
        message: str,
        options: Union[List[str], dict],
        default: Union[str, int] = None,
        sort: bool = False,
        include_exit: bool = True,
        allow_text_entry: bool = True
) -> tuple:
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
    if include_exit:
        options_list.append("Exit")

    selection, picked = option(options=options_list, default=default, allow_text_entry=allow_text_entry)
    if include_exit and picked == "Exit":
        exit()
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
    options = ["No", "Yes"]
    opt, _ = select_option(message=message, options=options, include_exit=True)
    if opt == 0:
        return False
    if opt == 1:
        return True


def user_input(message: str, input_type: type = str, default=None):
    inp = None
    if default is not None:
        if type(default) is not input_type:
            try:
                default = input_type(default)

            except ValueError:
                print(f"Default ({default}) could not be cast to {input_type}. Proceeding without default value.")

        message += f" [default: {default}]"

    print(message)
    while type(inp) is not input_type:
        inp = input()
        if inp == "":
            inp = default
        if type(inp) is not input_type:
            try:
                inp = input_type(inp)
            except ValueError:
                print(f"Could not cast {inp} to {input_type}. Try again:")
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
        command: str,
        arguments: Union[str, list] = None,
        suppress_print: bool = False,
        error_on_exit_code: bool = True,
        force_single_dash: bool = False,
        flags: list = (),
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
        else:
            print(len(flag))

    return system_command_verbose(command=sys_str, suppress_print=suppress_print, error_on_exit_code=error_on_exit_code)


def system_command_verbose(
        command: str,
        suppress_print: bool = False,
        error_on_exit_code: bool = True,
        suppress_path: bool = False,
        go_to_working_directory: str = None
):
    """
    A convenience function for executing terminal commands.

    :param command: The full command to send to the terminal.
    :param suppress_print: Do not print command, result, etc. This will not turn off stdout, so the command you run may
        still print to terminal.
    :param error_on_exit_code: If True, will raise a Python error on a non-0 exit code, ie if the command fails.
        If False, the Python code can continue even if the terminal command fails.
    :param suppress_path: If True, the working directory will not print even if `suppress_print` is False.
        `suppress_print=True` overrides.
    :param go_to_working_directory: The path to switch the working directory to while the command executes.
        The old working directory will be returned to upon completion. Useful if, for example,  there are output files
        that get written to the working directory.
    :return:
    """
    cwd = ""
    if go_to_working_directory is not None:
        cwd = os.getcwd()
        os.chdir(go_to_working_directory)
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
    if go_to_working_directory is not None:
        os.chdir(cwd)
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


def split_uncertainty_string(string: str, delim: str = "+/-"):
    value = float(string[:string.find(delim)])
    uncertainty = float(string[string.find(delim) + len(delim):])
    return value, uncertainty


def polar_to_cartesian(
        r: float,
        theta: float,
        centre_x: units.Quantity = 0,
        centre_y: units.Quantity = 0
) -> tuple:
    """Transforms polar (r, theta) coordinate to cartesian (x, y). Works with astropy Quantities, so long as r has the
    same units as centre_x and centre_y and theta has valid angular units.

    :param r: Radial polar coordinate
    :param theta: Angular polar coordinate
    :param centre_x: x coordinate of centre of polar coordinate system
    :param centre_y: y coordinate of centre of polar coordinate system
    :return: x, y with same units as r.
    """
    x = r * np.cos(theta) + centre_x
    y = r * np.sin(theta) + centre_y
    return x, y


def mod_latex_table(
        path: str,
        short_caption: str = None,
        caption: str = None,
        label: str = None,
        longtable: bool = False,
        coltypes: str = None,
        landscape: bool = False,
        sub_colnames: list = None,
        second_path: str = None,
        multicolumn=None
):
    with open(path, 'r') as f:
        file = f.readlines()
    tab_invoc = file[1]
    if coltypes is not None:
        tab_invoc = r"\begin{tabular}{" + coltypes + "}\n"
        file[1] = tab_invoc
    if longtable:
        file[1] = "% " + file[1]

    if sub_colnames is not None:
        under_col_str = ""
        for under_col in sub_colnames:
            under_col_str += under_col + " & "
        under_col_str = under_col_str[:-2]
        under_col_str += r"\\ \hline" + "\n"
        file.insert(3, under_col_str)
    else:
        file[2] = file[2].replace("\n", r"\hline" + "\n")

    if multicolumn is not None:
        multicol_str = ""
        for t in multicolumn:
            multicol_str += r"\multicolumn{" + str(t[0]) + "}{" + str(t[1]) + "}{" + str(t[2]) + "} & "
        multicol_str = multicol_str[:-2] + "\\\\ \n"
        file.insert(2, multicol_str)

    if label is not None:
        if not label.startswith("tab:"):
            label = "tab:" + label
        file.insert(
            1,
            r"\label{" + label + "}\n"
        )
        if longtable:
            file[1] = file[1].replace("\n", r"\\" + "\n")

    if caption is not None:
        if short_caption is None:
            cap_str = r"\caption{" + caption + "}\n"
        else:
            cap_str = r"\caption" + "[" + short_caption + "]{" + caption + "}\n"
        file.insert(
            1,
            cap_str
        )

    if longtable:
        tab_invoc = tab_invoc.replace("tabular", "longtable")
        file[0] = tab_invoc
        file.pop(-1)
        file.pop(-1)
        file.append(r"\end{longtable}" + "\n")
        file.insert(0, r"\begin{singlespace}" + "\n")
        file.append(r"\end{singlespace}" + "\n")

    if landscape:
        file.insert(
            0,
            r"\begin{landscape}" + "\n"
        )
        file.append(
            r"\end{landscape}" + "\n"
        )

    with open(path, 'w') as f:
        f.writelines(file)
    if second_path is not None:
        with open(second_path, 'w') as f:
            f.writelines(file)
    return file


def latexise_table(
        tbl: table.Table,
        column_dict: dict = None,
        output_path: str = None,
        sub_colnames: dict = None,
        exclude_from_unc: list = (),
        round_cols: list = (),
        round_digits: int = 1,
        ra_col: str = None,
        dec_col: str = None,
        ra_err_col: str = None,
        dec_err_col: str = None,
        err_suffix: str = "_err",
        coord_kwargs: dict = None,
        uncertainty_kwargs: dict = None,
        **kwargs
) -> Union[table.Table, List[str]]:
    tbl = tbl.copy()

    # Make appropriate RA and Dec columns
    if ra_col is not None and dec_col is not None:
        if ra_err_col is None:
            ra_err_col = ra_col + err_suffix
        if dec_err_col is None:
            dec_err_col = dec_col + err_suffix
        default_coord_kwargs = dict(
            n_digits_err=1,
            brackets=True,
            ra_err_seconds=False,
        )
        if coord_kwargs is not None:
            default_coord_kwargs.update(coord_kwargs)
        coord_kwargs = default_coord_kwargs

        ra_strs = []
        dec_strs = []
        for row in tbl:
            if row[ra_err_col] > 0:
                ra_str, dec_str = uncertainty_str_coord(
                    coord=SkyCoord(ra=row[ra_col], dec=row[dec_col], unit="deg"),
                    uncertainty_ra=row[ra_err_col].to("arcsec"),
                    uncertainty_dec=row[dec_err_col].to("arcsec"),
                    **coord_kwargs
                )
            else:
                print(row)
                print(row[ra_col])
                ra_str = Longitude(row[ra_col]).to_string("h", format="latex")
                dec_str = Latitude(row[dec_col]).to_string(format="latex")
            ra_strs.append(ra_str)
            dec_strs.append(dec_str)
        tbl[ra_col] = ra_strs
        tbl[dec_col] = dec_strs
        tbl.remove_column(ra_err_col)
        tbl.remove_column(dec_err_col)

    # Get rid of units
    def to_str(v):
        if v in (None, -999., -99.) or not np.isfinite(v):
            return "--"
        else:
            return str(v)
    for col in round_cols:
        new_col = []
        for row in tbl:
            val = dequantify(row[col])
            new_col.append(to_str(val.round(round_digits)))
        tbl[col] = new_col

    # Replace booleans with Y/N
    for col in tbl.colnames:
        if isinstance(tbl[col][0], np.bool_):
            yn = {True: "Y", False: "N"}
            tbl[col] = [yn[b] for b in tbl[col]]

    # Produce combined value(error) strings
    err_colnames = list(filter(lambda c: c.endswith(err_suffix), tbl.colnames))
    for err_col in err_colnames:
        val_col = err_col[:-len(err_suffix)]
        if val_col in exclude_from_unc:
            continue
        # print(colname, do_err_str, err_col, err_col in tbl.colnames)
        new_col = []
        default_unc_kwargs = dict(
            n_digits_lim=3,
            n_digits_err=1,
            n_digits_no_err=None,
            limit_type="upper",
        )
        if uncertainty_kwargs is not None:
            default_unc_kwargs.update(uncertainty_kwargs)
        uncertainty_kwargs = default_unc_kwargs
        for row in tbl:
            this_str, value, uncertainty = uncertainty_string(
                value=row[val_col],
                uncertainty=row[err_col],
                **uncertainty_kwargs
            )
            new_col.append(this_str)

        tbl[val_col] = new_col
        tbl.remove_column(err_col)

    val_cols = list(filter(lambda c: type(tbl[c][0]) in (int, float, np.float_), tbl.colnames))

    for col in val_cols:
        tbl[col] = [to_str(v) for v in tbl[col]]

    # # Add columns for reference
    # if ref_prefix is not None:
    #     ref_colnames = list(filter(lambda c: c.startswith(ref_prefix), tbl.colnames))
    #     for row in tbl:
    #         ref_key =

    # Stick some extra text under column names, e.g. units
    under_list = None
    if sub_colnames is not None:
        under_list = []
        for colname in tbl.colnames:
            if colname in sub_colnames:
                under_list.append(sub_colnames[colname])
            else:
                under_list.append(" ")

    # Rename columns
    if column_dict is not None:
        # Sort by the provided dictionary, then the rest
        # not_in_dict = set(tbl.colnames) - set(column_dict.keys())
        # tbl = tbl[list(column_dict.keys())]  + list(not_in_dict)]
        nems = []
        for original in tbl.colnames:
            if original in column_dict:
                new = column_dict[original]
                tbl[new] = tbl[original]
                tbl.remove_column(original)
                nems.append(new)
            else:
                nems.append(original)
        tbl = tbl[nems]

    # Add various other components to the .tex output
    if output_path is not None:
        tbl.write(output_path, format="ascii.latex", overwrite=True)
        if set(kwargs.keys()).intersection({"caption", "short_caption", "label", "landscape"}):
            tbl = mod_latex_table(path=output_path, sub_colnames=under_list, **kwargs)
    return tbl

def add_stats(
        tbl,
        name_col: str,
        cols_exclude: list,
        exclude_err: Union[bool, str] = True,
        round_n: int = 2
):
    if name_col not in tbl.colnames:
        raise ValueError(f"No column '{name_col}' in table.")
    if exclude_err:
        if not isinstance(exclude_err, str):
            exclude_err = "_err"
        cols_exclude += list(filter(lambda c: c.endswith(exclude_err), tbl.colnames))
    tbl_ = tbl.copy()
    median_dict = {}
    mean_dict = {}
    min_dict = {}
    max_dict = {}
    sigma_dict = {name_col: r"Std. Deviation"}
    median_dict[name_col] = "Median"
    mean_dict[name_col] = "Mean"
    max_dict[name_col] = "Maximum"
    min_dict[name_col] = "\hline Minimum"
    for col in tbl_.colnames:
        if col not in cols_exclude:
            if isinstance(tbl_[col], units.Quantity):
                un = tbl_[col].unit
            else:
                un = 1.
            samp = tbl_[col][tbl_[col] != -999 * un]
            median_dict[col] = np.round(np.nanmedian(samp), round_n)
            mean_dict[col] = np.round(np.nanmean(samp), round_n)
            max_dict[col] = np.round(np.nanmax(samp), round_n)
            min_dict[col] = np.round(np.nanmin(samp), round_n)
            sigma_dict[col] = np.round(np.nanstd(samp), round_n)
            # col_dict[col] =cxZ
            # print(col, ":\t\t", np.median(tbl[col]))
        # elif col in cols_exclude and isinstance(tbl_[col][0], numbers.Number) or isinstance(tbl_[col][0], units.Quantity):
        #     median_dict[col] = np.NaN
        #     mean_dict[col] = np.NaN
        #     max_dict[col] = np.NaN
        #     min_dict[col] = np.NaN
        #     sigma_dict[col] = np.NaN

    tbl_.add_row(min_dict)
    tbl_.add_row(max_dict)
    tbl_.add_row(sigma_dict)
    tbl_.add_row(mean_dict)
    tbl_.add_row(median_dict)

    return tbl_
