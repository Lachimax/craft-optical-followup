# Code by Lachlan Marnoch, 2019-2021

from datetime import datetime as dt
import numpy as np
import math
import os
from typing import List, Union

from astropy.coordinates import SkyCoord
import astropy.table as tbl
from astropy.io import fits
from astropy import units


# TODO: Arrange these into some kind of logical order.
# TODO: Also comment.

def dequantify(number):
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


def error_product(value, measurements, errors):
    """
    Produces the absolute error of a value calculated as a product or as a quotient.
    :param value: The final calculated value.
    :param measurements: An array of the measurements used to calculate the value.
    :param errors: An array of the respective errors of the measurements. Expected to be in the same order as
        measurements.
    :return:
    """
    measurements = np.array(measurements)
    errors = np.array(errors)
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
    # One of these should come out negative - that becomes the minus error, and the positive the plus error.
    error_plus = func(arg + err) - measurement
    error_minus = func(arg - err) - measurement

    error_plus_actual = []
    error_minus_actual = []
    for i, _ in enumerate(error_plus):
        error_plus_actual.append(np.max([error_plus[i], error_minus[i]]))
        error_minus_actual.append(np.min([error_plus[i], error_minus[i]]))

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

    return tbl.hstack([matches_both_cat, matches_both_1, matches_both_2],
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
            tables.append(tbl.Table.read(file, format='ascii.csv'))
        elif file[-5:] == '.fits':
            tables.append(tbl.Table(fits.open(file)[1].data))
        else:
            raise ValueError('File format not recognised.')

    output_tbl = tbl.hstack(tables)
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


def select_option(message: str, options: List[str], default: Union[str, int] = None):
    if type(default) is str:
        default = options.index(default)
    if default is not None:
        message += f" [default: {default} {options[default]}]"
    print()
    print(message)

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
            print(f"Response is not in provided options. Please pick an integer between 0 and {len(options) - 1}")
    print(f"You have selected {selection}: {picked}")
    return picked


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
    print()
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
