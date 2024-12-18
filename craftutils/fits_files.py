# Code by Lachlan Marnoch, 2019-2021

import os
import shutil as sh
import copy
from datetime import datetime as dt
from typing import Union

import numpy as np
import matplotlib.pyplot as plt

import astropy.wcs as wcs
import astropy.io.fits as fits
import astropy.units as units
from astropy.nddata import CCDData
from astropy.table import Table
from astropy.visualization import ImageNormalize, ZScaleInterval, SqrtStretch

import craftutils.utils as u

__all__ = []


# TODO: Fill in docstrings.

@u.export
def get_rotation_angle(header: fits.header, astropy_units=False):
    """
    Special thanks to `this StackExchange solution <https://math.stackexchange.com/questions/301319/derive-a-rotation-from-a-2d-rotation-matrix>`
    :param header:
    :return theta: rotation angle, in degrees.
    """
    # if 'CROTA2' in header:
    #     theta = header['CROTA2']
    # else:
    wcs_ob = wcs.WCS(header)
    matrix = wcs_ob.pixel_scale_matrix
    theta = np.arctan2(matrix[1, 1], matrix[1, 0]) * 180 / np.pi - 90

    if astropy_units:
        theta *= units.deg

    return theta


def path_or_hdu(hdu: Union[fits.HDUList, str], update=False):
    path = None
    if isinstance(hdu, str):
        path = u.sanitise_file_ext(filename=hdu, ext=".fits")
        u.debug_print(1, f"Loading HDU at {path}")
        if update:
            hdu = fits.open(hdu, mode='update')
        else:
            hdu = fits.open(hdu)
    elif hdu is None:
        raise ValueError("hdu is None; must be a path or a HDUList")

    return hdu, path


def trim_nan(hdu: fits.HDUList, second_hdu: fits.HDUList = None):
    image = hdu[0].data

    bottom = 0
    row = image[bottom]
    while np.sum(np.isnan(row)) == image.shape[1]:
        bottom += 1
        row = image[bottom]

    top = image.shape[0] - 1
    row = image[top]
    while np.sum(np.isnan(row)) == image.shape[1]:
        top -= 1
        row = image[top]

    left = 0
    column = image[:, left]
    while np.sum(np.isnan(column)) == image.shape[0]:
        left += 1
        column = image[:, left]

    right = image.shape[1] - 1
    column = image[:, right]
    while np.sum(np.isnan(column)) == image.shape[0]:
        right -= 1
        column = image[:, right]

    image = trim(hdu=hdu, left=left, right=right, bottom=bottom, top=top)
    if second_hdu is not None:
        second_image = trim(hdu=second_hdu, left=left, right=right, bottom=bottom, top=top)
    else:
        second_image = None

    return image, second_image


wcs_keys = [
    'AP_0_0', 'AP_0_1', 'AP_0_2', 'AP_1_0', 'AP_1_1', 'AP_2_0', 'AP_ORDER',
    'A_0_0', 'A_0_1', 'A_0_2', 'A_1_0', 'A_1_1', 'A_2_0', 'A_ORDER',
    'BP_0_0', 'BP_0_1', 'BP_0_2', 'BP_1_0', 'BP_1_1', 'BP_2_0', 'BP_ORDER',
    'B_0_0', 'B_0_1', 'B_0_2', 'B_1_0', 'B_1_1', 'B_2_0', 'B_ORDER',
    'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
    'CDELT1', 'CDELT2',
    'CROTA1', 'CROTA2',
    'CRPIX1', 'CRPIX2',
    'CRVAL1', 'CRVAL2',
    'CTYPE1', 'CTYPE2',
    'CUNIT1', 'CUNIT2',
    'EQUINOX',
    'IMAGEH', 'IMAGEW',
    'LONPOLE', 'LATPOLE',
    'NAXIS', 'NAXIS1', 'NAXIS2',
    'WCSAXES',
]


def wcs_transfer(header_template: Union[dict, fits.Header], header_update: dict):
    """
    Using the list of header keys in fits_files.wcs_keys, overwrites the WCS header elements of header_update with those
    in header_template.
    :param header_template: Header from which to copy WCS.
    :param header_update: Header to which to copy WCS.
    :return: Update header_update.
    """
    update = {}
    for key in wcs_keys:
        if key in header_update and key not in header_template:
            header_update.pop(key)
        if key in header_template:
            update[key] = header_template[key]

    header_update.update(update)

    return header_update


def divide_by_exp_time(file: Union['fits.hdu_list.hdulist.HDUList', str], output: 'str' = None, ext: int = 0):
    """
    Convert a fits file from total counts to counts/second.
    :param file: Path or HDU object of the file.
    :param output: Path to save altered file to.
    :return:
    """
    path = False
    file, path = path_or_hdu(file)

    old_exp_time = get_exp_time(file)

    file[ext].data = file[ext].data / old_exp_time

    # Set header entries to match.
    change_header(file=file, key='EXPTIME', value=1.)
    change_header(file=file, key='OLD_EXPTIME', value=old_exp_time)

    old_gain = get_header_attribute(file=file, attribute='GAIN')
    if old_gain is None:
        old_gain = 0.8
    change_header(file=file, key='GAIN', value=old_gain * old_exp_time)

    old_saturate = get_header_attribute(file=file, attribute='SATURATE')
    if old_saturate is None:
        old_saturate = 65535
    # Set 'saturate' at 10% lower than stated value, as the detector frequently behaves non-linearly well below the
    # max value.
    new_saturate = 0.9 * old_saturate / old_exp_time
    change_header(file=file, key='SATURATE', value=new_saturate)
    change_header(file=file, key='BUNIT', value="ct / s")

    if output is not None:
        file.writeto(output, overwrite=True)

    if path is not None:
        file.close()

    return file


def find_data(hdu: fits.HDUList):
    for i, layer in enumerate(hdu):
        if layer.data is not None:
            return layer.data


def subtract(hdu: fits.HDUList, subtract_hdu: fits.HDUList, offset: float = 0.0):
    """

    :return:
    """

    if type(hdu) is not fits.HDUList or type(subtract_hdu) is not fits.HDUList:
        raise TypeError("Both arguments must be fits.HDUList")

    data_prime = find_data(hdu=hdu)
    data_sub = find_data(hdu=subtract_hdu)

    if data_prime.shape != data_sub.shape:
        raise ValueError("The files are not the same shape.")

    return data_prime - data_sub + offset


def subtract_file(file: Union[str, fits.HDUList], sub_file: Union[str, fits.HDUList], output: str = None,
                  in_place: bool = False, offset: float = 0.0):
    hdu, path = path_or_hdu(hdu=file)
    sub_hdu, sub_path = path_or_hdu(hdu=sub_file)
    if in_place:
        subbed = hdu
    else:
        subbed = copy.deepcopy(hdu)
    print(f"Subtracting:")
    print(f"\t {sub_path} from")
    print(f"\t {path}")

    print(hdu)
    print(sub_hdu)
    subbed[0].data = subtract(hdu=hdu, subtract_hdu=sub_hdu, offset=offset)

    add_log(file=subbed, action=f'Subtracted {sub_path} from {path}')

    if output is not None:
        subbed.writeto(output, overwrite=True)

    hdu.close()
    sub_hdu.close()

    return subbed


def detect_frame_value(file: Union['fits.HDUList', 'str'], ext: int = 0):
    """
    For images that have
    :param file:
    :param value:
    :param ext:
    :return:
    """
    file, path = path_or_hdu(file)
    data = file[ext].data
    outer = (data[:, 0], data[:, -1], data[0], data[-1])

    value = None
    for edge in outer:
        is_uniform = np.all(edge == edge[0])
        if is_uniform:
            value = edge[0]

    if path:
        file.close()

    return value

@u.export
def detect_edges(file: Union['fits.HDUList', 'str'], value: float = 0.0, ext: int = 0):
    """
    Detects the edges of a rectangular non-zero block, where the frame consists of a single value. For use with
    background files with an obvious frame.
    :param file:
    :param value: the value of the frame.
    :return:
    """

    file, path = path_or_hdu(file)

    data = file[ext].data

    height = data.shape[0]
    mid_y = int(height / 2)

    slice_hor = data[mid_y]
    print(slice_hor, value)
    slice_hor_nonzero = np.nonzero(slice_hor - value)[0]

    while len(slice_hor_nonzero) == 0:
        mid_y = int(mid_y / 2)
        if mid_y == 0:
            raise ValueError("mid_y got stuck.")
        slice_hor = data[mid_y]
        slice_hor_nonzero = np.nonzero(slice_hor - value)[0]

    left = slice_hor_nonzero[0]
    right = slice_hor_nonzero[-1]

    width = data.shape[1]
    mid_x = int(width / 2)
    slice_vert = data[:, mid_x]
    slice_vert_nonzero = np.nonzero(slice_vert - value)[0]
    bottom = slice_vert_nonzero[0]
    top = slice_vert_nonzero[-1]

    if path:
        file.close()

    return left, right, bottom, top


def detect_edges_area(file: Union['fits.HDUList', 'str']):
    if type(file) is str:
        print("Area file:", file)
        file = fits.open(file)

    data = file[0].data

    # We round to the 13th decimal place
    keep_val = u.bucket_mode(data, 13)

    height = data.shape[0]
    mid_y = int(height / 2)
    # Take a horizontal slice right across the middle of the image.
    slice_hor = np.round(data[mid_y], 13)
    slice_hor_keep = np.nonzero(slice_hor == keep_val)[0]
    # This is here just in case mid_y finds a row that does not have maximum coverage.
    while len(slice_hor_keep) == 0:
        mid_y = int(mid_y / 2)
        if mid_y == 0:
            raise ValueError("mid_y got stuck.")
        u.debug_print(2, "detect_edges_area(): mid_y==", mid_y)
        # Take a horizontal slice right across the middle of the image.
        slice_hor = np.round(data[mid_y], 13)
        slice_hor_keep = np.nonzero(slice_hor > 0.75 * keep_val)[0]
        u.debug_print(2, "detect_edges_area(): slice_hor.max()", slice_hor.max())
    left = slice_hor_keep[0]
    right = slice_hor_keep[-1]

    width = data.shape[1]
    mid_x = int(width / 2)
    slice_vert = np.round(data[:, mid_x], 13)
    slice_vert_keep = np.nonzero(slice_vert == keep_val)[0]
    while len(slice_hor_keep) == 0:
        mid_x = int(mid_x / 2)
        # Take a horizontal slice right across the middle of the image.
        slice_hor = np.round(data[mid_x], 13)
        slice_hor_keep = np.nonzero(slice_hor == keep_val)[0]
    bottom = slice_vert_keep[0]
    top = slice_vert_keep[-1]

    if type(file) is str:
        file.close()

    return left, right, bottom, top


def get_filter(file: Union['fits.hdu_list.hdulist.HDUList', str]):
    path = False
    if type(file) is str:
        path = True
        file = fits.open(file)

    header = file[0].header
    filters = [header["ESO INS OPTI5 NAME"], header["ESO INS OPTI6 NAME"], header["ESO INS OPTI7 NAME"],
               header["ESO INS OPTI9 NAME"], header["ESO INS OPTI10 NAME"]]

    if path:
        file.close()

    while 'free' in filters:
        filters.remove('free')

    if len(filters) == 1:
        return filters[0]
    else:
        return filters


def get_header_attribute(file: Union['fits.hdu_list.hdulist.HDUList', 'str'], attribute: 'str', ext: 'int' = 0):
    file, path = path_or_hdu(file)

    header = file[ext].header
    if attribute in header:
        value = header[attribute]
    else:
        value = None

    if path:
        file.close()

    return value


def get_chip_num(file: Union['fits.hdu_list.hdulist.HDUList', 'str']):
    """
    For use with FORS2 images only. Returns 1 if image is from upper CCD, 2 if lower, and 0 if the necessary information
    is not present in the FITS file (likely indicating a stacked or non-FORS2 image).
    :param file: May be a string containing the path to the file, or the file itself as an astropy.fits HDUList object.
    :return:
    """
    chip_string = get_header_attribute(file=file, attribute='HIERARCH ESO DET CHIP1 ID')
    chip = 0
    if chip_string == 'CCID20-14-5-3':
        chip = 1
    elif chip_string == 'CCID20-14-5-6':
        chip = 2
    return chip


def get_exp_time(file: Union['fits.hdu_list.hdulist.HDUList', 'str']):
    return get_header_attribute(file=file, attribute='EXPTIME')


def get_airmass(file: Union['fits.hdu_list.hdulist.HDUList', 'str']):
    return get_header_attribute(file=file, attribute='AIRMASS')


def get_object(file: Union['fits.hdu_list.hdulist.HDUList', 'str']):
    return get_header_attribute(file=file, attribute='OBJECT')


def mean_exp_time(path: 'str'):
    files = os.listdir(path)

    exp_times = []

    for file in files:
        if file[-5:] == '.fits':
            exp_times.append(get_exp_time(path + "/" + file))

    mean_exp = np.mean(exp_times)
    error = 2 * np.nanstd(exp_times)

    return mean_exp, error


def mean_airmass(path: 'str'):
    files = os.listdir(path)

    airmasses = []

    for file in files:
        if file[-5:] == '.fits':
            airmasses.append(get_airmass(path + "/" + file))

    airmass = np.mean(airmasses)
    error = 2 * np.nanstd(airmasses)

    return airmass, error


def sort_by_filter(path: 'str'):
    if path[-1] != "/":
        path = path + "/"
    files = os.listdir(path)
    filters = {}
    # Gather the information we need.
    for file in files:
        if file[-5:] == ".fits":
            filter = get_filter(path + file)
            if type(filter) is list:
                filter = filter[0]
            if filter not in filters:
                filters[filter] = []
            filters[filter].append(file)
    # Do sort
    for filter in filters:
        # Create a directory for each filter
        filter_path = path + filter + "/"
        u.mkdir_check(filter_path)
        # Move all the files with that filter
        for file in filters[filter]:
            if os.path.isfile(filter_path + file):
                os.remove(filter_path + file)
            sh.move(path + file, filter_path)


def get_pixel_scale(
        file: Union['fits.hdu_list.hdulist.HDUList', 'str'],
        ext: int = 0,
        astropy_units: bool = False
):
    """
    Using the FITS file header, obtains the pixel scale of the file (in degrees).
    Declination scale is the true angular size of the pixel.
    Assumes that the change in spherical distortion in RA over the width of the image is negligible.
    :param file:
    :return: Tuple containing the pixel scale of the FITS file: (ra scale, dec scale)
        If astropy_units is True, returns it as an astropy pixel_scale equivalency.
    """
    file, path = path_or_hdu(file)

    header = file[ext].header
    image = file[ext].data

    w = wcs.WCS(header)

    dec = (header["CRVAL2"] * units.deg).to(units.rad)
    ra_pixel_scale, dec_pixel_scale = wcs.utils.proj_plane_pixel_scales(w)
    ra_pixel_scale /= np.cos(dec.value)

    if path:
        file.close()

    if astropy_units:
        ra_pixel_scale = units.pixel_scale(ra_pixel_scale * units.deg / units.pixel)
        dec_pixel_scale = units.pixel_scale(dec_pixel_scale * units.deg / units.pixel)

    return ra_pixel_scale, dec_pixel_scale


def add_log(file: Union[fits.hdu.hdulist.HDUList, str], action: str):
    change_header(file, key='history', value=dt.now().strftime('%Y-%m-%dT%H:%M:%S'))
    change_header(file, key='history', value=action)


def change_header(file: Union[fits.hdu.hdulist.HDUList, str], key: str, value, ext: int = 0):
    """
    Changes the value of a header entry, if it already exists; if not, adds an entry to the bottom of a given fits
    header. Format is NAME: 'entry'
    :param file:
    :param key:
    :param value:
    :return:
    """
    key = key.upper()
    path = ''
    if type(file) is str:
        path = file
        file = fits.open(path, mode='update')
    file[ext].header[key] = value
    if path != '':
        file.close(output_verify='ignore')
    return file


def pix_to_world(x: float, y: float, header: fits.header.Header):
    w = wcs.WCS(header)
    ra, dec = w.all_pix2world(x, y, 0)
    return ra, dec


def world_to_pix(ra: "float", dec: "float", header: "fits.header.Header"):
    w = wcs.WCS(header)
    x, y = w.all_world2pix(ra, dec, 0)
    return x, y


def trim(hdu: fits.hdu.hdulist.HDUList,
         left: Union[int, units.Quantity] = None, right: Union[int, units.Quantity] = None,
         bottom: Union[int, units.Quantity] = None, top: Union[int, units.Quantity] = None,
         update_wcs: bool = True, in_place: bool = False, ext: int = 0):
    """

    :param hdu:
    :param left:
    :param right:
    :param bottom:
    :param top:
    :return:
    """

    if in_place:
        new_hdu = hdu
    else:
        new_hdu = copy.deepcopy(hdu)

    if update_wcs:
        new_hdu[ext].header['CRPIX1'] = hdu[ext].header['CRPIX1'] - u.dequantify(left, units.pix)
        new_hdu[ext].header['CRPIX2'] = hdu[ext].header['CRPIX2'] - u.dequantify(bottom, units.pix)
    new_hdu[ext].data = u.trim_image(
        data=hdu[ext].data,
        left=left, right=right, bottom=bottom, top=top
    )

    return new_hdu


def subimage_edges(data: np.ndarray, x, y, frame):
    bottom = y - frame
    top = y + frame
    left = x - frame
    right = x + frame
    bottom, top, left, right = check_subimage_edges(
        data=data, bottom=bottom, top=top, left=left, right=right,
    )
    return bottom, top, left, right


def check_subimage_edges(data: np.ndarray, bottom, top, left, right):
    u.debug_print(1, "fits_files.check_subimage_edges(): bottom, top, left, right == ", bottom, top, left, right)
    if (bottom < 0 and top < 0) or (bottom > data.shape[0] and top > data.shape[0]):
        raise ValueError(f"Both y-axis edges ({bottom}, {top}) are outside the image.")
    if (left < 0 and right < 0) or (left > data.shape[1] and right > data.shape[1]):
        raise ValueError(f"Both x-axis edges ({left}, {right}) are outside the image.")
    bottom = max(int(bottom), 0)
    top = min(int(top), data.shape[0])
    left = max(int(left), 0)
    right = min(int(right), data.shape[1])
    return bottom, top, left, right


def trim_frame_point(hdu: fits.hdu.hdulist.HDUList, ra: float, dec: float,
                     frame: Union[int, float], world_frame: bool = False, ext: int = 0):
    """
    Trims a fits file to frame a single point.
    :param hdu:
    :param ra:
    :param dec:
    :param frame:
    :param world_frame:
    :return:
    """
    wcs_image = wcs.WCS(header=hdu[ext].header)
    x, y = wcs_image.all_world2pix(ra, dec, 0)

    if world_frame:
        _, scale = get_pixel_scale(hdu)
        frame = frame / scale

    bottom, top, left, right = subimage_edges(data=hdu[ext].data, x=x, y=y, frame=frame)

    hdu_cut = trim(hdu=hdu, left=left, right=right, bottom=bottom, top=top, ext=ext)
    return hdu_cut


def trim_ccddata(
        ccddata: CCDData,
        left: 'int' = None, right: 'int' = None, bottom: 'int' = None, top: 'int' = None,
        update_wcs=True):
    """

    :param ccddata:
    :param left:
    :param right:
    :param bottom:
    :param top:
    :return:
    """

    shape = ccddata.shape
    if left is None:
        left = 0
    if right is None:
        right = shape[1]
    if bottom is None:
        bottom = 0
    if top is None:
        top = shape[0]

    if right < left:
        raise ValueError('Improper inputs; right is smaller than left.')
    if top < bottom:
        raise ValueError('Improper inputs; top is smaller than bottom.')

    if update_wcs:
        ccddata.header['CRPIX1'] = ccddata.header['CRPIX1'] - left
        ccddata.header['CRPIX2'] = ccddata.header['CRPIX2'] - bottom

    ccddata.data = ccddata.data[bottom:top, left:right]

    return ccddata


def trim_file(
        path: Union[str, fits.HDUList],
        left: int = None,
        right: int = None,
        bottom: int = None,
        top: int = None,
        new_path: str = None
):
    """
    Trims the edges of a .fits file while retaining its WCS information.
    :param path:
    :param left:
    :param right:
    :param top:
    :param bottom:
    :return:
    """

    file, path = path_or_hdu(path)
    if new_path is not None:
        new_path = u.sanitise_file_ext(filename=new_path, ext='.fits')

    file = trim(hdu=file, left=left, right=right, bottom=bottom, top=top)

    if new_path is None:
        new_path = path.replace(".fits", "_trim.fits")

    print('Trimming: \n' + str(path))
    print('left', left, 'right', right, 'bottom', bottom, 'top', top)
    print('Moving to: \n' + str(new_path))
    # add_log(file=file,
    #         action='Trimmed using craftutils.fits_files.trim() with borders at x = ' + str(left) + ', ' + str(
    #             right) + '; y=' + str(bottom) + ', ' + str(top) + '; moved from ' + str(path) + ' to ' + str(new_path))
    print()

    print(new_path)
    file.writeto(new_path, overwrite=True)

    if path is not None:
        file.close()
        return new_path
    else:
        return file


def correct_naxis(file: Union[fits.hdu.hdulist.HDUList, CCDData, str], x: int, y: int, write=False):
    """

    :param file:
    :param write: Save changes. Currently Only works if file is a path string.
    :return:
    """
    if type(file) is str:
        path = file
        file = fits.open(path, mode='update')

    if type(file) is fits.hdu.hdulist.HDUList:
        file[0].header['NAXIS1'] = x
        file[0].header['NAXIS2'] = y
        file = trim(hdu=file, left=0, bottom=0, right=x, top=y, update_wcs=False)
    else:  # elif type(file) is CCDData:
        file.header['NAXIS1'] = x
        file.header['NAXIS2'] = y
        file = trim_ccddata(ccddata=file, left=0, bottom=0, right=x, top=y, update_wcs=False)

    return file


def blank_out(path, left, right, top, bottom, newpath=None):
    """
    Blanks out the edges of a .fits file.
    :param path:
    :param left:
    :param right:
    :param top:
    :param bottom:
    :param newpath:
    :return:
    """
    file = fits.open(path)
    file[0].data[top:, :] = 0
    file[0].data[:bottom, :] = 0
    file[0].data[:, right:] = 0
    file[0].data[:, :left] = 0

    if newpath is None:
        newpath = path.replace(".fits", "_blanked.fits")

    file.writeto(newpath, overwrite=True)
    file.close()


def write_sextractor_script(table: Union['str', Table], output_path: 'str' = 'sextract_multi.sh',
                            criterion: 'str' = None,
                            value: 'str' = None, sex_params: 'list' = None, sex_param_values: 'list' = None,
                            cat_name='sextracted', cats_dir='cats'):
    """

    :param table:
    :param output_path:
    :param criterion: The column in the fits table to filter by.
    :param value: The value which the files need to have to not be filtered out.
    :return:
    """
    if os.path.isfile(output_path):
        os.remove(output_path)

    print('Writing SExtractor script to: \n', output_path)

    if type(table) is str:
        table = Table.read(table, format="ascii.csv")

    if criterion is not None:
        table = table[table[criterion] == value]

    files = table['identifier']

    param_line = ''
    if sex_params is not None:
        for i, param in enumerate(sex_params):
            param_line = param_line + ' -' + str(param) + ' ' + str(sex_param_values[i])

    with open(output_path, 'a') as output:
        output.writelines('#!/usr/bin/env bash\n')
        for i, file in enumerate(files):
            output.writelines(
                'sextractor ' + file + param_line + ' -CATALOG_NAME ' + cat_name + '_' + str(i) + '.cat\n')
        output.writelines('mkdir ' + cats_dir + '\n')
        output.writelines('mv *.cat ' + cats_dir)


def write_sextractor_script_shift(file: 'str', shift_param: 'str', shift_param_values: 'list',
                                  output_path: 'str' = 'sextract.sh',
                                  sex_params: 'list' = None,
                                  sex_param_values: 'list' = None, cat_name: 'str' = 'sextracted',
                                  cats_dir: 'str' = 'cats'):
    if os.path.isfile(output_path):
        os.remove(output_path)

    print('Writing SExtractor script to: \n', output_path)

    param_line = ''
    if sex_params is not None:
        for i, param in enumerate(sex_params):
            param_line = param_line + f' -{param} {sex_param_values[i]}'

    with open(output_path, 'a') as output:
        output.writelines('#!/usr/bin/env bash\n')
        for i, par in enumerate(shift_param_values):
            output.writelines('sextractor ' + file + param_line + ' -' + shift_param + ' ' + str(
                par) + ' -CATALOG_NAME ' + cat_name + '_' + str(i + 1) + '.cat\n')
        output.writelines('mkdir ' + cats_dir + '\n')
        output.writelines('mv *.cat ' + cats_dir)


def stack(files: list, output: str = None, directory: str = '', stack_type: str = 'median', inherit: bool = True,
          show: bool = False, normalise: bool = False):
    accepted_stack_types = ['mean', 'median', 'add']
    if stack_type not in accepted_stack_types:
        raise ValueError('stack_type must be in ' + str(accepted_stack_types))
    if directory != '' and directory[-1] != '/':
        directory = directory + '/'

    data = []
    template = None

    print('Stacking:')

    for f in files:
        # Extract image data and append to list.
        if type(f) is str:
            f = u.sanitise_file_ext(f, 'fits')
            print(' ' + f)
            f = fits.open(directory + f)

        if type(f) is fits.hdu.hdulist.HDUList:
            data_append = f[0].data
            if template is None and inherit:
                # Get a template file to use for output.
                # TODO: Refine this - keep a record of which files went in in the header.
                template = f.copy()
        elif type(f) is CCDData:
            data_append = f.data

        else:
            raise TypeError('files must contain only strings, HDUList or CCDData objects.')

        if normalise:
            data_append = data_append / np.nanmedian(data_append[np.isfinite(data_append)])
        if show:
            norm = ImageNormalize(data_append, interval=ZScaleInterval(), stretch=SqrtStretch())
            plt.imshow(data_append, origin='lower', norm=norm)
            plt.show()

        data.append(data_append)

    data = np.array(data)
    if stack_type == 'mean':
        stacked = np.mean(data, axis=0)
    elif stack_type == 'median':
        stacked = np.median(data, axis=0)
    else:
        stacked = np.sum(data, axis=0)

    if show:
        norm = ImageNormalize(stacked, interval=ZScaleInterval(), stretch=SqrtStretch())
        plt.imshow(stacked, origin='lower', norm=norm)
        plt.show()

    if inherit:
        template[0].data = stacked
    else:
        template = fits.PrimaryHDU(stacked)
        template = fits.HDUList([template])
    add_log(template, f'Stacked.')
    if output is not None:
        print('Writing stacked image to', output)
        template[0].header['BZERO'] = 0
        template.writeto(output, overwrite=True, output_verify='warn')

    return template

    # def imacs_debias(file: str, bias: str, output: str = None):


def find_coordinates(ra: float, dec: float, path: str):
    """
    Looks through all of the FITS files in a directory and returns a list of those that include the target coordinates.
    :param ra: Right ascension of target, in degrees.
    :param dec: Declination of target, in degrees.
    :param path: Directory to search.
    :return:
    """

    files = filter(lambda f: f[-5:] == '.fits', os.listdir(path))
    passed = []
    for file in files:
        print(file)
        fit = fits.open(path + '/' + file)
        header = fit[0].header
        n_x = header['NAXIS1']
        n_y = header['NAXIS2']
        wcs_info = wcs.WCS(header)
        # Get pixel coordinates of the target.
        x, y = wcs_info.all_world2pix(ra, dec, 0)
        print(x, y)

        if 0 < x < n_x and 0 < y < n_y:
            passed.append(file)

    return passed


def swap_gmos_frames(path: str, output: str):
    hdu = fits.open(path)
    hdu[0].data = hdu[1].data
    hdu.writeto(output)
    return hdu
