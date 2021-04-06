# Code by Lachlan Marnoch, 2019-2021

import os
import shutil as sh
from copy import deepcopy

from astropy import wcs
from astropy.nddata import CCDData
from astropy.io import fits
from astropy import units

from datetime import datetime as dt
from typing import Union
from numbers import Number
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from craftutils import utils as u
from craftutils import plotting as pl


# TODO: Fill in docstrings.
# TODO: Sanitise pipeline inputs (ie check if object name is valid)

def get_rotation_angle(header: fits.header, astropy_units=False):
    """
    Special thanks to https://math.stackexchange.com/questions/301319/derive-a-rotation-from-a-2d-rotation-matrix
    :param header:
    :return theta: rotation angle, in degrees.
    """
    if 'CROTA2' in header:
        theta = header['CROTA2']
    else:
        wcs_ob = wcs.WCS(header)
        matrix = wcs_ob.pixel_scale_matrix
        theta = np.arctan2(matrix[1, 1], matrix[1, 0]) * 180 / np.pi - 90

    if astropy_units:
        theta *= units.deg

    return theta


def projected_pix_scale(hdu: Union[fits.HDUList, str], ang_size_distance: float):
    """
    Calculate the per-pixel projected distance of a given angular size distance in an image.
    :param hdu:
    :param ang_size_distance:
    :return: Pixel distance scale, in parsecs (per pixel)
    """
    hdu, path = path_or_hdu(hdu=hdu)

    _, angular_scale = get_pixel_scale(file=hdu)

    distance_scale = u.size_from_ang_size_distance(theta=angular_scale, ang_size_distance=ang_size_distance)

    if path:
        hdu.close()

    return distance_scale


def path_or_hdu(hdu: Union[fits.HDUList, str], update=False):
    # TODO: Propagate this method to where it's needed.
    path = None
    if type(hdu) is str:
        path = u.sanitise_file_ext(filename=hdu, ext=".fits")
        if update:
            hdu = fits.open(hdu, mode='update')
        else:
            hdu = fits.open(hdu)

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


def wcs_transfer(header_template: dict, header_update: dict):
    keys = ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2', 'CROTA1', 'CROTA2', 'CTYPE1', 'CTYPE2', 'CD1_1',
            'CD1_2', 'CD2_1', 'CD2_2', 'EQUINOX', 'CUNIT1', 'CUNIT2']

    update = {}
    for key in keys:
        if key in header_template:
            update[key] = header_template[key]
        if key in header_update:
            del header_update[key]

    header_update.update(update)

    print(update)

    return header_update


def reproject(image_1: Union[fits.HDUList, str], image_2: Union[fits.HDUList, str], image_1_output: str = None,
              image_2_output: str = None, show: bool = False, force: int = None):
    """
    Determines which image has the best pixel scale, and reprojects it onto the other; in the process, its spatial
    resolution will be downgraded to match the worse.
    :param image_1:
    :param image_2:
    :param image_1_output:
    :param image_2_output:
    :param show:
    :return:
    """
    import reproject as rp
    image_1, path_1 = path_or_hdu(image_1)
    image_2, path_2 = path_or_hdu(image_2)
    pix_scale_1 = get_pixel_scale(image_1)
    pix_scale_2 = get_pixel_scale(image_2)

    # Take the image with the better spatial resolution and down-sample it to match the worse-resolution one
    # (unfortunately)
    # TODO: The header transfer is coarse and won't convey necessary information about the downgraded image. Take the
    #  time to go through and select which header elements to keep and which to take from the other image.
    print('Reprojecting...')
    if force is None:
        if pix_scale_1 <= pix_scale_2:
            reprojected, footprint = rp.reproject_exact(image_1, image_2[0].header)
            image_1[0].data = reprojected
            image_1, image_2 = trim_nan(image_1, image_2)
            n_reprojected = 1
            image_1[0].header = wcs_transfer(image_2[0].header, image_1[0].header)
        else:
            reprojected, footprint = rp.reproject_exact(image_2, image_1[0].header)
            n_reprojected = 2
            image_2[0].data = reprojected
            image_2, image_1 = trim_nan(image_2, image_1)
            image_2[0].header = wcs_transfer(image_1[0].header, image_2[0].header)
    elif force == 1:
        reprojected, footprint = rp.reproject_exact(image_1, image_2[0].header)
        image_1[0].data = reprojected
        image_1, image_2 = trim_nan(image_1, image_2)
        n_reprojected = 1
        image_1[0].header = wcs_transfer(image_2[0].header, image_1[0].header)
    elif force == 2:
        reprojected, footprint = rp.reproject_exact(image_2, image_1[0].header)
        n_reprojected = 2
        image_2[0].data = reprojected
        image_2, image_1 = trim_nan(image_2, image_1)
        image_2[0].header = wcs_transfer(image_1[0].header, image_2[0].header)
    else:
        raise ValueError('force must be 1, 2 or None')

    print(image_1_output)
    print(image_2_output)
    if image_1_output is not None:
        image_1.writeto(image_1_output, overwrite=True)
    if image_2_output is not None:
        image_2.writeto(image_2_output, overwrite=True)

    if path_1:
        image_1.close()
    if path_2:
        image_2.close()

    return n_reprojected


# def trim_nan(contains_nans: fits.HDUList, other: fits.HDUList):
#     contains_nans_data = contains_nans[0].data
#     other_data = other[0].data
#
#     col_mask = np.ones(dtype=bool)
#     for col in contains_nans_data:
#         if np.sum(np.isnan(col)) > 0:

def align(comparison: Union[fits.hdu.hdulist.HDUList, str], template: Union[fits.hdu.hdulist.HDUList, str],
          comparison_output: str = 'comparison_shifted.fits', template_output: str = 'template_shifted.fits',
          axis: Union[tuple, str] = 'midpoint', wcs_coords=False):
    axis_list = ['midpoint', 'zero']
    if type(axis) is str and axis not in axis_list:
        raise ValueError('Axis type must be in', axis_list)

    comparison, comparison_path = path_or_hdu(comparison)
    template, template_path = path_or_hdu(template)

    template_data = template[0].data
    template_shape = template_data.shape

    # Get wcs solutions for each image.
    wcs_template = wcs.WCS(template[0].header)
    wcs_comparison = wcs.WCS(comparison[0].header)

    # Find centre pixel of the template.
    width_template = template_shape[1]
    mid_x_template = int(width_template / 2)
    height_template = template_shape[0]
    mid_y_template = int(height_template / 2)
    print('Midpoint in template:', mid_x_template, mid_y_template)
    if type(axis) is str:
        if axis == 'midpoint':
            # We take the midpoint of the template.
            axis = (mid_x_template, mid_y_template)
        elif axis == 'zero':
            axis = (0, 0)
    # If we have been passed world coordinates instead of pixel, transform to pixel coordinates.
    elif wcs_coords:
        axis = wcs_template.all_world2pix(axis[0], axis[1], 0)

    # Retrieve the RA and DEC of the axis pixel for each image.
    axis_ra_template, axis_dec_template = wcs_template.all_pix2world(axis[0], axis[1], 0)
    axis_ra_comparison, axis_dec_comparison = wcs_comparison.all_pix2world(axis[0], axis[1], 0)

    # Get the difference, in degrees.
    delta_ra_axis = axis_ra_comparison - axis_ra_template
    delta_dec_axis = axis_dec_comparison - axis_dec_template

    pix_scale_ra_template, pix_scale_dec_template = get_pixel_scale(template)
    pix_scale_ra_comparison, pix_scale_dec_comparison = get_pixel_scale(comparison)

    print('Template pixel scales:', pix_scale_ra_template, pix_scale_dec_template)
    print('Comparison pixel scales:', pix_scale_ra_comparison, pix_scale_dec_comparison)

    # Convert offset back to pixels
    pixel_offset_x = delta_ra_axis / pix_scale_ra_template
    pixel_offset_y = delta_dec_axis / pix_scale_dec_template
    print('Pixel offset:', pixel_offset_x, pixel_offset_y)
    pixel_offset_x = int(np.round(pixel_offset_x))
    pixel_offset_y = int(np.round(pixel_offset_y))

    comparison_cut = comparison.copy()
    template_cut = template.copy()
    print(template_cut[0].data.shape, comparison_cut[0].data.shape)
    # Trim left and bottom of either image to align their pixels.
    if pixel_offset_y > 0:
        template_cut = trim(hdu=template_cut, bottom=pixel_offset_y)
        template_cut[0].header['DELTA_Y'] = pixel_offset_y
        comparison_cut[0].header['DELTA_Y'] = 0
    else:
        comparison_cut = trim(hdu=comparison_cut, bottom=-pixel_offset_y)
        template_cut[0].header['DELTA_Y'] = 0
        comparison_cut[0].header['DELTA_Y'] = -pixel_offset_y
    print('Template shape:', template_cut[0].data.shape, '; Comparison shape:', comparison_cut[0].data.shape)

    if pixel_offset_x > 0:
        comparison_cut = trim(hdu=comparison_cut, left=pixel_offset_x)
        template_cut[0].header['DELTA_X'] = 0
        comparison_cut[0].header['DELTA_X'] = pixel_offset_x
    else:
        template_cut = trim(hdu=template_cut, left=-pixel_offset_x)
        template_cut[0].header['DELTA_X'] = -pixel_offset_x
        comparison_cut[0].header['DELTA_X'] = 0
    print('Template shape:', template_cut[0].data.shape, '; Comparison shape:', comparison_cut[0].data.shape)

    comparison_cut_shape = comparison_cut[0].data.shape
    template_cut_shape = template_cut[0].data.shape

    # Now that they are aligned, trim right and top of either image so that they come out the same size.
    if comparison_cut_shape[1] > template_cut_shape[1]:
        comparison_cut = trim(hdu=comparison_cut, right=template_cut_shape[1])

    elif comparison_cut_shape[1] < template_cut_shape[1]:
        template_cut = trim(hdu=template_cut, right=comparison_cut_shape[1])
    print('Template shape:', template_cut[0].data.shape, '; Comparison shape:', comparison_cut[0].data.shape)

    if comparison_cut_shape[0] > template_cut_shape[0]:
        comparison_cut = trim(hdu=comparison_cut, top=template_cut_shape[0])
    elif comparison_cut_shape[0] < template_cut_shape[0]:
        template_cut = trim(hdu=template_cut, top=comparison_cut_shape[0])
    print('Template shape:', template_cut[0].data.shape, '; Comparison shape:', comparison_cut[0].data.shape)

    add_log(comparison_cut, f'Aligned astrometrically to {template}.')
    add_log(template_cut, f'Aligned astrometrically to {comparison}.')

    print('Writing new comparison file to:\n', comparison_output)
    comparison_cut.writeto(comparison_output, overwrite=True)
    print('Writing new template file to:\n', template_output)
    template_cut.writeto(template_output, overwrite=True)

    if template_path:
        template.close()
    if comparison_path:
        comparison.close()


def divide_by_exp_time(file: Union['fits.hdu.hdulist.HDUList', 'str'], output: 'str' = None):
    """
    Convert a fits file from total counts to counts/second.
    :param file: Path or HDU object of the file.
    :param output: Path to save altered file to.
    :return:
    """
    path = False
    file, path = path_or_hdu(file)

    old_exp_time = get_exp_time(file)

    file[0].data = file[0].data / old_exp_time

    # Set header entries to match.
    change_header(file=file, name='EXPTIME', entry=1.)
    change_header(file=file, name='OLD_EXPTIME', entry=old_exp_time)

    old_gain = get_header_attribute(file=file, attribute='GAIN')
    if old_gain is None:
        old_gain = 0.8
    change_header(file=file, name='GAIN', entry=old_gain * old_exp_time)

    old_saturate = get_header_attribute(file=file, attribute='SATURATE')
    if old_saturate is None:
        old_saturate = 65535
    # Set 'saturate' at 10% lower than stated value, as the detector frequently behaves non-linearly well below the
    # max value.
    new_saturate = 0.9 * old_saturate / old_exp_time
    change_header(file=file, name='SATURATE', entry=new_saturate)

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
        subbed = deepcopy(hdu)
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


def detect_edges(file: Union['fits.hdu.hdulist.HDUList', 'str']):
    """
    Detects the edges of a rectangular non-zero block, where the frame consists of zeroed pixels. For use with
    background files with an obvious frame.
    :param file:
    :return:
    """

    if type(file) is str:
        file = fits.open(file)

    data = file[0].data

    height = data.shape[0]
    mid_y = int(height / 2)
    slice_hor = data[mid_y]
    slice_hor_nonzero = np.nonzero(slice_hor)[0]
    left = slice_hor_nonzero[0]
    right = slice_hor_nonzero[-1]

    width = data.shape[1]
    mid_x = int(width / 2)
    slice_vert = data[:, mid_x]
    slice_vert_nonzero = np.nonzero(slice_vert)[0]
    bottom = slice_vert_nonzero[0]
    top = slice_vert_nonzero[-1]

    if type(file) is str:
        file.close()

    print(left, right, bottom, top)

    return left, right, bottom, top


def detect_edges_area(file: Union['fits.hdu.hdulist.HDUList', 'str']):
    if type(file) is str:
        file = fits.open(file)

    data = file[0].data

    # We round to the 13th decimal place
    keep_val = np.round(np.max(data), 13)
    print(keep_val)

    height = data.shape[0]
    mid_y = int(height / 2)
    # Take a horizontal slice right across the middle of the image.
    slice_hor = np.round(data[mid_y], 13)
    slice_hor_keep = np.nonzero(slice_hor == keep_val)[0]
    # This is here just in case mid_y finds a row that does not have maximum coverage.
    while len(slice_hor_keep) == 0:
        mid_y = int(mid_y / 2)
        # Take a horizontal slice right across the middle of the image.
        slice_hor = np.round(data[mid_y], 13)
        slice_hor_keep = np.nonzero(slice_hor == keep_val)[0]
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


def get_filter(file: Union['fits.hdu.hdulist.HDUList', 'str']):
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


def get_header_attribute(file: Union['fits.hdu.hdulist.HDUList', 'str'], attribute: 'str', ext: 'int' = 0):
    file, path = path_or_hdu(file)

    header = file[ext].header
    if attribute in header:
        value = header[attribute]
    else:
        value = None

    if path:
        file.close()

    return value


def get_chip_num(file: Union['fits.hdu.hdulist.HDUList', 'str']):
    """
    For use with FORS2 images only. Returns 1 if image is from upper CCD, 2 if lower, and 0 if the necessary information
    is not present in the FITS file (likely indicating a non-FORS2 image).
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


def get_exp_time(file: Union['fits.hdu.hdulist.HDUList', 'str']):
    return get_header_attribute(file=file, attribute='EXPTIME')


def get_airmass(file: Union['fits.hdu.hdulist.HDUList', 'str']):
    return get_header_attribute(file=file, attribute='AIRMASS')


def get_object(file: Union['fits.hdu.hdulist.HDUList', 'str']):
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


def get_pixel_scale(file: Union['fits.hdu.hdulist.HDUList', 'str'], layer: int = 0, astropy_units: bool = False):
    """
    Using the FITS file header, obtains the pixel scale of the file (in degrees).
    Declination scale is the true angular size of the pixel.
    Assumes that the change in spherical distortion in RA over the width of the image is negligible.
    :param file:
    :return: Tuple containing the pixel scale of the FITS file: (ra scale, dec scale)
        If astropy_units is True, returns it as an astropy pixel_scale equivalency.
    """
    # TODO: Rewrite photometry functions to use this

    file, path = path_or_hdu(file)

    header = file[layer].header
    image = file[layer].data

    # To take (very roughly) into account the spherical distortion to the RA, we obtain an RA pixel scale by dividing
    # the difference in RA across the image by the number of pixels. It's good enough to give an average value, and the
    # difference SHOULD be pretty tiny across the image.

    w = wcs.WCS(header)
    end = image.shape[0] - 1
    ra_pixel_scale = (w.pixel_to_world(0, 0).ra.deg - w.pixel_to_world(end, 0).ra.deg) / end

    # By comparison the pixel scale in declination is easy to obtain - as DEC is undistorted, it is simply the true
    # pixel scale of the image, which the header stores.
    dec_pixel_scale = w.pixel_scale_matrix[1, 1]
    if dec_pixel_scale == 0:
        dec_pixel_scale = w.pixel_scale_matrix[1, 0]

    if path:
        file.close()

    ra_pixel_scale = abs(ra_pixel_scale)
    dec_pixel_scale = abs(dec_pixel_scale)

    if astropy_units:
        ra_pixel_scale = units.pixel_scale(ra_pixel_scale * units.deg / units.pixel)
        dec_pixel_scale = units.pixel_scale(dec_pixel_scale * units.deg / units.pixel)

    return ra_pixel_scale, dec_pixel_scale


def add_log(file: Union[fits.hdu.hdulist.HDUList, str], action: str):
    change_header(file, name='history', entry=dt.now().strftime('%Y-%m-%dT%H:%M:%S'))
    change_header(file, name='history', entry=action)


def change_header(file: Union[fits.hdu.hdulist.HDUList, str], name: str, entry):
    """
    Changes the value of a header entry, if it already exists; if not, adds an entry to the bottom of a given fits
    header. Format is NAME: 'entry'
    :param file:
    :param name:
    :param entry:
    :return:
    """
    name = name.upper()
    path = ''
    if type(file) is str:
        path = file
        file = fits.open(path, mode='update')
    file[0].header[name] = entry
    if path != '':
        file.close(output_verify='ignore')
    return file


def pix_to_world(x: "float", y: "float", header: "fits.header.Header"):
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
         update_wcs: bool = True, in_place: bool = False, quiet: bool = False):
    """

    :param hdu:
    :param left:
    :param right:
    :param bottom:
    :param top:
    :return:
    """
    shape = hdu[0].data.shape
    if left is None:
        left = 0
    else:
        left = int(u.dequantify(left))

    if right is None:
        right = shape[1]
    else:
        right = int(u.dequantify(right))

    if bottom is None:
        bottom = 0
    else:
        bottom = int(u.dequantify(bottom))

    if top is None:
        top = shape[0]
    else:
        top = int(u.dequantify(top))

    if right < left:
        raise ValueError('Improper inputs; right is smaller than left.')
    if top < bottom:
        raise ValueError('Improper inputs; top is smaller than bottom.')

    if in_place:
        new_hdu = hdu
    else:
        new_hdu = deepcopy(hdu)

    if update_wcs:
        new_hdu[0].header['CRPIX1'] = hdu[0].header['CRPIX1'] - left
        new_hdu[0].header['CRPIX2'] = hdu[0].header['CRPIX2'] - bottom
    if not quiet:
        print(bottom, top, left, right)
    new_hdu[0].data = hdu[0].data[bottom:top, left:right]

    return new_hdu


def subimage_edges(data: np.ndarray, x, y, frame, quiet: bool = True):
    bottom = y - frame
    top = y + frame
    left = x - frame
    right = x + frame
    bottom, top, left, right = check_subimage_edges(data=data, bottom=bottom, top=top, left=left, right=right,
                                                    quiet=quiet)
    return bottom, top, left, right


def check_subimage_edges(data: np.ndarray, bottom, top, left, right, quiet: bool = True):
    if not quiet:
        print(bottom, top, left, right)
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
                     frame: Union[int, float], world_frame: bool = False, quiet: bool = False):
    """
    Trims a fits file to frame a single point.
    :param hdu:
    :param ra:
    :param dec:
    :param frame:
    :param world_frame:
    :return:
    """
    wcs_image = wcs.WCS(header=hdu[0].header)
    x, y = wcs_image.all_world2pix(ra, dec, 0)

    if world_frame:
        _, scale = get_pixel_scale(hdu)
        frame = frame / scale

    bottom, top, left, right = subimage_edges(data=hdu[0].data, x=x, y=y, frame=frame)

    hdu_cut = trim(hdu=hdu, left=left, right=right, bottom=bottom, top=top, quiet=quiet)
    return hdu_cut


def trim_ccddata(ccddata: CCDData,
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


def trim_file(path: Union[str, fits.HDUList], left: int = None, right: int = None, bottom: int = None, top: int = None,
              new_path: str = None):
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
    add_log(file=file, action='Trimmed using PyCRAFT.fits_files.trim() with borders at x = ' + str(left) + ', ' + str(
        right) + '; y=' + str(bottom) + ', ' + str(top) + '; moved from ' + str(path) + ' to ' + str(new_path))
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


# TODO: Make this list all fits files, then write wrapper that eliminates non-science images and use that in scripts.
def fits_table(input_path: str, output_path: str = "", science_only: bool = True):
    """
    Produces and writes to disk a table of .fits files in the given path, with the vital statistics of each. Intended
    only for use with raw ESO data.
    :param input_path:
    :param output_path:
    :param science_only: If True, we are writing a list for a folder that also contains calibration files, which we want
     to ignore.
    :return:
    """

    # If there's no trailing slash in the paths, add one.
    input_path = u.check_trailing_slash(input_path)

    if output_path == "":
        output_path = input_path + "fits_table.csv"
    elif output_path[-4:] != ".csv":
        if output_path[-1] == "/":
            output_path = output_path + "fits_table.csv"
        else:
            output_path = output_path + ".csv"

    print('Writing table of fits files to: \n', output_path)

    files = os.listdir(input_path)
    files.sort()
    files_fits = []

    # Keep only the relevant fits files

    for f in files:
        if f[-5:] == ".fits":
            files_fits.append(f)

    # Create list of dictionaries to be used as the output data
    output = []

    ids = string.ascii_lowercase
    if len(ids) < len(files_fits):
        ids = ids + string.ascii_uppercase
    if len(ids) < len(files_fits):
        ids = ids + string.digits

    for i, f in enumerate(files_fits):
        data = {}
        file = fits.open(input_path + f)
        header = file[0].header
        data['identifier'] = f
        if science_only and ('ESO DPR CATG' not in header or 'SCIENCE' not in header['ESO DPR CATG']):
            continue
        if len(ids) >= len(files_fits):
            data['id'] = ids[i]
        if "OBJECT" in header:
            data['object'] = header["OBJECT"]
        if "ESO OBS NAME" in header:
            data['obs_name'] = header["ESO OBS NAME"]
        if "EXPTIME" in header:
            data['exp_time'] = header["EXPTIME"]
        if "AIRMASS" in header:
            data['airmass'] = header["AIRMASS"]
        elif "ESO TEL AIRM START" in header and "ESO TEL AIRM END":
            data['airmass'] = (header["ESO TEL AIRM START"] + header["ESO TEL AIRM END"]) / 2
        if "CRVAL1" in header:
            data['ref_ra'] = header["CRVAL1"]
        if "CRVAL2" in header:
            data['ref_dec'] = header["CRVAL2"]
        if "CRPIX1" in header:
            data['ref_pix_x'] = header["CRPIX1"]
        if "CRPIX2" in header:
            data['ref_pix_y'] = header["CRPIX2"]
        if "EXTNAME" in header:
            data['chip'] = header["EXTNAME"]
        elif "ESO DET CHIP1 ID" in header:
            if header["ESO DET CHIP1 ID"] == 'CCID20-14-5-3':
                data['chip'] = 'CHIP1'
            if header["ESO DET CHIP1 ID"] == 'CCID20-14-5-6':
                data['chip'] = 'CHIP2'
        if "GAIN" in header:
            data['gain'] = header["GAIN"]
        if "INSTRUME" in header:
            data['instrument'] = header["INSTRUME"]
        if "ESO TEL AIRM START" in header:
            data['airmass_start'] = header["ESO TEL AIRM START"]
        if "ESO TEL AIRM END" in header:
            data['airmass_end'] = header["ESO TEL AIRM END"]
        if "ESO INS OPTI3 NAME" in header:
            data['collimater'] = header["ESO INS OPTI3 NAME"]
        if "ESO INS OPTI5 NAME" in header:
            data['filter1'] = header["ESO INS OPTI5 NAME"]
        if "ESO INS OPTI6 NAME" in header:
            data['filter2'] = header["ESO INS OPTI6 NAME"]
        if "ESO INS OPTI7 NAME" in header:
            data['filter3'] = header["ESO INS OPTI7 NAME"]
        if "ESO INS OPTI9 NAME" in header:
            data['filter4'] = header["ESO INS OPTI9 NAME"]
        if "ESO INS OPTI10 NAME" in header:
            data['filter5'] = header["ESO INS OPTI10 NAME"]
        if "ESO INS OPTI8 NAME" in header:
            data['camera'] = header["ESO INS OPTI8 NAME"]
        if "NAXIS1" in header:
            data['pixels_x'] = header["NAXIS1"]
        if "NAXIS2" in header:
            data['pixels_y'] = header["NAXIS2"]
        if "SATURATE" in header:
            data['saturate'] = header["SATURATE"]
        if "MJD-OBS" in header:
            data['mjd_obs'] = header["MJD-OBS"]
        output.append(data)
        file.close()

    output.sort(key=lambda a: a['identifier'])

    out_file = pd.DataFrame(output)
    out_file.to_csv(output_path)

    return out_file


def fits_table_all(input_path: str, output_path: str = "", science_only: bool = True):
    """
    Produces and writes to disk a table of .fits files in the given path, with the vital statistics of each. Intended
    only for use with raw ESO data.
    :param input_path:
    :param output_path:
    :param science_only: If True, we are writing a list for a folder that also contains calibration files, which we want
     to ignore.
    :return:
    """

    # If there's no trailing slash in the paths, add one.
    if not output_path.endswith(".csv"):
        output_path = u.check_trailing_slash(output_path)

    if output_path == "":
        output_path = input_path + "fits_table.csv"
    elif output_path[-4:] != ".csv":
        if output_path[-1] == "/":
            output_path = output_path + "fits_table.csv"
        else:
            output_path = output_path + ".csv"

    print('Writing table of fits files to: \n', output_path)

    files = os.listdir(input_path)
    files.sort()
    files_fits = list(filter(lambda x: x[-5:] == '.fits', files))

    # Create list of dictionaries to be used as the output data
    output = []

    ids = string.ascii_lowercase
    if len(ids) < len(files_fits):
        ids = ids + string.ascii_uppercase
    if len(ids) < len(files_fits):
        ids = ids + string.digits

    for i, f in enumerate(files_fits):
        data = {}
        file = fits.open(input_path + f)
        header = file[0].header
        for key in header:
            data[key] = header[key]
        if 'ESO TEL AIRM END' in data and 'ESO TEL AIRM START' in data:
            data['AIRMASS'] = (float(data['ESO TEL AIRM END']) + float(data['ESO TEL AIRM START'])) / 2
        if science_only and 'SCIENCE' in data['ESO DPR CATG']:
            output.append(data)
        elif not science_only:
            output.append(data)
        file.close()

    output.sort(key=lambda a: a['ARCFILE'])

    out_file = pd.DataFrame(output)
    out_file.to_csv(output_path)

    return out_file


def write_sextractor_script(table: Union['str', pd.DataFrame], output_path: 'str' = 'sextract_multi.sh',
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
        table = pd.read_csv(table)

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


def write_sof(table_path: str, output_path: str = 'bias.sof', sof_type: str = 'fors_bias', chip: int = 1,
              cat_path: str = ""):
    """
    Requires that fits_table has already been run on the directory.
    For fors_zeropoint, if there are multiple STD-type files, it will use the first one listed in the table.
    :param table_path:
    :param output_path:
    :param sof_type:
    :param chip:
    :return:
    """
    sof_types = ['fors_bias', 'fors_img_sky_flat', 'fors_zeropoint']

    chip = ['CHIP1', 'CHIP2'][chip - 1]

    if cat_path != "" and cat_path[-1] != "/":
        cat_path = cat_path + "/"

    if sof_type not in sof_types:
        raise ValueError(sof_type + ' is not a recognised sof_type. Recognised types are:' + str(sof_types))

    if output_path[-4:] != ".sof":
        output_path = output_path + ".sof"

    if os.path.isfile(output_path):
        os.remove(output_path)

    files = pd.read_csv(table_path)
    files = files[files['chip'] == chip]

    # Bias
    if sof_type == 'fors_bias':
        bias_files = files[files['object'] == 'BIAS']['identifier']

        with open(output_path, 'a') as output:
            for file in bias_files:
                output.writelines(file + " BIAS\n")

    # Flats
    if sof_type == 'fors_img_sky_flat':
        flat_files = files[files['object'] == 'FLAT,SKY']['identifier']

        with open(output_path, 'a') as output:
            for file in flat_files:
                output.writelines(file + " SKY_FLAT_IMG\n")
            if chip == 'CHIP1':
                suffix = "up"
            else:
                suffix = "down"
            output.writelines("master_bias_" + suffix + ".fits MASTER_BIAS")

    # Zeropoint
    if sof_type == 'fors_zeropoint':
        if chip == 'CHIP1':
            suffix = "up"
            chip_id = "1453"
        else:
            suffix = "down"
            chip_id = "1456"

        std_image = files[files['object'] == 'STD']['identifier'].values[0]

        with open(output_path, 'a') as output:
            output.writelines(std_image + " STANDARD_IMG\n")
            output.writelines("master_bias_" + suffix + ".fits MASTER_BIAS\n")
            output.writelines("master_sky_flat_img_" + suffix + ".fits MASTER_SKY_FLAT_IMG\n")
            output.writelines(cat_path + "fors2_landolt_std_UBVRI.fits FLX_STD_IMG\n")
            output.writelines(cat_path + "fors2_" + chip_id + "_phot.fits PHOT_TABLE\n")


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
            norm = pl.nice_norm(data_append)
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
        norm = pl.nice_norm(stacked)
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
