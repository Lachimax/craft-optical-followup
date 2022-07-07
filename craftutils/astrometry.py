# Code by Lachlan Marnoch, 2019 - 2021
import copy
from typing import Union, Iterable
import os

import matplotlib.pyplot as plt
import numpy as np

import astropy.table as table
import astropy.io.fits as fits
import astropy.wcs as wcs
from astropy.coordinates import SkyCoord
import astropy.units as units
import astropy.time as time
from astropy.visualization import ImageNormalize, ZScaleInterval, SqrtStretch

import craftutils.fits_files as ff
import craftutils.params as p
import craftutils.utils as u
import craftutils.wrap.astrometry_net as astrometry_net
from craftutils.retrieve import cat_columns, load_catalogue


def jname(coord: SkyCoord, ra_precision: int = 2, dec_precision: int = 1):
    s_ra, s_dec = coord_string(coord)
    ra_second = np.round(float(s_ra[s_ra.find("m") + 1:s_ra.find("s")]), ra_precision)
    if ra_precision <= 0:
        ra_second = int(ra_second)
    ra_second = str(ra_second)  # .ljust(6, "0")
    dec_second = np.round(float(s_dec[s_dec.find("m") + 1:s_dec.find("s")]), dec_precision)
    if dec_precision == 0:
        dec_second = int(dec_second)
    dec_second = str(dec_second)  # .ljust(5, "0")
    s_ra = s_ra[:s_ra.find("m")].replace("h", "")
    s_dec = s_dec[:s_dec.find("m")].replace("d", "")
    name = f"J{s_ra}{ra_second}{s_dec}{dec_second}"
    return name


def correct_gaia_to_epoch(gaia_cat: Union[str, table.QTable], new_epoch: time.Time):
    gaia_cat = load_catalogue(cat_name="gaia", cat=gaia_cat)
    epochs = list(map(lambda y: f"J{y}", gaia_cat['ref_epoch']))
    gaia_coords = SkyCoord(
        ra=gaia_cat["ra"], dec=gaia_cat["dec"],
        pm_ra_cosdec=gaia_cat["pmra"], pm_dec=gaia_cat["pmdec"],
        obstime=epochs)
    u.debug_print(2, "astrometry.correct_gaia_to_epoch(): new_epoch ==", new_epoch)
    gaia_coords_corrected = gaia_coords.apply_space_motion(new_obstime=new_epoch)
    gaia_cat_corrected = copy.deepcopy(gaia_cat)
    gaia_cat_corrected["ra"] = gaia_coords_corrected.ra
    gaia_cat_corrected["dec"] = gaia_coords_corrected.dec
    new_epoch.format = "jyear"
    gaia_cat_corrected["ref_epoch"] = new_epoch.value
    return gaia_cat_corrected


def generate_astrometry_indices(
        cat_name: str, cat: Union[str, table.Table],
        output_file_prefix: str,
        unique_id_prefix: int,
        index_output_dir: str,
        fits_cat_output: str = None,
        p_lower: int = -1, p_upper: int = 2):
    u.mkdir_check(index_output_dir)
    cat_name = cat_name.lower()
    if fits_cat_output is None and isinstance(cat, str):
        if cat.endswith(".csv"):
            fits_cat_output = cat.replace(".csv", ".fits")
        else:
            fits_cat_output = cat + ".fits"
    elif fits_cat_output is None:
        raise ValueError("fits_cat_output must be provided if cat is given as a Table instead of a path.")
    cat = u.path_or_table(cat, fmt="ascii.csv")
    cols = cat_columns(cat=cat_name, f="rank")
    cat.write(fits_cat_output, format='fits', overwrite=True)
    unique_id_prefix = str(unique_id_prefix)
    for scale in range(p_lower, p_upper + 1):
        unique_id = unique_id_prefix + str(scale).replace("-", "0")
        unique_id = int(unique_id)
        output_file_name_scale = f"{output_file_prefix}_{scale}"
        try:
            astrometry_net.build_astrometry_index(
                input_fits_catalog=fits_cat_output,
                unique_id=unique_id,
                output_index=os.path.join(index_output_dir, output_file_name_scale),
                scale_number=scale,
                sort_column=cols["mag_auto"],
                scan_through_catalog=True
            )
        except SystemError:
            print(f"Building index for scale {scale} failed.")


def attempt_skycoord(coord: Union[SkyCoord, str, tuple, list, np.ndarray]):
    if type(coord) is SkyCoord:
        return coord
    elif type(coord) is str:
        return SkyCoord(coord)
    elif type(coord) in [tuple, list, np.ndarray]:
        if isinstance(coord[0], float):
            coord = (coord[0] * units.deg, coord[1] * units.deg)
        elif isinstance(coord[0], str):
            if coord[0][-1].isnumeric():
                coord = (coord[0] + "d", coord[1] + "d")
        return SkyCoord(coord[0], coord[1])
    else:
        raise TypeError(f"coord is {type(coord)}; must be of type SkyCoord, str, tuple, list, or numpy array")


def coord_string(coord: SkyCoord):
    s = coord.to_string("hmsdms")
    ra = s[:s.find(" ")]
    dec = s[s.find(" ") + 1:]
    return ra, dec


def calculate_error_ellipse(frb: Union[str, dict], error: str = 'quadrature'):
    """
    Calculates the parameters of the uncertainty ellipse of an FRB, for use in plotting.
    :param frb: Either a string specifying the FRB, which must have a corresponding .yaml file in /param/FRBs, or a
        dictionary containing the same information.
    :param error: String specifying the type of error calculation to use. Available options are 'quadrature', which
        provides the quadrature sum of statistical and systematic uncertainty; 'systematic'; and 'statistical'.
    :return: (a, b, theta) as floats in a tuple, in units of degrees.
    """
    if error == 'quadrature':

        if type(frb) is str:
            frb = p.object_params_frb(frb)

        if frb['burst_err_stat_a'] == 0.0 or frb['burst_err_stat_b'] == 0.0:

            ra_frb = frb['burst_dec']
            dec_frb = frb['burst_dec']

            ra_stat = frb['burst_err_stat_ra']
            ra_sys = frb['burst_err_sys_ra']
            ra = np.sqrt(ra_stat ** 2 + ra_sys ** 2)

            a = SkyCoord(f'0h0m0s {dec_frb}d').separation(SkyCoord(f'0h0m{ra}s {dec_frb}d')).value

            dec_stat = frb['burst_err_stat_dec'] / 3600
            dec_sys = frb['burst_err_sys_dec'] / 3600
            dec = np.sqrt(dec_stat ** 2 + dec_sys ** 2)
            b = SkyCoord(f'{ra_frb}d {dec_frb}d').separation(SkyCoord(f'{ra_frb}d {dec_frb + dec}d')).value

            theta = 0.0

        else:

            a_stat = frb['burst_err_stat_a']
            a_sys = frb['burst_err_sys_a']
            a = np.sqrt(a_stat ** 2 + a_sys ** 2) / 3600

            b_stat = frb['burst_err_stat_b']
            b_sys = frb['burst_err_sys_b']
            b = np.sqrt(b_stat ** 2 + b_sys ** 2) / 3600

            theta = frb['burst_err_theta']

    elif error == 'systematic':

        if frb['burst_err_stat_a'] == 0.0 or frb['burst_err_stat_b'] == 0.0:

            ra_frb = frb['burst_dec']
            dec_frb = frb['burst_dec']

            ra_sys = frb['burst_err_sys_ra']

            a = SkyCoord(f'0h0m0s {dec_frb}d').separation(SkyCoord(f'0h0m{ra_sys}s {dec_frb}d')).value

            dec_sys = frb['burst_err_sys_dec'] / 3600
            b = SkyCoord(f'{ra_frb}d {dec_frb}d').separation(SkyCoord(f'{ra_frb}d {dec_frb + dec_sys}d')).value

            theta = 0.0

        else:
            a = frb['burst_err_sys_a'] / 3600
            b = frb['burst_err_sys_b'] / 3600
            theta = frb['burst_err_theta']

    elif error == 'statistical':

        if frb['burst_err_stat_a'] == 0.0 or frb['burst_err_stat_b'] == 0.0:

            ra_frb = frb['burst_dec']
            dec_frb = frb['burst_dec']

            ra_stat = frb['burst_err_stat_ra']

            a = SkyCoord(f'0h0m0s {dec_frb}d').separation(SkyCoord(f'0h0m{ra_stat}s {dec_frb}d')).value

            dec_stat = frb['burst_err_stat_dec'] / 3600
            b = SkyCoord(f'{ra_frb}d {dec_frb}d').separation(SkyCoord(f'{ra_frb}d {dec_frb + dec_stat}d')).value

            theta = 0.0

        else:
            a = frb['burst_err_stat_a'] / 3600
            b = frb['burst_err_stat_b'] / 3600
            theta = frb['burst_err_theta']

    else:
        raise ValueError('error type not recognised.')

    # print(a * 3600)
    # print(b * 3600)

    return a, b, theta


def offset_astrometry(hdu: fits.hdu, offset_ra: float, offset_dec: float, output: str):
    """
    Offsets the astrometric solution of a fits HDU by the specified amounts.
    :param hdu: Astropy HDU object to be offset.
    :param offset_ra: Amount to offset Right Ascension by, in degrees.
    :param offset_dec: Amount to offset Declination by, in degrees.
    :param output: String specifying the output directory.
    :return:
    """
    hdu, path = ff.path_or_hdu(hdu)
    print(offset_ra, offset_dec)
    print('Writing tweaked file to:')
    print('\t', output)
    print(hdu[0].header['CRVAL1'], hdu[0].header['CRVAL2'])
    hdu[0].header['CRVAL1'] = hdu[0].header['CRVAL1'] - offset_ra
    hdu[0].header['CRVAL2'] = hdu[0].header['CRVAL2'] - offset_dec
    print(hdu[0].header['CRVAL1'], hdu[0].header['CRVAL2'])

    hdu.writeto(output, overwrite=True)

    if path:
        hdu.close()

    return hdu


def find_nearest(coord: SkyCoord, search_coords: SkyCoord):
    separations = coord.separation(search_coords)
    match_id = np.argmin(separations)
    return match_id, separations[match_id]


def sanitise_coord(
        cat: table.Table,
        dec_col: str,
):
    print("len(cat) sanitise_coord:", len(cat))
    if isinstance(cat[dec_col][0], units.Quantity):
        upper = 90 * units.deg
        lower = -90 * units.deg
    else:
        upper = 90
        lower = -90

    cat = cat[cat[dec_col] <= upper]
    cat = cat[cat[dec_col] >= lower]
    return cat


def match_catalogs(
        cat_1: table.Table, cat_2: table.Table,
        ra_col_1: str = "RA", dec_col_1: str = "DEC",
        ra_col_2: str = "ra", dec_col_2: str = "dec",
        tolerance: units.Quantity = 1 * units.arcsec
):
    # Clean out any invalid declinations
    u.debug_print(2, "match_catalogs(): type(cat_1) ==", type(cat_1), "type(cat_2) ==", type(cat_2))
    print("len(cat_1) match_catalogs:", len(cat_1))
    cat_1 = sanitise_coord(cat_1, dec_col_1)
    cat_2 = sanitise_coord(cat_2, dec_col_2)

    coords_1 = SkyCoord(cat_1[ra_col_1], cat_1[dec_col_1])
    coords_2 = SkyCoord(cat_2[ra_col_2], cat_2[dec_col_2])

    idx, distance, _ = coords_2.match_to_catalog_sky(coords_1)
    keep = distance < tolerance
    idx = idx[keep]
    matches_2 = cat_2[keep]
    distance = distance[keep]

    matches_1 = cat_1[idx]

    return matches_1, matches_2, distance
