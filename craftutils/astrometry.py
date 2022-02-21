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
        p_lower: int = 0, p_upper: int = 2):
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


def tweak(sextractor_path: str, destination: str, image_path: str, cat_path: str, cat_name: str, tolerance: float = 10.,
          show: bool = False,
          stars_only: bool = False,
          manual: bool = False, offset_x: float = None, offset_y: float = None, offsets_world: bool = False,
          psf: bool = True,
          specific_star: bool = False, star_ra: float = None, star_dec: float = None
          ):
    """
    For tweaking the astrometric solution of a fits image using a catalogue; either matches as many stars as possible
    and uses the median offset, uses a single star at a specified position (specific_star=True) or a manual offset.
    :param sextractor_path: Path to SExtractor-generated catalogue.
    :param destination: Directory to write tweaked image to.
    :param image_path: Path to image FITS file
    :param cat_path: Path to file containing catalogue.
    :param cat_name: Name of catalogue used.
    :param tolerance: Tolerance, in pixels, within which matches will be accepted.
    :param show: Plot matches onscreen?
    :param stars_only: Only match using stars, determined using SExtractor's 'class_star' output.
    :param manual: Use manual offset?
    :param offset_x: Offset in x to use; only if 'manual' is set to True.
    :param offset_y: Offset in y to use; only if 'manual' is set to True.
    :param offsets_world: If True, interprets the offsets as being given in World Coordinates (RA and DEC)
    :param psf: Use the PSF-fitting position?
    :param specific_star: Use a specific star to tweak? This means, instead of finding the closest matches for many
        stars, alignment is attempted with a single star nearest the given position.
    :param star_ra: Right Ascension of star for single-star alignment.
    :param star_dec: Declination of star for single-star alignment.
    :return: None
    """
    param_dict = {}

    print(image_path)
    image = fits.open(image_path)
    header = image[0].header
    data = image[0].data

    wcs_info = wcs.WCS(header=header)
    # Set the appropriate column names depending on the format of the catalogue to use.
    if cat_name == 'DES':
        ra_name = 'RA'
        dec_name = 'DEC'
    elif cat_name == 'Gaia' or cat_name == 'SDSS':
        ra_name = 'ra'
        dec_name = 'dec'
    else:
        if psf:
            ra_name = 'ra_psf'
            dec_name = 'dec_psf'
        else:
            ra_name = 'ra'
            dec_name = 'dec'

    if psf:
        names = p.sextractor_names_psf()
        sextractor_ra_name = 'ra_psf'
        sextractor_dec_name = 'dec_psf'

    else:
        names = p.sextractor_names()
        sextractor_ra_name = 'ra'
        sextractor_dec_name = 'dec'

    if show or not manual:
        if cat_name == 'SExtractor':
            cat = table.Table(np.genfromtxt(cat_path, names=names))
        else:
            cat = table.Table()
            cat = cat.read(cat_path, format='ascii.csv')

        sextracted = table.Table(np.genfromtxt(sextractor_path, names=names))
        if stars_only:
            sextracted = sextracted[sextracted['class_star'] > 0.9]

        cat['x'], cat['y'] = wcs_info.all_world2pix(cat[ra_name], cat[dec_name], 0)
        x, y = wcs_info.all_world2pix(sextracted[sextractor_ra_name], sextracted[sextractor_dec_name], 0)

    norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=SqrtStretch())
    if show:
        # plt.subplot(projection=wcs_info)
        plt.imshow(data, norm=norm, origin='lower', cmap='viridis')
        plt.scatter(x, y, label='SExtractor', c='violet')
        plt.scatter(cat['x'], cat['y'], label=cat_name, c='red')
        plt.legend()
        plt.title('Pre-Correction')
        plt.show()

    if manual:

        if offset_x is not None and offset_y is not None:

            if not offsets_world:

                scale_ra, scale_dec = ff.get_pixel_scale(image)

                offset_ra = -offset_x * abs(scale_ra)
                offset_dec = -offset_y * abs(scale_dec)

                print(offset_x, offset_y)

            else:

                offset_ra = offset_x
                offset_dec = offset_y

        else:

            print('Set offsets in epoch .yaml file')

            offset_ra = offset_dec = None

    elif specific_star:

        if star_ra is not None and star_dec is not None:
            print(star_ra, star_dec)
            star_cat, d = u.find_object(x=star_ra, y=star_dec, x_search=cat[ra_name], y_search=cat[dec_name])
            print('MD:', d)
            star_cat = cat[star_cat]
            star_sex, d = u.find_object(x=star_ra, y=star_dec, x_search=sextracted[sextractor_ra_name],
                                        y_search=sextracted[sextractor_dec_name])
            print('MD:', d)
            star_sex = sextracted[star_sex]
            offset_ra = (star_sex[sextractor_ra_name] - star_cat[ra_name])
            offset_dec = (star_sex[sextractor_dec_name] - star_cat[dec_name])
        else:
            raise ValueError('If doing specific_star, must specify star_ra and star_dec')

    else:

        match_ids, match_ids_cat = u.match_cat(x_match=sextracted['x'], y_match=sextracted['y'],
                                               x_cat=cat['x'],
                                               y_cat=cat['y'], tolerance=tolerance)

        sextracted = sextracted[match_ids]
        cat = cat[match_ids_cat]

        offsets_ra = sextracted[sextractor_ra_name] - cat[ra_name]
        offsets_dec = sextracted[sextractor_dec_name] - cat[dec_name]

        offset_ra = np.nanmedian(offsets_ra)
        offset_dec = np.nanmedian(offsets_dec)

    # TODO: This is quick and dirty and will only work if the image is approximately oriented along x=RA
    #  and y=DEC (CROTA ~ 0)

    if offset_ra is not None and offset_dec is not None:

        param_dict['offset_ra'] = float(offset_ra)
        param_dict['offset_dec'] = float(offset_dec)

        image = offset_astrometry(hdu=image, offset_ra=offset_ra, offset_dec=offset_dec, output=destination)

        if show and not manual:
            wcs_info = wcs.WCS(header=header)
            cat['x'], cat['y'] = wcs_info.all_world2pix(cat[ra_name], cat[dec_name], 0)
            x, y = wcs_info.all_world2pix(sextracted[sextractor_ra_name] - offset_ra,
                                          sextracted[sextractor_dec_name] - offset_dec, 0)

        if show:
            plt.subplot(projection=wcs_info)
            plt.imshow(data, norm=norm, origin='lower', cmap='viridis')
            plt.scatter(x, y, label='SExtractor', c='violet')
            plt.scatter(cat['x'], cat['y'], label=cat_name, c='red')
            plt.legend()
            plt.title('Post-Correction')
            plt.show()

    image.close()
    return param_dict


def tweak_final(sextractor_path: str, destination: str,
                epoch: str, instrument: str,
                show: bool, tolerance: float = 10.,
                output_suffix: str = 'astrometry', input_suffix='coadded',
                stars_only: bool = False, path_add: str = 'subtraction_image',
                manual: bool = False, specific_star: bool = False):
    """
    A wrapper for tweak, to interface with the .yaml param files and provide offsets to all filters used in an
    observation.
    :param sextractor_path: Path to SExtractor-generated catalogue.
    :param destination: Directory to write tweaked image to.
    :param epoch: The epoch number of the observation to be tweaked.
    :param instrument: The instrument on which the observation was taken.
    :param show: Plot matches onscreen?
    :param tolerance: Tolerance, in pixels, within which matches will be accepted.
    :param output_suffix: String to append to filename of output file.
    :param input_suffix: Suffix appended to filenmae of input file.
    :param stars_only: Only match using stars, determined using SExtractor's 'class_star' output.
    :param path_add: Key under which to add the output path in the 'output_paths.yaml' file.
    :param manual: Use manual offset?
    :param specific_star: Use a specific star to tweak? This means, instead of finding the closest matches for many
        stars, alignment is attempted with a single star nearest the given position.
    :return: None
    """
    u.mkdir_check(destination)
    properties = p.object_params_instrument(obj=epoch, instrument=instrument)
    frb_properties = p.object_params_frb(obj=epoch[:-2])
    cat_name = properties['cat_field_name']
    manual = properties['manual_astrometry'] and manual
    cat_path = frb_properties['data_dir'] + cat_name.upper() + "/" + cat_name.upper() + ".csv"

    if cat_path is not None:

        outputs = p.object_output_params(obj=epoch, instrument=instrument)
        filters = outputs['filters']

        param_dict = {}

        for filt in filters:

            print(filt)

            f = filt[0]

            if manual:

                offset_x = properties[f + '_offset_x']
                offset_y = properties[f + '_offset_y']

            else:
                offset_x = None
                offset_y = None

            if specific_star:
                burst_properties = p.object_params_frb(epoch[:-2])
                star_ra = burst_properties['alignment_ra']
                star_dec = burst_properties['alignment_dec']

            else:
                star_ra = star_dec = None

            param = tweak(sextractor_path=sextractor_path + f + '_psf-fit.cat',
                          destination=destination + f + '_' + output_suffix + '.fits',
                          image_path=destination + f + '_' + input_suffix + '.fits',
                          cat_path=cat_path,
                          cat_name=cat_name, tolerance=tolerance, show=show, stars_only=stars_only, manual=manual,
                          offset_x=offset_x, offset_y=offset_y, specific_star=specific_star, star_ra=star_ra,
                          star_dec=star_dec)

            for par in param:
                param_dict[f + '_' + par] = param[par]

            p.add_output_path(obj=epoch, key=f + '_' + path_add,
                              path=destination + f + '_' + output_suffix + '.fits',
                              instrument=instrument)

        p.add_params(properties['data_dir'] + 'output_values.yaml', params=param_dict)

    else:
        print('No catalogue found for this alignment.')


def find_nearest(coord: SkyCoord, search_coords: SkyCoord):
    separations = coord.separation(search_coords)
    match_id = np.argmin(separations)
    return match_id, separations[match_id]


def match_catalogs(
        cat_1: table.Table, cat_2: table.Table,
        ra_col_1: str = "ALPHAPSF_SKY", dec_col_1: str = "DELTAPSF_SKY",
        ra_col_2: str = "ra", dec_col_2: str = "dec",
        tolerance: units.Quantity = 1 * units.arcsec
):
    # Clean out any invalid declinations
    u.debug_print(2, "match_catalogs(): type(cat_1) ==", type(cat_1), "type(cat_2) ==", type(cat_2))
    cat_1 = cat_1[cat_1[dec_col_1] <= 90 * units.deg]
    cat_1 = cat_1[cat_1[dec_col_1] >= -90 * units.deg]

    cat_2 = cat_2[cat_2[dec_col_2] <= 90 * units.deg]
    cat_2 = cat_2[cat_2[dec_col_2] >= -90 * units.deg]

    coords_1 = SkyCoord(cat_1[ra_col_1], cat_1[dec_col_1])
    coords_2 = SkyCoord(cat_2[ra_col_2], cat_2[dec_col_2])

    idx, distance, _ = coords_2.match_to_catalog_sky(coords_1)
    keep = distance < tolerance
    idx = idx[keep]
    matches_2 = cat_2[keep]
    distance = distance[keep]

    matches_1 = cat_1[idx]

    return matches_1, matches_2, distance
