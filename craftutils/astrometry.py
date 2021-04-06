# Code by Lachlan Marnoch, 2019

from astropy import table
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt
import numpy as np

from craftutils import fits_files as ff
from craftutils import params as p
from craftutils import utils as u
from craftutils import plotting as pl

from typing import Union


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

    norm = pl.nice_norm(data)
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
