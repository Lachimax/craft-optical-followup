# Code by Lachlan Marnoch, 2019-2023
import copy
import os
import math
from typing import Union, List, Iterable
from copy import deepcopy

import numpy as np
import photutils as ph
from photutils.datasets import make_model_image
import matplotlib.pyplot as plt
from scipy.ndimage import shift

import astropy.stats as stats
import astropy.wcs as wcs
import astropy.table as table
import astropy.io.fits as fits
import astropy.time as time
import astropy.units as units
import astropy.constants as constants
from astropy.coordinates import SkyCoord
from astropy.modeling import models, fitting
from astropy.modeling.functional_models import Sersic1D
from astropy.stats import sigma_clip
from astropy.visualization import quantity_support
import astropy.cosmology as cosmology

import craftutils.fits_files as ff
import craftutils.params as p
import craftutils.utils as u
import craftutils.plotting as plotting
import craftutils.astrometry as a

gain_unit = units.electron / units.ct

__all__ = []


@u.export
def image_psf_diagnostics(
        hdu: Union[str, fits.HDUList],
        cat: Union[str, table.Table],
        star_class_tol: float = 0.95,
        mag_max: float = 0.0 * units.mag,
        mag_min: float = -50. * units.mag,
        match_to: table.Table = None,
        match_tolerance: units.Quantity = 1 * units.arcsec,
        frame: float = 15,
        ext: int = 0,
        near_centre: SkyCoord = None,
        near_radius: units.Quantity = None,
        ra_col: str = "RA",
        dec_col: str = "DEC",
        output: str = None,
        min_stars: int = 30,
        plot_file_prefix: str = "",
        debug_plots: bool = False
):
    """

    :param hdu:
    :param cat:
    :param star_class_tol:
    :param mag_max:
    :param mag_min:
    :param match_to:
    :param match_tolerance:
    :param frame:
    :param ext:
    :param near_centre:
    :param near_radius:
    :param ra_col:
    :param dec_col:
    :param output:
    :param min_stars:
    :param plot_file_prefix:
    :return:
    """
    hdu, path = ff.path_or_hdu(hdu=hdu)
    hdu = copy.deepcopy(hdu)
    cat = u.path_or_table(cat)

    # stars = u.trim_to_class(cat=cat, modify=True, allowed=np.arange(0, star_class_tol + 1))
    stars = cat[cat["CLASS_STAR"] >= star_class_tol]
    print(f"Initial num stars:", len(stars))
    print("star_class_tol:", star_class_tol)
    if "MAG_PSF" in stars.colnames:
        mag_col = "MAG_PSF"
    else:
        mag_col = "MAG_AUTO"
    stars = stars[stars[mag_col] < mag_max]
    print(f"Num stars with {mag_col} < {mag_max}:", len(stars))
    stars = stars[stars[mag_col] > mag_min]
    print(f"Num stars with {mag_col} > {mag_min}:", len(stars))

    if near_radius is not None:
        header = hdu[ext].header
        wcs_this = wcs.WCS(header)
        if near_centre is None:
            near_centre = SkyCoord.from_pixel(header["NAXIS1"] / 2, header["NAXIS2"] / 2, wcs_this, origin=1)

        ra = stars[ra_col]
        dec = stars[dec_col]
        coords = SkyCoord(ra, dec)

        img_width = max(header["NAXIS1"], header["NAXIS2"]) * units.pix
        _, scale = ff.get_pixel_scale(hdu, ext=ext, astropy_units=True)
        img_width_ang = img_width.to(units.arcsec, scale)
        stars_near = stars[near_centre.separation(coords) < near_radius]
        print("len(stars_near):", len(stars_near), "near_radius:", near_radius, "img_width_ang:", img_width_ang)
        while len(stars_near) < min_stars and near_radius < img_width_ang:
            stars_near = stars[near_centre.separation(coords) < near_radius]
            print(f"Num stars within {near_radius} of {near_centre}:", len(stars_near))
            near_radius += 0.5 * units.arcmin
        stars = stars_near

    if match_to is not None:
        stars, stars_match, distance = a.match_catalogs(
            cat_1=stars,
            cat_2=match_to,
            ra_col_1=ra_col,
            dec_col_1=dec_col,
            ra_col_2=ra_col,
            dec_col_2=dec_col,
            tolerance=match_tolerance)

        print(f"Num stars after match to other sextractor cat:", len(stars))

    for colname in ["GAUSSIAN_FWHM_FITTED", "MOFFAT_FWHM_FITTED", "MOFFAT_ALPHA_FITTED", "MOFFAT_GAMMA_FITTED"]:
        if colname not in stars.colnames:
            stars.add_column(np.zeros(len(stars)), name=colname)

    if type(stars) is table.QTable:
        if not isinstance(stars["GAUSSIAN_FWHM_FITTED"], units.Quantity):
            stars["GAUSSIAN_FWHM_FITTED"] *= units.arcsec
        if not isinstance(stars["MOFFAT_FWHM_FITTED"], units.Quantity):
            stars["MOFFAT_FWHM_FITTED"] *= units.arcsec

    print()

    for j, star in enumerate(stars):
        ra = star[ra_col]
        dec = star[dec_col]

        # print(star)

        # print(ra)
        # print(dec)

        window = ff.trim_frame_point(hdu=hdu, ra=ra, dec=dec, frame=frame, ext=ext)
        if debug_plots and output is not None:
            plot_dir = os.path.join(output, "debug_plots")
            u.mkdir_check(plot_dir)
            # window.writeto(os.path.join(plot_dir, f"{star['NUMBER']}.fits", overwrite=True))
        data = window[ext].data
        if debug_plots and output is not None:
            fig, ax = plt.subplots()
            ax.imshow(data)
            fig.savefig(os.path.join(plot_dir, f"{star['NUMBER']}_data.png"))
        _, scale = ff.get_pixel_scale(hdu, astropy_units=True, ext=ext)

        mean, median, stddev = stats.sigma_clipped_stats(data)
        data -= median
        data[~np.isfinite(data)] = np.nanmedian(data)

        y, x = np.mgrid[:data.shape[0], :data.shape[1]]

        # Fit the data using astropy.modeling

        # First, a Moffat function
        model_init = models.Moffat2D(x_0=frame, y_0=frame)
        fitter = fitting.LevMarLSQFitter()
        try:
            model = fitter(model_init, x, y, data)
            # print("fit_info:", fitter.fit_info)
            # If astropy thinks that the fit is bad, who are we to argue?
            if "Number of calls to function has reached maxfev = 100" in fitter.fit_info['message']:
                star["MOFFAT_FWHM_FITTED"] = np.nan
                star["MOFFAT_GAMMA_FITTED"] = np.nan
                star["MOFFAT_ALPHA_FITTED"] = np.nan
            else:
                fwhm = (model.fwhm * units.pixel).to(units.arcsec, scale)
                star["MOFFAT_FWHM_FITTED"] = fwhm
                star["MOFFAT_GAMMA_FITTED"] = model.gamma.value
                star["MOFFAT_ALPHA_FITTED"] = model.alpha.value

                if debug_plots and output is not None:
                    fig = plt.figure()
                    ax_data_moffat = fig.add_subplot(2, 3, 1)
                    ax_data_moffat.imshow(data)
                    ax_model_moffat = fig.add_subplot(2, 3, 2)
                    ax_model_moffat.imshow(model(x, y))
                    ax_residuals_moffat = fig.add_subplot(2, 3, 3)
                    ax_residuals_moffat.imshow(data - model(x, y))

        except fitting.NonFiniteValueError:
            star["MOFFAT_FWHM_FITTED"] = np.nan
            star["MOFFAT_GAMMA_FITTED"] = np.nan
            star["MOFFAT_ALPHA_FITTED"] = np.nan

        # Then a good-old-fashioned Gaussian, with the x and y axes tied together.
        model_init = models.Gaussian2D(x_mean=frame, y_mean=frame)

        def tie_stddev(mod):
            return mod.x_stddev

        model_init.y_stddev.tied = tie_stddev

        model = fitter(model_init, x, y, data)
        fwhm = (model.x_fwhm * units.pixel).to(units.arcsec, scale)
        # print(fwhm)
        # print(star["GAUSSIAN_FWHM_FITTED"])
        star["GAUSSIAN_FWHM_FITTED"] = fwhm
        # print(star["GAUSSIAN_FWHM_FITTED"])
        stars[j] = star
        # print(stars[j]["GAUSSIAN_FWHM_FITTED"])
        # print(stars[j])
        # print()
        if debug_plots and output is not None:
            ax_data_gauss = fig.add_subplot(2, 3, 4)
            ax_data_gauss.imshow(data)
            ax_model_gauss = fig.add_subplot(2, 3, 5)
            ax_model_gauss.imshow(model(x, y))
            ax_residuals_gauss = fig.add_subplot(2, 3, 6)
            ax_residuals_gauss.imshow(data - model(x, y))
            fig.savefig(os.path.join(plot_dir, str(star["NUMBER"]) + ".png"))

    # print()

    cols_check = ("MOFFAT_FWHM_FITTED", "GAUSSIAN_FWHM_FITTED", "FWHM_WORLD")
    clip_dict = {}

    for col in cols_check:
        if col in stars.colnames:
            clipped = sigma_clip(stars[col], masked=True, sigma=2)
            stars_clip = stars[~clipped.mask]
            stars_clip = stars_clip[np.isfinite(stars_clip[col])]
            stars_clip = stars_clip[stars_clip[col] > 0.1 * units.arcsec]
            clip_dict[col] = stars_clip
            print(f"Num stars after sigma clipping {col}:", len(stars_clip))

    plt.close()
    u.debug_print(2, f"image_psf_diagnostics(): {output=}")
    if output is not None:

        with quantity_support():
            for colname in clip_dict:
                fig, ax = plt.subplots()
                ax.hist(
                    stars[colname][np.isfinite(stars[colname])].to(units.arcsec),
                    label="Full sample",
                    bins="auto"
                )
                ax.legend()
                fig.savefig(os.path.join(output, f"{plot_file_prefix}_psf_histogram_{colname}_full.png"))

                fig, ax = plt.subplots()
                ax.hist(
                    clip_dict[colname][colname].to(units.arcsec),
                    edgecolor='black',
                    linewidth=1.2,
                    label="Sigma-clipped",
                    fc=(0, 0, 0, 0),
                    bins="auto"
                )
                ax.legend()
                fig.savefig(os.path.join(output, f"{plot_file_prefix}_psf_histogram_{colname}_clipped.png"))

    return clip_dict


def get_median_background(image: Union[str, fits.HDUList], ra: float = None, dec: float = None, frame: int = 100,
                          show: bool = False, output: str = None):
    file, path = ff.path_or_hdu(image)
    data = file[0].data
    if ra is not None and dec is not None and frame is not None:
        wcs_this = wcs.WCS(header=file[0].header)
        if not SkyCoord(f"{ra}d {dec}d").contained_by(wcs_this):
            raise ValueError("RA and DEC must be within image footprint")
        x, y = wcs_this.all_world2pix(ra, dec, 0)
        bottom, top, left, right = ff.subimage_edges(data=data, x=x, y=y, frame=frame)
        back_patch = data[bottom:top, left:right]
    else:
        back_patch = data

    if path is not None:
        file.close()
    plt.imshow(back_patch, origin='lower')
    if output is not None:
        plt.savefig(output + "patch.jpg")
    if show:
        plt.show()
    plt.close()
    plt.hist(back_patch.flatten())
    if output is not None:
        plt.savefig(output + "hist.jpg")
    if show:
        plt.show()
    plt.close()
    return np.nanmedian(back_patch)


def fit_background(
        data: np.ndarray,
        model_type='polynomial',
        deg: int = 2,
        footprint: List[int] = None,
        weights: np.ndarray = None):
    """

    :param data:
    :param model_type:
    :param deg:
    :param footprint: Piece of image to use in the fit. Should have format of (y_min, y_max, x_min, x_max). Blame numpy for the ordering.
    :return:
    """
    if footprint is not None and len(footprint) != 4:
        raise ValueError("Footprint should be a tuple of four integers.")
    if footprint is None:
        footprint = [0, data.shape[0], 0, data.shape[1]]
    else:
        footprint[0], footprint[1], footprint[2], footprint[3] = ff.check_subimage_edges(data=data,
                                                                                         bottom=footprint[0],
                                                                                         top=footprint[1],
                                                                                         left=footprint[2],
                                                                                         right=footprint[3])
    accepted_models = ['polynomial', 'gaussian']
    for i, side in enumerate(footprint):
        if side < 0:
            footprint[i] = 0
        elif side > data.shape[1]:
            footprint[i] = data.shape[1]
    y, x = np.mgrid[footprint[0]:footprint[1], footprint[2]:footprint[3]]
    y_large, x_large = np.mgrid[:data.shape[0], :data.shape[1]]
    if model_type.lower() == 'polynomial':
        init = models.Polynomial2D(degree=deg)
    elif model_type.lower() == 'gaussian':
        init = models.Gaussian2D()
    else:
        raise ValueError("Unrecognised model; must be in", accepted_models)
    fitter = fitting.LevMarLSQFitter()
    if weights is not None:
        weights = weights[footprint[0]:footprint[1], footprint[2]:footprint[3]]
    model = fitter(init, x, y, data[footprint[0]:footprint[1], footprint[2]:footprint[3]],
                   weights=weights)

    # std_err = u.root_mean_squared_error(model_values=model(x,y).flatten(), obs_values=data[footprint[0]:footprint[1], footprint[2]:footprint[3]].flatten())
    return model(x, y), model(x_large, y_large), model


def fit_background_fits(
        image: Union[str, fits.HDUList], model_type='polynomial', local: bool = True, global_sub=False,
        centre_x: int = None, centre_y: int = None,
        frame: int = 50,
        deg: int = 3, weights: np.ndarray = None
):
    image, _ = ff.path_or_hdu(image)
    data = image[0].data

    if local:
        if centre_x is None:
            centre_x = int(data.shape[1] / 2)
        if centre_y is None:
            centre_y = int(data.shape[0] / 2)
        bottom, top, left, right = ff.subimage_edges(data=data, x=centre_x, y=centre_y, frame=frame)
        footprint = [bottom, top, left, right]
    else:
        footprint = None
    background, background_large, model = fit_background(data=data, model_type=model_type, deg=deg, footprint=footprint,
                                                         weights=weights)
    background_image = deepcopy(image)
    if local:
        if global_sub:
            background_image[0].data = background_large
        else:
            background_image[0].data = np.zeros(shape=image[0].data.shape)
            background_image[0].data[footprint[0]:footprint[1], footprint[2]:footprint[3]] = background
    else:
        background_image[0].data = background
    return background_image


def gain_median_combine(old_gain: float = 0.8, n_frames: int = 1):
    """
    Only valid if you have a median-combined image.
    :param old_gain: Gain of the individual images.
    :param n_frames: Number of stacked images.
    :return:
    """
    return 2 * n_frames * old_gain / 3


def gain_mean_combine(old_gain: float = 0.8, n_frames: int = 1):
    return n_frames * old_gain


AB_zeropoint = 3631 * units.Jy


def redshift_frequency(nu: units.Quantity, z: float, z_new: float):
    return nu * (1 + z) / (1 + z_new)


def redshift_wavelength(wavelength: units.Quantity, z: float, z_new: float):
    return wavelength * (1 + z_new) / (1 + z)


def redshift_flux_nu(
        flux,
        z: float,
        z_new: float,
        cosmo: cosmology.LambdaCDM = cosmology.Planck18,
):
    d_l = cosmo.luminosity_distance(z)
    d_l_shift = cosmo.luminosity_distance(z_new)

    return flux * ((1 + z_new) * d_l ** 2) / ((1 + z) * d_l_shift ** 2)


def redshift_flux_lambda(
        flux,
        z: float,
        z_new: float,
        cosmo: cosmology.LambdaCDM = cosmology.Planck18,
):
    d_l = cosmo.luminosity_distance(z)
    d_l_shift = cosmo.luminosity_distance(z_new)

    return flux * ((1 + z) * d_l ** 2) / ((1 + z_new) * d_l_shift ** 2)


def flux_ab(
        tbl: table.QTable = None,
        transmission: Union[np.ndarray, units.Quantity] = None,
        frequency: units.Quantity = None,
):
    """Calculates the total integrated flux of the flat AB source (3631 Jy) as seen through a given filter;
    that is, the denominator of the AB Magnitude formula.
    `transmission` and `frequency`, whether provided in `tbl` or as separate arguments, must be the same length and
    correspond 1-to-1.

    :param tbl: an astropy `Table` with `transmission` and/or `frequency` as columns; when these columns are provided,
        their respective arguments can be ignored, but if either is missing from the table it must be provided in an
        argument.
    :param transmission: the filter profile as an array of transmission fractions as a function of frequency.
    :param frequency: the corresponding frequency coordinates for the transmission argument.
    :return: The integrated AB flux.
    """

    if tbl is None:
        if transmission is None:
            raise ValueError(
                "If `tbl` is not provided (with transmission included as column 'transmission'), `transmission` must be provided.")

        if frequency is None:
            raise ValueError(
                "If `tbl` is not provided (with frequency included as column 'frequency'), `frequency` must be provided.")

        tbl = table.QTable(
            data={
                "frequency": frequency,
                "transmission": transmission,
            }
        )
    tbl["flux"] = np.ones(len(tbl)) * AB_zeropoint

    return flux_from_band(
        tbl,
        use_quantum_factor=True
    )


def flux_from_band(
        flux: Union[units.Quantity, table.QTable],
        transmission: Union[np.ndarray, units.Quantity] = None,
        frequency: units.Quantity = None,
        use_quantum_factor: bool = True
):
    """
    Calculates the integrated flux of an object with an SED `flux` as a function of `frequency`, as seen through a
    filter with transmission profile `transmission`.
    `flux`, `transmission` and `frequency` must all be of the same length, with entries corresponding 1-to-1.

    :param flux: flux per unit frequency.
    :param transmission: the filter profile as an array of transmission fractions as a function of frequency.
    :param frequency: the corresponding frequency coordinates for the transmission argument.
    :param use_quantum_factor: For use if flux is in energy-related units (eg, Jy, erg/s, etc.) to convert to photon
        counts. If you are passing an SED flux, then you should set this to True.
    :return: Integrated flux in the bandpass.
    """
    if not isinstance(flux, table.Table):
        if transmission is None:
            raise ValueError(
                "If `flux` is not a table (with transmission included as column 'transmission'), `band_transmission` must be provided.")
        if frequency is None:
            raise ValueError(
                "If `flux` is not a table (with frequency included as column 'frequency'), `frequency` must be provided.")
        if not u.is_iterable(flux):
            flux = flux * np.ones(len(frequency))
        flux_tbl = table.QTable(
            data={
                "frequency": frequency,
                "transmission": transmission,
                "flux": flux
            }
        )
    else:
        if "frequency" not in flux.colnames:
            raise ValueError("If `flux` is a table, frequency must be included as column `frequency`")
        if "transmission" not in flux.colnames:
            raise ValueError("If `flux` is a table, transmission must be included as column `transmission`")
        flux_tbl = flux

    flux_tbl.sort("frequency")

    if use_quantum_factor:
        quantum_factor = (constants.h * flux_tbl["frequency"]) ** -1
    else:
        quantum_factor = 1

    return np.trapz(
        y=flux_tbl["flux"] * quantum_factor * flux_tbl["transmission"],
        x=flux_tbl["frequency"]
    )


def magnitude_AB(
        flux: units.Quantity,
        transmission: Union[np.ndarray, units.Quantity],
        frequency: units.Quantity,
        use_quantum_factor: bool = True,
        mag_unit: bool = False

):
    """
    `flux`, `transmission` and `frequency` must all be of the same length, with entries corresponding 1-to-1.

        :param flux: flux per unit frequency.
    :param transmission: the filter profile as an array of transmission fractions as a function of frequency.
    :param frequency: the corresponding frequency coordinates for the transmission argument.
    :param use_quantum_factor: For use if flux is in energy-related units (eg, Jy, erg/s, etc.) to convert to photon
        counts. If you are passing an SED flux, then you should set this to True.
    """
    flux_tbl = table.QTable(
        data={
            "frequency": frequency,
            "transmission": transmission,
            "flux": flux
        }
    )
    flux_tbl.sort("frequency")

    flux_band = flux_from_band(flux=flux_tbl, use_quantum_factor=use_quantum_factor)
    flux_ab_this = flux_ab(flux_tbl)
    val = -2.5 * np.log10(flux_band / flux_ab_this)
    if mag_unit:
        val *= units.mag
    return val


def magnitude_absolute_from_luminosity(
        luminosity_nu: units.Quantity,
        transmission: Union[np.ndarray, units.Quantity],
        frequency: units.Quantity,
):
    """
    Derived from Equation (5) of Hogg et al 2002 (https://arxiv.org/abs/astro-ph/0210394v1)
    :param luminosity_nu:
    :param transmission:
    :param frequency:
    :return:
    """
    lum_tbl = table.QTable(
        data={
            "frequency": frequency,
            "transmission": transmission,
            "luminosity_nu": luminosity_nu
        }
    )
    lum_tbl.sort("frequency")

    luminosity_band = np.trapz(
        y=lum_tbl["luminosity_nu"] * lum_tbl["transmission"] / lum_tbl["frequency"],
        x=lum_tbl["frequency"]
    )

    ab = np.trapz(
        y=AB_zeropoint * lum_tbl["transmission"] / lum_tbl["frequency"],
        x=lum_tbl["frequency"]
    )

    return -2.5 * np.log10(luminosity_band / (ab * 4 * np.pi * 100 * units.pc ** 2))


def luminosity_in_band(
        luminosity_nu: units.Quantity,
        transmission: Union[np.ndarray, units.Quantity],
        frequency: units.Quantity,
        use_quantum_factor: bool = False
):
    lum_tbl = table.QTable(
        data={
            "frequency": frequency,
            "transmission": transmission,
            "luminosity_nu": luminosity_nu
        }
    )
    lum_tbl.sort("frequency")

    if use_quantum_factor:
        quantum_factor = (constants.h * lum_tbl["frequency"]) ** -1
    else:
        quantum_factor = 1

    return np.trapz(
        y=lum_tbl["luminosity_nu"] * lum_tbl["transmission"] * quantum_factor,
        x=lum_tbl["frequency"]
    )


def magnitude_instrumental(
        flux: units.Quantity,
        flux_err: units.Quantity = 0.0 * units.ct,
        exp_time: units.Quantity = 1. * units.second,
        exp_time_err: units.Quantity = 0.0 * units.second,
        zeropoint: units.Quantity = 0.0 * units.mag,
        zeropoint_err: units.Quantity = 0.0 * units.mag,
        airmass: float = 0.0,
        airmass_err: float = 0.0,
        ext: units.Quantity = 0.0 * units.mag,
        ext_err: units.Quantity = 0.0 * units.mag,
        colour_term: float = 0.0,
        colour_term_err: float = 0.0,
        colour: units.Quantity = 0.0 * units.mag,
        colour_err: units.Quantity = 0.0 * units.mag):
    """
    Returns a photometric magnitude and the calculated error.
    :param flux: Total flux of the object over the exposure time, in counts.
    :param exp_time: Exposure time of the image, in seconds. Set this to 1.0 if the image is normalised to
    counts / second.
    :param zeropoint: The zero point of the instrument / filter.
    :param airmass:
    :param ext:
    :param colour_term:
    :param colour:
    :return:
    """

    # print('Calculating magnitudes...')

    if colour is None:
        colour = 0.0
    if airmass is None:
        airmass = 0.0

    flux = u.check_quantity(flux, units.ct)
    flux_err = u.check_quantity(flux_err, units.ct)
    zeropoint = u.check_quantity(zeropoint, units.mag)
    zeropoint_err = u.check_quantity(zeropoint_err, units.mag)
    exp_time = u.check_quantity(exp_time, units.s)
    exp_time_err = u.check_quantity(exp_time_err, units.s)

    u.debug_print(2, "photometry.magnitude_instrumental():")
    u.debug_print(2, "\texp_time ==", exp_time, "+/-", exp_time_err)
    u.debug_print(2, "\tzeropoint ==", zeropoint, "+/-", zeropoint_err)
    u.debug_print(2, "\tairmass", airmass, "+/-", airmass_err)
    u.debug_print(2, "\textinction", ext, "+/-", ext_err)

    mag_inst, mag_error = magnitude_uncertainty(
        flux=flux, flux_err=flux_err,
        exp_time=exp_time, exp_time_err=exp_time_err
    )
    magnitude = mag_inst + zeropoint - ext * airmass - colour_term * colour
    error_extinction = u.uncertainty_product(ext * airmass, (ext, ext_err), (airmass, airmass_err))
    error_colour = u.uncertainty_product(colour_term * colour, (colour_term, colour_term_err), (colour, colour_err))
    error = u.uncertainty_sum(mag_error, zeropoint_err, error_extinction, error_colour)

    u.debug_print(2, "\textinction", magnitude, "+/-", mag_error)

    return magnitude, error


def magnitude_uncertainty(
        flux: units.Quantity,
        flux_err: units.Quantity = 0.0 * units.ct,
        exp_time: units.Quantity = 1. * units.second,
        exp_time_err: units.Quantity = 0.0 * units.second
):
    flux = u.check_quantity(flux, unit=units.ct)
    flux_err = u.check_quantity(flux_err, unit=units.ct)
    exp_time = u.check_quantity(exp_time, unit=units.second)
    exp_time_err = u.check_quantity(exp_time_err, unit=units.second)

    flux_per_sec = flux / exp_time
    error_fps = u.uncertainty_product(flux_per_sec, (flux, flux_err), (exp_time, exp_time_err))
    mag = units.Magnitude(flux_per_sec).value * units.mag
    error = u.uncertainty_log10(arg=flux_per_sec, uncertainty_arg=error_fps, a=-2.5) * units.mag
    return mag, error


@u.export
def distance_modulus(distance: units.Quantity):
    return (5 * np.log10(distance / units.pc) - 5) * units.mag


def determine_zeropoint_sextractor(
        sextractor_cat: Union[str, table.QTable],
        cat: str,
        image: Union[str, fits.HDUList],
        output_path: str,
        cat_name: str = 'Catalogue',
        image_name: str = 'FORS2',
        show: bool = False,
        cat_ra_col: str = 'RA',
        cat_dec_col: str = 'DEC',
        cat_mag_col: str = 'WAVG_MAG_PSF_',
        cat_mag_col_err: str = 'WAVGERR_MAG_PSF_',
        sex_ra_col='ALPHA_SKY',
        sex_dec_col='DELTA_SKY',
        sex_x_col: str = 'XPSF_IMAGE',
        sex_y_col: str = 'YPSF_IMAGE',
        dist_tol: units.Quantity = 2 * units.arcsec,
        flux_column: str = 'FLUX_PSF',
        flux_err_column: str = 'FLUXERR_PSF',
        mag_range_sex_upper: units.Quantity = 100 * units.mag,
        mag_range_sex_lower: units.Quantity = -100 * units.mag,
        stars_only: bool = True,
        star_class_tol: float = 0.95,
        star_class_type: str = "CLASS_STAR",
        star_class_kwargs={},
        exp_time: float = None,
        y_lower: units.Quantity = 0 * units.pix,
        y_upper: units.Quantity = 100000 * units.pix,
        cat_type: str = 'csv',
        cat_zeropoint: units.Quantity = 0.0 * units.mag,
        cat_zeropoint_err: units.Quantity = 0.0 * units.mag,
        snr_cut: float = 10.,
        snr_col: str = 'SNR_WIN',
        iterate_uncertainty: bool = True,
        do_x_shift: bool = True
):
    """This function expects your catalogue to be a .csv.

    :param sextractor_cat:
    :param cat:
    :param image:
    :param output_path:
    :param cat_name:
    :param image_name:
    :param show:
    :param cat_ra_col:
    :param cat_dec_col:
    :param cat_mag_col:
    :param cat_mag_col_err:
    :param sex_ra_col:
    :param sex_dec_col:
    :param sex_x_col:
    :param sex_y_col:
    :param dist_tol:
    :param flux_column:
    :param flux_err_column:
    :param mag_range_sex_upper:
    :param mag_range_sex_lower:
    :param stars_only:
    :param star_class_tol:
    :param star_class_type:
    :param star_class_kwargs:
    :param exp_time:
    :param y_lower:
    :param y_upper:
    :param cat_type:
    :param cat_zeropoint:
    :param cat_zeropoint_err:
    :param snr_cut:
    :param snr_col:
    :param iterate_uncertainty:
    :param do_x_shift:
    :return:
    """

    plt.rc('text', usetex=False)

    output_path = u.check_trailing_slash(output_path)

    params = {}

    image, path = ff.path_or_hdu(hdu=image)
    params['image_path'] = str(path)

    print("Running zeropoint determination, with:")
    if isinstance(sextractor_cat, str):
        print('SExtractor catalogue path:', sextractor_cat)
    print('Image path:', path)
    print('Output:', output_path)
    print()

    u.mkdir_check(output_path)

    if exp_time is None:
        exp_time = ff.get_exp_time(image)

    # Extract pixel scales from images.
    _, pix_scale = ff.get_pixel_scale(image, astropy_units=True)
    # Get tolerance as angle.
    if dist_tol.unit.is_equivalent(units.pix):
        tolerance = dist_tol.to(units.deg, pix_scale)
    else:
        tolerance = dist_tol

    # Import the catalogue of the sky region.

    if isinstance(cat, str):
        cat_path = cat
        if cat_type != 'sextractor':
            cat = table.QTable.read(cat, format='ascii.csv')
            if cat_mag_col not in cat.colnames:
                print(f"{cat_mag_col} not found in {cat_name}; is this band included?")
                p.save_params(file=output_path + 'parameters.yaml', dictionary=params)
                return None
            cat = cat.filled(fill_value=-999.)
        else:
            cat = table.QTable.read(cat, format="ascii.sextractor")

    elif isinstance(cat, table.QTable):
        cat_path = None

    else:
        cat_path = None
        cat = table.QTable(cat)

    if len(cat) == 0:
        raise ValueError("The reference catalogue is empty.")

    cat[cat_ra_col] = u.check_quantity(cat[cat_ra_col], units.deg)
    cat[cat_dec_col] = u.check_quantity(cat[cat_dec_col], units.deg)
    cat[cat_mag_col] = u.check_quantity(cat[cat_mag_col], units.mag)

    params['time'] = str(time.Time.now())
    params['catalogue'] = str(cat_name)
    params['airmass'] = 0.0
    params['exp_time'] = exp_time = u.check_quantity(exp_time, units.second)
    params['pix_tol'] = dist_tol
    params['ang_tol'] = tolerance = u.check_quantity(tolerance, units.deg)
    #    params['mag_cut_min'] = float(mag_range_cat_lower)
    #    params['mag_cut_max'] = float(mag_range_cat_upper)
    params['cat_path'] = str(cat_path)
    params['cat_ra_col'] = str(cat_ra_col)
    params['cat_dec_col'] = str(cat_dec_col)
    params['cat_flux_col'] = str(cat_mag_col)
    # params['sextractor_path'] = str(sextractor_cat)
    params['sex_ra_col'] = str(sex_ra_col)
    params['sex_dec_col'] = str(sex_dec_col)
    params['sex_flux_col'] = str(flux_column)
    params['stars_only'] = stars_only
    params['y_lower'] = y_lower
    params['y_upper'] = y_upper
    params['cat_zeropoint'] = u.check_quantity(cat_zeropoint, units.mag)
    params['cat_zeropoint_err'] = u.check_quantity(cat_zeropoint_err, units.mag)
    if stars_only:
        params['star_class_kwargs'] = star_class_kwargs

    if isinstance(sextractor_cat, str):
        source_tbl = table.QTable.read(sextractor_cat, format="ascii.sextractor")
    else:
        source_tbl = sextractor_cat

    if flux_column not in sextractor_cat.colnames:
        print(f"Flux column {flux_column} not in provided SE catalogue.")
        p.save_params(file=output_path + 'parameters.yaml', dictionary=params)
        return None

    source_tbl['mag'], source_tbl['mag_err'] = magnitude_instrumental(flux=source_tbl[flux_column],
                                                                      flux_err=source_tbl[flux_err_column],
                                                                      exp_time=exp_time)

    # Plot all stars found by SExtractor.
    plt.close()
    source_tbl = source_tbl[source_tbl[sex_ra_col] != 0.0]
    source_tbl = source_tbl[source_tbl[sex_dec_col] != 0.0]
    plt.scatter(u.dequantify(source_tbl[sex_ra_col]), u.dequantify(source_tbl[sex_dec_col]), label='SExtractor')
    plt.xlim()
    plt.title('Objects found by SExtractor')
    plt.savefig(output_path + '1-sextracted-positions.png')
    if show:
        plt.show()
    plt.close()

    # Plot all catalogue stars.
    plt.scatter(u.dequantify(cat[cat_ra_col]), u.dequantify(cat[cat_dec_col]))
    plt.title('Objects in catalogue')
    plt.savefig(output_path + '2-catalogue-positions.png')
    if show:
        plt.show()
    plt.close()

    plt.scatter(u.dequantify(source_tbl[sex_ra_col]), u.dequantify(source_tbl[sex_dec_col]), label='SExtractor')
    plt.scatter(cat[cat_ra_col], cat[cat_dec_col], label=cat_name)
    plt.legend()
    plt.title('Matches with ' + cat_name + ' Catalogue')
    if show:
        plt.show()
    plt.close()

    # Match stars to catalogue.
    matches, matches_cat, _ = a.match_catalogs(
        cat_1=source_tbl, cat_2=cat,
        ra_col_1=sex_ra_col, ra_col_2=cat_ra_col,
        dec_col_1=sex_dec_col, dec_col_2=cat_dec_col,
        tolerance=tolerance)

    # Plot all matches with catalogue.
    plt.scatter(matches[sex_ra_col], matches[sex_dec_col], label='SExtractor MAG\\_AUTO')
    plt.scatter(matches_cat[cat_ra_col], matches_cat[cat_dec_col], c='green', label=cat_name + ' Catalogue')
    plt.legend()
    plt.title('Matches with ' + cat_name + ' Catalogue')
    plt.savefig(output_path + "3-matches.png")
    if show:
        plt.show()
    plt.close()

    # Consolidate tables for cleanliness

    matches = table.hstack([matches_cat, matches], table_names=[cat_name, 'img'])

    # Plot positions on image, referring back to g image
    wcst = wcs.WCS(header=image[0].header)
    # If the hstack process has changed the ra and dec column names, we adjust our variables.
    if cat_ra_col not in matches.colnames:
        cat_ra_col = cat_ra_col + '_' + cat_name
    if cat_dec_col not in matches.colnames:
        cat_dec_col = cat_dec_col + '_' + cat_name
    matches_cat_pix_x, matches_cat_pix_y = wcst.all_world2pix(matches[cat_ra_col], matches[cat_dec_col], 0, quiet=False)

    # Clean undesirable objects from consideration:
    print(len(matches), 'total matches')
    n_match = 0

    params[f'matches_{n_match}_total'] = len(matches)
    n_match += 1

    remove = np.isnan(matches[cat_mag_col])
    print(sum(np.invert(remove)), 'matches after removing catalogue mag nans')
    params[f'matches_{n_match}_nans_cat'] = int(sum(np.invert(remove)))
    n_match += 1

    star_class_col = "CLASS_STAR"
    if stars_only:
        if star_class_type == "SPREAD_MODEL":
            if "class_flag_col" in star_class_kwargs:
                star_class_col = star_class_kwargs["class_flag_col"]
            else:
                star_class_col = "CLASS_FLAG"
            is_star = u.trim_to_class(
                matches,
                classify_kwargs=star_class_kwargs,
                modify=False,
                allowed=np.arange(0, star_class_tol + 1)
            )
            remove = remove + np.invert(is_star)
            print(sum(np.invert(remove)), 'matches after removing non-stars (class_flag < ' + str(star_class_tol) + ')')
        else:
            if "class_flag_col" in star_class_kwargs:
                star_class_col = star_class_kwargs["class_flag_col"]
            remove = remove + (matches[star_class_col] < star_class_tol)
            print(sum(np.invert(remove)),
                  'matches after removing non-stars (class_star >= ' + str(star_class_tol) + ')')
        params[f'matches_{n_match}_non_stars'] = int(sum(np.invert(remove)))
        n_match += 1

    plt.imshow(image[0].data, origin='lower', norm=plotting.nice_norm(image[0].data))
    plt.scatter(matches_cat_pix_x, matches_cat_pix_y, label=cat_name + 'Catalogue', c=matches[star_class_col],
                cmap="plasma")
    plt.colorbar()
    plt.legend()
    plt.title(f'Matches with {cat_name} Catalogue against {image_name} Image (Using SExtractor)')
    plt.savefig(output_path + "4-matches_back.png")
    if show:
        plt.show()
    plt.close()

    matches[cat_mag_col] = matches[cat_mag_col] + cat_zeropoint

    remove = remove + np.isnan(matches['mag'])
    print(sum(np.invert(remove)), 'matches after removing SExtractor mag nans')
    params[f'matches_{n_match}_nans_sex'] = int(sum(np.invert(remove)))
    n_match += 1

    remove = remove + (matches[sex_y_col] < y_lower)
    remove = remove + (matches[sex_y_col] > y_upper)
    print(sum(np.invert(remove)), 'matches after removing objects in y-exclusion zone')
    params[f'matches_{n_match}_y_exclusion'] = int(sum(np.invert(remove)))
    n_match += 1

    remove = remove + (matches['mag'] < mag_range_sex_lower)
    print(sum(np.invert(remove)),
          'matches after removing objects with SExtractor mags > ' + str(mag_range_sex_upper))
    params[f'matches_{n_match}_sex_mag_upper'] = int(sum(np.invert(remove)))

    remove = remove + (mag_range_sex_upper < matches['mag'])
    print(sum(np.invert(remove)),
          'matches after removing objects with SExtractor mags < ' + str(mag_range_sex_lower))
    params[f'matches_{n_match}_sex_mag_upper'] = int(sum(np.invert(remove)))
    n_match += 1

    remove = remove + (matches[cat_mag_col] < -98 * units.mag)
    print(sum(np.invert(remove)),
          'matches after removing objects with mags < -98')
    params[f'matches_{n_match}_sex_mag_upper'] = int(sum(np.invert(remove)))
    n_match += 1

    remove = remove + (matches[cat_mag_col] > 90 * units.mag)
    print(sum(np.invert(remove)),
          'matches after removing objects with mags > 98')
    params[f'matches_{n_match}_sex_mag_upper'] = int(sum(np.invert(remove)))
    n_match += 1

    remove = remove + (matches[snr_col] < snr_cut)
    print(sum(np.invert(remove)),
          f'matches after removing objects with {snr_col} < {snr_cut}')
    params[f'matches_{n_match}_snr'] = int(sum(np.invert(remove)))
    n_match += 1

    keep_these = np.invert(remove)
    matches_clean = matches[keep_these]

    if len(matches_clean) < 3:
        print('Not enough valid matches to calculate zeropoint.')
        p.save_params(file=output_path + 'parameters.yaml', dictionary=params)
        return None

    # Plot remaining matches
    plt.imshow(image[0].data, origin='lower', norm=plotting.nice_norm(image[0].data))
    plt.scatter(
        u.dequantify(matches_clean[sex_x_col]),
        u.dequantify(matches_clean[sex_y_col]),
        label='SExtractor',
        c=matches_clean[star_class_col],
        cmap="plasma"
    )
    plt.colorbar()
    plt.legend()
    plt.title('Matches with ' + cat_name + ' Catalogue against image (Using SExtractor)')
    plt.savefig(output_path + "5-matches_back_clean.png")
    if show:
        plt.show()
    plt.close()

    # Linear fit of catalogue magnitudes vs sextractor magnitudes

    x = matches_clean[cat_mag_col]
    # Change coordinates to get centred
    x_shift = 0.
    if do_x_shift:
        x_shift = np.nanmean(x)
    params["x_shift"] = x_shift
    x -= x_shift
    x_uncertainty = matches_clean[cat_mag_col_err]
    y = matches_clean['mag']
    y_uncertainty = matches_clean['mag_err']

    # Use a geometric mean with x_uncertainty? ie uncertainty from catalogue:
    # weights = 1./np.sqrt(x_uncertainty**2 + y_uncertainty**2)
    # Might not be sensible.

    y_weights = 1. / y_uncertainty
    x_weights = 1. / x_uncertainty

    linear_model_free = models.Linear1D(slope=1.0)
    linear_model_fixed = models.Linear1D(slope=1.0, fixed={"slope": True})

    fitter = fitting.LinearLSQFitter()

    mag_max = np.inf
    mag_min = -np.inf

    matches = matches_clean

    if iterate_uncertainty:

        delta = -np.inf
        std_err_prior = np.inf
        mag_min = np.min(x)
        x_iter = x[:]
        y_iter = y[:]
        y_uncertainty_iter = y_uncertainty[:]
        y_weights_iter = y_weights[:]
        x_weights_iter = x_weights[:]
        matches_iter = matches_clean[:]

        x_prior = x_iter
        y_prior = y_iter
        y_uncertainty_prior = y_iter
        y_weights_prior = y_weights_iter * 1.
        x_weights_prior = x_weights_iter * 1.
        matches_prior = matches_iter

        keep = np.ones(x_iter.shape, dtype=bool)
        n = 0
        u.mkdir_check(output_path + "min_mag_iterations/")
        while (std_err_prior > 1.0 * units.mag or delta <= 0) and sum(keep) > 3:
            x_prior = x_iter
            y_prior = y_iter
            y_uncertainty_prior = y_uncertainty_iter
            y_weights_prior = y_weights_iter
            x_weights_prior = x_weights_iter
            matches_prior = matches_iter

            keep = x_iter >= mag_min
            if sum(keep) < 3:
                break
            x_iter = x_iter[keep]
            y_iter = y_iter[keep]
            y_uncertainty_iter = y_uncertainty_iter[keep]
            y_weights_iter = y_weights_iter[keep]
            x_weights_iter = x_weights_iter[keep]
            matches_iter = matches_iter[keep]
            zps_iter = x_iter - y_iter
            zp_mean_iter = np.average(zps_iter, weights=y_weights_iter)
            err_this = u.root_mean_squared_error(
                model_values=zp_mean_iter * np.ones_like(zps_iter).value,
                obs_values=zps_iter,
                weights=y_weights_iter,
                dof_correction=1
            ) / np.sqrt(len(zps_iter))

            #         fitted_iter = fitter(linear_model_fixed, x_iter, y_iter, weights=y_weights_iter)
            #         line_iter = fitted_iter(x_iter)
            #
            #         err_this = u.std_err_intercept(
            #             y_model=line_iter,
            #             y_obs=y_iter,
            #             x_obs=x_iter,
            #             y_weights=y_weights_iter,
            #             x_weights=x_weights_iter,
            #             dof_correction=1
            #         )
            #
            #         plt.scatter(x_iter, y_iter, c='blue')
            #         plt.plot(x_iter, line_iter, c='green')
            #         plt.suptitle("")
            #         plt.xlabel(f"Magnitude in {cat_name}")
            #         plt.ylabel("SExtractor Magnitude in " + image_name)
            #         plt.savefig(
            #             f"{output_path}min_mag_iterations/{n}_{cat_name}vsex_std_err_{err_this}_delta{delta}_magmin_{mag_min}.png")
            #         plt.close()
            #
            delta = err_this - std_err_prior

            mag_min += 0.1 * units.mag

            print("Iterating min cat mag:", mag_min, std_err_prior, delta, sum(keep))

            std_err_prior = err_this
            n += 1

        mag_min -= 0.1 * units.mag

        x = x_prior
        y = y_prior
        y_uncertainty = y_uncertainty_prior[:]
        y_weights = y_weights_prior
        x_weights = x_weights_prior
        matches = matches_prior

        print(len(x), f'matches after iterating minimum mag ({mag_min})')
        params[f'matches_{n_match}_min_cut'] = int(len(x))
        n_match += 1

        delta = -np.inf
        std_err_prior = np.inf
        mag_max = np.max(x)
        x_iter = x[:]
        y_iter = y[:]
        y_uncertainty_iter = y_uncertainty[:]
        y_weights_iter = y_weights[:]
        x_weights_iter = x_weights[:]
        matches_iter = matches[:]

        keep = np.ones(x_iter.shape, dtype=bool)
        n = 0
        u.mkdir_check(output_path + "max_mag_iterations/")
        while delta <= 0 and sum(keep) > 3:
            x_prior = x_iter
            y_prior = y_iter
            y_uncertainty_prior = y_uncertainty_iter
            y_weights_prior = y_weights_iter
            x_weights_prior = x_weights_iter
            matches_prior = matches_iter

            keep = x_iter <= mag_max
            if sum(keep) < 3:
                break
            x_iter = x_iter[keep]
            y_iter = y_iter[keep]
            y_uncertainty_iter = y_uncertainty_iter[keep]
            y_weights_iter = y_weights_iter[keep]
            x_weights_iter = x_weights_iter[keep]
            matches_iter = matches_iter[keep]

            zps_iter = x_iter - y_iter
            zp_mean_iter = np.average(zps_iter, weights=y_weights_iter)
            err_this = u.root_mean_squared_error(
                model_values=zp_mean_iter * np.ones_like(zps_iter).value,
                obs_values=zps_iter,
                weights=y_weights_iter,
                dof_correction=1
            ) / np.sqrt(len(zps_iter))

            #         fitted_iter = fitter(linear_model_fixed, x_iter, y_iter, weights=y_weights_iter)
            #         line_iter = fitted_iter(x_iter)
            #
            #         err_this = u.std_err_intercept(
            #             y_model=line_iter,
            #             y_obs=y_iter,
            #             x_obs=x_iter,
            #             y_weights=y_weights_iter,
            #             x_weights=x_weights_iter
            #         )
            #
            #         plt.scatter(x_iter, y_iter, c='blue')
            #         plt.plot(x_iter, line_iter, c='green')
            #         plt.suptitle("")
            #         plt.xlabel(f"Magnitude in {cat_name}")
            #         plt.ylabel("SExtractor Magnitude in " + image_name)
            #         plt.savefig(f"{output_path}max_mag_iterations/{n}_{cat_name}vsex_std_err_{err_this}_magmax_{mag_max}.png")
            #         plt.close()
            #
            delta = err_this - std_err_prior

            mag_max -= 0.1 * units.mag
            std_err_prior = err_this

            print("Iterating max cat mag:", mag_max, std_err_prior, delta)

            n += 1

        mag_max += 0.1 * units.mag

        x = x_prior
        y = y_prior
        y_uncertainty = y_uncertainty_prior[:]
        y_weights = y_weights_prior
        x_weights = x_weights_prior
        matches = matches_prior

    params["mag_cut_min"] = mag_min
    params["mag_cut_max"] = mag_max
    print(len(x), f'matches after iterating maximum mag ({mag_max})')
    params[f'matches_{n_match}_mag_cut'] = int(len(x))
    n_match += 1

    fitted_free = fitter(linear_model_free, x, y, weights=y_weights)
    fitted_fixed = fitter(linear_model_fixed, x, y, weights=y_weights)

    zps = x - y
    zp_mean_shifted = np.average(zps, weights=y_weights)
    zp_mean = zp_mean_shifted + x_shift
    zp_mean_err = u.root_mean_squared_error(
        model_values=zp_mean_shifted * np.ones_like(zps).value,
        obs_values=zps,
        weights=y_weights,
        dof_correction=1
    ) / np.sqrt(len(zps))

    line_free = fitted_free(x)
    std_err_free = u.std_err_intercept(
        y_model=line_free,
        y_obs=y,
        y_weights=y_weights,
        x_obs=x,
        x_weights=x_weights,
        dof_correction=2
    )
    std_err_free_slope = u.std_err_slope(
        y_model=line_free,
        y_obs=y,
        x_obs=x,
        y_weights=y_weights,
        x_weights=x_weights,
        dof_correction=2
    )

    line_fixed = fitted_fixed(x)
    std_err_fixed = u.std_err_intercept(
        y_model=line_fixed,
        y_obs=y,
        y_weights=y_weights,
        x_obs=x,
        x_weights=x_weights,
        dof_correction=2
    )

    plt.plot(u.dequantify(x), line_free, c='red', label='Line of best fit')
    plt.scatter(u.dequantify(x), u.dequantify(y), c='blue')
    #    plt.errorbar(x, y, yerr=y_uncertainty, linestyle="None")
    plt.plot(u.dequantify(x), u.dequantify(line_fixed), c='green', label='Fixed slope = 1')
    plt.legend()
    plt.suptitle("Magnitude Comparisons")
    plt.xlabel("Magnitude in " + cat_name)
    plt.ylabel("SExtractor Magnitude in " + image_name)
    plt.savefig(output_path + "6-" + cat_name + "catvsex.png")
    if show:
        plt.show()
    plt.close()

    zp_free = -fitted_free.intercept.value * units.mag + x_shift
    slope_free = fitted_free.slope.value
    zp_fixed = -fitted_fixed.intercept.value * units.mag + x_shift
    print("UNCLIPPED")
    print("Linear fit (slope free):")
    print(f"\tZeropoint = {zp_free} +/- {std_err_free}")
    print(f"\tSlope = {slope_free} +/- {std_err_free_slope}")
    print(f"Linear fit (slope fixed): Zeropoint = {zp_fixed} +/- {std_err_fixed}")
    print(f"Mean: {zp_mean} +/- {zp_mean_err}")

    params["zeropoint_raw"] = zp_mean
    params["zeropoint_err_raw"] = zp_mean_err
    params["free_zeropoint"] = zp_free
    params["free_zeropoint_err"] = std_err_free
    params["free_slope"] = slope_free
    params["free_slope_err"] = std_err_free_slope
    params["fixed_zeropoint"] = zp_fixed
    params["fixed_zeropoint_err"] = std_err_fixed

    zps_masked = sigma_clip(zps, sigma=2.0, masked=True, copy=True)
    mask = zps_masked.mask
    zps_clipped = zps[~mask]

    y_clipped = y[~mask]
    x_clipped = x[~mask]
    y_uncertainty_clipped = y_uncertainty[~mask]
    y_weights_clipped = y_weights[~mask]
    x_weights_clipped = x_weights[~mask]

    zp_mean_clipped_shifted = np.average(zps, weights=y_weights)
    zp_mean_clipped = zp_mean_clipped_shifted + x_shift
    zp_mean_clipped_err = u.root_mean_squared_error(
        model_values=zp_mean_clipped_shifted * np.ones_like(zps_clipped).value,
        obs_values=zps_clipped,
        weights=y_weights_clipped,
        dof_correction=1
    ) / np.sqrt(len(zps_clipped))

    fitted_clipped = fitter(linear_model_fixed, x_clipped, y_clipped, weights=y_weights_clipped)
    fitted_free_clipped = fitter(linear_model_free, x_clipped, y_clipped, weights=y_weights_clipped)

    line_free_clipped = fitted_free_clipped(x_clipped)
    std_err_free_clipped = u.std_err_intercept(
        y_model=line_free_clipped,
        y_obs=y_clipped,
        y_weights=y_weights_clipped,
        x_obs=x_clipped,
        x_weights=y_weights_clipped,
        dof_correction=2
    )
    std_err_free_slope_clipped = u.std_err_slope(
        y_model=line_free_clipped,
        y_obs=y_clipped,
        x_obs=x_clipped,
        y_weights=y_weights_clipped,
        x_weights=x_weights_clipped,
        dof_correction=2
    )

    line_fixed_clipped = fitted_clipped(x_clipped)
    std_err_fixed_clipped = u.std_err_intercept(
        y_model=line_fixed_clipped,
        y_obs=y_clipped,
        y_weights=y_weights_clipped,
        x_obs=x_clipped,
        x_weights=y_weights_clipped,
        dof_correction=2
    )

    plt.plot(u.dequantify(x_clipped), u.dequantify(line_free_clipped), c='red', label='Line of best fit')
    plt.scatter(u.dequantify(x_clipped), u.dequantify(y_clipped), c='blue')
    # plt.errorbar(x_clipped, y_clipped, yerr=y_uncertainty_clipped, linestyle="None")
    plt.plot(u.dequantify(x_clipped), u.dequantify(line_fixed_clipped), c='green', label='Fixed slope = 1')
    plt.legend()
    plt.suptitle("Magnitude Comparisons")
    plt.xlabel("Magnitude in " + cat_name)
    plt.ylabel("SExtractor Magnitude in " + image_name)
    plt.savefig(output_path + "7-" + cat_name + "catvsex_clipped.png")
    if show:
        plt.show()
    plt.close()

    print(sum(~mask), 'matches after clipping outliers from mean')
    params[f'matches_{n_match}_mag_clipped'] = int(sum(~mask))
    n_match += 1
    if sum(~mask) < 3:
        print('Not enough valid matches to calculate zeropoint.')
        p.save_params(file=output_path + 'parameters.yaml', dictionary=params)
        return None

    zp_free_clipped = -fitted_free_clipped.intercept.value * units.mag + x_shift
    slope_free = fitted_free.slope.value
    zp_fixed_clipped = -fitted_fixed.intercept.value * units.mag + x_shift
    print("CLIPPED:")
    print("Linear fit:")
    print(f"\tZeropoint = {zp_free_clipped} +/- {std_err_free_clipped}")
    print(f"\tSlope = {slope_free} +/- {std_err_free_slope_clipped}")
    print(f"Linear fit (slope fixed): Zeropoint = {zp_fixed_clipped} +/- {std_err_fixed_clipped}")
    print(f"Mean: {zp_mean_clipped} +/- {zp_mean_clipped_err}")

    matches_final = matches[~mask]

    params["zeropoint"] = zp_mean_clipped
    params["zeropoint_err"] = np.sqrt(zp_mean_clipped_err ** 2 + cat_zeropoint_err ** 2)
    params["free_zeropoint_clipped"] = zp_free_clipped
    params["free_zeropoint_clipped_err"] = std_err_free_clipped
    params["free_slope_clipped"] = slope_free
    params["free_slope_clipped_err"] = std_err_free_slope_clipped
    params["fixed_zeropoint_clipped"] = zp_fixed_clipped
    params["fixed_zeropoint_clipped_err"] = std_err_fixed_clipped

    #     if latex_plot:
    #         plot_params = p.plotting_params()
    #         size_font = plot_params['size_font']
    #         size_label = plot_params['size_label']
    #         size_legend = plot_params['size_legend']
    #         weight_line = plot_params['weight_line']

    #         plotting.latex_setup()

    #         major_ticks = np.arange(-30, 30, 1)
    #         minor_ticks = np.arange(-30, 30, 0.1)

    #         fig = plt.figure(figsize=(6, 6))
    #         plot = fig.add_subplot(1, 1, 1)
    #         plot.set_xticks(major_ticks)
    #         plot.set_yticks(major_ticks)
    #         plot.set_xticks(minor_ticks, minor=True)
    #         plot.set_yticks(minor_ticks, minor=True)

    #         plot.tick_params(axis='x', labelsize=size_label, pad=5)
    #         plot.tick_params(axis='y', labelsize=size_label)
    #         plot.tick_params(which='both', width=2)
    #         plot.tick_params(which='major', length=4)
    #         plot.tick_params(which='minor', length=2)

    #         plot.plot(x_clipped[cat_mag_col], line_clipped, c='red', label='', lw=weight_line)
    #         plot.scatter(matches_sub_outliers[cat_mag_col], matches_sub_outliers['mag'], c='blue', s=16)
    #         # plt.legend()
    #         # plt.suptitle("Magnitude Comparisons without outliers")
    #         plot.set_xlabel("Magnitude in " + cat_name + " catalogue ($g$-band)", fontsize=size_font, fontweight='bold')
    #         plot.set_ylabel("Magnitude from SExtractor (image)", fontsize=size_font, fontweight='bold')
    #         # fig.savefig(output_path + "7-catvaper-outliers.pdf")
    #         fig.savefig(output_path + "8-catvmag-nice.png")
    #         if show:
    #             plt.show(plot)
    #         plt.close()

    params["matches_cat_path"] = output_path + "matches.csv"
    params["n_matches"] = len(matches_final)

    matches_final["mag_cat"] = matches_final[cat_mag_col]

    matches_final.write(output_path + "matches.csv", format='ascii.csv', overwrite=True)
    # u.rm_check(output_path + 'parameters.yaml')
    p.save_params(file=output_path + 'parameters.yaml', dictionary=params)

    print('Zeropoint - kx: ' + str(params['zeropoint']) + ' +/- ' + str(
        params['zeropoint_err']))
    print()

    if path is not None:
        image.close()

    return params


def single_aperture_photometry(
        data: np.ndarray,
        aperture: ph.aperture.Aperture,
        annulus: ph.aperture.Aperture,
        exp_time: float = 1.0,
        zeropoint: float = 0.0,
        extinction: float = 0.0,
        airmass: float = 0.0
):
    # Use background annulus to obtain a median sky background
    mask = annulus.to_mask()
    annulus_data = mask.multiply(data)[mask.data > 0]
    _, median, _ = stats.sigma_clipped_stats(annulus_data)
    # Get the photometry of the aperture
    cat_photutils = ph.aperture_photometry(data=data, apertures=aperture)
    # Multiply the median by the aperture area, so that we subtract the right amount for the aperture:
    subtract_flux = median * aperture.area
    # Correct:
    flux_photutils = cat_photutils['aperture_sum'] - subtract_flux
    # Convert to magnitude, with uncertainty propagation:
    mag_photutils, _, _ = magnitude_instrumental(
        flux=flux_photutils,
        # flux_err=cat_photutils['aperture_sum_err'],
        exp_time=exp_time,  # exp_time_err=exp_time_err,
        zeropoint=zeropoint,  # zeropoint_err=zeropoint_err,
        ext=extinction,  # ext_err=extinction_err,
        airmass=airmass,  # airmass_err=airmass_err
    )

    return mag_photutils, flux_photutils, subtract_flux, median


# TODO: Implement error properly here. Not needed right this minute because I'm currently relying on SExtractor.
#  There are some multiplications in this function that might need erroring.
def aperture_photometry(data: np.ndarray, x: float = None, y: float = None, fwhm: float = 2.,
                        exp_time: float = 1., exp_time_err: float = 0.0,
                        zeropoint: float = 0.0, zeropoint_err: float = 0.0,
                        ext: float = 0.0, ext_err: float = 0.0,
                        airmass: float = 0.0, airmass_err: float = 0.0,
                        colour_term: float = 0.0, colour_term_err: float = 0.0,
                        colours=0.0, colours_err: float = 0.0,
                        plot: bool = False,
                        r_ap: float = None,
                        r_ann_in: float = None,
                        r_ann_out: float = None,
                        r_type='fwhm', sky: bool = False, wcs_obj: wcs.WCS = None):
    """
    Constructed using this tutorial: https://photutils.readthedocs.io/en/latest/aperture.html
    :param data:
    :param x:
    :param y:
    :param fwhm: in arcsec
    :param exp_time:
    :param zeropoint:
    :param ext:
    :param airmass:
    :param colour_term:
    :param colour:
    :param plot:
    :param r_ap: Aperture radius, in arcsec.
    :param r_ann_in: Inner background-correction annulus radius, in arcsec if or in multiples of fwhm.
    :param r_ann_out: Outer background-correction annulus radius, in arcsec.
    :param r_type: Accepts 'fwhm' or 'abs' - if 'fwhm', it calculates the aperture size as a multiple of the image
    full-width half-maximum. If 'abs', it uses a simple pixel distance.
    :param sky: if True, x and y should be sky coordinates; x=RA, y=DEC. Must provide wcs_obj if True.
    :param wcs_obj: the WCS solution for the data. Only necessary if sky is True.
    :return:
    """

    # Set defaults for aperture and annulus size; if annulus sizes are not given, they default to a fixed value
    # above the aperture size. This is likely not advantageous, so do try to provide your own fixed values.

    if r_type == 'fwhm':

        if r_ap is None:
            r_ap = 5. * fwhm
        else:
            r_ap = r_ap * fwhm

        if r_ann_in is None:
            r_ann_in = 10. * fwhm
        else:
            r_ann_in = r_ann_in * fwhm

        if r_ann_out is None:
            r_ann_out = 15. * fwhm
        else:
            r_ann_out = r_ann_out * fwhm

    elif r_type == 'abs':
        if r_ap is None:
            r_ap = 10.
        if r_ann_in is None:
            r_ann_in = r_ap + 5.
        if r_ann_out is None:
            r_ann_out = r_ann_in + 10.

    else:
        raise ValueError("r_type not recognised")

    positions = np.array([x, y]).transpose()
    if sky:
        coord = SkyCoord(ra=x, dec=y)
        apertures = ph.aperture.SkyCircularAperture(positions=coord, r=r_ap)
    else:
        apertures = ph.aperture.CircularAperture(positions=positions, r=r_ap)

    # Calculate background median for each aperture using a concentric annulus.
    annuli = ph.CircularAnnulus(positions=positions, r_in=r_ann_in, r_out=r_ann_out)
    masks = annuli.to_mask(method='center')
    bg_medians = []
    for mask in masks:
        ann_data = mask.multiply(data)[mask.data > 0]
        _, median, _, = stats.sigma_clipped_stats(ann_data)
        bg_medians.append(median)
    bg_medians = np.array(bg_medians)
    # Use photutils' aperture photometry function to produce a flux for each aperture.
    cat = ph.aperture_photometry(data, apertures)
    # Add a new column to cat with the median of each annulus (corresponding to the median background for each aperture)
    cat['annulus_median'] = bg_medians
    # 'subtract' is the total amount to subtract from the flux of each aperture, given the area.
    cat['subtract'] = bg_medians * apertures.area
    # 'flux' is then the corrected flux of the aperture.
    cat['flux'] = cat['aperture_sum'] - cat['subtract']
    # 'mag' is the aperture magnitude.
    cat['mag'], cat['mag_err'] = magnitude_instrumental(flux=cat['flux'],
                                                        # flux_err=cat['aperture_sum_err'],
                                                        exp_time=exp_time,
                                                        exp_time_err=exp_time_err,
                                                        zeropoint=zeropoint,
                                                        zeropoint_err=zeropoint_err,
                                                        ext=ext, ext_err=ext_err,
                                                        airmass=airmass,
                                                        airmass_err=airmass_err,
                                                        colour_term=colour_term,
                                                        colour_term_err=colour_term_err,
                                                        colour=colours, colour_err=colours_err)
    # If selected, plot the apertures and annuli against the image.
    if plot:
        plt.imshow(data, origin='lower')
        apertures.plot(color='blue', lw=1.5, alpha=0.5)
        annuli.plot(color='red', lw=1.5, alpha=0.5)
        plt.show()
    return cat, apertures, annuli


def mask_from_area(area_file: Union['fits.HDUList', 'str'], plot=False):
    """
    Creates a photometry mask using only the highest-valued pixels in a Montage area file.
    :param area_file: Either an astropy.io.fits hdu data object or a path to a fits file, which should be the Montage
    area file.
    :param plot: To plot the mask or not to plot
    :return:
    """

    string = False
    if type(area_file) is str:
        area_file = fits.open(area_file)
        string = True

    data = area_file[0].data
    keep_val = np.round(np.max(data), 12)
    mask = data.round(12) != keep_val
    if plot:
        plt.imshow(mask, origin='lower')
    if string:
        area_file.close()

    return mask


def create_mask(file, left, right, bottom, top, plot=False):
    string = False
    if type(file) is str:
        file = fits.open(file)
        string = True
    data = file[0].data
    mask = np.zeros(data.shape, dtype=bool)
    mask[:bottom, :] = True
    mask[top:, :] = True
    mask[:, :left] = True
    mask[:, right:] = True
    if plot:
        plt.imshow(mask)
    if string:
        file.close()

    return mask


def source_table(file: Union[fits.HDUList, str],
                 bg_file: Union[fits.HDUList, str] = None, output: str = None,
                 plot: bool = False,
                 algorithm: str = 'DAO',
                 exp_time: float = None, zeropoint: float = 0., ext: float = 0.0, airmass: float = None,
                 colour_coeff: float = 0.0, colours=None, fwhm: float = 2.0, fwhm_override: bool = False,
                 mask: np.ndarray = None, r_ap: float = None, r_ann_in: float = None, r_ann_out: float = None,
                 r_type='fwhm'):
    """
    Finds sources in a .fits file using photutils and returns a catalogue table. If output is given, writes the
    catalogue to disk.
    :param file:
    :param bg_file: The path of the background fits file. If none is provided, the median across the whole image is
    taken as an initial background level.
    :param output: The path of the output file. If None, a file is not written.
    :param plot: Plots the positions of sources against the image if True.
    :param algorithm: "DAO" or "IRAF" - complete table using either DAOStarFinder or IRAFStarFinder, both built in to
    photutils.
    :param exp_time: Overrides the exposure time for the image. If left as None, this is extracted from the .fits file.
    Caution when using 'None' - not all images list this in the header, and some that do have data normalised to 1-second exposure, which makes this misleading.
    :param zeropoint:
    :param ext:
    :param airmass:
    :param colour_coeff:
    :param colours:
    :param fwhm: The fwhm to use for the initial StarFinder. If fwhm_override is True, will use this value for all
    calculations.
    :param fwhm_override: If True, will use the given fwhm for all calculations instead of calculating the mean for the
    image.
    :param mask:
    :param r_ap:
    :param r_ann_in:
    :param r_ann_out:
    :param r_type:
    :return:
    """

    bg = None
    filename = None

    # If file is a path, we want to load the fits file at that location; if it's an astropy HDU, that's already
    # been done for us.

    if type(file) is str:
        filename = file
        file = fits.open(file)

    if bg_file is None:
        bg = None
    else:
        if type(bg_file) is str:
            bg_file = fits.open(bg_file)
            bg = bg_file[0].data

    data = file[0].data
    header = file[0].header
    if exp_time is None:
        exp_time = ff.get_exp_time(file)
    if airmass is None:
        airmass = ff.get_airmass(file)
        if airmass is None:
            airmass = 0.0

    sources = find_sources(algorithm=algorithm, data=data, mask=mask, fwhm=fwhm, bg=bg)

    x = sources['xcentroid']
    y = sources['ycentroid']

    if not fwhm_override and algorithm == 'IRAF':
        fwhm = np.mean(sources['fwhm'])

    # Feed locations to our aperture photometry function.
    phot, _, _ = aperture_photometry(data=data, x=x, y=y, fwhm=fwhm, exp_time=exp_time, plot=plot,
                                     zeropoint=zeropoint, ext=ext, airmass=airmass, colour_term=colour_coeff,
                                     colours=colours,
                                     r_ap=r_ap, r_ann_in=r_ann_in, r_ann_out=r_ann_out, r_type=r_type)
    sources['mag'] = phot['mag']
    sources['flux'] = phot['flux']

    # Find world coordinates, because DAOStarFinder doesn't just do this for some reason
    w = wcs.WCS(header)
    sources['ra'], sources['dec'] = w.all_pix2world(x, y, 0)

    for col in sources.colnames:
        sources[col].info.format = '%.8g'

    if output is not None:
        sources.write(output, format='ascii', overwrite=True)

    if filename is None:
        file.close()

    # sources = np.array(sources)

    return sources


def find_sources(data: np.ndarray,
                 mask: np.ndarray = None,
                 algorithm: str = 'DAO',
                 fwhm: float = 2.0,
                 bg: float = None,
                 threshold: float = 5):
    if algorithm not in ['DAO', 'IRAF']:
        raise ValueError(str(algorithm) + " is not a recognised algorithm.")

    mean, median, std = stats.sigma_clipped_stats(data)

    print("Mean: ", mean, " Median: ", median, " Std dev: ", std)

    find = None

    if bg is None:
        bg = median

    # Find star locations using photutils StarFinder
    if algorithm == 'DAO':
        find = ph.DAOStarFinder(fwhm=fwhm, threshold=threshold * std)
    elif algorithm == 'IRAF':
        find = ph.IRAFStarFinder(fwhm=fwhm, threshold=threshold * std)

    if mask is None:
        mask = np.zeros(data.shape, dtype=bool)

    sources = find(data - bg, mask=mask)

    return sources


def match_sources_filters(file_1: 'str', file_2: 'str', path: 'str' = "", tolerance: 'float' = 1.,
                          output: 'str' = "match_table", plot: 'bool' = True, filter_1: 'str' = 'A',
                          filter_2: 'str' = 'B',
                          algorithm: 'str' = 'DAO'):
    """
    Tries to match sources between fits files using their sky coordinates, and writes to disk a table of sources in both
     - including their magnitudes. Returns a numpy array with the same information.
    :param file_1: Path of the first fits image.
    :param file_2: Path of the second fits image.
    :param path: The directory containing the input files. The directories can also be specified individually in file_1
    and file_2.
    :param tolerance: The maximum distance, in pixels, sources in can be apart between images to be declared a match.
    :param output: Path to write output .csv file to. If None, does not write to disk.
    :param plot: If True, plots the positions of all sources followed by matched sources.
    :param filter_1: Name of the filter or image used for the first image, for plotting and naming of table.
    :param filter_2: Name of the filter or image used for the second image, for plotting and naming of table.
    :param algorithm: "DAO" or "IRAF" - do photometry using either DAOStarFinder or IRAFStarFinder, both built in to
    photutils.
    :return: numpy array with each object assigned an id, containing the right ascension and declination of the object
    in each image, the average of the two, the pixel positions in x and y in each image, the magnitude in each, and the
    second magnitude subtracted from the first.
    """

    if algorithm not in ['DAO', 'IRAF']:
        raise ValueError(str(algorithm) + " is not a recognised algorithm.")

    # Produce source_table
    sources_1 = source_table(path + file_1, algorithm=algorithm)
    sources_2 = source_table(path + file_2, algorithm=algorithm)
    fit_f = fits.open(path + file_1)
    image = fit_f[0].data
    header = fit_f[0].header

    # To take (very roughly) into account the spherical distortion to the RA, we obtain an RA pixel scale by dividing
    # the difference in RA across the image by the number of pixels. It's good enough to give an average value, and the
    # difference SHOULD be pretty tiny across the image.
    w = wcs.WCS(header)
    end = image.shape[0] - 1
    ra_pixel_scale = (w.pixel_to_world(0, 0).ra.deg - w.pixel_to_world(end, 0).ra.deg) / end
    ra_tolerance = tolerance * ra_pixel_scale

    # By comparison the pixel scale in declination is easy to obtain - as DEC is undistorted, it is simply the true
    # pixel scale of the image, which the header stores.
    dec_pixel_scale = w.pixel_scale_matrix[1, 1]
    dec_tolerance = tolerance * dec_pixel_scale

    # Plot all sources picked up in source_table over one of the images.
    if plot:
        plt.imshow(image, cmap='Greys', origin='lower')
        plt.scatter(sources_1['xcentroid'], sources_1['ycentroid'], marker='+', label=filter_1)
        plt.scatter(sources_2['xcentroid'], sources_2['ycentroid'], marker='+', label=filter_2)
        plt.legend()
        plt.show()

        plt.scatter(sources_1['ra'], sources_1['dec'], marker='+', label=filter_1)
        plt.scatter(sources_2['ra'], sources_2['dec'], marker='+', label=filter_2)
        plt.legend()
        plt.show()

    matched_id_1, matched_id_2 = match_coordinates(ids_1=sources_1['id'], ras_1=sources_1['ra'],
                                                   decs_1=sources_1['dec'],
                                                   ids_2=sources_2['id'], ras_2=sources_2['ra'],
                                                   decs_2=sources_2['dec'],
                                                   ra_tolerance=ra_tolerance, dec_tolerance=dec_tolerance)

    data = np.zeros(len(matched_id_1),
                    dtype=[('id', np.int64),
                           ('ra_' + filter_1, np.float64),
                           ('ra_' + filter_2, np.float64),
                           ('ra_avg', np.float64),
                           ('x_' + filter_1, np.float64),
                           ('x_' + filter_2, np.float64),
                           ('dec_' + filter_1, np.float64),
                           ('dec_' + filter_2, np.float64),
                           ('dec_avg', np.float64),
                           ('y_' + filter_1, np.float64),
                           ('y_' + filter_2, np.float64),
                           ('mag_' + filter_1, np.float64),
                           ('mag_' + filter_2, np.float64),
                           ('mag_diff', np.float64)
                           ])

    count = 0
    for i, x in enumerate(matched_id_1):
        data['id'][i] = count
        count += 1
        source = sources_1[x - 1]
        data['ra_' + filter_1][i] = source['ra']
        data['dec_' + filter_1][i] = source['dec']
        data['mag_' + filter_1][i] = source['mag']
        data['x_' + filter_1][i] = source['xcentroid']
        data['y_' + filter_1][i] = source['ycentroid']

    for i, x in enumerate(matched_id_2):
        source = sources_2[x - 1]
        data['ra_' + filter_2][i] = source['ra']
        data['dec_' + filter_2][i] = source['dec']
        data['mag_' + filter_2][i] = source['mag']
        data['x_' + filter_2][i] = source['xcentroid']
        data['y_' + filter_2][i] = source['ycentroid']

    data['ra_avg'] = (data['ra_' + filter_1] + data['ra_' + filter_2]) / 2
    data['dec_avg'] = (data['dec_' + filter_1] + data['dec_' + filter_2]) / 2
    data['mag_diff'] = data['mag_' + filter_1] - data['mag_' + filter_2]

    # Plot only the matched sources over the first image.
    if plot:
        plt.imshow(image, cmap='Greys', origin='lower')
        plt.scatter(data['x_' + filter_1], data['y_' + filter_1], marker='+', label=filter_1)
        plt.scatter(data['x_' + filter_2], data['y_' + filter_1], marker='x', label=filter_2)
        plt.legend()
        plt.show()

        plt.scatter(data['ra_' + filter_1], data['dec_' + filter_1], marker='+', label=filter_1)
        plt.scatter(data['ra_' + filter_2], data['dec_' + filter_1], marker='x', label=filter_2)
        plt.legend()
        plt.show()

    to_csv = table.Table(data)
    to_csv.write(output + ".csv", format="ascii.csv")

    return data


def match_coordinates(ids_1, ras_1, decs_1, ids_2, ras_2, decs_2, ra_tolerance, dec_tolerance):
    matched_id_1 = []
    matched_id_2 = []

    # Check the distance of every source in the first image against every source; if it's within tolerance, register
    # as a match

    for i in range(len(ras_1)):
        ra_1 = ras_1[i]
        dec_1 = decs_1[i]
        for j in range(len(ras_2)):
            ra_2 = ras_2[j]
            dec_2 = decs_2[j]
            ra_diff = abs(ra_1 - ra_2)
            dec_diff = abs(dec_1 - dec_2)

            if dec_diff < dec_tolerance and ra_diff < ra_tolerance:
                matched_id_1.append(ids_1[i])
                matched_id_2.append(ids_2[j])

    return matched_id_1, matched_id_2


# TODO: Make this more general
# TODO: DOCSTRINGS
def match_coordinates_filters_multi(prime, match_tables, ra_tolerance, dec_tolerance, ra_name: 'str' = 'ra',
                                    dec_name: str = 'dec', name_1='1', name_2='2'):
    """

    :param prime:
    :param match_tables:
    :param ra_tolerance:
    :param dec_tolerance:
    :param ra_name: Name of the columns in match_tables containing the right ascensions.
    :param dec_name: Name of the columns in match_tables containing the declinations.
    :param name_1:
    :param name_2:
    :return: pandas.DataFrame
    """

    keys = sorted(match_tables)

    ras_1 = prime[ra_name]
    decs_1 = prime[dec_name]

    data = table.Table()
    data['ra'] = ras_1
    data['dec'] = decs_1
    data['mag_prime'] = prime['mag_diff']

    for i, key in enumerate(keys):
        data['mag_' + name_1 + '_' + str(i)] = np.nan
        data['mag_' + name_2 + '_' + str(i)] = np.nan
        data['mag_diff_' + str(i)] = np.nan
        ras_2 = match_tables[key][ra_name]
        decs_2 = match_tables[key][dec_name]
        mags_2 = match_tables[key]['mag_diff']
        for j in range(len(ras_1)):
            ra_1 = ras_1[j]
            dec_1 = decs_1[j]
            for k in range(len(ras_2)):
                ra_2 = ras_2[k]
                dec_2 = decs_2[k]
                ra_diff = abs(ra_1 - ra_2)
                dec_diff = abs(dec_1 - dec_2)

                if dec_diff < dec_tolerance and ra_diff < ra_tolerance:
                    data['mag_' + name_1 + '_' + str(i)][j] = match_tables[key]['mag_' + name_1][k]
                    data['mag_' + name_2 + '_' + str(i)][j] = match_tables[key]['mag_' + name_2][k]
                    data['mag_diff_' + str(i)][j] = mags_2[k]

    return data


def mag_to_instrumental_flux(
        mag: Union[float, units.Quantity],
        exp_time: Union[float, units.Quantity] = 1.0 * units.second,
        zeropoint: Union[float, units.Quantity] = 0.0 * units.mag,
        extinction: Union[float, units.Quantity] = 0.0 * units.mag,
        airmass: float = 0.0):
    zeropoint = u.dequantify(zeropoint, units.mag)
    extinction = u.dequantify(extinction, units.mag)
    mag = u.dequantify(mag, units.mag)
    u.debug_print(2, "mag", mag)
    exp_time = u.dequantify(exp_time, units.second)
    u.debug_print(2, exp_time * 10 ** (-(mag - zeropoint + extinction * airmass) / 2.5))
    return exp_time * 10 ** (-(mag - zeropoint + extinction * airmass) / 2.5)


def insert_synthetic_point_sources_gauss(
        image: np.ndarray, x: np.float64, y: np.float64,
        fwhm: float,
        mag: units.Quantity = 0.0 * units.mag,
        exp_time: units.Quantity = 1.0 * units.second,
        zeropoint: units.Quantity = 0.0 * units.mag,
        extinction: units.Quantity = 0.0 * units.mag,
        airmass: float = 0.0,
        saturate: float = None):
    """
    Using a simplified Gaussian point-spread function, insert a synthetic point source in an image.
    :param image:
    :param x:
    :param y:
    :param fwhm: In pixels.
    :param mag:
    :param exp_time:
    :param zeropoint:
    :param extinction:
    :param airmass:
    :param saturate:
    :return:
    """

    mag = units.Quantity(u.check_iterable(mag))
    x = units.Quantity(u.check_iterable(x))
    y = units.Quantity(u.check_iterable(y))

    fwhm = u.dequantify(fwhm, unit=units.pix)

    gaussian_model = ph.psf.IntegratedGaussianPRF(sigma=u.fwhm_to_std(fwhm=fwhm))
    sources = table.QTable()
    sources.add_column(x, name="x_0")
    sources.add_column(y, name="y_0")
    flux = mag_to_instrumental_flux(mag=mag, exp_time=exp_time, zeropoint=zeropoint, extinction=extinction,
                                    airmass=airmass)

    u.debug_print(1, "sources:\n", sources)
    u.debug_print(2, "x:", x)
    u.debug_print(2, "sources[x_0]:", sources['x_0'])
    u.debug_print(1, "len(mag):", len(mag))
    u.debug_print(1, "len(flux):", len(flux))
    u.debug_print(2, "flux:", flux)
    u.debug_print(1, "len(sources):", len(sources))

    sources['flux'] = u.dequantify(flux)
    print('Generating additive image...')

    add = make_model_image(shape=image.shape, model=gaussian_model, params_table=sources)
    sources['x_inserted'] = sources['x_0']
    sources['y_inserted'] = sources['y_0']
    sources['flux_inserted'] = sources['flux']

    # plt.imshow(add)

    combine = image + add

    if saturate is not None:
        print('Saturating...')
        combine[combine > saturate] = saturate
    print('Done.')

    return combine, sources


def insert_synthetic_point_sources_psfex(
        image: np.ndarray,
        x: np.float64, y: np.float64,
        model_path: str, mag: np.float64 = 0.0,
        exp_time: units.Quantity = 1.0 * units.second,
        zeropoint: units.Quantity = 0.0 * units.mag,
        extinction: units.Quantity = 0.0 * units.mag,
        airmass: float = 0.0,
        saturate: float = None):
    """
    Use a PSFEx psf model to insert a synthetic point source into the file.
    :param image:
    :param x:
    :param y:
    :param model_path: Path to model .psf file.
    :param mag:
    :param exp_time:
    :param zeropoint:
    :param extinction:
    :param airmass:
    :param saturate:
    :return:
    """

    import psfex

    if not isinstance(mag, Iterable):
        mag = np.array([mag])
    if not isinstance(x, Iterable):
        x = np.array([x])
    if not isinstance(y, Iterable):
        y = np.array([y])

    psfex_model = psfex.PSFEx(model_path)

    combine = np.zeros(image.shape)
    print('Generating additive image...')
    for i in range(len(x)):
        flux = mag_to_instrumental_flux(
            mag=mag[i],
            exp_time=exp_time,
            zeropoint=zeropoint,
            extinction=extinction,
            airmass=airmass
        )

        row = (x[i], y[i], flux)
        source = table.QTable(rows=[row], names=('x_inserted', 'y_inserted', 'flux_inserted'))
        psf = psfex_model.get_rec(y[i], x[i])
        y_cen, x_cen = psfex_model.get_center(y[i], x[i])
        psf[psf < 0] = 0
        psf *= flux / np.sum(psf)
        add = np.zeros(image.shape)
        add[0:psf.shape[0], 0:psf.shape[1]] += psf

        source["flux_inserted"] *= units.ct

        combine += shift(add, (y[i] - y_cen, x[i] - x_cen))

        if i == 0:
            sources = source
        else:
            sources = table.vstack([sources, source])

    combine += image

    if saturate is not None:
        print('Saturating...')
        combine[combine > saturate] = saturate
    print('Done.')

    return combine, sources


def insert_point_sources_to_file(
        file: Union[fits.hdu.HDUList, str],
        x: np.float64, y: np.float64,
        mag: np.float64,
        fwhm: float = None,
        psf_model: str = None,
        zeropoint: units.Quantity = 0.0 * units.mag,
        extinction: units.Quantity = 0.0 * units.mag,
        airmass: float = None,
        exp_time: units.Quantity = None,
        saturate: float = None,
        world_coordinates: bool = False,
        extra_values: table.Table = None,
        output: str = None,
        output_cat: str = None,
        overwrite: bool = True,
):
    """

    :param file:
    :param x: x position, in pixels or RA degrees
    :param y:
    :param mag:
    :param fwhm: In pixels.
    :param output:
    :param overwrite:
    :param zeropoint:
    :param extinction:
    :param airmass:
    :param exp_time:
    :param saturate:
    :param world_coordinates: If True, converts x and y from RA/DEC to pixel coordinates using image header before
    insertion. Set to True if using RA and DEC.
    :return:
    """
    output = u.sanitise_file_ext(output, '.fits')
    file, path = ff.path_or_hdu(file)

    if saturate is None and 'SATURATE' in file[0].header:
        saturate = file[0].header['SATURATE']
    if airmass is None and 'AIRMASS' in file[0].header:
        airmass = file[0].header['AIRMASS']
    if exp_time is None and 'EXPTIME' in file[0].header:
        exp_time = file[0].header['EXPTIME']

    wcs_info = wcs.WCS(file[0].header)
    if world_coordinates:
        ra = x
        dec = y
        x, y = wcs_info.all_world2pix(x, y, 0)
        x = np.array(x)
        y = np.array(y)
    else:
        ra, dec = wcs_info.all_pix2world(x, y, 0)

    if psf_model is not None:
        file[0].data, sources = insert_synthetic_point_sources_psfex(
            image=file[0].data, x=x, y=y, mag=mag,
            exp_time=exp_time,
            zeropoint=zeropoint, extinction=extinction,
            airmass=airmass,
            saturate=saturate, model_path=psf_model)

    elif fwhm is not None:
        file[0].data, sources = insert_synthetic_point_sources_gauss(
            image=file[0].data, x=x, y=y,
            fwhm=fwhm, mag=mag,
            exp_time=exp_time,
            zeropoint=zeropoint, extinction=extinction,
            airmass=airmass,
            saturate=saturate)

    else:
        raise ValueError("Either fwhm or psf_model must be given")

    if extra_values is not None:
        sources = table.hstack([sources, extra_values], join_type='exact')

    print('Saving to', output)
    if output is not None:
        file.writeto(output, overwrite=overwrite)

    if path:
        file.close()

    sources['mag_inserted'], _ = magnitude_instrumental(
        flux=sources['flux_inserted'], exp_time=exp_time, zeropoint=zeropoint,
        airmass=airmass,
        ext=extinction)
    sources['ra_inserted'] = ra * units.deg
    sources['dec_inserted'] = dec * units.deg

    if output_cat is None:
        output_cat = output.replace('.fits', '.ecsv')
    u.debug_print(1, "insert_point_sources_to_file: output_cat", output_cat)
    sources.write(filename=output_cat, format='ascii.ecsv', overwrite=overwrite)

    print('Done.')

    return file, sources


def insert_random_point_sources_to_file(
        file: Union[fits.hdu.HDUList, str],
        fwhm: float,
        output: str,
        n: int = 1000,
        exp_time: float = 1.,
        zeropoint: float = 0.,
        max_mag: float = 30,
        min_mag: float = 20.,
        extinction: float = 0.,
        airmass: float = None,
        overwrite: bool = True,
        saturate: float = None
):
    # TODO: For these methods, make it read from file header if None for some of the arguments.

    file, path = ff.path_or_hdu(file)

    print('Generating objects...')
    n_x, n_y = file[0].data.shape
    x = np.random.uniform(0, n_x, size=n)
    y = np.random.uniform(0, n_y, size=n)
    mag = np.random.uniform(min_mag, max_mag, size=n)

    return insert_point_sources_to_file(
        file=file,
        x=x,
        y=y,
        fwhm=fwhm,
        mag=mag,
        exp_time=exp_time,
        zeropoint=zeropoint,
        extinction=extinction,
        airmass=airmass,
        output=output,
        overwrite=overwrite,
        saturate=saturate
    )


def insert_synthetic_at_frb(obj: Union[str, dict], test_path, filters: list, magnitudes: list, add_path=False,
                            psf: bool = False, paths: dict = None, output_properties: dict = None,
                            instrument: str = 'FORS2'):
    u.mkdir_check(test_path)

    obj, epoch_properties = p.path_or_params_obj(obj=obj, instrument=instrument)
    if paths is None:
        paths = p.object_output_paths(obj, instrument=instrument)
    if output_properties is None:
        output_properties = p.object_output_params(obj, instrument=instrument)

    burst_properties = p.object_params_frb(obj[:-2])

    ra = burst_properties['burst_ra']
    dec = burst_properties['burst_dec']

    for i, f in enumerate(filters):

        psf_model = None
        if psf:
            psf_model = paths[f[0] + '_psf_model']
            psf_model = fits.open(psf_model)

        f_0 = f[0]
        fwhm = output_properties[f_0 + '_fwhm_pix']
        zeropoint, _, airmass, _, extinction, _ = select_zeropoint(obj, f, instrument=instrument,
                                                                   outputs=output_properties)

        base_path = paths[f_0 + '_' + epoch_properties['subtraction_image']]

        output_path = test_path + f_0 + '_frb_source.fits'

        insert_point_sources_to_file(file=base_path, fwhm=fwhm,
                                     output=output_path,
                                     airmass=airmass, zeropoint=zeropoint, mag=magnitudes[i],
                                     x=ra, y=dec, world_coordinates=True, psf_model=psf_model)

        if add_path:
            p.add_output_path(obj=obj, key=f_0 + '_subtraction_image_synth_frb', path=output_path)

        if psf:
            psf_model.close()


# TODO: These 'insert' functions are confusingly named. Fix that.

def insert_synthetic(obj: Union[dict, str], x, y, test_path, filters: list, magnitudes: list, suffix: str = 'synth',
                     extra_values: table.Table = None, paths: dict = None, output_properties: dict = None,
                     psf_models: list = None, instrument: str = 'FORS2'):
    """
    Insert synthetic in multiple filters.
    :param obj:
    :param x:
    :param y:
    :param test_path:
    :param filters:
    :param magnitudes:
    :param suffix:
    :param extra_values:
    :param paths:
    :param output_properties:
    :param psf_models:
    :return:
    """
    u.mkdir_check(test_path)
    obj, params = p.path_or_params_obj(obj, instrument=instrument)
    if output_properties is None:
        output_properties = p.object_output_params(obj)
    if paths is None:
        paths = p.object_output_paths(obj)

    for i, f in enumerate(filters):
        f_0 = f[0]

        fwhm = output_properties[f_0 + '_fwhm_pix']
        zeropoint, _, airmass, _, extinction, _ = select_zeropoint(obj, f, instrument=instrument)

        base_path = paths[f_0 + '_' + params['subtraction_image']]

        output_path = test_path + f_0 + '_' + suffix + '.fits'
        if psf_models is not None:
            psf_model = psf_models[i]
        else:
            psf_model = None
        insert_point_sources_to_file(file=base_path, fwhm=fwhm,
                                     output=output_path,
                                     airmass=airmass, zeropoint=zeropoint, mag=magnitudes[i],
                                     x=x, y=y, world_coordinates=False, extra_values=extra_values, psf_model=psf_model,
                                     )


def select_zeropoint(obj: str, filt: str, instrument: str, outputs: dict = None):
    instrument = instrument.lower()
    if instrument == 'fors2' or instrument == 'xshooter':
        f_0 = filt[0]
        f_output = f_0
    elif instrument == 'imacs':
        f_0 = filt[-1]
        f_output = filt
    else:
        raise ValueError('Invalid instrument.')
    if outputs is None:
        outputs = p.object_output_params(obj=obj, instrument=instrument)

    if 'provided' in outputs[f'{f_output}_zeropoints'] and outputs[f'{f_output}_zeropoints']['provided'] is not None:
        # If there is a zeropoint provided by the telescope, use that and associated parameters.
        zeropoint_dict = outputs[f'{f_output}_zeropoints']['provided']
        zeropoint = zeropoint_dict['zeropoint']
        zeropoint_err = zeropoint_dict['zeropoint_err']
        airmass = outputs[f_output + '_airmass_mean']
        airmass_err = outputs[f_output + '_airmass_err']
        extinction = outputs[f_output + '_extinction']
        extinction_err = outputs[f_output + '_extinction_err']
        print('Using provided zeropoint.')

    elif 'best' in outputs[f'{f_output}_zeropoints'] and outputs[f'{f_output}_zeropoints']['best'] is not None:

        # If neither of those exist, use the zeropoint derived from provided standard fields and correct for the
        # difference in airmass.
        zeropoint_dict = outputs[f'{f_output}_zeropoints']['best']
        zeropoint = zeropoint_dict['zeropoint']
        zeropoint_err = zeropoint_dict['zeropoint_err']
        typ = zeropoint_dict['type']

        if typ == 'science_field':
            airmass = 0.0
            airmass_err = 0.0
            extinction = 0.0
            extinction_err = 0.0
            print('Using science-field zeropoint.')

        elif typ == 'standard_field':
            airmass_field = outputs[f_output + '_airmass_mean']
            airmass_field_err = outputs[f_output + '_airmass_err']
            airmass_std = zeropoint_dict['airmass']
            airmass_std_err = 0.0
            airmass = airmass_field - airmass_std
            airmass_err = airmass_field_err + airmass_std_err
            print('Standard airmass:', airmass_std)
            print('Science airmass:', airmass_field)
            print('Delta airmass:', airmass)

            extinction = outputs[f_output + '_extinction']
            extinction_err = outputs[f_output + '_extinction_err']

        else:
            raise ValueError('No zeropoint found in output_values file.')
    else:
        raise ValueError('No zeropoint found in output_values file.')

    if instrument == 'IMACS':
        airmass = airmass - 1.0

    return zeropoint, zeropoint_err, airmass, airmass_err, extinction, extinction_err


def subtract(
        template_origin: str, comparison_origin: str,
        template_fwhm: float, comparison_fwhm: float,
        output: str,
        template_title: str, comparison_title: str,
        template_epoch: int, comparison_epoch: int,
        field: str,
        force_subtract_better_seeing: bool = True,
        sextractor_threshold: float = None
):
    """

    :param template_origin:
    :param comparison_origin:
    :param template_fwhm: In arcseconds.
    :param comparison_fwhm: In arcseconds.
    :param output:
    :param comparison_title:
    :param template_title:
    :param comparison_epoch:
    :param template_epoch:
    :param field:
    :param force_subtract_better_seeing:
    :return:
    """
    values = {'fwhm_pix_comparison': float(comparison_fwhm), 'fwhm_pix_template': float(template_fwhm),
              'comparison_file': comparison_origin, 'template_file': template_origin}
    if sextractor_threshold is not None:
        values['sextractor_threshold'] = sextractor_threshold

    comparison_file = f'{output}{comparison_title}_comparison_aligned.fits'
    template_file = f'{output}{template_title}_template_aligned.fits'

    # After reprojection, both should have the same image scale.
    _, scale = ff.get_pixel_scale(comparison_file)

    values['fwhm_pix_template'] = float((template_fwhm / 3600) / scale)
    values['fwhm_arcsec_template'] = float(template_fwhm)
    values['fwhm_pix_comparison'] = float((comparison_fwhm / 3600) / scale)
    values['fwhm_arcsec_comparison'] = float(comparison_fwhm)

    # We need this in pixels for the subtraction.
    template_fwhm = values['fwhm_pix_template']
    comparison_fwhm = values['fwhm_pix_comparison']

    sigma_comparison = u.fwhm_to_std(fwhm=comparison_fwhm)
    sigma_template = u.fwhm_to_std(fwhm=template_fwhm)
    values['sigma_template'] = sigma_template
    values['sigma_comparison'] = sigma_comparison

    # We want to always subtract the better-seeing image from the worse-seeing, to avoid false positives.
    if sigma_comparison < sigma_template:
        if force_subtract_better_seeing:
            # Subtract comparison from template.
            difference_file = f'{output}{field}_{template_epoch}-{comparison_epoch}_difference.fits'

            sigma_match = math.sqrt(sigma_template ** 2 - sigma_comparison ** 2)

            os.system(
                f'hotpants -inim {template_file}'
                f' -tmplim {comparison_file}'
                f' -outim {difference_file}'
                f' -ng 3 6 {0.5 * sigma_match} 4 {sigma_match} 2 {2 * sigma_match}'
                f' -oki {output}kernel.fits'
                f' -n i'
            )

            # We then reverse the pixels of the difference image, giving transients positive flux (so that SExtractor
            # can see them)
            difference_image = fits.open(difference_file, mode='update')
            difference_image[0].data = -difference_image[0].data
            difference_image.close()

            values['convolved'] = 'template'

        else:
            # Subtract template from comparison, but force convolution on comparison image.
            difference_file = f'{output}{field}_{comparison_epoch}-{template_epoch}_difference.fits'

            os.system(f'hotpants -inim {comparison_file}'
                      f' -tmplim {template_file}'
                      f' -outim {difference_file}'
                      f' -oki {output}kernel.fits'
                      f' -c i'
                      f' -n i')

            values['convolved'] = 'comparison'

    else:
        # Subtract template from comparison.
        difference_file = f'{output}{field}_{comparison_epoch}-{template_epoch}_difference.fits'

        sigma_match = math.sqrt(sigma_comparison ** 2 - sigma_template ** 2)
        values['sigma_match'] = sigma_match

        os.system(f'hotpants -inim {comparison_file}'
                  f' -tmplim {template_file}'
                  f' -outim {difference_file}'
                  f' -ng 3 6 {0.5 * sigma_match} 4 {sigma_match} 2 {2 * sigma_match}'
                  f' -oki {output}kernel.fits'
                  f' -n i')

        values['convolved'] = 'template'

    values['force_sub_better_seeing'] = False

    print(f'Wrote difference file to ', difference_file)
    # Save a .yaml file with some outputs.
    p.add_params(output + 'output_values', values)
    p.yaml_to_json(output + 'output_values')

    return fits.open(difference_file), difference_file


def find_brightest_pixel(image, start_x, start_y, radius=5):
    """
    Finds the brightest pixel in a subimage centred on start_x, start_y of given radius
    :param image:
    :param start_x:
    :param start_y:
    :param radius:
    :return:
    """
    image_copy = image.copy()[start_y - radius:start_y + radius, start_x - radius:start_x + radius]
    h, w = image_copy.shape
    index_brightest = np.unravel_index(np.argmax(image_copy), image_copy.shape)

    index_x, index_y = index_brightest
    index_x += start_y - radius
    index_y += start_x - radius
    return index_y, index_x, image[index_x, index_y]


def sersic_profile(r, *para):
    ie, re, n = para
    bn = 2 * n - 1 / 3
    return ie * np.exp(-bn * ((r / re) ** (1 / n) - 1))


def fit_sersic_profile(radii, intensities):
    model = Sersic1D()
    fit = fitting.LevMarLSQFitter()
    new_model = fit(model, radii, intensities)
    return new_model.amplitude.value, new_model.r_eff.value, new_model.n.value


def intensity_radius(image, centre_x, centre_y, noise: float = None, limit: float = None):
    """
    Creates an intensity profile of a subimage over a given radius from a given pixel.
    :param image:
    :param centre_x:
    :param centre_y:
    :param limit:
    :return:
    """
    if noise is None:
        noise = np.median(image)
    intensities = []
    h, w = image.shape
    y, x = np.ogrid[:h, :w]
    pix_distance = np.sqrt((x - centre_x) ** 2 + (y - centre_y) ** 2).astype(int)

    if limit is None:
        intensity = np.inf
        radii = []
        r = 1
        while intensity > noise:
            radii.append(r)
            pixels = image[pix_distance == r]
            intensity = np.mean(pixels)
            intensities.append(intensity)

            r += 1
        radii = np.array(radii)

    else:
        radii = np.arange(1, limit + 1)
        for r in radii:
            pixels = image[pix_distance == r]
            intensities.append(np.mean(pixels))

    return radii, np.array(intensities)


def signal_to_noise_ccd_equ(
        rate_target: units.Quantity,
        rate_sky: units.Quantity,
        rate_read: units.Quantity,
        exp_time: units.Quantity,
        gain: units.Quantity,
        n_pix: units.Quantity,
        rate_dark: units.Quantity = 0.0 * (units.electron / (units.second * units.pixel)),
):
    """
    Calculate the signal-to-noise ratio of an observed object using the Howell 1989 formulation.
    :param rate_target:
    :param rate_sky:
    :param rate_read:
    :param exp_time:
    :param gain:
    :param n_pix:
    :param rate_dark:
    :return:
    """
    rate_target = u.check_quantity(rate_target, units.ct / units.second)
    rate_sky = u.check_quantity(rate_sky, units.ct * units.second ** -1 * units.pixel ** -1)
    rate_read = u.check_quantity(rate_read, units.electron / units.pixel)
    rate_dark = u.check_quantity(rate_dark, units.electron / (units.second * units.pixel))
    exp_time = u.check_quantity(exp_time, units.second)
    gain = u.check_quantity(gain, gain_unit)
    n_pix = u.check_quantity(n_pix, units.pix)

    u.debug_print(2, rate_target)
    u.debug_print(2, rate_sky)
    u.debug_print(2, rate_dark)
    u.debug_print(2, exp_time)
    u.debug_print(2, gain)
    u.debug_print(2, n_pix)

    u.debug_print(2, rate_read ** 2 / (exp_time * gain))

    snr = rate_target * np.sqrt(exp_time * gain) / np.sqrt(
        rate_target + n_pix * (rate_sky + rate_dark / gain + rate_read / (exp_time * gain)))
    return snr
