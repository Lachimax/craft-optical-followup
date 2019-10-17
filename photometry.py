# Code by Lachlan Marnoch, 2019

import os
import shutil
import math
from typing import Union

import numpy as np
import pandas as pd
import photutils as ph
from photutils import datasets
import pylab as pl

from astropy import stats
from astropy import wcs
from astropy.modeling import fitting
from astropy.modeling.functional_models import Sersic1D
import astropy.table as table
from astropy.io import fits as fits
import astropy.time as time
from astropy import convolution

from matplotlib import pyplot as plt
from datetime import datetime as dt
import reproject as rp

from PyCRAFT import fits_files as ff
import PyCRAFT.params as p
import PyCRAFT.utils as u
from PyCRAFT import plotting
import operator
from functools import reduce


# TODO: End-to-end pipeline script?

# TODO: Change expected types to Union


def gain_median_combine(old_gain=0.8, n_frames=1):
    """
    Only valid if you have a median-combined image normalised to counts per second.
    :param old_gain:
    :param n_frames:
    :return:
    """

    return 2 * n_frames * old_gain / 3


def magnitude_complete(flux: 'float', flux_err: 'float' = 0.0,
                       exp_time: 'float' = 1., exp_time_err: 'float' = 0.0,
                       zeropoint: 'float' = 0.0, zeropoint_err: 'float' = 0.0,
                       airmass: 'float' = 0.0, airmass_err: 'float' = 0.0,
                       ext: 'float' = 0.0, ext_err: 'float' = 0.0,
                       colour_term: 'float' = 0.0, colour_term_err: 'float' = 0.0,
                       colour: 'float' = 0.0, colour_err: 'float' = 0.0):
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

    print('Calculating magnitudes...')

    if colour is None:
        colour = 0.0
    if airmass is None:
        airmass = 0.0

    mag_inst, mag_error_plus, mag_error_minus = magnitude_error(flux=flux, flux_err=flux_err,
                                                                exp_time=exp_time, exp_time_err=exp_time_err)

    magnitude = mag_inst + zeropoint - ext * airmass - colour_term * colour

    # TODO: Since colour will be an array, the below might cause problems. Check this.

    error_extinction = u.error_product(value=ext * airmass,
                                       measurements=[ext, airmass],
                                       errors=[ext_err, airmass_err])
    error_colour = u.error_product(value=colour_term * colour,
                                   measurements=[colour_term, colour],
                                   errors=[colour_term_err, colour_err])

    error_plus = mag_error_plus + zeropoint_err + error_extinction + error_colour
    error_minus = mag_error_minus - zeropoint_err - error_extinction - error_colour

    return magnitude, error_plus, error_minus


def magnitude_error(flux: 'float', flux_err: 'float' = 0.0,
                    exp_time: 'float' = 1., exp_time_err: 'float' = 0.0,
                    absolute: 'bool' = False):
    flux_per_sec = flux / exp_time
    error_fps = u.error_product(value=flux_per_sec, measurements=[flux, exp_time], errors=[flux_err, exp_time_err])
    mag, error_plus, error_minus = u.error_func(arg=flux_per_sec, err=error_fps,
                                                func=lambda x: -2.5 * np.log10(x),
                                                absolute=absolute)

    return np.array([mag, error_plus, error_minus])


def determine_zeropoint_sextractor(sextractor_cat_path: 'str',
                                   cat_path: 'str',
                                   image: 'str',
                                   output_path: 'str',
                                   cat_name: 'str' = 'Catalogue',
                                   image_name: 'str' = 'FORS2',
                                   show: 'bool' = False,
                                   cat_ra_col: 'str' = 'RA',
                                   cat_dec_col: 'str' = 'DEC',
                                   cat_mag_col: 'str' = 'WAVG_MAG_PSF_',
                                   sex_ra_col='ra',
                                   sex_dec_col='dec',
                                   sex_x_col: 'str' = 'x',
                                   sex_y_col: 'str' = 'y',
                                   pix_tol: 'float' = 10.,
                                   mag_tol: 'float' = 0.1,
                                   flux_column: 'str' = 'flux_aper',
                                   mag_range_cat_upper: 'float' = 20.,
                                   mag_range_cat_lower: 'float' = 30.,
                                   mag_range_sex_upper: 'float' = 100,
                                   mag_range_sex_lower=-100,
                                   stars_only: 'bool' = True,
                                   star_class_tol: 'float' = 0.95,
                                   star_class_col: 'str' = 'class_star',
                                   exp_time: 'float' = None,
                                   y_lower: 'int' = 0,
                                   y_upper: 'int' = 100000,
                                   get_sextractor_names=False,
                                   sextractor_names=None,
                                   cat_type='csv',
                                   cat_zeropoint=0.0,
                                   cat_zeropoint_err=0.0):
    """
    This function expects your sextractor columns to be in the same format as found in params.sextractor_columns and
    default.param under sextractor.
    It also expects your catalogue to be a .csv.
    :param sextractor_cat_path:
    :param cat_path:
    :param image:
    :param cat_name:
    :param image_name:
    :param output_path:
    :param show:
    :param cat_ra_col:
    :param cat_dec_col:
    :param cat_mag_col:
    :param sex_ra_col:
    :param sex_dec_col:
    :param sex_x_col:
    :param sex_y_col:
    :param pix_tol:
    :param mag_tol:
    :param flux_column:
    :param mag_range_cat_upper:
    :param mag_range_cat_lower:
    :param mag_range_sex_upper:
    :param mag_range_sex_lower:
    :param stars_only:
    :param star_class_tol:
    :param star_class_col:
    :param exp_time:
    :param y_lower:
    :param y_upper:
    :param get_sextractor_names:
    :param sextractor_names:
    :param cat_type:
    :return:
    """

    params = {}

    if type(image) is str:
        params['image_path'] = image
        image_path = image
        image = fits.open(image_path)

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    if exp_time is None:
        exp_time = ff.get_exp_time(image)

    if get_sextractor_names:
        sextractor_names = u.get_column_names_sextractor(sextractor_cat_path)
    elif sextractor_names is None:
        sextractor_names = p.load_params('param/sextractor_names')

    # Extract pixel scales from images.
    _, pix_scale = ff.get_pixel_scale(image)
    print('PIXEL SCALE:', pix_scale)
    # Get tolerance as angle.
    tolerance = pix_tol * pix_scale
    print('TOLERANCE:', tolerance)

    # Import the catalogue of the sky region.
    if cat_type != 'sextractor':
        cat = table.Table()
        cat = cat.read(cat_path, format='ascii.csv')
        cat = cat.filled(fill_value=-999.)

    else:
        sextractor_names_mod = sextractor_names.copy()
        sextractor_names_mod[sextractor_names.index('ra')] = 'ra_cat'
        sextractor_names_mod[sextractor_names.index('dec')] = 'dec_cat'
        cat = table.Table(np.genfromtxt(cat_path, names=sextractor_names_mod))

    params['time'] = dt.now().strftime('%Y-%m-%dT%H:%M:%S')
    params['catalogue'] = cat_name
    params['airmass'] = ff.get_airmass(image)
    print('Airmass:', params['airmass'])
    params['exp_time'] = exp_time
    params['pix_tol'] = pix_tol
    params['ang_tol'] = tolerance
    params['mag_cut_min'] = mag_range_cat_lower
    params['mag_cut_max'] = mag_range_cat_upper
    params['mag_tol'] = mag_tol
    params['cat_path'] = cat_path
    params['cat_ra_col'] = cat_ra_col
    params['cat_dec_col'] = cat_dec_col
    params['cat_flux_col'] = cat_mag_col
    params['sextractor_path'] = sextractor_cat_path
    params['sex_ra_col'] = sex_ra_col
    params['sex_dec_col'] = sex_dec_col
    params['sex_flux_col'] = flux_column
    params['stars_only'] = stars_only
    params['pix_scale_deg_image'] = pix_scale
    params['pix_scale_arc_image'] = pix_scale * 60 * 60
    params['y_lower'] = y_lower
    params['y_upper'] = y_upper
    params['cat_zeropoint'] = cat_zeropoint
    params['cat_zeropoint_err'] = cat_zeropoint_err
    if stars_only:
        params['star_class_tol'] = star_class_tol

    source_tbl = table.Table(np.genfromtxt(sextractor_cat_path, names=sextractor_names))

    source_tbl['mag'], _, _ = magnitude_complete(flux=source_tbl[flux_column], exp_time=exp_time)

    # Plot all stars found by SExtractor.
    plt.scatter(source_tbl[sex_ra_col], source_tbl[sex_dec_col], label='SExtractor')
    plt.title('Objects found by SExtractor')
    plt.savefig(output_path + '1-sextracted-positions.png')
    if show:
        plt.show()
    plt.close()

    # Plot all catalogue stars.
    plt.scatter(cat[cat_ra_col], cat[cat_dec_col])
    plt.title('Objects in catalogue')
    plt.savefig(output_path + '2-catalogue-positions.png')
    if show:
        plt.show()
    plt.close()

    plt.scatter(source_tbl[sex_ra_col], source_tbl[sex_dec_col], label='SExtractor')
    plt.scatter(cat[cat_ra_col], cat[cat_dec_col], label=cat_name)
    plt.legend()
    plt.title('Matches with ' + cat_name + ' Catalogue')
    if show:
        plt.show()
    plt.close()

    # Match stars to catalogue.
    match_ids, match_ids_cat = u.match_cat(x_match=source_tbl[sex_ra_col], y_match=source_tbl[sex_dec_col],
                                           x_cat=cat[cat_ra_col],
                                           y_cat=cat[cat_dec_col], tolerance=tolerance)

    matches = source_tbl[match_ids]
    matches_cat = cat[match_ids_cat]

    # Plot all matches with catalogue.
    plt.scatter(matches[sex_ra_col], matches[sex_dec_col], label='SExtractor MAG\_AUTO')
    plt.scatter(matches_cat[cat_ra_col], matches_cat[cat_dec_col], c='green', label=cat_name + ' Catalogue')
    plt.legend()
    plt.title('Matches with ' + cat_name + ' Catalogue')
    plt.savefig(output_path + "3-matches.png")
    if show:
        plt.show()
    plt.close()

    # Consolidate tables for cleanliness

    matches = table.hstack([matches_cat, matches], table_names=[cat_name, 'fors'])

    # Plot positions on image, referring back to g image
    wcst = wcs.WCS(header=image[0].header)
    # If the hstack process has changed the ra and dec column names, we adjust our variables.
    if cat_ra_col not in matches.colnames:
        cat_ra_col = cat_ra_col + '_' + cat_name
    if cat_dec_col not in matches.colnames:
        cat_dec_col = cat_dec_col + '_' + cat_name
    matches_cat_pix_x, matches_cat_pix_y = wcst.all_world2pix(matches[cat_ra_col], matches[cat_dec_col], 0, quiet=False)

    plt.imshow(image[0].data, origin='lower', norm=plotting.nice_norm(image[0].data))
    plt.scatter(matches_cat_pix_x, matches_cat_pix_y, label=cat_name + 'Catalogue', c=matches[star_class_col])
    plt.colorbar()
    plt.legend()
    plt.title('Matches with ' + cat_name + ' Catalogue against ' + image_name + ' Image (Using SExtractor)')
    plt.savefig(output_path + "4-matches_back.png")
    if show:
        plt.show()
    plt.close()

    matches[cat_mag_col] = matches[cat_mag_col] + cat_zeropoint

    # Clean undesirable objects from consideration:
    print(len(matches), 'total matches')
    params['matches_1_total'] = len(matches)

    remove = np.isnan(matches[cat_mag_col])
    print(sum(np.invert(remove)), 'matches after removing catalogue mag nans')
    params['matches_2_nans_cat'] = sum(np.invert(remove))

    remove = remove + np.isnan(matches['mag'])
    print(sum(np.invert(remove)), 'matches after removing SExtractor mag nans')
    params['matches_3_nans_sex'] = sum(np.invert(remove))

    remove = remove + (matches[sex_y_col] < y_lower)
    remove = remove + (matches[sex_y_col] > y_upper)
    print(sum(np.invert(remove)), 'matches after removing objects in y-exclusion zone')
    params['matches_4_y_exclusion'] = sum(np.invert(remove))

    remove = remove + (matches[cat_mag_col] < mag_range_cat_lower)
    print(sum(np.invert(remove)),
          'matches after removing objects objects with mags > ' + str(mag_range_cat_upper))
    params['matches_5_cat_mag_upper'] = sum(np.invert(remove))

    remove = remove + (mag_range_cat_upper < matches[cat_mag_col])
    print(sum(np.invert(remove)),
          'matches after removing objects objects with mags < ' + str(mag_range_cat_lower))
    params['matches_6_cat_mag_lower'] = sum(np.invert(remove))

    remove = remove + (matches['mag'] < mag_range_sex_lower)
    print(sum(np.invert(remove)),
          'matches after removing objects objects with SExtractor mags > ' + str(mag_range_sex_upper))
    params['matches_7_sex_mag_upper'] = sum(np.invert(remove))

    remove = remove + (mag_range_sex_upper < matches['mag'])
    print(sum(np.invert(remove)),
          'matches after removing objects objects with SExtractor mags < ' + str(mag_range_sex_lower))
    params['matches_8_sex_mag_upper'] = sum(np.invert(remove))

    if stars_only:
        if star_class_col == 'spread_model':
            remove = remove + (np.abs(matches[star_class_col]) > star_class_tol)
            print(sum(np.invert(remove)),
                  f'matches after removing non-stars (abs({star_class_tol}) > spread_model)')
        else:
            remove = remove + (matches[star_class_col] < star_class_tol)
            print(sum(np.invert(remove)), 'matches after removing non-stars (star_class < ' + str(star_class_tol) + ')')
        params['matches_9_non_stars'] = sum(np.invert(remove))
    keep_these = np.invert(remove)
    matches_clean = matches[keep_these]

    if len(matches_clean) < 3:
        print('Not enough valid matches to calculate zeropoint.')
        return None

    # Calculate individual zeropoint for each star
    matches_clean['zeropoint_ind'] = matches_clean[cat_mag_col] - matches_clean['mag']

    # Plot remaining matches
    plt.imshow(image[0].data, origin='lower', norm=plotting.nice_norm(image[0].data))
    plt.scatter(matches_clean[sex_x_col], matches_clean[sex_y_col], label='SExtractor',
                c=matches_clean['class_star'])  # c=matches_clean[star_class_col])
    plt.colorbar()
    plt.legend()
    plt.title('Matches with ' + cat_name + ' Catalogue against image (Using SExtractor)')
    plt.savefig(output_path + "5-matches_back_clean.png")
    if show:
        plt.show()
    plt.close()

    # Linear fit of catalogue magnitudes vs sextractor magnitudes
    linfit = pl.polyfit(matches_clean[cat_mag_col],
                        matches_clean['mag'], 1)
    linfit_array = linfit[0] * matches_clean[cat_mag_col] + linfit[1]
    ideal_intercept = np.mean(matches_clean['mag'] - 1. * matches_clean[cat_mag_col])
    # linfit_ideal is the model when we fix the slope to 1, as it should be in theory.
    linfit_ideal = 1. * matches_clean[cat_mag_col] + ideal_intercept
    params['linfit_unfixed'] = str(linfit)
    rmse = u.root_mean_squared_error(model_values=linfit_ideal, obs_values=matches_clean['mag'])
    params['zeropoint_raw'] = -ideal_intercept
    params['rmse_raw'] = rmse

    plt.plot(matches_clean[cat_mag_col], linfit_array, c='red', label='Line of best fit')
    plt.scatter(matches_clean[cat_mag_col], matches_clean['mag'], c='blue')
    plt.plot(matches_clean[cat_mag_col], linfit_ideal, c='green', label='Fixed slope = 1')
    plt.legend()
    plt.suptitle("Magnitude Comparisons")
    plt.xlabel("Magnitude in " + cat_name + " g-band")
    plt.ylabel("SExtractor MAG\_APER in " + image_name)
    plt.savefig(output_path + "6-" + cat_name + "catvaper.png")
    if show:
        plt.show()
    plt.close()

    # Discard outliers
    delta_mag = np.abs(matches_clean['mag'] - linfit_ideal)
    keep = delta_mag < mag_tol

    matches_sub_outliers = matches_clean[keep]

    print(len(matches_sub_outliers), 'matches after stripping outliers from linear fit')
    params['matches_10_-outliers'] = len(matches_sub_outliers)

    if len(matches_sub_outliers) < 3:
        print('Not enough valid matches to calculate zeropoint.')
        return None

    linfit = pl.polyfit(matches_sub_outliers[cat_mag_col],
                        matches_sub_outliers['mag'], 1)
    linfit_array = linfit[0] * matches_sub_outliers[cat_mag_col] + linfit[1]
    ideal_intercept = np.mean(matches_sub_outliers['mag'] - 1. * matches_sub_outliers[cat_mag_col])
    # linfit_ideal is the model when we fix the slope to 1, as it should be in theory.
    linfit_ideal = 1. * matches_sub_outliers[cat_mag_col] + ideal_intercept
    params['linfit_unfixed-outliers'] = str(linfit)
    params['linfit_slope'] = str(linfit[0])
    params['d_linfit_slope'] = str(abs(1. - float(linfit[0])))
    rmse = u.root_mean_squared_error(model_values=linfit_ideal, obs_values=matches_sub_outliers['mag'])
    params['zeropoint_sub_outliers'] = -ideal_intercept
    params['rmse_sub_outliers'] = rmse

    plot_params = p.plotting_params()
    size_font = plot_params['size_font']
    size_label = plot_params['size_label']
    size_legend = plot_params['size_legend']
    weight_line = plot_params['weight_line']

    plotting.latex_setup()

    major_ticks = np.arange(-30, 30, 1)
    minor_ticks = np.arange(-30, 30, 0.1)

    fig = plt.figure(figsize=(6, 6))
    plot = fig.add_subplot(1, 1, 1)
    plot.set_xticks(major_ticks)
    plot.set_yticks(major_ticks)
    plot.set_xticks(minor_ticks, minor=True)
    plot.set_yticks(minor_ticks, minor=True)

    plot.tick_params(axis='x', labelsize=size_label, pad=5)
    plot.tick_params(axis='y', labelsize=size_label)
    plot.tick_params(which='both', width=2)
    plot.tick_params(which='major', length=4)
    plot.tick_params(which='minor', length=2)

    # plt.plot(matches_sub_outliers[cat_mag_col], linfit_array, c='red', label='Line of best fit')
    plot.plot(matches_sub_outliers[cat_mag_col], linfit_ideal, c='red', label='', lw=weight_line)
    plot.scatter(matches_sub_outliers[cat_mag_col], matches_sub_outliers['mag'], c='blue', s=16)
    # plt.legend()
    # plt.suptitle("Magnitude Comparisons without outliers")
    plot.set_xlabel("Magnitude in " + cat_name + " catalogue ($g$-band)", fontsize=size_font, fontweight='bold')
    plot.set_ylabel("Magnitude from SExtractor (image)", fontsize=size_font, fontweight='bold')
    fig.savefig(output_path + "7-catvaper-outliers.pdf")
    if show:
        plt.show(plot)
    plt.close()

    params['zeropoint_median'] = np.median(matches_sub_outliers['zeropoint_ind'])
    params['zeropoint_median_err'] = 2 * np.std(matches_sub_outliers['zeropoint_ind'] + cat_zeropoint_err)
    params['zeropoint_err'] = cat_zeropoint_err + params['rmse_sub_outliers']

    matches.write(output_path + "matches.csv", format='ascii.csv')
    if os.path.isfile(output_path + 'parameters'):
        os.remove(output_path + 'parameters')
    for par in params:
        with open(output_path + 'parameters', 'a') as file:
            file.write(par + ": " + str(params[par]) + "\n")

    print('Zeropoint - kx: ' + str(params['zeropoint_sub_outliers']) + ' +/- ' + str(
        params['zeropoint_err']))
    print('Zeropoint - kx (median): ' + str(params['zeropoint_median']) + ' +/- ' + str(
        params['zeropoint_median_err']))
    print()

    return params


def jy_to_mag(jy: 'float'):
    """
    Converts a flux density in janskys to a magnitude.
    :param jy: value in janskys.
    :return: Magnitude
    """

    return -2.5 * np.log10(jy)


# TODO: Implement error properly here. Not needed right this minute because I'm currently relying on SExtractor.
#  There are some multiplications in this function that might need erroring.
def aperture_photometry(data: np.ndarray, x: float, y: float, fwhm: float = 2.,
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
                        r_type='fwhm'):
    """
    Constructed using this tutorial: https://photutils.readthedocs.io/en/latest/aperture.html
    :param data:
    :param x:
    :param y:
    :param fwhm: in pixels
    :param exp_time:
    :param zeropoint:
    :param ext:
    :param airmass:
    :param colour_term:
    :param colour:
    :param plot:
    :param r_ap: Aperture radius.
    :param r_ann_in: Inner background-correction annulus radius.
    :param r_ann_out: Outer background-correction annulus radius.
    :param r_type: Accepts 'fwhm' or 'abs' - if 'fwhm', it calculates the aperture size as a multiple of the image
    full-width half-maximum. If 'abs', it uses a simple pixel distance.
    :return:
    """

    # TODO: throw error if r_type is not "abs" or "fwhm"

    # Set defaults for aperture and annulus size; if annulus sizes are not given, they default to a fixed value
    # above the aperture size. This is likely not advantageous, so do try to provide your own fixed values.

    if r_type is 'fwhm':
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

    elif r_type is 'abs':
        if r_ap is None:
            r_ap = 10.
        if r_ann_in is None:
            r_ann_in = r_ap + 5.
        if r_ann_out is None:
            r_ann_out = r_ann_in + 10.

    positions = np.array([x, y])
    apertures = ph.CircularAperture(positions=positions, r=r_ap)

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
    cat['subtract'] = bg_medians * apertures.area()
    # 'flux' is then the corrected flux of the aperture.
    cat['flux'] = cat['aperture_sum'] - cat['subtract']
    # 'mag' is the aperture magnitude.
    cat['mag'], cat['mag_err_plus'], cat['mag_err_minus'] = magnitude_complete(flux=cat['flux'],
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


def mask_from_area(area_file: Union['fits.hdu.hdulist.HDUList', 'str'], plot=False):
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


def source_table(file: Union['fits.hdu.hdulist.HDUList', 'str'],
                 bg_file: Union['fits.hdu.hdulist.HDUList', 'str'] = None, output: 'str' = None, plot: 'bool' = False,
                 algorithm: 'str' = 'DAO',
                 exp_time: 'float' = None, zeropoint: 'float' = 0., ext: 'float' = 0.0, airmass: 'float' = None,
                 colour_coeff: 'float' = 0.0, colours=None, fwhm: 'float' = 2.0, fwhm_override: 'bool' = False,
                 mask: 'np.ndarray' = None, r_ap: 'float' = None, r_ann_in: 'float' = None, r_ann_out: 'float' = None,
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
        bg = 0.0
    else:
        if type(file) is str:
            bg_file = fits.open(bg_file)
            bg = bg_file[0].data

    if algorithm not in ['DAO', 'IRAF']:
        raise ValueError(str(algorithm) + " is not a recognised algorithm.")

    data = file[0].data
    header = file[0].header
    if exp_time is None:
        exp_time = ff.get_exp_time(file)
    if airmass is None:
        airmass = ff.get_airmass(file)
        if airmass is None:
            airmass = 0.0

    mean, median, std = stats.sigma_clipped_stats(data)

    print("Mean: ", mean, " Median: ", median, " Std dev: ", std)

    find = None

    # Find star locations using photutils StarFinder

    if algorithm == 'DAO':
        find = ph.DAOStarFinder(fwhm=fwhm, threshold=5. * std)
    elif algorithm == 'IRAF':
        find = ph.IRAFStarFinder(fwhm=fwhm, threshold=5. * std)

    if mask is None:
        mask = np.zeros(data.shape, dtype=bool)

    sources = find(data - bg, mask=mask)

    x = sources['xcentroid']
    y = sources['ycentroid']
    if not fwhm_override and algorithm == 'IRAF':
        fwhm = np.mean(sources['fwhm'])
    print("FWHM:", fwhm)

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

    to_csv = pd.DataFrame(data)
    to_csv.to_csv(output + ".csv", index=False)

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
                                    dec_name: 'str' = 'dec', name_1='1', name_2='2'):
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

    data = pd.DataFrame()
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


def match_coordinates_multi(prime, match_tables, ra_tolerance: 'float', dec_tolerance: 'float', ra_name: 'str' = 'ra',
                            dec_name: 'str' = 'dec', mag_name: 'str' = 'mag', extra_cols: 'list' = None,
                            x_name: 'str' = 'xcentroid', y_name: 'str' = 'ycentroid'):
    """

    :param prime:
    :param match_tables:
    :param ra_tolerance:
    :param dec_tolerance:
    :param ra_name:
    :param dec_name:
    :param extra_cols: the names of any extra columns in the tables of prime and match_tables you wish to be appended
            to the returned data.
    :return:
    """
    # TODO: Oh god why does this use a pandas dataframe
    if extra_cols is None:
        extra_cols = []
    keys = sorted(match_tables)

    ras_1 = prime[ra_name]
    decs_1 = prime[dec_name]
    mags_1 = prime[mag_name]

    data = pd.DataFrame()
    data[ra_name] = ras_1
    data[dec_name] = decs_1
    data['x'] = prime[x_name]
    data['y'] = prime[y_name]
    data['mag_prime'] = prime[mag_name]
    for name in extra_cols:
        data[name + '_prime'] = prime[name]

    for i, key in enumerate(keys):
        data['mag_' + str(i)] = np.nan
        for name in extra_cols:
            data[name + '_' + str(i)] = np.nan
        ras_2 = match_tables[key][ra_name]
        decs_2 = match_tables[key][dec_name]
        mags_2 = match_tables[key][mag_name]

        for j in range(len(ras_1)):
            candidate_mag = float("inf")
            ra_1 = ras_1[j]
            dec_1 = decs_1[j]
            mag_1 = mags_1[j]
            for k in range(len(ras_2)):
                ra_2 = ras_2[k]
                dec_2 = decs_2[k]
                mag_2 = mags_2[k]
                ra_diff = abs(ra_1 - ra_2)
                dec_diff = abs(dec_1 - dec_2)

                # This makes sure that, in case of multiple matches, the closest match in terms of magnitude is
                # written in.
                if dec_diff < dec_tolerance and ra_diff < ra_tolerance \
                        and abs(mag_1 - mag_2) < abs(mag_1 - candidate_mag):
                    candidate_mag = mag_2
                    for name in extra_cols:
                        data[name + '_' + str(i)][j] = match_tables[key][name][k]

            if candidate_mag != float("inf"):
                data['mag_' + str(i)][j] = candidate_mag
    return data.dropna()


def mag_to_flux(mag: np.float, exp_time: float = 1., zeropoint: float = 0.0, extinction: float = 0.0,
                airmass: float = 0.0):
    return exp_time * 10 ** (-(mag - zeropoint + extinction * airmass) / 2.5)


def insert_synthetic_point_sources_gauss(image: np.ndarray, x: np.float, y: np.float, fwhm: float, mag: np.float = 0.0,
                                         exp_time: float = 1.,
                                         zeropoint: float = 0.0, extinction: float = 0.0, airmass: float = 0.0,
                                         saturate: float = None, model: str = 'gauss'):
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

    if type(mag) is not np.ndarray:
        mag = np.array(mag)
    if type(x) is not np.ndarray:
        x = np.array([x])
    if type(y) is not np.ndarray:
        y = np.array([y])

    gaussian_model = ph.psf.IntegratedGaussianPRF(sigma=u.fwhm_to_std(fwhm=fwhm))
    sources = table.Table()
    sources['x_0'] = x
    sources['y_0'] = y
    sources['flux'] = mag_to_flux(mag=mag, exp_time=exp_time, zeropoint=zeropoint, extinction=extinction,
                                  airmass=airmass)
    print('Generating additive image...')
    add = datasets.make_model_sources_image(shape=image.shape, model=gaussian_model, source_table=sources)

    combine = image + add

    if saturate is not None:
        print('Saturating...')
        combine[combine > saturate] = saturate
    print('Done.')

    return combine, sources


def insert_synthetic_point_sources_psfex(image: np.ndarray, x: np.float, y: np.float,
                                         model: Union[str, fits.hdu.HDUList], mag: np.float = 0.0,
                                         exp_time: float = 1.,
                                         zeropoint: float = 0.0, extinction: float = 0.0, airmass: float = 0.0,
                                         saturate: float = None):
    """
    Use a PSFEx psf model to insert a synthetic point source into the file.
    :param image:
    :param x:
    :param y:
    :param model: Path to model .psf file.
    :param mag:
    :param exp_time:
    :param zeropoint:
    :param extinction:
    :param airmass:
    :param saturate:
    :return:
    """

    if type(mag) is not np.ndarray:
        mag = np.array(mag)
    if type(x) is not np.ndarray:
        x = np.array([x])
    if type(y) is not np.ndarray:
        y = np.array([y])

    model, path = ff.path_or_hdu(model)
    psf_fwhm = model[1].header['PSF_FWHM']
    gauss_fwhm = 0.41 * psf_fwhm

    gaussian_model = ph.psf.IntegratedGaussianPRF(sigma=u.fwhm_to_std(fwhm=gauss_fwhm))

    sources = table.Table()
    combine = np.zeros(image.shape)
    print('Generating additive image...')
    for i in range(len(x)):
        source = table.Table()

        source['x_0'] = x
        source['y_0'] = y
        source['flux'] = mag_to_flux(mag=mag, exp_time=exp_time, zeropoint=zeropoint, extinction=extinction,
                                     airmass=airmass)

        print('Convolving...')
        add = datasets.make_model_sources_image(shape=image.shape, model=gaussian_model, source_table=source)
        kernel = convolution.CustomKernel(psfex(model=model, x=source['x_0'], y=source['y_0']))
        add = convolution.convolve(add, kernel)

        combine = combine + add

        if i == 0:
            sources = source
        else:
            sources = table.vstack([sources, source])

    combine = image + combine

    if saturate is not None:
        print('Saturating...')
        combine[combine > saturate] = saturate
    print('Done.')

    return combine, sources


def psfex(model: str, x, y):
    model, path = ff.path_or_hdu(model)

    header = model[1].header

    a = model[1].data[0][0]

    x = (x - header['POLZERO1']) / header['POLSCAL1']
    y = (y - header['POLZERO2']) / header['POLSCAL2']

    print(len(a))

    if len(a) == 3:
        psf = a[0] + a[1] * x + a[2] * y

    elif len(a) == 6:
        psf = a[0] + a[1] * x + a[2] * x ** 2 + a[3] * y + a[4] * y ** 2 + a[5] * x * y

    elif len(a) == 10:
        psf = a[0] + a[1] * x + a[2] * x ** 2 + a[3] * x ** 3 + a[4] * y + a[5] * x * y + a[6] * x ** 2 * y + \
              a[7] * y ** 2 + a[8] * x * y ** 2 + a[9] * y ** 3

    else:
        raise ValueError("I haven't accounted for polynomials of order > 3. My bad.")

    if path:
        model.close()

    return psf


def insert_point_sources_to_file(file: Union[fits.hdu.HDUList, str], x: np.float, y: np.float, mag: np.float,
                                 fwhm: float,
                                 output: str = None, overwrite: bool = True,
                                 zeropoint: float = 0.0,
                                 extinction: float = 0.0,
                                 airmass: float = None,
                                 exp_time: float = None,
                                 saturate: float = None,
                                 world_coordinates: bool = False,
                                 extra_values: table.Table = None,
                                 psf_model: Union[str, fits.HDUList] = None):
    """

    :param file:
    :param x:
    :param y:
    :param mag:
    :param fwhm:
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
    path = False
    if type(file) is str:
        path = True
        file = fits.open(file)

    if saturate is None and 'SATURATE' in file[0].header:
        saturate = file[0].header['SATURATE']
        print('SATURATE:', saturate)
    if airmass is None and 'AIRMASS' in file[0].header:
        airmass = file[0].header['AIRMASS']
        print('AIRMASS:', airmass)
    if exp_time is None and 'EXPTIME' in file[0].header:
        exp_time = file[0].header['EXPTIME']
        print('EXPTIME:', exp_time)

    wcs_info = wcs.WCS(file[0].header)
    if world_coordinates:
        ra = x
        dec = y
        x, y = wcs_info.all_world2pix(x, y, 0)
        x = np.array([x])
        y = np.array([y])
    else:
        ra, dec = wcs_info.all_pix2world(x, y, 0)

    if psf_model is not None:
        file[0].data, sources = insert_synthetic_point_sources_psfex(image=file[0].data, x=x, y=y, mag=mag,
                                                                     exp_time=exp_time,
                                                                     zeropoint=zeropoint, extinction=extinction,
                                                                     airmass=airmass,
                                                                     saturate=saturate, model=psf_model)

    else:
        file[0].data, sources = insert_synthetic_point_sources_gauss(image=file[0].data, x=x, y=y, fwhm=fwhm, mag=mag,
                                                                     exp_time=exp_time,
                                                                     zeropoint=zeropoint, extinction=extinction,
                                                                     airmass=airmass,
                                                                     saturate=saturate)

    if extra_values is not None:
        sources = table.hstack([sources, extra_values], join_type='exact')

    print('Saving to', output)
    if output is not None:
        file.writeto(output, overwrite=overwrite)

    if path:
        file.close()

    sources['mag'], _, _ = magnitude_complete(flux=sources['flux'], exp_time=exp_time, zeropoint=zeropoint,
                                              airmass=airmass,
                                              ext=extinction)
    sources['ra'] = ra
    sources['dec'] = dec

    sources.write(filename=output.replace('.fits', '.csv'), format='csv', overwrite=overwrite)

    print('Done.')

    return file, sources


def insert_random_point_sources_to_file(file: Union[fits.hdu.HDUList, str], fwhm: float, output: str, n: int = 1000,
                                        exp_time: float = 1.,
                                        zeropoint: float = 0., max_mag: float = 30, min_mag: float = 20.,
                                        extinction: float = 0., airmass: float = None, overwrite: bool = True,
                                        saturate: float = None):
    # TODO: For these methods, make it read from file header if None for some of the arguments.

    if type(file) is str:
        file = fits.open(file)

    print('Generating objects...')
    n_x, n_y = file[0].data.shape
    x = np.random.uniform(0, n_x, size=n)
    y = np.random.uniform(0, n_y, size=n)
    mag = np.random.uniform(min_mag, max_mag, size=n)

    return insert_point_sources_to_file(file=file, x=x, y=y, fwhm=fwhm, mag=mag, exp_time=exp_time, zeropoint=zeropoint,
                                        extinction=extinction, airmass=airmass, output=output, overwrite=overwrite,
                                        saturate=saturate)


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
        print(instrument)
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
    if instrument == 'FORS2' or instrument == 'XSHOOTER':
        f_0 = filt[0]
        f_output = f_0
    elif instrument == 'IMACS':
        f_0 = filt[-1]
        f_output = filt
    else:
        raise ValueError('Invalid instrument.')
    if outputs is None:
        outputs = p.object_output_params(obj=obj, instrument=instrument)

    if f_output + '_zeropoint_provided' in outputs and outputs[f_output + '_zeropoint_provided'] is not None:
        # If there is a zeropoint provided by the telescope, use that and associated parameters.
        zeropoint = outputs[f_output + '_zeropoint_provided']
        zeropoint_err = outputs[f_output + '_zeropoint_provided_err']
        airmass = outputs[f_output + '_airmass_mean']
        airmass_err = outputs[f_output + '_airmass_err']
        extinction = outputs[f_output + '_extinction']
        extinction_err = outputs[f_output + '_extinction_err']

    elif f_output + '_zeropoint_derived' in outputs and outputs[f_output + '_zeropoint_derived'] is not None:
        # If not, defer to the zeropoint derived from the field. Extinction and airmass set to zero because extinction
        # correction has already been performed for this quantity.
        zeropoint = outputs[f_output + '_zeropoint_derived']
        zeropoint_err = outputs[f_output + '_zeropoint_derived_err']
        airmass = 0.0
        airmass_err = 0.0
        extinction = 0.0
        extinction_err = 0.0

    elif f_output + '_zeropoint_std' in outputs and outputs[f_output + '_zeropoint_std'] is not None:
        # If neither of those exist, use the zeropoint derived from provided standard fields and correct for the
        # difference in airmass.
        zeropoint = outputs[f_output + '_zeropoint_std']
        zeropoint_err = outputs[f_output + '_zeropoint_std_err']

        airmass_field = outputs[f_output + '_airmass_mean']
        airmass_field_err = outputs[f_output + '_airmass_err']
        airmass_std = outputs[f_output + '_airmass_std']
        airmass_std_err = 0.0
        airmass = airmass_field - airmass_std
        airmass_err = airmass_field_err + airmass_std_err
        print('Standard airmass:', airmass_std)
        print('Delta airmass:', airmass)

        extinction = outputs[f_output + '_extinction']
        extinction_err = outputs[f_output + '_extinction_err']

    else:
        raise ValueError('No zeropoint found in output_values file.')

    if instrument == 'IMACS':
        airmass = airmass - 1.0

    return zeropoint, zeropoint_err, airmass, airmass_err, extinction, extinction_err


def subtract(template_origin: str, comparison_origin: str,
             template_fwhm: float,
             comparison_fwhm: float, output: str, comparison_title: str, template_title: str, comparison_epoch: int,
             template_epoch: int,
             field: str,
             force_subtract_better_seeing: bool = True, sextractor_threshold: float = None):
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

            os.system(f'hotpants -inim {template_file}'
                      f' -tmplim {comparison_file}'
                      f' -outim {difference_file}'
                      f' -ng 3 6 {0.5 * sigma_match} 4 {sigma_match} 2 {2 * sigma_match}'
                      f' -oki {output}kernel.fits'
                      f' -n i')

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
