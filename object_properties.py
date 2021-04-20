# Code by Lachlan Marnoch, 2019-2021
from craftutils import params
from craftutils import utils as u, photometry as ph, plotting as p, fits_files as ff
from craftutils.astrometry import calculate_error_ellipse
from craftutils.retrieve import update_frb_des_cutout

from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import time
from astropy.visualization import ImageNormalize, SquaredStretch, SqrtStretch, ZScaleInterval, MinMaxInterval
from astropy import units
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
import photutils as pu

import pandas

import matplotlib.pyplot as plt
import numpy as np
from os.path import isfile
from math import cos, sin


# TODO: Integrate into pipeline
# TODO: Switch to astropy tables

def main(obj,
         show,
         frame,
         instrument,
         cat_name,
         image_spec,
         cone):
    print(obj)

    frame *= units.pixel

    frb = obj[:-2]

    epoch_properties = params.object_params_instrument(obj=obj, instrument=instrument)
    burst_properties = params.object_params_frb(obj=frb)
    print()
    burst_outputs = params.frb_output_params(obj=frb)
    paths = params.object_output_paths(obj=obj, instrument=instrument)

    filters = epoch_properties['filters']
    fixed_aperture = epoch_properties['photometry_apertures'][0]

    burst_coord = SkyCoord(burst_properties["burst_ra"] * units.degree, burst_properties["burst_dec"] * units.degree)

    for f in filters:
        if burst_outputs is None or f"{f}_ext_gal" not in burst_outputs:
            print(f"\nGalactic extinction missing for {f}; calculating now.")
            import extinction_galactic
            extinction_galactic.main(obj=frb)
            burst_outputs = params.frb_output_params(obj=frb)

    galaxies = burst_properties['other_objects']
    if galaxies is None:
        galaxies = {}
    for i, ra in enumerate(epoch_properties['test_synths']['ra']):
        galaxies[f"fake_star_{i}"] = {"ra": ra, "dec": epoch_properties['test_synths']['dec'][i]}
    hg_ra = burst_properties['hg_ra']
    hg_dec = burst_properties['hg_dec']
    if hg_ra == 0.0 or hg_dec == 0.0:
        hg_ra = burst_properties['burst_ra']
        hg_dec = burst_properties['burst_dec']
    galaxies[obj + ' Host'] = {'ra': hg_ra, 'dec': hg_dec}

    a, b, theta = calculate_error_ellipse(frb=frb)
    line_style = '-'
    if a == 0.0:
        a = 0.5 / 3600
        line_style = ":"
    if b == 0.0:
        a = 0.5 / 3600
        line_style = ":"

    now = time.Time.now()
    now.format = 'isot'

    des_cat_path = burst_properties[f'data_dir'] + "DES/DES.csv"
    if not isfile(des_cat_path):
        des_cat_path = None

    cat_name_for_path = cat_name
    if type(cat_name_for_path) is str:
        cat_name_for_path = cat_name_for_path.replace('/', '')

    output_path = f"{epoch_properties['data_dir']}/analysis/object_properties/"
    u.mkdir_check(output_path)
    output_path = f"{epoch_properties['data_dir']}/analysis/object_properties/" \
                  f"{str(now)}_{cat_name_for_path}_{image_spec}/"
    u.mkdir_check(output_path)

    instrument = instrument.upper()

    for f in filters:
        print()
        print('FILTER:', f)
        if instrument == 'FORS2' or instrument == 'XSHOOTER':
            f_0 = f[0]
            f_output = f_0
        elif instrument == 'IMACS':
            f_0 = f[-1]
            f_output = f
        else:
            raise ValueError('Invalid instrument.')
        f_up = f_0.upper()

        if image_spec is None:
            image_path = paths[f_output + '_' + epoch_properties['subtraction_image']]
        else:
            image_path = paths[f_output + '_' + image_spec]

        if cat_name is None:
            cat_path = paths[f'{f_output}_cat_path']
        else:
            if epoch_properties['do_dual_mode'] and f != epoch_properties['deepest_filter']:
                cat_path = epoch_properties[
                               'data_dir'] + 'analysis/sextractor/' + cat_name + '/' + f_0 + '_dual-mode.cat'

            else:
                cat_path = epoch_properties['data_dir'] + 'analysis/sextractor/' + cat_name + '/' + f_0 + '_psf-fit.cat'

        cat_path_local = cat_path.replace(".cat", "_back_local.cat")

        print('Catalogue:', cat_path)

        des_path = f"{burst_properties['data_dir']}/DES/0-data/{f_0.lower()}_cutout.fits"
        if not isfile(des_path):
            if not update_frb_des_cutout(frb=frb):
                des_path = None

        print(f)

        zeropoint, zeropoint_err, airmass, airmass_err, extinction, extinction_err \
            = ph.select_zeropoint(obj=obj, filt=f, instrument=instrument)

        print(f"{f} zeropoint:", zeropoint, '+/-', zeropoint_err)
        print(f'{f_0} airmass:', airmass, '+/-', airmass_err)
        print(f'{f_0} extinction co-efficient:', extinction, '+/-', extinction_err)

        colour_term = epoch_properties[f'{f_0}_colour_term']
        colour_term_err = epoch_properties[f'{f_0}_colour_term_err']

        exp_time = ff.get_exp_time(image_path)
        exp_time_err = 0.
        print(f"{f} exposure time:", exp_time, 's')

        ext_gal = burst_outputs[f"{f}_ext_gal"]

        output_params = {"zeropoint": zeropoint, "zeropoint_err": zeropoint_err, "airmass": airmass,
                         "airmass_err": airmass_err, "extinction": extinction, "extinction_err": extinction_err,
                         "kX": extinction * airmass,
                         "kX_err": float(u.error_product(extinction * airmass, measurements=[airmass, extinction],
                                                         errors=[airmass_err, extinction_err])),
                         "colour_term": colour_term, "colour_term_err": colour_term_err,
                         "exp_time": exp_time, "exp_time_err": exp_time_err, "ext_gal": ext_gal,
                         "sextractor_catalogue": cat_path}
        output_catalogue = {}

        print('Loading FITS file from', image_path)
        image = fits.open(image_path)
        data = image[0].data
        header = image[0].header
        _, pix_scale = ff.get_pixel_scale(image, astropy_units=True)
        wcs_main = wcs.WCS(header=header)
        norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=SqrtStretch())

        # Analysis
        cat = Table.read(cat_path, format="ascii.sextractor")

        # print(cat.colnames)
        # np.genfromtxt(cat_path, names=params.sextractor_names_psf())
        mag_auto_true, mag_auto_err_plus, mag_auto_err_minus = ph.magnitude_complete(flux=cat['FLUX_AUTO'],
                                                                                     flux_err=cat['FLUXERR_AUTO'],
                                                                                     exp_time=exp_time,
                                                                                     exp_time_err=exp_time_err,
                                                                                     zeropoint=zeropoint,
                                                                                     zeropoint_err=zeropoint_err,
                                                                                     colour_term=colour_term,
                                                                                     colour_term_err=colour_term_err,
                                                                                     ext=extinction,
                                                                                     ext_err=extinction_err,
                                                                                     airmass=airmass,
                                                                                     airmass_err=airmass_err
                                                                                     )
        mag_psf, mag_psf_err_plus, mag_psf_err_minus = ph.magnitude_complete(flux=cat['FLUX_PSF'],
                                                                             flux_err=cat['FLUXERR_PSF'],
                                                                             exp_time=exp_time,
                                                                             exp_time_err=exp_time_err,
                                                                             zeropoint=zeropoint,
                                                                             zeropoint_err=zeropoint_err,
                                                                             colour_term=colour_term,
                                                                             colour_term_err=colour_term_err,
                                                                             ext=extinction,
                                                                             ext_err=extinction_err,
                                                                             airmass=airmass,
                                                                             airmass_err=airmass_err
                                                                             )

        mag_aper, mag_aper_err_plus, mag_aper_err_minus = ph.magnitude_complete(flux=cat['FLUX_APER'],
                                                                                flux_err=cat['FLUXERR_APER'],
                                                                                exp_time=exp_time,
                                                                                exp_time_err=exp_time_err,
                                                                                zeropoint=zeropoint,
                                                                                zeropoint_err=zeropoint_err,
                                                                                colour_term=colour_term,
                                                                                colour_term_err=colour_term_err,
                                                                                ext=extinction,
                                                                                ext_err=extinction_err,
                                                                                airmass=airmass,
                                                                                airmass_err=airmass_err
                                                                                )

        cat_local = Table.read(cat_path_local, format="ascii.sextractor")
        mag_auto_local, mag_auto_local_err_plus, mag_auto_local_err_minus = ph.magnitude_complete(
            flux=cat_local['FLUX_AUTO'],
            flux_err=cat_local['FLUXERR_AUTO'],
            exp_time=exp_time,
            exp_time_err=exp_time_err,
            zeropoint=zeropoint,
            zeropoint_err=zeropoint_err,
            colour_term=colour_term,
            colour_term_err=colour_term_err,
            ext=extinction,
            ext_err=extinction_err,
            airmass=airmass,
            airmass_err=airmass_err
        )

        mag_aper_local, mag_aper_local_err_plus, mag_aper_local_err_minus = ph.magnitude_complete(
            flux=cat_local['FLUX_APER'],
            flux_err=cat_local['FLUXERR_APER'],
            exp_time=exp_time,
            exp_time_err=exp_time_err,
            zeropoint=zeropoint,
            zeropoint_err=zeropoint_err,
            colour_term=colour_term,
            colour_term_err=colour_term_err,
            ext=extinction,
            ext_err=extinction_err,
            airmass=airmass,
            airmass_err=airmass_err
        )

        p.plot_all_params(image=image, cat=cat, kron=True, show=show)

        if cone is not None:
            cat['dist_from_burst'] = SkyCoord(cat['ALPHA_SKY'], cat['DELTA_SKY']).separation(burst_coord).arcsec
            cat_cone = cat[cat['dist_from_burst'] < cone]
            for i, thing in enumerate(cat_cone):
                galaxies[f'cone_{i}'] = {"ra": thing['ALPHA_SKY'], "dec": thing['DELTA_SKY']}

        for o in galaxies:
            ra = galaxies[o]['ra']
            dec = galaxies[o]['dec']
            coord = SkyCoord(ra * units.deg, dec * units.deg)
            if not coord.contained_by(wcs_main):
                print(f"{o} is not in this image's footprint; skipping.")
                continue
            print('Matching...')
            index, dist = u.find_object(ra, dec, cat['ALPHA_SKY'], cat['DELTA_SKY'])
            index_local, dist_local = u.find_object(ra, dec, cat_local['ALPHA_SKY'], cat_local['DELTA_SKY'])
            this = cat[index]
            print(f'{o} SExtractor {f}')
            print()

            mag_err = max(abs(mag_auto_err_plus[index]), abs(mag_auto_err_minus[index]))
            mag_psf_err = max(abs(mag_psf_err_plus[index]), abs(mag_psf_err_minus[index]))
            mag_auto_local_err = max(abs(mag_auto_local_err_minus[index_local]),
                                     abs(mag_auto_local_err_plus[index_local]))
            mag_aper_err = max(abs(mag_aper_err_minus[index_local]),
                               abs(mag_aper_err_plus[index_local]))
            mag_aper_local_err = max(abs(mag_aper_local_err_minus[index_local]),
                                     abs(mag_aper_err_plus[index_local]))

            mag_ins, mag_ins_err_1, mag_ins_err_2 = ph.magnitude_error(flux=np.array([this['FLUX_AUTO']]),
                                                                       flux_err=np.array([this['FLUXERR_AUTO']]),
                                                                       exp_time=exp_time, exp_time_err=exp_time_err)
            mag_ins = mag_ins[0]
            mag_ins_err = max(abs(mag_ins_err_1[0]), abs(mag_ins_err_2[0]))

            # Do photutils photometry
            # Define aperture using mag_auto kron_aperture:
            kron_a = this['KRON_RADIUS'] * this['A_WORLD'] * units.deg
            kron_b = this['KRON_RADIUS'] * this['B_WORLD'] * units.deg
            # Convert theta to the units and frame photutils likes
            kron_theta = this['THETA_WORLD'] * units.deg
            kron_theta = -kron_theta + ff.get_rotation_angle(header=header, astropy_units=True)
            kron_theta = kron_theta.to(units.rad)
            # Establish initial values so that Python doesn't barf
            mag_photutils = np.nan
            flux_photutils = np.nan
            subtract = np.nan
            median = np.nan
            # Convert from SExtractor origin (1,1) to numpy origin (0,0)
            x_image_mod = this['X_IMAGE'] - 1
            y_image_mod = this['Y_IMAGE'] - 1
            if kron_a >= kron_b > 0:

                aperture = pu.EllipticalAperture(positions=(x_image_mod, y_image_mod),
                                                 a=kron_a.to(units.pixel, pix_scale).value,
                                                 b=kron_b.to(units.pixel, pix_scale).value,
                                                 theta=kron_theta.value)
                # Define background annulus:
                annulus = pu.EllipticalAnnulus(positions=(x_image_mod, y_image_mod),
                                               a_in=2 * kron_a.to(units.pixel, pix_scale).value,
                                               a_out=3 * kron_a.to(units.pixel, pix_scale).value,
                                               b_out=3 * kron_b.to(units.pixel, pix_scale).value,
                                               theta=kron_theta.value
                                               )
                mag_photutils, flux_photutils, subtract_photutils, median_photutils = ph.single_aperture_photometry(
                    data=data, aperture=aperture, annulus=annulus,
                    exp_time=exp_time, zeropoint=zeropoint,
                    extinction=extinction,
                    airmass=airmass)

                plt.imshow(data, origin='lower', norm=norm, )
                aperture.plot(color='violet', label='Kron aperture')
                annulus.plot(color='cyan', label='Photutils background annulus')
                print(this['X_IMAGE'], this['Y_IMAGE'])
                plt.legend()
                plt.title(f"{o} (photutils), {f_0}-band image")
                plt.savefig(output_path + f + '_' + o + "_photutils")
                if show:
                    plt.show()
                plt.close()

            # Do the above for the fixed aperture.

            aperture_radius = fixed_aperture / 2.
            aperture = pu.CircularAperture(positions=(x_image_mod, y_image_mod), r=aperture_radius)
            # Define background annulus:
            annulus = pu.CircularAnnulus(positions=(x_image_mod, y_image_mod),
                                         r_in=2 * aperture_radius,
                                         r_out=3 * aperture_radius,
                                         )
            # Use background annulus to obtain a median sky background
            mag_photutils_fixed, flux_photutils_fixed, subtract_photutils_fixed, median_photutils_fixed = \
                ph.single_aperture_photometry(
                    data=data, aperture=aperture, annulus=annulus,
                    exp_time=exp_time, zeropoint=zeropoint,
                    extinction=extinction,
                    airmass=airmass)

            # Again, but with the aperture forced to the given coordinates.

            x_given, y_given = coord.to_pixel(wcs=wcs_main)

            aperture = pu.CircularAperture(positions=(x_given, y_given), r=aperture_radius)
            # Define background annulus:
            annulus = pu.CircularAnnulus(positions=(x_image_mod, y_image_mod),
                                         r_in=2 * aperture_radius,
                                         r_out=3 * aperture_radius,
                                         )
            # Use background annulus to obtain a median sky background
            mag_photutils_forced, flux_photutils_forced, subtract_photutils_forced, median_photutils_forced = \
                ph.single_aperture_photometry(
                    data=data, aperture=aperture, annulus=annulus,
                    exp_time=exp_time, zeropoint=zeropoint,
                    extinction=extinction,
                    airmass=airmass)

            # plt.imshow(data, origin='lower', norm=norm, )
            # aperture.plot(color='violet', label='Photutils aperture')
            # annulus.plot(color='cyan', label='Background annulus')
            # plt.legend()
            # plt.title(f"{o} (fixed aperture), {f_0}-band image")
            # plt.savefig(output_path + f + '_' + o + "_fixed_aperture")
            # if show:
            #     plt.show()
            # plt.close()

            output_catalogue_this = {'id': o,
                                     'ra': float(this['ALPHA_SKY']), 'ra_err_2': float(this['ERRX2_WORLD']),
                                     'dec': float(this['DELTA_SKY']), 'dec_err_2': float(this['ERRY2_WORLD']),
                                     'ra_given': float(ra), 'dec_given': float(dec),
                                     'matching_distance_sex': float(dist * 3600),
                                     'kron_radius': float(this['KRON_RADIUS']),
                                     'a': float(this['A_WORLD'] * 3600), 'a_err': float(this['ERRA_WORLD'] * 3600),
                                     'b': float(this['B_WORLD'] * 3600), 'b_err': float(this['ERRB_WORLD'] * 3600),
                                     'theta': float(this['THETA_WORLD']), 'theta_err': float(this['ERRTHETA_WORLD']),
                                     'mag_auto': float(mag_auto_true[index]),
                                     'mag_auto_err': float(mag_err), 'mag_ins': float(mag_ins),
                                     'mag_auto_gal_correct': float(mag_auto_true[index]) - ext_gal,
                                     'mag_auto_local': float(mag_auto_local[index_local]),
                                     'mag_auto_local_err': float(mag_auto_local_err),
                                     'matching_distance_local': float(dist_local * 3600),
                                     'mag_photutils': float(mag_photutils),
                                     'mag_aper': float(mag_aper[index_local]),
                                     'mag_aper_err': float(mag_aper_err),
                                     'mag_aper_local': float(mag_aper_local[index_local]),
                                     'mag_aper_local_err': float(mag_aper_local_err),
                                     'mag_photutils_aper': float(mag_photutils_fixed),
                                     'mag_photutils_forced': float(mag_photutils_forced),
                                     # 'mag_photutils_err': float(mag_photutils_err),
                                     'ext_gal': float(ext_gal),
                                     'mag_ins_err': float(mag_ins_err), 'flux_auto': float(this['FLUX_AUTO']),
                                     'flux_auto_err': float(this['FLUXERR_AUTO']),
                                     'flux_photutils': float(flux_photutils),
                                     'flux_offset_photutils': float(subtract),
                                     # flux_offset_photutils is the total background flux subtracted from the aperture flux.
                                     'median_background_photutils': float(median),
                                     'flux_offset_auto': float(this['FLUX_BACKOFFSET']),
                                     'flux_offset_auto_err': float(this['FLUXERR_BACKOFFSET']),
                                     'mag_psf': float(mag_psf[index]),
                                     'mag_psf_err': float(mag_psf_err),
                                     'flux_psf': float(this['FLUX_PSF']), 'fluxerr_psf': float(this['FLUXERR_PSF']),
                                     'x': float(this['X_IMAGE']), 'y': float(this['Y_IMAGE']),
                                     'background': float(this['BACKGROUND'])
                                     }

            print('RA (deg):', output_catalogue_this['ra'])
            print('DEC (deg):', output_catalogue_this['dec'])
            print('RA burst (deg):', ra)
            print('DEC burst (deg):', dec)
            print(f'{o} Matching distance (arcsec):', output_catalogue_this['matching_distance_sex'])
            print('Kron radius (a, b):', output_catalogue_this['kron_radius'])
            print('a (arcsec):', output_catalogue_this['a'], '+/-', output_catalogue_this['a_err'])
            print('b (arcsec):', output_catalogue_this['b'], '+/-', output_catalogue_this['b_err'])
            print('theta (degrees):', output_catalogue_this['theta'], '+/-', output_catalogue_this['theta_err'])
            print(f'{o} {f} mag auto:', output_catalogue_this['mag_auto'], '+/-',
                  output_catalogue_this['mag_auto_err'])
            print(f'{o} {f} mag auto local back:', output_catalogue_this['mag_auto_local'], '+/-',
                  output_catalogue_this['mag_auto_local_err'])
            print(f'{o} {f} mag photutils:', output_catalogue_this['mag_photutils'])
            print(f'{o} {f} mag aper:', output_catalogue_this['mag_aper'], '+/-', output_catalogue_this['mag_aper_err'])
            print(f'{o} {f} mag aper local:', output_catalogue_this['mag_aper_local'], '+/-',
                  output_catalogue_this['mag_aper_local_err'])
            print(f'{o} {f} mag photutils fixed:', output_catalogue_this['mag_photutils_aper'])
            print(f'{o} {f} mag photutils forced:', output_catalogue_this['mag_photutils_forced'])
            print(f'{o} {f} mag auto corrected for Galactic extinction:', output_catalogue_this['mag_auto_gal_correct'])
            print(f'Galactic extinction used:', ext_gal)
            print(f'{o} {f} mag psf:', output_catalogue_this['mag_psf'], '+/-',
                  output_catalogue_this['mag_psf_err'])
            print(f'{o} {f} flux auto:', output_catalogue_this['flux_auto'], '+/-',
                  output_catalogue_this['flux_auto_err'])
            print(f'{o} {f} flux photutils:', output_catalogue_this['flux_photutils'])
            print()
            print()

            print('Plotting...')

            # p.plot_all_params(image=image, cat=cat)

            # Plot fixed aperture

            mid_x = output_catalogue_this['x'] * units.pix
            mid_y = output_catalogue_this['y'] * units.pix

            print(f"{kron_a} * cos({kron_theta})")

            # Set the frame using the extent of the ellipse.

            this_frame = (aperture_radius * 3 + 1) * units.pix

            print(f"{mid_x} - {this_frame}")

            left = mid_x - this_frame
            right = mid_x + this_frame
            bottom = mid_y - this_frame
            top = mid_y + this_frame

            image_cut = ff.trim(hdu=image, left=left, right=right, bottom=bottom, top=top)

            rad_deg = ((aperture_radius * units.pix).to(units.deg, pix_scale)).value

            plt.imshow(image_cut[0].data, origin='lower')
            # , norm=ImageNormalize(image_cut[0].data, stretch=SqrtStretch(), interval=ZScaleInterval()))
            p.plot_gal_params(hdu=image_cut,
                              ras=[output_catalogue_this['ra_given']],
                              decs=[output_catalogue_this['dec_given']],
                              a=[0],
                              b=[0],
                              theta=[0],
                              show_centre=True,
                              colour='red',
                              label='Given coordinates')
            p.plot_gal_params(hdu=image_cut,
                              ras=[output_catalogue_this['ra']],
                              decs=[output_catalogue_this['dec']],
                              a=[rad_deg],
                              b=[rad_deg],
                              theta=[0],
                              show_centre=True,
                              colour='blue',
                              label=f'SExtractor ellipse')
            p.plot_gal_params(hdu=image_cut,
                              ras=[output_catalogue_this['ra']],
                              decs=[output_catalogue_this['dec']],
                              a=[rad_deg * 2],
                              b=[rad_deg * 2],
                              theta=[0],
                              show_centre=True,
                              colour='violet',
                              label=f'Photutils annulus')
            p.plot_gal_params(hdu=image_cut,
                              ras=[output_catalogue_this['ra']],
                              decs=[output_catalogue_this['dec']],
                              a=[rad_deg * 3],
                              b=[rad_deg * 3],
                              theta=[0],
                              show_centre=True,
                              colour='violet')

            if SkyCoord(burst_properties['burst_ra'] * units.deg,
                        burst_properties['burst_dec'] * units.deg).contained_by(
                wcs.WCS(header=image_cut[0].header)):
                p.plot_gal_params(hdu=image_cut,
                                  ras=[burst_properties['burst_ra']],
                                  decs=[burst_properties['burst_dec']],
                                  a=[a],
                                  b=[b],
                                  theta=[theta],
                                  colour="orange",
                                  label='frb',
                                  line_style=line_style,
                                  show_centre=True)

            plt.legend()
            plt.title(f"{output_catalogue_this['id']}, {f_0}-band image, fixed aperture")
            plt.savefig(output_path + f + '_' + output_catalogue_this['id'] + "_fixed_aperture")
            if show:
                plt.show()
            plt.close()

            this_frame = max(kron_a.to(units.pixel, pix_scale) * np.cos(kron_theta) + 10 * units.pix,
                             kron_a.to(units.pixel, pix_scale) * np.sin(kron_theta) + 10 * units.pix,
                             frame)

            print(f"{mid_x} - {this_frame}")

            left = mid_x - this_frame
            right = mid_x + this_frame
            bottom = mid_y - this_frame
            top = mid_y + this_frame

            image_cut = ff.trim(hdu=image, left=left, right=right, bottom=bottom, top=top)

            plt.imshow(image_cut[0].data, origin='lower')
            # , norm=ImageNormalize(image_cut[0].data, stretch=SqrtStretch(), interval=ZScaleInterval()))
            p.plot_gal_params(hdu=image_cut,
                              ras=[output_catalogue_this['ra_given']],
                              decs=[output_catalogue_this['dec_given']],
                              a=[0],
                              b=[0],
                              theta=[0],
                              show_centre=True,
                              colour='red',
                              label='Given coordinates')
            p.plot_gal_params(hdu=image_cut,
                              ras=[output_catalogue_this['ra']],
                              decs=[output_catalogue_this['dec']],
                              a=[output_catalogue_this['a'] / 3600],
                              b=[output_catalogue_this['b'] / 3600],
                              theta=[output_catalogue_this['theta']],
                              show_centre=True,
                              colour='blue',
                              label=f'SExtractor ellipse')
            p.plot_gal_params(hdu=image_cut,
                              ras=[output_catalogue_this['ra']],
                              decs=[output_catalogue_this['dec']],
                              a=[kron_a.value],
                              b=[kron_b.value],
                              theta=[output_catalogue_this['theta']],
                              colour='violet',
                              label='Kron aperture', line_style='-')
            print(mid_x, output_catalogue_this['x'])
            print(mid_y, output_catalogue_this['y'])

            if SkyCoord(burst_properties['burst_ra'] * units.deg,
                        burst_properties['burst_dec'] * units.deg).contained_by(
                wcs.WCS(header=image_cut[0].header)):
                p.plot_gal_params(hdu=image_cut,
                                  ras=[burst_properties['burst_ra']],
                                  decs=[burst_properties['burst_dec']],
                                  a=[a],
                                  b=[b],
                                  theta=[theta],
                                  colour="orange",
                                  label='frb',
                                  line_style=line_style,
                                  show_centre=True)

            plt.legend()
            plt.title(f"{output_catalogue_this['id']}, {f_0}-band image")
            plt.savefig(output_path + f + '_' + output_catalogue_this['id'])
            if show:
                plt.show()
            plt.close()

            # Plot forced aperture

            mid_x = x_given * units.pix
            mid_y = y_given * units.pix

            # Set the frame using the extent of the ellipse.

            this_frame = (aperture_radius * 3 + 5) * units.pix

            print(f"{mid_x} - {this_frame}")

            left = mid_x - this_frame
            right = mid_x + this_frame
            bottom = mid_y - this_frame
            top = mid_y + this_frame

            image_cut = ff.trim(hdu=image, left=left, right=right, bottom=bottom, top=top)

            rad_deg = ((aperture_radius * units.pix).to(units.deg, pix_scale)).value

            plt.imshow(image_cut[0].data, origin='lower')
            # , norm=ImageNormalize(image_cut[0].data, stretch=SqrtStretch(), interval=ZScaleInterval()))
            p.plot_gal_params(hdu=image_cut,
                              ras=[ ],
                              decs=[output_catalogue_this['dec_given']],
                              a=[0],
                              b=[0],
                              theta=[0],
                              show_centre=True,
                              colour='red',
                              label='Given coordinates')
            p.plot_gal_params(hdu=image_cut,
                              ras=[output_catalogue_this['ra_given']],
                              decs=[output_catalogue_this['dec_given']],
                              a=[rad_deg],
                              b=[rad_deg],
                              theta=[0],
                              show_centre=True,
                              colour='blue',
                              label=f'Forced ellipse')
            p.plot_gal_params(hdu=image_cut,
                              ras=[output_catalogue_this['ra_given']],
                              decs=[output_catalogue_this['dec_given']],
                              a=[rad_deg * 2],
                              b=[rad_deg * 2],
                              theta=[0],
                              show_centre=True,
                              colour='violet',
                              label=f'Photutils annulus')
            p.plot_gal_params(hdu=image_cut,
                              ras=[output_catalogue_this['ra_given']],
                              decs=[output_catalogue_this['dec_given']],
                              a=[rad_deg * 3],
                              b=[rad_deg * 3],
                              theta=[0],
                              show_centre=True,
                              colour='violet')

            if SkyCoord(burst_properties['burst_ra'] * units.deg,
                        burst_properties['burst_dec'] * units.deg).contained_by(
                wcs.WCS(header=image_cut[0].header)):
                p.plot_gal_params(hdu=image_cut,
                                  ras=[burst_properties['burst_ra']],
                                  decs=[burst_properties['burst_dec']],
                                  a=[a],
                                  b=[b],
                                  theta=[theta],
                                  colour="orange",
                                  label='frb',
                                  line_style=line_style,
                                  show_centre=True)

            plt.legend()
            plt.title(f"{output_catalogue_this['id']}, {f_0}-band image, forced-position aperture")
            plt.savefig(output_path + f + '_' + output_catalogue_this['id'] + "_fixed_aperture")
            if show:
                plt.show()
            plt.close()

            this_frame = max(kron_a.to(units.pixel, pix_scale) * np.cos(kron_theta) + 10 * units.pix,
                             kron_a.to(units.pixel, pix_scale) * np.sin(kron_theta) + 10 * units.pix,
                             frame)

            print(f"{mid_x} - {this_frame}")

            left = mid_x - this_frame
            right = mid_x + this_frame
            bottom = mid_y - this_frame
            top = mid_y + this_frame

            image_cut = ff.trim(hdu=image, left=left, right=right, bottom=bottom, top=top)

            plt.imshow(image_cut[0].data, origin='lower')
            # , norm=ImageNormalize(image_cut[0].data, stretch=SqrtStretch(), interval=ZScaleInterval()))
            p.plot_gal_params(hdu=image_cut,
                              ras=[output_catalogue_this['ra_given']],
                              decs=[output_catalogue_this['dec_given']],
                              a=[0],
                              b=[0],
                              theta=[0],
                              show_centre=True,
                              colour='red',
                              label='Given coordinates')
            p.plot_gal_params(hdu=image_cut,
                              ras=[output_catalogue_this['ra']],
                              decs=[output_catalogue_this['dec']],
                              a=[output_catalogue_this['a'] / 3600],
                              b=[output_catalogue_this['b'] / 3600],
                              theta=[output_catalogue_this['theta']],
                              show_centre=True,
                              colour='blue',
                              label=f'SExtractor ellipse')
            p.plot_gal_params(hdu=image_cut,
                              ras=[output_catalogue_this['ra']],
                              decs=[output_catalogue_this['dec']],
                              a=[kron_a.value],
                              b=[kron_b.value],
                              theta=[output_catalogue_this['theta']],
                              colour='violet',
                              label='Kron aperture', line_style='-')
            print(mid_x, output_catalogue_this['x'])
            print(mid_y, output_catalogue_this['y'])

            if SkyCoord(burst_properties['burst_ra'] * units.deg,
                        burst_properties['burst_dec'] * units.deg).contained_by(
                wcs.WCS(header=image_cut[0].header)):
                p.plot_gal_params(hdu=image_cut,
                                  ras=[burst_properties['burst_ra']],
                                  decs=[burst_properties['burst_dec']],
                                  a=[a],
                                  b=[b],
                                  theta=[theta],
                                  colour="orange",
                                  label='frb',
                                  line_style=line_style,
                                  show_centre=True)

            plt.legend()
            plt.title(f"{output_catalogue_this['id']}, {f_0}-band image")
            plt.savefig(output_path + f + '_' + output_catalogue_this['id'])
            if show:
                plt.show()
            plt.close()

            if des_cat_path is not None and des_path is not None:
                des = np.genfromtxt(des_cat_path, names=True, delimiter=',')
                _, des_pix_scale = ff.get_pixel_scale(des_path)

                print('Matching...')
                des_ind_other, dist = u.find_object(ra, dec, des['RA'], des['DEC'])
                des_other = des[des_ind_other]

                output_catalogue_this['id_des'] = des_other['COADD_OBJECT_ID']
                output_catalogue_this['ra_des'] = des_other['RA']
                output_catalogue_this['dec_des'] = des_other['DEC']
                output_catalogue_this['matching_distance_des'] = dist * 3600
                output_catalogue_this['kron_radius_des'] = des_other[f'KRON_RADIUS_{f_up}']
                output_catalogue_this['a_des'] = des_other['A_IMAGE'] * 3600 * des_pix_scale
                output_catalogue_this['a_des_err'] = des_other['ERRA_IMAGE'] * 3600 * des_pix_scale
                output_catalogue_this['b_des'] = des_other['B_IMAGE'] * 3600 * des_pix_scale
                output_catalogue_this['b_des_err'] = des_other['ERRB_IMAGE'] * 3600 * des_pix_scale
                output_catalogue_this['theta_des'] = des_other['THETA_J2000']
                output_catalogue_this['theta_des_err'] = des_other['ERRTHETA_IMAGE']
                output_catalogue_this['mag_auto_des'] = des_other[f'MAG_AUTO_{f_up}']
                output_catalogue_this['mag_auto_des_err'] = des_other[f'MAGERR_AUTO_{f_up}']
                output_catalogue_this['mag_psf_des'] = des_other[f'WAVG_MAG_PSF_{f_up}']
                output_catalogue_this['mag_psf_des_err'] = des_other[f'WAVG_MAGERR_PSF_{f_up}']

                print('DES:', output_catalogue_this['id_des'])
                print('RA (deg):', output_catalogue_this['ra_des'])
                print('DEC (deg):', output_catalogue_this['dec_des'])
                print(' Matching distance (arcsec):', output_catalogue_this['matching_distance_des'])
                print('Kron radius (pixels):', output_catalogue_this['kron_radius_des'])
                print('a (arcsec):', output_catalogue_this['a_des'], '+/-',
                      output_catalogue_this['a_des_err'])
                print('b (arcsec):', output_catalogue_this['b_des'], '+/-',
                      output_catalogue_this['b_des_err'])
                print('theta (degrees):', output_catalogue_this['theta_des'], '+/-', )
                print(o + f' DES {f_0} mag auto:', output_catalogue_this['mag_auto_des'], '+/-',
                      output_catalogue_this['mag_auto_des_err'])
                print(o + f' DES {f_0} mag psf:', output_catalogue_this['mag_psf_des'], '+/-',
                      output_catalogue_this['mag_psf_des_err'])
                print()
                print()

                # TODO: Make PEP 8 Happy.
                # TODO: Generalise this to other catalogues.

                # Plotting

                print('Loading FITS file...')
                des_image = fits.open(des_path)
                print('Plotting...')
                wcs_des = wcs.WCS(header=des_image[0].header)
                mid_x, mid_y = wcs_des.all_world2pix(output_catalogue_this['ra_des'],
                                                     output_catalogue_this['dec_des'], 0)
                mid_x = int(mid_x)
                mid_y = int(mid_y)

                left = mid_x - frame.value
                right = mid_x + frame.value
                bottom = mid_y - frame.value
                top = mid_y + frame.value

                des_image_cut = ff.trim(hdu=des_image, left=left, right=right, bottom=bottom, top=top)

                plt.imshow(des_image_cut[0].data, origin='lower',
                           norm=ImageNormalize(des_image_cut[0].data, stretch=SqrtStretch(),
                                               interval=MinMaxInterval()), )

                p.plot_gal_params(hdu=des_image_cut, ras=[output_catalogue_this['ra_given']],
                                  decs=[output_catalogue_this['dec_given']],
                                  a=[0],
                                  b=[0],
                                  theta=[0], show_centre=True, colour='red',
                                  label='Given coordinates', world_axes=True)
                p.plot_gal_params(hdu=des_image_cut,
                                  ras=[output_catalogue_this['ra_des']],
                                  decs=[output_catalogue_this['dec_des']],
                                  a=[output_catalogue_this['kron_radius_des']],
                                  b=[output_catalogue_this['kron_radius_des']
                                     * output_catalogue_this['b_des'] / output_catalogue_this['a_des']],
                                  theta=[output_catalogue_this['theta_des']],
                                  show_centre=True,
                                  colour='violet',
                                  label='Kron aperture',
                                  world_axes=False)
                p.plot_gal_params(hdu=des_image_cut, ras=[output_catalogue_this['ra_des']],
                                  decs=[output_catalogue_this['dec_des']],
                                  a=[output_catalogue_this['a_des'] / 3600],
                                  b=[output_catalogue_this['b_des'] / 3600],
                                  theta=[output_catalogue_this['theta_des']],
                                  show_centre=True,
                                  colour='blue',
                                  label='DES ellipse',
                                  world_axes=True)

                plt.title(f"{output_catalogue_this['id']}, DES {f_0}-band image")
                plt.legend()
                plt.savefig(output_path + f + '_des_' + output_catalogue_this['id'])
                if show:
                    plt.show()

            output_catalogue[o] = output_catalogue_this
            # params.add_params(output_path + o + "_" + f + '_object_properties', params=output_catalogue_this)

        ind_depth = np.nanargmax(mag_auto_true)
        depth = mag_auto_true[ind_depth]

        print()
        print(f'{f} Depth:', depth)
        print()

        output_catalogue_csv = pandas.DataFrame(output_catalogue).transpose()
        output_catalogue_csv.to_csv(output_path + f + '_catalogue.csv')
        params.add_params(output_path + f + '_params.yaml', params=output_params)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Print out and write photometric properties of host galaxy and other listed '
                    'field objects.')
    parser.add_argument('-e',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        type=str)
    parser.add_argument('-no_show',
                        help='Don\'t show plots onscreen.',
                        action='store_false')
    parser.add_argument('--frame',
                        help='Padding from object coordinates to plot.',
                        type=int,
                        default=20)
    parser.add_argument('--instrument',
                        help='Name of instrument.',
                        type=str,
                        default='FORS2')
    parser.add_argument('--cat',
                        help='Name of Sextractor subdirectory.',
                        type=str,
                        default=None)
    parser.add_argument('--image',
                        help='Name of image.',
                        type=str,
                        default=None)
    parser.add_argument('--output',
                        help='Path to directory in which to save information and plots.',
                        type=str,
                        default=None)
    parser.add_argument('--cone',
                        help='Radius within which to extract all magnitudes. If not given, only the "other_objects" list is used.',
                        type=float,
                        default=None)

    args = parser.parse_args()
    main(obj=args.e,
         show=args.no_show,
         frame=args.frame,
         instrument=args.instrument,
         cat_name=args.cat,
         image_spec=args.image,
         cone=args.cone)

# TODO: Turn this script into a function.
