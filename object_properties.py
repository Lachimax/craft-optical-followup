# Code by Lachlan Marnoch, 2019-2020
from craftutils import params
from craftutils import utils as u, photometry as ph, plotting as p, fits_files as ff
from craftutils.astrometry import calculate_error_ellipse
from craftutils.retrieve import update_frb_des_cutout

from astropy import wcs
from astropy.io import fits
from astropy import time
from astropy.visualization import (ImageNormalize, SquaredStretch, SqrtStretch, ZScaleInterval, MinMaxInterval)

import pandas

import matplotlib.pyplot as plt
import numpy as np
from os.path import isfile


# TODO: Integrate into pipeline

def main(obj,
         plot,
         frame,
         instrument,
         cat_name,
         image_spec):
    print(obj)

    frb = obj[:-2]

    epoch_properties = params.object_params_instrument(obj=obj, instrument=instrument)
    burst_properties = params.object_params_frb(obj=frb)
    print()
    burst_outputs = params.frb_output_params(obj=frb)
    paths = params.object_output_paths(obj=obj, instrument=instrument)

    filters = epoch_properties['filters']

    for f in filters:
        if burst_outputs is None or f"{f}_ext_gal" not in burst_outputs:
            print(f"\nGalactic extinction missing for {f}; calculating now.")
            import extinction_galactic
            extinction_galactic.main(obj=frb)
            burst_outputs = params.frb_output_params(obj=frb)

    galaxies = burst_properties['other_objects']
    if galaxies is None:
        galaxies = {}
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

    output_path = f"{epoch_properties['data_dir']}/analysis/object_properties/"
    u.mkdir_check(output_path)
    output_path = f"{epoch_properties['data_dir']}/analysis/object_properties/{str(now)}_{cat_name}_{image_spec}/"
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

        # Analysis
        cat = np.genfromtxt(cat_path, names=params.sextractor_names_psf())
        mag_auto_true, mag_auto_err_plus, mag_auto_err_minus = ph.magnitude_complete(flux=cat['flux_auto'],
                                                                                     flux_err=cat['fluxerr_auto'],
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
        mag_psf, mag_psf_err_plus, mag_psf_err_minus = ph.magnitude_complete(flux=cat['flux_psf'],
                                                                             flux_err=cat['fluxerr_psf'],
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

        # Find index of other galaxy
        for o in galaxies:
            ra = galaxies[o]['ra']
            dec = galaxies[o]['dec']
            print('Matching...')
            index, dist = u.find_object(ra, dec, cat['ra'], cat['dec'])
            this = cat[index]
            print(f'{o} SExtractor {f}')
            print()

            mag_err = max(abs(mag_auto_err_plus[index]), abs(mag_auto_err_minus[index]))
            mag_psf_err = max(abs(mag_psf_err_plus[index]), abs(mag_psf_err_minus[index]))

            mag_ins, mag_ins_err_1, mag_ins_err_2 = ph.magnitude_error(flux=np.array([this['flux_auto']]),
                                                                       flux_err=np.array([this['fluxerr_auto']]),
                                                                       exp_time=exp_time, exp_time_err=exp_time_err)
            mag_ins = mag_ins[0]
            mag_ins_err = max(abs(mag_ins_err_1[0]), abs(mag_ins_err_2[0]))

            output_catalogue_this = {'id': o,
                                     'ra': float(this['ra']), 'dec': float(this['dec']),
                                     'ra_given': float(ra), 'dec_given': float(dec),
                                     'matching_distance_sex': float(dist * 3600),
                                     'kron_radius': float(this['kron_radius']),
                                     'a': float(this['a'] * 3600), 'a_err': float(this['a_err'] * 3600),
                                     'b': float(this['b'] * 3600), 'b_err': float(this['b_err'] * 3600),
                                     'theta': float(this['theta']), 'theta_err': float(this['theta_err']),
                                     'mag_auto': float(mag_auto_true[index]),
                                     'mag_auto_err': float(mag_err), 'mag_ins': float(mag_ins),
                                     'mag_auto_gal_correct': float(mag_auto_true[index]) - ext_gal,
                                     'ext_gal': float(ext_gal),
                                     'mag_ins_err': float(mag_ins_err), 'flux': float(this['flux_auto']),
                                     'flux_err': float(this['fluxerr_auto']), 'mag_psf': float(mag_psf[index]),
                                     'mag_psf_err': float(mag_psf_err),
                                     'flux_psf': float(this['flux_psf']), 'fluxerr_psf': float(this['fluxerr_psf']),
                                     'x_err': float(this['x_deg_err']), 'y_err': float(this['y_deg_err'])}

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
            print(f'{o} {f} mag auto corrected for Galactic extinction:', output_catalogue_this['mag_auto_gal_correct'])
            print(f'Galactic extinction used:', ext_gal)
            print(f'{o} {f} mag psf:', output_catalogue_this['mag_psf'], '+/-',
                  output_catalogue_this['mag_psf_err'])
            print()

            if plot:
                print('Loading FITS file...')
                image = fits.open(image_path)
                print('Plotting...')
                wcs_main = wcs.WCS(header=image[0].header)
                # p.plot_all_params(image=image, cat=cat)

                mid_x, mid_y = wcs_main.all_world2pix(output_catalogue_this['ra'], output_catalogue_this['dec'], 0)
                mid_x = int(mid_x)
                mid_y = int(mid_y)

                left = mid_x - frame
                right = mid_x + frame
                bottom = mid_y - frame
                top = mid_y + frame

                image_cut = ff.trim(hdu=image, left=left, right=right, bottom=bottom, top=top)

                plt.imshow(image_cut[0].data, origin='lower',
                           norm=ImageNormalize(image_cut[0].data, stretch=SqrtStretch(), interval=ZScaleInterval()))
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
                                  ras=[this['ra']],
                                  decs=[this['dec']],
                                  a=[this['kron_radius'] * this['a']],
                                  b=[this['kron_radius'] * this['b']],
                                  theta=[output_catalogue_this['theta']],
                                  colour='violet',
                                  label='Kron aperture')
                p.plot_gal_params(hdu=image_cut,
                                  ras=[output_catalogue_this['ra']],
                                  decs=[output_catalogue_this['dec']],
                                  a=[this['a']],
                                  b=[this['b']],
                                  theta=[output_catalogue_this['theta']],
                                  show_centre=True,
                                  colour='blue',
                                  label=f'SExtractor ellipse')
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
                plt.show()

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


                if plot:
                    print('Loading FITS file...')
                    des_image = fits.open(des_path)
                    print('Plotting...')
                    wcs_des = wcs.WCS(header=des_image[0].header)
                    mid_x, mid_y = wcs_des.all_world2pix(output_catalogue_this['ra_des'],
                                                         output_catalogue_this['dec_des'], 0)
                    mid_x = int(mid_x)
                    mid_y = int(mid_y)

                    left = mid_x - frame
                    right = mid_x + frame
                    bottom = mid_y - frame
                    top = mid_y + frame

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
        description='Print out photometric properties of host galaxy and other listed '
                    'field objects.')
    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        type=str)
    parser.add_argument('-no_show',
                        help='Don\'t show plots onscreen.',
                        action='store_true')
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

    args = parser.parse_args()
    main(obj=args.op,
         plot=not args.no_show,
         frame=args.frame,
         instrument=args.instrument,
         cat_name=args.cat,
         image_spec=args.image)

# TODO: Turn this script into a function.
