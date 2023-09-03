# Code by Lachlan Marnoch, 2019
from astropy.io import fits
import shutil
import astropy.time as time
from astropy import table
from astropy import wcs
import numpy as np
import sncosmo

from craftutils import params as p
from craftutils import utils as u
from craftutils import photometry as ph
from craftutils import sne


def main(obj, test, n, filter_dist, instrument, limit):
    properties = p.object_params_instrument(obj, instrument=instrument)
    burst_properties = p.object_params_frb(obj=obj[:-2])
    output = p.object_output_params(obj=obj, instrument=instrument)
    paths = p.object_output_paths(obj=obj, instrument=instrument)

    z = burst_properties['z']
    mjd_burst = burst_properties['mjd_burst']
    ebv_mw = burst_properties['dust_ebv']

    mjd_obs = properties['mjd']
    synth_path = properties['data_dir'] + 'synthetic/'
    u.mkdir_check(synth_path)
    synth_path = synth_path + 'sn_random/'
    u.mkdir_check(synth_path)

    epoch = mjd_obs - mjd_burst

    f_0 = filter_dist[0]

    hg_ra = burst_properties['hg_ra']
    hg_dec = burst_properties['hg_dec']

    burst_ra = burst_properties['burst_ra']
    burst_dec = burst_properties['burst_dec']

    image_path = paths[f_0 + '_' + properties['subtraction_image']]

    image = fits.open(image_path)

    wcs_info = wcs.WCS(image[0].header)

    burst_x, burst_y = wcs_info.all_world2pix(burst_ra, burst_dec, 0)


    now = time.Time.now()
    now.format = 'isot'
    synth_path += test + '_' + str(now) + '/'
    u.mkdir_check(synth_path)

    filters = output['filters']

    psf_models = []
    for i, f in enumerate(filters):
        sn.register_filter(f=f, instrument=instrument)
        psf_model = fits.open(paths[f[0] + '_psf_model'])
        psf_models.append(psf_model)

    for i in range(n):
        test_spec = test + '_' + str(i)
        test_path = synth_path + test_spec + '/'

        magnitudes = []
        mags_filters, model, x, y, tbl = sn.random_light_curves_type_ia(filters=filters,
                                                                        image=image,
                                                                        hg_ra=hg_ra,
                                                                        hg_dec=hg_dec,
                                                                        z=z,
                                                                        ebv_mw=ebv_mw,
                                                                        output_path=test_path,
                                                                        output_title=test,
                                                                        limit=limit,
                                                                        x=burst_x,
                                                                        y=burst_y,
                                                                        ra=burst_ra,
                                                                        dec=burst_dec)
        days = mags_filters['days']

        for f in filters:
            magnitude = sn.magnitude_at_epoch(epoch=epoch, days=days, mags=mags_filters[f])
            print(f, 'mag:', magnitude)
            magnitudes.append(magnitude)

        ph.insert_synthetic(x=float(x), y=float(y), obj=properties, test_path=test_path, filters=filters,
                            magnitudes=magnitudes, suffix='sn_random_random_ia', extra_values=tbl, paths=paths,
                            output_properties=output, psf_models=psf_models, instrument=instrument)

    p.add_output_path(obj=obj, key='subtraction_image_synth_sn_random_ia', path=synth_path, instrument=instrument)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        type=str)
    parser.add_argument('--test',
                        help='Name of test.',
                        type=str,
                        default='random_sn_ia')
    parser.add_argument('--n',
                        help='Number of sn to generate.',
                        type=int,
                        default=100)
    parser.add_argument('--filter_dist',
                        help='The filter to use for position determination.',
                        type=str,
                        default='g_HIGH')
    parser.add_argument('--force_r',
                        help='Radial separation in pixels. Set to force a certain separation. If not given, will be generated.',
                        type=float,
                        default=None)
    parser.add_argument('--instrument',
                        help='Name of instrument.',
                        default='FORS2',
                        type=str)
    parser.add_argument('--limit',
                        help='Size of cutout for position distribution',
                        type=int)

    args = parser.parse_args()

    main(obj=args.op, test=args.test, n=args.n, filter_dist=args.filter_dist, instrument=args.instrument,
         limit=args.limit)
