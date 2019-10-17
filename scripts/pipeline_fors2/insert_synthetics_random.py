# Code by Lachlan Marnoch, 2019
from astropy.io import fits
import shutil
import astropy.time as time
import numpy as np

from PyCRAFT import params as p
from PyCRAFT import utils as u
from PyCRAFT import photometry as ph


def main(obj, test, n, mag_lower, mag_upper, colour_upper, colour_lower):
    properties = p.object_params_fors2(obj)
    output = p.object_output_params(obj=obj, instrument='FORS2')
    paths = p.object_output_paths(obj)
    burst_properties = p.object_params_frb(obj[:-2])

    synth_path = properties['data_dir'] + 'synthetic/'

    u.mkdir_check(synth_path)
    synth_path = synth_path + 'random/'
    u.mkdir_check(synth_path)
    now = time.Time.now()
    now.format = 'isot'
    test = str(now) + '_' + test
    test_path = synth_path + test + '/'
    u.mkdir_check(test_path)

    filters = {}
    bluest = None
    bluest_lambda = np.inf
    for f in output['filters']:
        filter_properties = p.filter_params(f=f, instrument='FORS2')
        filters[f] = filter_properties
        lambda_eff = filter_properties['lambda_eff']
        if lambda_eff < bluest_lambda:
            bluest_lambda = lambda_eff
            bluest = f

    # Insert random sources in the bluest filter.

    f_0 = bluest[0]
    output_properties = p.object_output_params(obj)
    fwhm = output_properties[f_0 + '_fwhm_pix']
    zeropoint, _, airmass, _ = ph.select_zeropoint(obj, bluest, instrument='fors2')

    base_path = paths[f_0 + '_subtraction_image']

    output_path = test_path + f_0 + '_random_sources.fits'
    _, sources = ph.insert_random_point_sources_to_file(file=base_path, fwhm=fwhm,
                                                        output=output_path, n=n,
                                                        airmass=airmass, zeropoint=zeropoint)

    p.add_output_path(obj=obj, key=f_0 + '_subtraction_image_synth_random', path=output_path)

    # Now insert sources at the same positions in other filters, but with magnitudes randomised.
    for f in filters:
        if f != bluest:
            f_0 = f[0]
            output_properties = p.object_output_params(obj)
            fwhm = output_properties[f_0 + '_fwhm_pix']
            zeropoint, _, airmass, _ = ph.select_zeropoint(obj, f, instrument='fors2')

            base_path = paths[f_0 + '_subtraction_image']

            mag = np.random.uniform(mag_lower, mag_upper, size=n)

            output_path = test_path + f_0 + '_random_sources.fits'
            ph.insert_point_sources_to_file(file=base_path, fwhm=fwhm,
                                            output=output_path, x=sources['x_0'],
                                            y=sources['y_0'], mag=mag,
                                            airmass=airmass, zeropoint=zeropoint)

            p.add_output_path(obj=obj, key=f_0 + '_subtraction_image_synth_random', path=output_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        type=str)
    parser.add_argument('--test',
                        help='Name of test.',
                        type=str,
                        default='random')
    parser.add_argument('--n',
                        help='Number of sources to insert.',
                        type=int,
                        default=100
                        )
    parser.add_argument('--mag_lower',
                        help='Lower magnitude limit.',
                        type=float,
                        default=20.)
    parser.add_argument('--mag_upper',
                        type=float,
                        default=30.)
    parser.add_argument('--colour_upper',
                        type=float,
                        default=3.)
    parser.add_argument('--colour_lower',
                        type=float,
                        default=-3.)

    args = parser.parse_args()

    main(obj=args.op, test=args.test, n=args.n, mag_lower=args.mag_lower, mag_upper=args.mag_upper,
         colour_lower=args.colour_lower, colour_upper=args.colour_upper)
