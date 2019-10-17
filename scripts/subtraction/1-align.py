# Code by Lachlan Marnoch, 2019
from PyCRAFT import fits_files as ff
from PyCRAFT import params as p
from PyCRAFT import utils as u
from PyCRAFT import photometry as ph
from PyCRAFT import astrometry as ast

import shutil
import os
import math
from astropy.io import fits


# TODO: Document!!!!!!!!!


def main(destination, show, field, epoch, skip_reproject, offsets_yaml, copy_other_vals, f, manual,
         instrument_template):
    print('manual:', manual)
    properties = p.object_params_frb(field)
    alignment_ra = properties['alignment_ra']
    alignment_dec = properties['alignment_dec']
    skip_reproject = properties['skip_reproject'] or skip_reproject

    specific_star = False
    if alignment_dec != 0.0 and alignment_ra != 0.0 and offsets_yaml is not None:
        specific_star = True

    if destination[-1] != '/':
        destination = destination + '/'
    comparison = f'{field}_{epoch}_comparison.fits'

    if skip_reproject:
        comparison_tweaked = comparison.replace('comparison.fits', 'comparison_aligned.fits')
    else:
        comparison_tweaked = comparison.replace('comparison.fits', 'comparison_tweaked.fits')

    if offsets_yaml is not None:

        offsets = p.load_params(offsets_yaml)
        offset_ra = offsets['offset_ra']
        offset_dec = offsets['offset_dec']
        values = {'offset_ra': offset_ra, 'offset_dec': offset_dec}
        ast.offset_astrometry(hdu=destination + comparison, offset_ra=offset_ra, offset_dec=offset_dec,
                              output=destination + comparison_tweaked)

    else:
        if not manual:
            print('Doing specific star offset...')
            values = ast.tweak(sextractor_path=destination + '/sextractor/alignment/comparison.cat',
                               destination=destination + comparison_tweaked,
                               image_path=destination + comparison,
                               cat_path=destination + '/sextractor/alignment/template.cat',
                               cat_name='SExtractor', tolerance=10., show=show, stars_only=True,
                               psf=True, specific_star=specific_star, star_dec=alignment_dec,
                               star_ra=alignment_ra)
        else:
            print('Doing manual offset...')
            instrument_template = instrument_template.lower()
            manual_x = properties['offset_' + instrument_template + '_' + f[0] + '_x']
            manual_y = properties['offset_' + instrument_template + '_' + f[0] + '_y']
            values = ast.tweak(sextractor_path=destination + '/sextractor/alignment/comparison.cat',
                               destination=destination + comparison_tweaked,
                               image_path=destination + comparison,
                               cat_path=destination + '/sextractor/alignment/template.cat',
                               cat_name='SExtractor', tolerance=10., show=show, stars_only=True,
                               psf=True, specific_star=specific_star, star_dec=alignment_dec,
                               star_ra=alignment_ra,
                               manual=True, offset_x=manual_x, offset_y=manual_y)

    if instrument_template == 'FORS2':
        force = 2
    else:
        force = None

    if not skip_reproject:
        template = filter(lambda f: "template.fits" in f, os.listdir(destination)).__next__()
        n = ff.reproject(image_1=destination + comparison_tweaked,
                         image_2=destination + template,
                         image_1_output=destination + comparison.replace('comparison.fits', 'comparison_aligned.fits'),
                         image_2_output=destination + template.replace('template.fits', 'template_aligned.fits'),
                         force=force)

        if n == 1:
            values['reprojected'] = 'comparison'
        else:
            values['reprojected'] = 'template'

    p.add_params(destination + 'output_values.yaml', values)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--field',
                        help='Name of object parameter file without .yaml, eg FRB190102.',
                        type=str)
    parser.add_argument('--destination',
                        help='Folder to copy to and operate on.',
                        type=str)
    parser.add_argument('--epoch',
                        help='Number of comparison epoch to subtract template from.',
                        type=int,
                        default=1)
    parser.add_argument('--instrument',
                        help='Instrument of comparison epoch.',
                        type=str,
                        default='FORS2')
    parser.add_argument('--instrument_template',
                        help='Instrument of comparison epoch.',
                        type=str,
                        default='FORS2')
    parser.add_argument('--filter',
                        help='Filter.',
                        type=str,
                        default=None)
    parser.add_argument('--type',
                        help='Specify synthetic insertion file to use. If unspecified, uses the unmodified file.',
                        type=str,
                        default=None)
    parser.add_argument('-manual',
                        help='Do alignment manually?',
                        action='store_true')
    parser.add_argument('-show',
                        help='Show plots onscreen?',
                        action='store_true')
    parser.add_argument('-skip_reproject',
                        help='Skip reproject?',
                        action='store_true')
    parser.add_argument('--offsets_yaml',
                        help='Path to yaml file containing RA and DEC offsets to apply.',
                        type=str,
                        default=None)
    parser.add_argument('-copy_other_vals',
                        help='Skip reproject?',
                        action='store_true')

    args = parser.parse_args()
    print('manual:', args.manual)
    main(destination=args.destination, show=args.show, field=args.field, epoch=args.epoch,
         skip_reproject=args.skip_reproject, offsets_yaml=args.offsets_yaml, copy_other_vals=args.copy_other_vals,
         manual=args.manual, instrument_template=args.instrument_template, f=args.filter)
