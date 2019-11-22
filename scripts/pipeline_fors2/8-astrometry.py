# Code by Lachlan Marnoch, 2019

import shutil
import os
import astropy.io.fits as fits

import PyCRAFT.fits_files as ff
import PyCRAFT.params as p
from PyCRAFT.astrometry import tweak_final


# TODO: Refactor all script inputs to match argparse inputs, for readability.

def main(obj, astrometry_path, sextractor_path, template, show):
    files = os.listdir(astrometry_path)
    template_file = template[0] + '_astrometry.fits'

    params = p.object_params_fors2(obj)

    for file in files:
        if 'coadded.fits' in file:
            astrometry_file = file.replace('_coadded', '_astrometry')
            coadd = fits.open(astrometry_path + file)
            header = coadd[0].header
            if os.path.isfile(astrometry_path + astrometry_file):
                astrometry = fits.open(astrometry_path + astrometry_file, mode='update')
                astrometry[0].header['AIRMASS'] = header['AIRMASS']
                astrometry[0].header['FILTER'] = header['FILTER']
                astrometry[0].header['OBJECT'] = header['OBJECT']
                gain = header['GAIN']
                astrometry[0].header['GAIN'] = gain
                astrometry[0].header['EXPTIME'] = header['EXPTIME']
                astrometry[0].header['RDNOISE'] = 0
                astrometry[0].header['MJD-OBS'] = header['MJD-OBS']
                ff.add_log(astrometry, 'Refined astrometric solution using Astrometry.net')
                coadd.close()
                astrometry.close()

                print()

                p.add_output_path(obj=obj, key=file[0] + '_astrometry_image',
                                  path=astrometry_path + astrometry_file)

                if file != template_file:
                    shutil.copy(astrometry_path + astrometry_file,
                                astrometry_path + astrometry_file.replace('.fits', '_orig.fits'))
                    shutil.copy(astrometry_path + template_file,
                                astrometry_path + template_file.replace('.fits', '_orig.fits'))

                    ff.align(comparison=astrometry_path + astrometry_file,
                             template=astrometry_path + template_file,
                             comparison_output=astrometry_path + astrometry_file,
                             template_output=astrometry_path + template_file)

            else:
                print(f'No astrometry file corresponding to {file} found.')

            # TODO: This will break if more than two filters have been used on this observation.
            #  Rewrite align to work with one template and multiple comparisons.

    if params['astrometry_tweak']:
        tweak_final(epoch=obj, sextractor_path=sextractor_path,
                    destination=astrometry_path, instrument='FORS2',
                    show=show, output_suffix='astrometry_tweaked', input_suffix='astrometry', stars_only=True,
                    specific_star=False,
                    path_add='tweaked_image', manual=False)

        tweak_final(epoch=obj, sextractor_path=sextractor_path,
                    destination=astrometry_path, instrument='FORS2',
                    show=show, output_suffix='astrometry_tweaked_hg', input_suffix='astrometry', stars_only=True,
                    specific_star=True,
                    path_add='tweaked_image_hg', manual=False)

    if params['manual_astrometry']:
        tweak_final(epoch=obj, sextractor_path=sextractor_path,
                    destination=astrometry_path, instrument='FORS2',
                    show=show, output_suffix='astrometry_tweaked_manual', input_suffix='astrometry', stars_only=True,
                    path_add='tweaked_image_manual', manual=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Copy header information from previous .fits file and produce "
                                                 "sextraction scripts.")
    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        type=str)
    parser.add_argument('--astrometry_path',
                        help='Directory containing the files to be processed.',
                        type=str)
    parser.add_argument('--sextractor_path',
                        help='Directory containing the sextractor files to be processed.',
                        type=str)
    parser.add_argument('--template',
                        type=str,
                        help='Filter to use as template for alignment.')
    parser.add_argument('-show',
                        action='store_true')
    # TODO: Make no_show consistent across all scripts
    args = parser.parse_args()
    main(obj=args.op, astrometry_path=args.astrometry_path, sextractor_path=args.sextractor_path,
         template=args.template, show=args.show)
