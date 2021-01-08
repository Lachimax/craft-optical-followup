# Code by Lachlan Marnoch, 2019

import craftutils.params as p
import craftutils.fits_files as ff
import sys
import shutil
import os
import astropy.io.fits as fits


# TODO: Refactor all script inputs to match argparse inputs, for readability.

def main(directory, psfex_file, image, prefix, output_file):

    print("\nExecuting Python script pipeline_fors2/9-psf.py, with:")
    print(f"\tmain data directory {directory}")
    print(f"\tPSFEx file {psfex_file}")
    print(f"\timage file {image}")
    print(f"\toutput file {output_file}")
    print(f"\tfilter prefix {prefix}")
    print()

    _, pix_scale = ff.get_pixel_scale(image)
    pix_scale = float(pix_scale)
    fwhm_pix = float(ff.get_header_attribute(file=psfex_file, attribute='PSF_FWHM', ext=1))
    fwhm_deg = fwhm_pix * pix_scale
    fwhm_arcsec = fwhm_deg * 3600.

    param_dict = {f'{prefix}_pixel_scale': pix_scale,
                  f'{prefix}_fwhm_pix': fwhm_pix,
                  f'{prefix}_fwhm_deg': fwhm_deg,
                  f'{prefix}_fwhm_arcsec': fwhm_arcsec,
                  }

    p.add_params(file=directory + '/' + output_file, params=param_dict)
    p.add_params(file=directory + '/' + 'output_paths', params={f'{prefix}_psf_model': psfex_file})
    p.yaml_to_json(directory + '/' + output_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Extract and write some image properties to file.")
    parser.add_argument('--directory', help='Main data directory (probably starts with "MJD")')
    parser.add_argument('--psfex_file',
                        help='Path of .psf fits file.')
    parser.add_argument('--image_file',
                        help='Path of image fits file.')
    parser.add_argument('--prefix',
                        help='Filter prefix to attach to variable names.',
                        default='')
    parser.add_argument('--output_file',
                        help='Name of file to write to.',
                        default='output_values')

    args = parser.parse_args()
    main(directory=args.directory, psfex_file=args.psfex_file, image=args.image_file, prefix=args.prefix,
         output_file=args.output_file)
