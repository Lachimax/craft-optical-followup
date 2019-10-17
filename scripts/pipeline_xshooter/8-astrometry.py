# Code by Lachlan Marnoch, 2019

from PyCRAFT.astrometry import tweak_final
from PyCRAFT import params as p

import shutil
import os


# TODO: Refactor all script inputs to match argparse inputs, for readability.

def main(op, origin, destination, show, tolerance):

    properties = p.object_params_instrument(op, 'XSHOOTER')

    origin = properties['data_dir'] + origin
    destination = properties['data_dir'] + destination

    for f in filter(lambda f: f[-13:] == '_coadded.fits', os.listdir(origin)):
        shutil.copyfile(origin + f, destination + f)

    tweak_final(sextractor_path=origin, destination=destination, epoch=op, instrument='XSHOOTER', show=show, tolerance=tolerance, stars_only=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Correct the astrometry of the image by bootstrapping from a catalogue.")
    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        type=str)
    parser.add_argument('--origin',
                        help='Path to the origin folder.',
                        type=str,
                        default="7-sextractor/")
    parser.add_argument('--destination',
                        help='Path to the destination folder.',
                        type=str,
                        default="8-astrometry/")
    parser.add_argument('-show', action='store_true')
    parser.add_argument('--tolerance',
                        help='Pixel tolerance for matching between catalogue and sextraction.',
                        type=float,
                        default=10.0)

    args = parser.parse_args()

    main(op=args.op,
         origin=args.origin,
         destination=args.destination,
         show=args.show,
         tolerance=args.tolerance)
