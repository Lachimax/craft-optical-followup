# Code by Lachlan Marnoch, 2019
import os
import shutil
import astropy.io.fits as fits

from PyCRAFT import params
from PyCRAFT import utils as u
from PyCRAFT import params as p


def main(data_title: str, origin: str, destination: str, redo: bool = False):
    properties = p.object_params_imacs(data_title)
    path = properties['data_dir']

    origin_path = path + origin
    astrometry_path = path + destination
    u.mkdir_check(astrometry_path)

    keys = params.load_params('param/keys')
    key = keys['astrometry']

    reduced_list = os.listdir(origin_path)
    astrometry_list = os.listdir(astrometry_path)

    if redo:
        to_send = list(filter(lambda f: f[-5:] == '.fits', reduced_list))
    else:
        to_send = list(filter(lambda f: f[-5:] == '.fits' and f not in astrometry_list, reduced_list))

    filters = list(filter(lambda f: os.path.isdir(f), os.listdir(origin)))

    for f in filters:
        reduced_path_filter = origin_path + f + '/'
        astrometry_path_filter = astrometry_path + f + '/'
        print(f'To send to Astrometry.net from {f}:')
        for file in to_send:
            print('\t' + file)

        for file in to_send:
            hdu = fits.open(origin_path + file)
            header = hdu[0].header
            ra = header['RA-D']
            dec = header['DEC-D']
            scale_upper = header['SCALE'] + 0.1
            scale_lower = header['SCALE'] - 0.1
            hdu.close()
            print('Sending to Astrometry.net:', file)
            os.system(
                f'python scripts/astrometry-client.py '
                f'--apikey {key} '
                f'-u {reduced_path_filter}{file} '
                f'-w '
                f'--newfits {astrometry_path_filter}{file} '
                f'--ra {ra} --dec {dec} --radius {1.} '
                f'--scale-upper {scale_upper} '
                f'--scale-lower {scale_lower} '
                f'--private --no_commercial')

    if os.path.isfile(origin_path + data_title + '.log'):
        shutil.copy(origin_path + data_title + '.log', astrometry_path + data_title + ".log")
    u.write_log(path=astrometry_path + data_title + ".log", action=f'Astrometry solved using 3-astrometry.py')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Reduce raw IMACS data.')

    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        type=str)
    parser.add_argument('-redo',
                        help='Redo all files; if left out, files that have already been done will not be uploaded to '
                             'astrometry.net',
                        action='store_true')
    parser.add_argument('--origin',
                        help='Path to the folder (within the main data directory) from which to draw files.',
                        type=str,
                        default='2-reduced/')
    parser.add_argument('--destination', help='Path to the folder (within the main data directory) for processed files.',
                        type=str,
                        default='3-astrometry/')

    args = parser.parse_args()

    main(data_title=args.op, origin=args.origin, destination=args.destination, redo=args.redo)
