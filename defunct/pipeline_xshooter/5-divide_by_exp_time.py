# Code by Lachlan Marnoch, 2019
import sys
import os
from shutil import copyfile

import craftutils.utils as u
import craftutils.fits_files as f
import craftutils.params as p


def main(data_title, sextractor_path, origin, destination):
    properties = p.object_params_xshooter(data_title)
    data_dir = properties['data_dir']
    if sextractor_path is not None:
        if not os.path.isdir(sextractor_path):
            os.mkdir(sextractor_path)
        do_sextractor = True
    else:
        do_sextractor = False

    origin_path = data_dir + origin
    destination_path = data_dir + destination
    u.mkdir_check(destination_path)
    filters = next(os.walk(origin_path))[1]

    for fil in filters:
        u.mkdir_check(destination_path + fil)
        if do_sextractor:
            if not os.path.isdir(sextractor_path + fil):
                os.mkdir(sextractor_path + fil)
        files = os.listdir(origin_path + fil + "/")
        for file_name in files:
            if file_name[-5:] == '.fits':
                science_origin = origin_path + fil + "/" + file_name
                science_destination = destination_path + fil + "/" + file_name
                print(science_origin)
                # Divide by exposure time to get an image in counts/second.
                f.divide_by_exp_time(file=science_origin, output=science_destination)
                if do_sextractor:
                    copyfile(science_origin, sextractor_path + fil + "/" + file_name)

    if os.path.isfile(origin_path + data_title + '.log'):
        copyfile(origin_path + data_title + '.log', destination_path + data_title + ".log")
    u.write_log(path=destination_path + data_title + ".log", action=f'Divided by exposure time.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Divide fits files by their exposure times.")
    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        type=str)
    parser.add_argument('--sextractor_directory', default=None,
                        help='Directory for sextractor scripts to be moved to. If you don\'t want to run sextractor, '
                             'leave this parameter empty.',
                        type=str)
    parser.add_argument('--origin',
                        help='Path to the destination folder.',
                        type=str,
                        default="4-trimmed/")
    parser.add_argument('--destination',
                        help='Path to the destination folder.',
                        type=str,
                        default="5-divided_by_exp_time/")

    # Load arguments

    args = parser.parse_args()

    main(data_title=args.op, sextractor_path=args.sextractor_directory, origin=args.origin,
         destination=args.destination)
