# Code by Lachlan Marnoch, 2019

import sys
import os
from shutil import copyfile

import PyCRAFT.utils as u
import PyCRAFT.fits_files as f
import PyCRAFT.params as p


def main(data_title, sextractor_path, origin, destination):
    properties = p.object_params_fors2(data_title)
    outputs = p.object_output_params(obj=data_title, instrument='FORS2')

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
    u.mkdir_check(destination_path + "science/")
    u.mkdir_check(destination_path + "backgrounds/")

    filters = outputs['filters']

    for fil in filters:
        u.mkdir_check(destination_path + "science/" + fil)
        u.mkdir_check(destination_path + "backgrounds/" + fil)
        if do_sextractor:
            if not os.path.isdir(sextractor_path + fil):
                os.mkdir(sextractor_path + fil)
        files = os.listdir(origin_path + "science/" + fil + "/")
        for file_name in files:
            if file_name[-5:] == '.fits':
                science_origin = origin_path + "science/" + fil + "/" + file_name
                science_destination = destination_path + "science/" + fil + "/" + file_name.replace("trim", "norm")

                background_origin = origin_path + "backgrounds/" + fil + "/" + file_name.replace("SCIENCE_REDUCED",
                                                                                                 "PHOT_BACKGROUND_SCI")
                background_destination = destination_path + "backgrounds/" + fil + "/" + \
                                         file_name.replace("SCIENCE_REDUCED", "PHOT_BACKGROUND_SCI").replace("trim",
                                                                                                             "norm")

                print(science_origin)
                # Divide by exposure time to get an image in counts/second.
                f.divide_by_exp_time(file=science_origin, output=science_destination)
                f.divide_by_exp_time(file=background_origin, output=background_destination)
                if do_sextractor:
                    copyfile(science_destination, sextractor_path + fil + "/" + file_name.replace("trim", "norm"))

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
                        default="3-trimmed_with_python/")
    parser.add_argument('--destination',
                        help='Path to the destination folder.',
                        type=str,
                        default="4-divided_by_exp_time/")

    # Load arguments

    args = parser.parse_args()

    main(data_title=args.op, sextractor_path=args.sextractor_directory, origin=args.origin,
         destination=args.destination)
