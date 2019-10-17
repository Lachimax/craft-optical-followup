# Code by Lachlan Marnoch, 2019

import PyCRAFT.fits_files as f
import PyCRAFT.utils as u
import PyCRAFT.params as p
import os
from shutil import copyfile
import sys


def main(data_title, origin, destination):
    properties = p.object_params_xshooter(data_title)
    path = properties['data_dir']

    destination = path + destination
    u.mkdir_check(destination)
    origin = path + origin
    u.mkdir_check(origin)

    dirs = next(os.walk(origin))[1]

    left = 27
    right = 537
    top = 526
    bottom = 15

    for fil in dirs:
        u.mkdir_check(destination + fil)
        print('HERE:')
        print(origin + fil)
        files = os.listdir(origin + fil)
        files.sort()

        for i, file in enumerate(files):
            # Split the files into upper CCD and lower CCD, with even-numbered being upper and odd-numbered being lower
            new_path = destination + fil + "/" + file.replace(".fits", "_trim.fits")
            # Add GAIN and SATURATE keywords to headers.
            path = origin + fil + "/" + file
            f.trim_file(path, left=left, right=right, top=top, bottom=bottom,
                        new_path=new_path)

    copyfile(origin + data_title + ".log", destination + data_title + ".log")
    u.write_log(path=destination + data_title + ".log", action='Edges trimmed using 4-trim.py\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Trim noise frame from individual files.")
    parser.add_argument('--op', help='Name of object parameter file without .yaml, eg FRB180924_1')
    parser.add_argument('--origin',
                        help='Path to the destination folder.',
                        type=str,
                        default="3-defringed/")
    parser.add_argument('--destination',
                        help='Path to the destination folder.',
                        type=str,
                        default="4-trimmed/")
    args = parser.parse_args()

    main(data_title=args.op, origin=args.origin, destination=args.destination)
