# Code by Lachlan Marnoch, 2019

import PyCRAFT.fits_files as f
import PyCRAFT.utils as u
import PyCRAFT.params as p
import os
from shutil import copyfile
import sys


def main(origin_dir, output_dir, data_title, sextractor_path):
    # If this is None, we don't want the SExtractor components to be performed.
    if sextractor_path is not None:
        if not os.path.isdir(sextractor_path):
            os.mkdir(sextractor_path)
        do_sextractor = True
        print(os.getcwd())
        ap_diams_sex = p.load_params(f'param/aperture_diameters_fors2')
    else:
        do_sextractor = False

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(output_dir + "backgrounds/"):
        os.mkdir(output_dir + "backgrounds/")
    if not os.path.isdir(output_dir + "science/"):
        os.mkdir(output_dir + "science/")

    wdir = origin_dir + "backgrounds/"

    dirs = next(os.walk(wdir))[1]

    edged = False

    up_left = 0
    up_right = 0
    up_bottom = 0
    up_top = 0

    dn_left = 0
    dn_right = 0
    dn_bottom = 0
    dn_top = 0

    for fil in dirs:
        print(output_dir + "backgrounds/" + fil)
        if not os.path.isdir(output_dir + "backgrounds/" + fil):
            os.mkdir(output_dir + "backgrounds/" + fil)
        print('HERE:')
        print(wdir + fil)
        files = os.listdir(wdir + fil)
        files.sort()
        if not edged:
            # Find borders of noise frame using backgrounds.
            up_left, up_right, up_bottom, up_top = f.detect_edges(wdir + fil + "/" + files[0])
            dn_left, dn_right, dn_bottom, dn_top = f.detect_edges(wdir + fil + "/" + files[1])
            up_left = up_left + 5
            up_right = up_right - 5
            up_top = up_top - 5
            dn_left = dn_left + 5
            dn_right = dn_right - 5
            dn_bottom = dn_bottom + 5

            edged = True

        for i, file in enumerate(files):
            # Split the files into upper CCD and lower CCD, with even-numbered being upper and odd-numbered being lower
            new_path = output_dir + "backgrounds/" + fil + "/" + file.replace(".fits", "_trim.fits")
            # Add GAIN and SATURATE keywords to headers.
            path = wdir + fil + "/" + file
            if i % 2 == 0:
                f.trim_file(path, left=up_left, right=up_right, top=up_top, bottom=up_bottom,
                            new_path=new_path)
            else:
                f.trim_file(path, left=dn_left, right=dn_right, top=dn_top, bottom=dn_bottom,
                            new_path=new_path)

    # Repeat for science images

    wdir = origin_dir + "science/"

    dirs = os.listdir(wdir)

    for fil in dirs:
        print(output_dir + "science/" + fil)
        if do_sextractor:
            if not os.path.isdir(sextractor_path + fil):
                os.mkdir(sextractor_path + fil)
        if not os.path.isdir(output_dir + "science/" + fil):
            os.mkdir(output_dir + "science/" + fil)

        files = os.listdir(wdir + fil)
        files.sort()

        for i, file in enumerate(files):
            # Split the files into upper CCD and lower CCD, with even-numbered being upper and odd-numbered being lower
            new_file = file.replace(".fits", "_trim.fits")
            new_path = output_dir + "science/" + fil + "/" + new_file
            path = wdir + fil + "/" + file
            f.change_header(file=path, name='GAIN', entry=0.8)
            f.change_header(file=path, name='SATURATE', entry=65535.)
            if i % 2 == 0:
                f.trim_file(path, left=up_left, right=up_right, top=up_top, bottom=up_bottom,
                            new_path=new_path)
                if do_sextractor:
                    copyfile(new_path, sextractor_path + fil + "/" + new_file)
            else:
                f.trim_file(path, left=dn_left, right=dn_right, top=dn_top, bottom=dn_bottom,
                            new_path=new_path)

        exp_time, _ = f.mean_exp_time(path=output_dir + "science/" + fil)
        gain = 0.8

    copyfile(origin_dir + data_title + ".log", output_dir + data_title + ".log")
    u.write_log(path=output_dir + data_title + ".log", action='Edges trimmed using 3-trim.py\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Trim noise frame from individual files using background files.")
    parser.add_argument('--origin',
                        help='Directory containing background and science subdirectories, which in turn contain the '
                             'necessary files.')
    parser.add_argument('--destination', help='Output directory.')
    parser.add_argument('-op', help='Name of object parameter file without .yaml, eg FRB180924_1')
    parser.add_argument('--sextractor_directory', default=None,
                        help='Directory for sextractor scripts to be moved to. If you don\'t want to run sextractor, '
                             'leave this parameter empty.')

    args = parser.parse_args()

    main(origin_dir=args.origin, output_dir=args.destination, data_title=args.op,
         sextractor_path=args.sextractor_directory)
