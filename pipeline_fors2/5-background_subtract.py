# Code by Lachlan Marnoch, 2019 - 2020

import os
from shutil import copyfile

import craftutils.utils as u
import craftutils.fits_files as f
import craftutils.params as p
from craftutils.photometry import fit_background_fits

from astropy.io import fits


def main(data_dir, data_title, origin, destination, local):
    print("\nExecuting Python script pipeline_fors2/5-background_subtract.py, with:")
    print(f"\tepoch {data_title}")
    print(f"\torigin directory {origin}")
    print(f"\tdestination directory {destination}")
    print()

    methods = ["eso_backs", "polynomial", "gaussian"]

    method = u.select_option(message="Please select the background subtraction method.", options=methods)
    degree = None
    if method == "polynomial":
        degree = u.user_input(message=f"Please enter the degree of {method} to use:", typ=int)
    local = u.select_yn(message="Use a local fit?")

    outputs = p.object_output_params(data_title, instrument='FORS2')

    destination = data_dir + "/" + destination + "/"
    u.mkdir_check_nested(destination)

    science_origin = data_dir + "/" + origin + "/science/"
    print(science_origin)

    filters = outputs['filters']

    if method == "eso_backs":
        background_origin = data_dir + "/" + origin + "/backgrounds/"
    else:
        background_origin = data_dir + "/" + origin + f"/backgrounds_{method}_degree_{degree}/"

    for fil in filters:
        fil_dir = background_origin + fil + "/"
        u.mkdir_check_nested(fil_dir)
        u.mkdir_check(destination + fil)
        files = os.listdir(science_origin + fil + "/")
        for file_name in files:
            if file_name[-5:] == '.fits':
                new_file = file_name.replace("norm", "bg_sub")
                new_path = destination + fil + "/" + new_file
                science = science_origin + fil + "/" + file_name
                if method == 'eso_backs':
                    background = background_origin + fil + "/" + file_name.replace("SCIENCE_REDUCED",
                                                                                   "PHOT_BACKGROUND_SCI")
                else:
                    background = fit_background_fits(image=science, model_type=method, deg=degree, local=local)

                    background_path = background_origin + fil + "/" + file_name.replace("SCIENCE_REDUCED",
                                                                                        "PHOT_BACKGROUND_SCI")
                    background.writeto(background_path, overwrite=True)

                f.subtract_file(file=science, sub_file=background, output=new_path)

    copyfile(data_dir + "/" + origin + "/" + data_title + ".log", destination + data_title + ".log")
    u.write_log(path=destination + data_title + ".log",
                action=f'Backgrounds subtracted using 4-background_subtract.py with method {method}\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Subtract backgrounds provided by ESO Reflex from science images, and also divide all values by "
                    "exposure time.")
    parser.add_argument('--directory', help='Main data directory(probably starts with "MJD"')
    parser.add_argument('--op', help='Name of object parameter file without .yaml, eg FRB180924_1')
    parser.add_argument('--origin',
                        help='Folder within data_dir to copy target files from.',
                        type=str,
                        default="4-divided_by_exp_time")
    parser.add_argument('--destination',
                        help='Folder within data_dir to copy processed files to.',
                        type=str,
                        default="5-background_subtracted_with_python")

    # Load arguments

    args = parser.parse_args()

    main(data_dir=args.directory, data_title=args.op, origin=args.origin,
         destination=args.destination)
