# Code by Lachlan Marnoch, 2019 - 2020

import os
from shutil import copyfile
import craftutils.utils as u
import craftutils.fits_files as f
import craftutils.params as p


def main(data_dir, data_title, origin, destination):
    print("\nExecuting Python script pipeline_fors2/5-background_subtract.py, with:")
    print(f"\tepoch {data_title}")
    print(f"\torigin directory {origin}")
    print(f"\tdestination directory {destination}")
    print()

    outputs = p.object_output_params(data_title, instrument='FORS2')

    destination = data_dir + "/" + destination + "/"
    if not os.path.isdir(destination):
        os.mkdir(destination)

    science_origin = data_dir + "/" + origin + "/science/"
    background_origin = data_dir + "/" + origin + "/backgrounds/"
    print(science_origin)
    filters = outputs['filters']

    for fil in filters:
        if not os.path.isdir(destination + fil):
            os.mkdir(destination + fil)
        files = os.listdir(science_origin + fil + "/")
        for file_name in files:
            if file_name[-5:] == '.fits':
                new_file = file_name.replace("norm", "bg_sub")
                new_path = destination + fil + "/" + new_file
                science = science_origin + fil + "/" + file_name
                background = background_origin + fil + "/" + file_name.replace("SCIENCE_REDUCED", "PHOT_BACKGROUND_SCI")
                # Divide by exposure time to get an image in counts/second.
                f.subtract_file(file=science, sub_file=background, output=new_path)

    copyfile(data_dir + "/" + origin + "/" + data_title + ".log", destination + data_title + ".log")
    u.write_log(path=destination + data_title + ".log",
                action='Backgrounds subtracted using 4-background_subtract.py\n')


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