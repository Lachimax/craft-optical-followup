# Code by Lachlan Marnoch, 2019

import sys
import os
from shutil import copyfile
import PyCRAFT.utils as u
import PyCRAFT.fits_files as f
import PyCRAFT.params as p


# TODO: Split normalise into its own step.


def main(data_dir, data_title, sextractor_path, origin, destination):
    if sextractor_path is not None:
        if not os.path.isdir(sextractor_path):
            os.mkdir(sextractor_path)
        do_sextractor = True
        ap_diams_sex = p.load_params(f'param/aperture_diameters_fors2')
    else:
        do_sextractor = False
        
    outputs = p.object_output_params(data_title, instrument='FORS2')

    destination = data_dir + "/" + destination + "/"
    if not os.path.isdir(destination):
        os.mkdir(destination)

    science_origin = data_dir + "/" + origin + "/science/"
    background_origin = data_dir + "/" + origin + "/backgrounds/"
    print(science_origin)
    filters = outputs['filters']

    for fil in filters:

        if do_sextractor:
            if not os.path.isdir(sextractor_path + fil):
                os.mkdir(sextractor_path + fil)
        if not os.path.isdir(destination + fil):
            os.mkdir(destination + fil)
        files = os.listdir(science_origin + fil + "/")
        for file_name in files:
            if file_name[-5:] == '.fits':
                new_file = file_name.replace("norm", "bg_sub")
                new_path = destination + fil + "/" + new_file
                science = science_origin + fil + "/" + file_name
                background = background_origin + fil + "/" + file_name.replace("SCIENCE_REDUCED", "PHOT_BACKGROUND_SCI")
                print(science)
                # Divide by exposure time to get an image in counts/second.
                f.subtract_file(file=science, sub_file=background, output=new_path)
                if do_sextractor:
                    copyfile(new_path, sextractor_path + fil + "/" + new_file)

        # Write a sextractor file for photometric testing of the data from the upper chip.
        if do_sextractor:
            # Write a csv table of file properties to each filter directory.
            tbl = f.fits_table(input_path=sextractor_path + fil,
                               output_path=sextractor_path + fil + "/" + fil + "_fits_tbl.csv",
                               science_only=False)
            for i, d in enumerate(ap_diams_sex):
                f.write_sextractor_script(table=tbl,
                                          output_path=sextractor_path + fil + "/sextract_aperture_" + str(d) + ".sh",
                                          sex_params=['c', 'PHOT_APERTURES'],
                                          sex_param_values=['im.sex', str(d)], cat_name='sextracted_' + str(d),
                                          cats_dir='aperture_' + str(d), criterion='chip', value='CHIP1')

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
    parser.add_argument('--sextractor_directory', default=None,
                        help='Directory for sextractor scripts to be moved to. If you don\'t want to run sextractor, '
                             'leave this parameter empty.')
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

    main(data_dir=args.directory, data_title=args.op, sextractor_path=args.sextractor_directory, origin=args.origin,
         destination=args.destination)
