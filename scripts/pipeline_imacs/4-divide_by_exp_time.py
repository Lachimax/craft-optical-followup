# Code by Lachlan Marnoch, 2019
import sys
import os
from shutil import copyfile

import PyCRAFT.utils as u
import PyCRAFT.fits_files as f
import PyCRAFT.params as p


def main(data_title, sextractor_path, origin, destination):
    properties = p.object_params_imacs(data_title)
    data_dir = properties['data_dir']
    if sextractor_path is not None:
        if not os.path.isdir(sextractor_path):
            os.mkdir(sextractor_path)
        do_sextractor = True
        ap_diams_sex = p.load_params(f'param/aperture_diameters_fors2')
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

        # Write a sextractor file for photometric testing of the data from the upper chip.
        if do_sextractor:
            # Write a csv table of file properties to each filter directory.
            tbl = f.fits_table(input_path=sextractor_path + fil,
                               output_path=sextractor_path + fil + "/" + fil + "_fits_tbl.csv",
                               science_only=False)
            # TODO: Rewrite to use psf-fitting (in FORS2 pipeline as well)
            for i, d in enumerate(ap_diams_sex):
                f.write_sextractor_script(table=tbl,
                                          output_path=sextractor_path + fil + "/sextract_aperture_" + str(d) + ".sh",
                                          sex_params=['c', 'PHOT_APERTURES'],
                                          sex_param_values=['im.sex', str(d)], cat_name='sextracted_' + str(d),
                                          cats_dir='aperture_' + str(d), criterion='chip', value='CHIP1')

    if os.path.isfile(origin_path + data_title + '.log'):
        copyfile(origin_path + data_title + '.log', destination_path + data_title + ".log")
    u.write_log(path=destination_path + data_title + ".log", action=f'Astrometry solved using 3-astrometry.py')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Divide fits files by their exposure times.")
    parser.add_argument('--op', help='Name of object parameter file without .yaml, eg FRB180924_1')
    parser.add_argument('--sextractor_directory', default=None,
                        help='Directory for sextractor scripts to be moved to. If you don\'t want to run sextractor, '
                             'leave this parameter empty.')
    parser.add_argument('--origin',
                        help='Path to the destination folder.',
                        type=str,
                        default="3-astrometry/")
    parser.add_argument('--destination',
                        help='Path to the destination folder.',
                        type=str,
                        default="4-divided_by_exp_time/")

    # Load arguments

    args = parser.parse_args()

    main(data_title=args.op, sextractor_path=args.sextractor_directory, origin=args.origin,
         destination=args.destination)
