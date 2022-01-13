# Code by Lachlan Marnoch, 2021

import sys
import os
from shutil import copyfile
from numpy import array

from astropy.io import fits

import craftutils.utils as u
import craftutils.fits_files as ff
import craftutils.params as p
import craftutils.photometry as ph


def main(epoch, origin, destination):
    print("\nExecuting Python script pipeline_fors2/4.1-insert_test_synth.py, with:")
    print(f"\tepoch {epoch}")
    print(f"\torigin directory {origin}")
    print(f"\tdestination directory {destination}")
    print()

    epoch_params = p.object_params_fors2(obj=epoch)
    outputs = p.object_output_params(obj=epoch, instrument='FORS2')

    data_dir = epoch_params['data_dir']

    insert = epoch_params['test_synths']

    origin_path = data_dir + "analysis/sextractor/" + origin
    destination_path = data_dir + destination

    u.mkdir_check(destination_path)
    u.mkdir_check(destination_path + "science/")
    u.mkdir_check(destination_path + "backgrounds/")

    filters = outputs['filters']

    for fil in filters:
        f = fil[0]
        path_fil_output = destination_path + "science/" + fil + "/"
        path_fil_input = origin_path + fil + "/"
        u.mkdir_check(path_fil_output)
        u.mkdir_check(destination_path + "backgrounds/" + fil)
        zeropoint, _, airmass, _, extinction, _ = ph.select_zeropoint(obj=epoch,
                                                                      filt=fil,
                                                                      instrument='fors2',
                                                                      outputs=outputs)

        print(path_fil_input)
        # print(os.listdir(path_fil_input))

        for fits_file in filter(lambda f: f.endswith("_norm.fits"), os.listdir(path_fil_input)):
            print(fits_file)
            path_fits_file_input = path_fil_input + fits_file
            path_fits_file_output = path_fil_output + fits_file
            path_psf_model = path_fits_file_input.replace(".fits", "_psfex.psf")

            try:
                ph.insert_point_sources_to_file(file=path_fits_file_input,
                                                x=array(insert["ra"]),
                                                y=array(insert["dec"]),
                                                mag=insert[f"{f}_mag"],
                                                output=path_fits_file_output,
                                                zeropoint=zeropoint,
                                                extinction=extinction,
                                                airmass=airmass,
                                                world_coordinates=True,
                                                psf_model=path_psf_model
                                                )
            except ValueError:
                ph.insert_point_sources_to_file(file=path_fits_file_input,
                                                x=array(insert["ra"]),
                                                y=array(insert["dec"]),
                                                mag=insert[f"{f}_mag"],
                                                output=path_fits_file_output,
                                                zeropoint=zeropoint,
                                                extinction=extinction,
                                                airmass=airmass,
                                                world_coordinates=True,
                                                fwhm=fits.open(path_psf_model)[1].header['PSF_FWHM']
                                                )

    if os.path.isfile(origin_path + epoch + '.log'):
        copyfile(origin_path + epoch + '.log', destination_path + epoch + ".log")
    u.write_log(path=destination_path + epoch + ".log", action=f'Divided by exposure time.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Divide fits files by their exposure times.")
    parser.add_argument('--op',
                        help='Name of epoch parameter file without .yaml, eg FRB180924_1',
                        type=str)
    parser.add_argument('--origin',
                        help='Path to the destination folder.',
                        type=str,
                        default="4-divided_by_exp_time/")
    parser.add_argument('--destination',
                        help='Path to the destination folder.',
                        type=str,
                        default="4.1-test_synth_inserted/")

    # Load arguments

    args = parser.parse_args()

    main(epoch=args.op, origin=args.origin,
         destination=args.destination)
