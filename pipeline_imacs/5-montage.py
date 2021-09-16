# Code by Lachlan Marnoch, 2019
import os
import numpy as np
import craftutils.utils as u
import craftutils.params as p
from craftutils.photometry import gain_median_combine
import craftutils.fits_files as ff


def main(data_dir, data_title, destination, fil, object):
    table = (destination + '/' + fil, science_only=False)

    fil = fil.replace('/', '')

    header_file = destination + '/' + fil + '_template.hdr'

    header_stream = open(header_file, 'r')
    header = header_stream.readlines()
    header_stream.close()

    # Transfer and/or modify header information

    params = p.object_output_params(data_title, instrument='IMACS')
    airmass = table['airmass'].mean()
    saturate = table['saturate'].mean()
    old_gain = table['gain'].mean()
    n_frames = params[fil + '_n_exposures']
    gain = gain_median_combine(old_gain=old_gain, n_frames=n_frames)

    header.insert(-1, f'AIRMASS = {airmass}\n')
    header.insert(-1, f'FILTER  = {fil}\n')
    header.insert(-1, f'OBJECT  = {object}\n')
    header.insert(-1, f'EXPTIME = 1.\n')
    header.insert(-1, f'GAIN    = {gain}\n')
    header.insert(-1, f'SATURATE= {saturate}\n')
    header.insert(-1, f'MJD-OBS = {float(np.nanmean(table["mjd_obs"]))}\n')

    os.remove(header_file)

    with open(header_file, 'w') as file:
        file.writelines(header)

    p.add_output_path(obj=data_title, key=fil + '_subtraction_image',
                      path=destination + '/' + fil + '_coadded.fits', instrument='IMACS')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Inject header information while doing Montage combine.")
    parser.add_argument('--directory', help='Main data directory(probably starts with "MJD"')
    parser.add_argument('-op', help='Name of object parameter file without .yaml, eg FRB180924_1', type=str)
    parser.add_argument('--origin', help='Path to the folder from which to copy individual files', type=str)
    parser.add_argument('--destination', help='Path to the Montage coaddition folder', type=str)
    parser.add_argument('--filter', help='Filter name, eg g_HIGH')
    parser.add_argument('--object', help='Object name, eg FRB-181112--Host')

    # Load arguments

    args = parser.parse_args()

    main(data_dir=args.directory, data_title=args.op, destination=args.destination, fil=args.filter,
         object=args.object)
