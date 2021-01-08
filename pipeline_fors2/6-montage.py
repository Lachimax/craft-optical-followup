# Code by Lachlan Marnoch, 2019

import os
import numpy as np

import craftutils.params as p
from craftutils.photometry import gain_median_combine
import craftutils.fits_files as ff


def main(data_dir, data_title, destination, fil):

    print("\nExecuting Python script pipeline_fors2/6-montage.py, with:")
    print(f"\tepoch {data_title}")
    print(f"\tdata directory {data_dir}")
    print(f"\tdestination directory {destination}")
    print(f"\tfilter {fil}")
    print()

    table = ff.fits_table(destination + '/' + fil, science_only=False)

    fil = fil.replace('/', '')

    header_file = destination + '/' + fil[0] + '_template.hdr'

    header_stream = open(header_file, 'r')
    header = header_stream.readlines()
    header_stream.close()

    # Transfer and/or modify header information

    params = p.load_params(data_dir + '/output_values')
    airmass = table['airmass'].mean()
    saturate = table['saturate'].mean()
    obj = table['object'][0]
    old_gain = table['gain'].mean()
    n_frames = params[fil[0] + '_n_exposures']
    gain = gain_median_combine(old_gain=old_gain, n_frames=n_frames)

    header.insert(-1, f'AIRMASS = {airmass}\n')
    header.insert(-1, f'FILTER  = {fil}\n')
    header.insert(-1, f'OBJECT  = {obj}\n')
    header.insert(-1, f'EXPTIME = 1.\n')
    header.insert(-1, f'GAIN    = {gain}\n')
    header.insert(-1, f'SATURATE= {saturate}\n')
    header.insert(-1, f'MJD-OBS = {float(np.nanmean(table["mjd_obs"]))}\n')

    os.remove(header_file)

    with open(header_file, 'w') as file:
        file.writelines(header)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Inject header information while doing Montage combine.")
    parser.add_argument('--directory', help='Main data directory(probably starts with "MJD"')
    parser.add_argument('-op', help='Name of object parameter file without .yaml, eg FRB180924_1')
    parser.add_argument('--destination', help='Path to the Montage coaddition folder')
    parser.add_argument('--filter', help='Filter name, eg g_HIGH')
    parser.add_argument('--object', help='Object name, eg FRB-181112--Host')

    # Load arguments

    args = parser.parse_args()

    main(data_dir=args.directory, data_title=args.op, destination=args.destination, fil=args.filter)
