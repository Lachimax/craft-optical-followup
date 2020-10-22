# Code by Lachlan Marnoch, 2019

import PyCRAFT.fits_files as ff
import PyCRAFT.params as p
import PyCRAFT.utils as u

import shutil
import numpy as np
import os


def main(output_dir: 'str', data_title: 'str'):
    output_dir = u.check_trailing_slash(output_dir)
    data_dir = output_dir + "/0-data_with_raw_calibs/"

    # Write tables of fits files to main directory; firstly, science images only:
    table = ff.fits_table(input_path=data_dir,
                          output_path=output_dir + data_title + "_fits_table_science.csv",
                          science_only=True)
    # Then including all calibration files
    table_full = ff.fits_table(input_path=data_dir,
                               output_path=output_dir + data_title + "_fits_table_all.csv",
                               science_only=False)

    # Clear output files for fresh start.
    u.rm_check(output_dir + '/output_values.yaml')
    u.rm_check(output_dir + '/output_values.json')

    # Collect list of filters used:
    filters = []
    columns = []

    for j in [1, 2, 3, 4, 5]:
        column = 'filter' + str(j)
        for name in table[column]:
            if name != 'free':

                if name not in filters:
                    filters.append(name)
                    columns.append(column)

    # Collect pointings of standard-star observations.
    std_ras = []
    std_decs = []
    std_pointings = []
    # TODO: This is a horrible way to do this. Take the time to find a better one.
    for ra in table_full[table_full['object'] == 'STD']['ref_ra']:
        if ra not in std_ras:
            std_ras.append(ra)
    for dec in table_full[table_full['object'] == 'STD']['ref_dec']:
        if dec not in std_decs:
            std_decs.append(dec)

    for i, ra in enumerate(std_ras):
        std_pointings.append(f'RA{ra}_DEC{std_decs[i]}')

    print(std_ras)
    print(std_decs)
    print(std_pointings)

    # Collect and save some stats on those filters:
    param_dict = {}
    exp_times = []
    ns_exposures = []

    param_dict['filters'] = filters

    for i, f in enumerate(filters):
        f_0 = f[0]
        exp_time = table['exp_time'][table[columns[i]] == f]
        exp_times.append(exp_time)

        airmass_col = table['airmass'][table[columns[i]] == f]
        n_frames = sum(table[columns[i]] == f)
        n_exposures = n_frames / 2
        ns_exposures.append(n_exposures)

        airmass = float(np.nanmean(airmass_col))

        param_dict[f_0 + '_exp_time_mean'] = float(np.nanmean(exp_time))
        param_dict[f_0 + '_exp_time_err'] = float(2 * np.nanstd(exp_time))
        param_dict[f_0 + '_airmass_mean'] = airmass
        param_dict[f_0 + '_airmass_err'] = float(max(np.nanmax(airmass_col) - airmass, airmass - np.nanmin(airmass_col)))
        param_dict[f_0 + '_n_frames'] = float(n_frames)
        param_dict[f_0 + '_n_exposures'] = float(n_exposures)
        param_dict[f_0 + '_mjd_obs'] = float(np.nanmean(table['mjd_obs'][table[columns[i]] == f]))

        std_filter_dir = f'{output_dir}calibration/std_star/{f}/'
        u.mkdir_check(std_filter_dir)
        print(f'Copying {f} calibration data to std_star folder...')

        # Sort the STD files by filter, and within that by pointing.
        for j, ra in enumerate(std_ras):
            at_pointing = False
            pointing = std_pointings[j]
            pointing_dir = std_filter_dir + pointing + '/'
            for file in \
                    table_full[
                        (table_full['object'] == 'STD') &
                        (table_full['ref_ra'] == ra) &
                        (table_full[columns[i]] == f)]['identifier']:
                at_pointing = True
                u.mkdir_check(pointing_dir)
                shutil.copyfile(data_dir + file, pointing_dir + file)
            if at_pointing:
                for file in table_full[table_full['object'] == 'BIAS']['identifier']:
                    shutil.copyfile(data_dir + file, pointing_dir + file)
                for file in table_full[(table_full['object'] == 'FLAT,SKY') & (table_full[columns[i]] == f)][
                    'identifier']:
                    shutil.copyfile(data_dir + file, pointing_dir + file)

    p.add_params(output_dir + '/output_values', param_dict)

    # Copy calibration data (including standard-field images) to calibration folder


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Perform some initial setup on data directory.')
    parser.add_argument('--output', help='High-level data path to operate on, probably starting with "MJD"')
    parser.add_argument('-op', help='Name of object parameter file without .yaml, eg FRB180924_1')
    args = parser.parse_args()
    main(output_dir=args.output, data_title=args.op)
