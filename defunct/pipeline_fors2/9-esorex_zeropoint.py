# Code by Lachlan Marnoch, 2019

import craftutils.fits_files as ff
from craftutils.observation.image import fits_table


def main(path, cat_path):

    print("\nExecuting Python script pipeline_fors2/9-esorex_zeropoint.py, with:")
    print(f"\tcalibration data path {path}")
    print(f"\tESO calibration path {cat_path}")
    print()

    print('ESO CALIB DIR:', cat_path)
    raw_path = path + '/0-data_with_raw_calibs/'
    table_path = raw_path + 'fits_table.csv'

    fits_table(input_path=raw_path, output_path=table_path, science_only=False)

    # Write bias sof files.
    ff.write_sof(table_path=table_path,
                 output_path=raw_path + 'bias_up.sof',
                 sof_type='fors_bias', chip=1)
    ff.write_sof(table_path=table_path,
                 output_path=raw_path + 'bias_down.sof',
                 sof_type='fors_bias', chip=2)
    # Write sky flats sof files
    ff.write_sof(table_path=table_path,
                 output_path=raw_path + 'flats_up.sof',
                 sof_type='fors_img_sky_flat', chip=1)
    ff.write_sof(table_path=table_path,
                 output_path=raw_path + 'flats_down.sof',
                 sof_type='fors_img_sky_flat', chip=2)
    # Write zeropoint sof file
    ff.write_sof(table_path=table_path,
                 output_path=raw_path + 'zp_up.sof',
                 sof_type='fors_zeropoint', chip=1, cat_path=cat_path)
    ff.write_sof(table_path=table_path,
                 output_path=raw_path + 'zp_down.sof',
                 sof_type='fors_zeropoint', chip=2, cat_path=cat_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Write the necessary .sof files for esorex to reduce the standard '
                                                 'star data.')
    parser.add_argument('--directory',
                        help='Directory of standard star and calibration data.')
    parser.add_argument('--eso_calib_dir',
                        help='Directory containing the necessary ESO catalogues, '
                             'likely <ESOReflex>/install/calib/fors-5.3.32/cal/')

    args = parser.parse_args()
    main(path=args.directory, cat_path=args.eso_calib_dir)
