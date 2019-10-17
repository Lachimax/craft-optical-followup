# Code by Lachlan Marnoch, 2019
from astropy.io import fits
import shutil
import astropy.time as time
from astropy import table
import numpy as np
import os

from PyCRAFT import params as p
from PyCRAFT import utils as u
from PyCRAFT import photometry as ph


def main(obj, test, curves_path):
    properties = p.object_params_fors2(obj)

    if curves_path[-1] != '/':
        curves_path += '/'

    mag_table_file = filter(lambda file: file[-4:] == '.csv', os.listdir(curves_path)).__next__()
    mag_table = table.Table.read(curves_path + mag_table_file)

    filters = mag_table.colnames.copy()
    filters.remove('model')

    synth_path = properties['data_dir'] + 'synthetic/'
    u.mkdir_check(synth_path)
    synth_path = synth_path + 'frb_position/'
    u.mkdir_check(synth_path)

    now = time.Time.now()
    now.format = 'isot'
    synth_path += f'sn_models_{now}/'
    u.mkdir_check(synth_path)

    for row in mag_table:
        model = row['model']
        magnitudes = []
        for f in filters:
            magnitudes.append(row[f])
        test_spec = test + '_' + model
        test_path = synth_path + test_spec + '/'
        ph.insert_synthetic_at_frb(obj=obj, test_path=test_path, filters=filters, magnitudes=magnitudes, add_path=False)

    for f in filters:
        p.add_output_path(obj=obj, key=f[0] + '_subtraction_image_synth_frb_sn_models', path=synth_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        type=str)
    parser.add_argument('--test',
                        help='Name of test.',
                        type=str,
                        default='frb_position')
    parser.add_argument('--curves_path',
                        help='Path to folder containing light curve data.',
                        type=str,
                        default='/home/lachlan/Data/FRB180924/sn_light_curves/FRB180924_ebvhost_0.0/')

    args = parser.parse_args()

    main(obj=args.op, test=args.test, curves_path=args.curves_path)
