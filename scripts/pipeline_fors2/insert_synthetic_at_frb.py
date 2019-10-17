# Code by Lachlan Marnoch, 2019
from astropy.io import fits
import shutil
import astropy.time as time
import numpy as np

from PyCRAFT import params as p
from PyCRAFT import utils as u
from PyCRAFT import photometry as ph


def main(obj, test, magnitude):
    print('Got here.')
    properties = p.object_params_fors2(obj)

    synth_path = properties['data_dir'] + 'synthetic/'
    u.mkdir_check(synth_path)
    synth_path = synth_path + 'frb_position/'
    u.mkdir_check(synth_path)

    now = time.Time.now()
    now.format = 'isot'
    test = str(now) + '_' + test
    test_path = synth_path + test + '/'

    print(properties['filters'])

    ph.insert_synthetic_at_frb(obj=obj, test_path=test_path, filters=properties['filters'],
                               magnitudes=[magnitude, magnitude], add_path=True)


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
    parser.add_argument('--magnitude',
                        help='Magnitude to insert.',
                        type=float,
                        default=20.)

    args = parser.parse_args()

    main(obj=args.op, test=args.test, magnitude=args.magnitude)
