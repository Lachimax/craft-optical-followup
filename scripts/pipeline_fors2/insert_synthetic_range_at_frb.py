# Code by Lachlan Marnoch, 2019
from astropy.io import fits
import shutil
import astropy.time as time
import numpy as np

from PyCRAFT import params as p
from PyCRAFT import utils as u
from PyCRAFT import photometry as ph


def main(obj, test, mag_min, mag_max, increment, instrument):
    properties = p.object_params_instrument(obj, instrument=instrument)
    output = p.object_output_params(obj=obj, instrument=instrument)
    paths = p.object_output_paths(obj=obj, instrument=instrument)

    synth_path = properties['data_dir'] + 'synthetic/'
    u.mkdir_check(synth_path)
    synth_path = synth_path + 'frb_position/'
    u.mkdir_check(synth_path)

    now = time.Time.now()
    now.format = 'isot'
    synth_path += f'range_{now}/'
    u.mkdir_check(synth_path)

    filters = output['filters']

    for magnitude in np.arange(mag_min, mag_max, increment):
        magnitudes = []
        for i in range(len(filters)):
            magnitudes.append(magnitude)
        test_spec = test + '_' + str(u.round_to_sig_fig(magnitude, 4))
        test_path = synth_path + test_spec + '/'
        ph.insert_synthetic_at_frb(obj=properties, test_path=test_path, filters=filters, magnitudes=magnitudes, add_path=False,
                                   psf=True, output_properties=output, instrument=instrument, paths=paths)

    p.add_output_path(obj=obj, key='subtraction_image_synth_frb_range', path=synth_path, instrument=instrument)


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
    parser.add_argument('--mag_min',
                        help='Brightest magnitude to insert.',
                        type=float,
                        default=20.)
    parser.add_argument('--mag_max',
                        help='Faintest magnitude to insert.',
                        type=float,
                        default=30.)
    parser.add_argument('--increment',
                        help='Increment between magnitudes.',
                        type=float,
                        default=0.1)
    parser.add_argument('--instrument',
                        help='Name of instrument.',
                        default='FORS2',
                        type=str)

    args = parser.parse_args()

    main(obj=args.op, test=args.test, mag_min=args.mag_min, mag_max=args.mag_max,
         increment=args.increment, instrument=args.instrument)
