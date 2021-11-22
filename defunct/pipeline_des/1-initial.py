# Code by Lachlan Marnoch, 2019

import os
import shutil

from craftutils import params as p
from craftutils import utils as u


def main(data_title: str):
    properties = p.object_params_des(data_title)
    path = properties['data_dir']

    data_path = path + '/0-data/'

    u.mkdir_check(data_path)
    os.listdir(path)

    print(path)

    for file in filter(lambda fil: fil[-5:] == '.fits', os.listdir(path)):
        f = file[-6]
        shutil.copy(path + file, data_path + str(f) + '_cutout.fits')

        p.add_output_path(obj=data_title, instrument='DES', key=f + '_subtraction_image', path=data_path + str(f) + '_cutout.fits')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Reduce raw DES data.')

    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        type=str)

    args = parser.parse_args()

    main(data_title=args.op)
