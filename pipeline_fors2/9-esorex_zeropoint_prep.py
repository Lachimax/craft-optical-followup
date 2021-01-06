# Code by Lachlan Marnoch, 2019

from craftutils.params import object_output_params
from craftutils.utils import mkdir_check


def main(ob, path):
    output = object_output_params(obj=ob, instrument='FORS2')
    filters = output['filters']
    for f in filters:
        mkdir_check(path + '/' + f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Do some initial sorting before reducing standard images.')
    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        type=str)
    parser.add_argument('--directory',
                        help='Directory of standard star and calibration data.')

    args = parser.parse_args()
    main(ob=args.op, path=args.directory)
