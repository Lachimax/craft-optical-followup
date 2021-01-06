# Code by Lachlan Marnoch, 2019

import craftutils.fits_files as ff
from craftutils import params as p


def main(obj, input, output, path, prefix):
    ff.trim_file(path=input, bottom=0, top=950, left=195, right=1825, new_path=output)
    airmass = ff.get_airmass(input)
    exptime = ff.get_exp_time(input)

    p.add_params(path + '/output_values',
                 {prefix + '_airmass': airmass, prefix + '_airmass_err': 0, prefix + '_exptime': exptime})
    p.add_output_path(obj=obj, instrument='FORS2', key=prefix + '_std_image', path=output)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Do some initial sorting before reducing standard images.')
    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        type=str)
    parser.add_argument('--input',
                        help='Path to the image to trim.')
    parser.add_argument('--output',
                        help='Path to save the output image to.')
    parser.add_argument('--path',
                        help='Path of main standard star folder.')
    parser.add_argument('--prefix',
                        help='Filter prefix to attach to variable names.',
                        default='')

    args = parser.parse_args()
    main(obj=args.op, input=args.input, output=args.output, path=args.path, prefix=args.prefix)
