# Code by Lachlan Marnoch, 2019

import os
import PyCRAFT.fits_files as ff


def main(data_dir: 'str'):
    directories = next(os.walk(data_dir))[1]

    for d in directories:
        ff.sort_by_filter(data_dir + d)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='After reduction with ESOReflex, sorts fits files by their filters.')
    parser.add_argument('--directory', help='Path to sort.')
    args = parser.parse_args()
    main(data_dir=args.directory)
