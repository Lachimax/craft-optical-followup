# Code by Lachlan Marnoch, 2019
from craftutils import params as p


def main(obj, key, path, instrument):
    p.add_output_path(obj=obj, key=key, path=path, instrument=instrument)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Add a path to this dataset's output_paths.yaml")

    parser.add_argument('--op',
                        help='Name of object parameter file without .yaml, eg FRB180924_1',
                        type=str)
    parser.add_argument('--instrument',
                        help='Instrument on which data was taken.',
                        type=str,
                        default='FORS2')
    parser.add_argument('--key',
                        help='Key of path to add.',
                        type=str)
    parser.add_argument('--path',
                        help='Path to add.',
                        type=str)

    args = parser.parse_args()

    main(obj=args.op, key=args.key, path=args.path, instrument=args.instrument)
